# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import threading
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.rr_utils import RRConstant, StatusReport, execution_failure


class _LearnTask:
    def __init__(self, task_name: str, task_data: Shareable, fl_ctx: FLContext):
        self.task_name = task_name
        self.task_data = task_data
        self.fl_ctx = fl_ctx
        self.abort_signal = Signal()


class CyclicExecutor(Executor):
    def __init__(
        self,
        rr_task_name=RRConstant.TASK_NAME_RR,
        submit_result_task_name=RRConstant.TASK_NAME_SUBMIT_RESULT,
        train_task_name=AppConstants.TASK_TRAIN,
        max_status_report_interval: float = 600.0,
        task_check_interval: float = 1.0,
    ):
        super().__init__()
        self.rr_task_name = rr_task_name
        self.submit_result_task_name = submit_result_task_name
        self.train_task_name = train_task_name

        self.max_status_report_interval = max_status_report_interval
        self.status_check_interval = 1.0  # for internal check
        self.status_report_thread = threading.Thread(target=self._check_status)
        self.status_report_thread.daemon = True
        self.current_status = StatusReport()
        self.learn_error = None
        self.last_status_report_time = 0  # time of last status report to server
        self.status_change_time = 0  # time of last status change

        self.learn_thread = threading.Thread(target=self._do_learn)
        self.learn_thread.daemon = True
        self.task_check_interval = task_check_interval
        self.learn_task = None
        self.current_task = None
        self.learn_executor = None
        self.learn_lock = threading.Lock()
        self.asked_to_stop = False
        self.status_lock = threading.Lock()
        self.engine = None
        self.me = None
        self.is_starting_client = False
        self.last_result = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            self.engine = fl_ctx.get_engine()
            if not self.engine:
                self.system_panic("no engine", fl_ctx)
                return

            runner = fl_ctx.get_prop(FLContextKey.RUNNER)
            if not runner:
                self.system_panic("no client runner", fl_ctx)
                return

            self.me = fl_ctx.get_identity_name()
            self.learn_executor = runner.task_table.get(self.train_task_name)
            if not self.learn_executor:
                self.system_panic(f"no executor for task {self.train_task_name}", fl_ctx)
                return

            self.engine.register_aux_message_handler(
                topic=RRConstant.TOPIC_LEARN, message_handle_func=self._process_learn_request
            )

            self.log_info(fl_ctx, "Started learn thread")
            self.learn_thread.start()
            self.status_report_thread.start()
        elif event_type in [EventType.ABORT_TASK, EventType.END_RUN]:
            self.asked_to_stop = True
            task = self.learn_task
            if task:
                task.abort_signal.trigger(True)

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self.rr_task_name:
            self.log_info(fl_ctx, "Starting RR!")
            self.is_starting_client = True
            return self._start_rr(shareable, fl_ctx)
        elif task_name == self.submit_result_task_name:
            self.log_info(fl_ctx, "Submitting my result")
            if not self.last_result:
                self.log_error(fl_ctx, "got request to submit result but I have no result to submit!")
                return make_reply(ReturnCode.BAD_REQUEST_DATA)
            return self.last_result
        else:
            self.log_error(fl_ctx, f"Could not handle task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def _check_status(self):
        while not self.asked_to_stop:
            must_report = False
            now = time.time()
            if self.last_status_report_time == 0 or self.status_change_time > self.last_status_report_time:
                # never reported or status has changed since last reported:
                must_report = True
            elif now - self.last_status_report_time > self.max_status_report_interval:
                # too long after last report time. report again even if no status change to keep alive
                must_report = True

            if must_report:
                # do status report
                with self.engine.new_context() as fl_ctx:
                    if self.learn_error:
                        # report error!
                        request = execution_failure(self.learn_error)
                        topic = RRConstant.TOPIC_FAILURE
                    else:
                        request = self.current_status.to_shareable()
                        topic = RRConstant.TOPIC_REPORT_STATUS

                    self.log_info(fl_ctx, f"send status report: {request}")
                    resp = self.engine.send_aux_request(
                        targets=[], topic=topic, request=request, timeout=2.0, fl_ctx=fl_ctx
                    )
                    reply = resp.get("server")
                    if reply and isinstance(reply, Shareable) and reply.get_return_code() == ReturnCode.OK:
                        self.last_status_report_time = now
            time.sleep(self.status_check_interval)

    def _start_rr(self, shareable: Shareable, fl_ctx: FLContext):
        shareable.set_header(AppConstants.CURRENT_ROUND, 1)
        self.learn_task = _LearnTask(task_name=self.train_task_name, task_data=shareable, fl_ctx=fl_ctx)
        return make_reply(ReturnCode.OK)

    def _do_learn(self):
        while not self.asked_to_stop:
            if self.learn_task:
                self.logger.info("Got a Learn task")
                self._do_task(self.learn_task)
                self.learn_task = None
            time.sleep(self.task_check_interval)

    def _do_task(self, task: _LearnTask):
        task_data = task.task_data
        if not isinstance(task_data, Shareable):
            raise ValueError(f"task data must be Shareable but got {type(task_data)}")

        fl_ctx = task.fl_ctx
        assert isinstance(fl_ctx, FLContext)

        # set status report of starting task
        current_round = task_data.get_header(AppConstants.CURRENT_ROUND)
        start_time = time.time()
        self.current_status = StatusReport(last_round=current_round, start_time=start_time)
        self.status_change_time = start_time

        # execute the task
        self.log_info(fl_ctx, f"executing round {current_round}")

        result = self.learn_executor.execute(task.task_name, task_data, fl_ctx, task.abort_signal)

        self.log_info(fl_ctx, f"finished round {current_round}")

        assert isinstance(result, Shareable)
        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"learn executor failed: {rc}")
            self._set_learn_error(rc)
            return

        self.last_result = result

        # see whether we need to send to next leg
        end_time = time.time()
        num_rounds = task_data.get_header(AppConstants.NUM_ROUNDS)
        current_round = task_data.get_header(AppConstants.CURRENT_ROUND)
        clients = task_data.get_header(RRConstant.CLIENTS)
        self.log_info(fl_ctx, f"RR CLIENTS: {clients}")

        result.set_header(AppConstants.NUM_ROUNDS, num_rounds)
        result.set_header(AppConstants.CURRENT_ROUND, current_round)
        result.set_header(RRConstant.CLIENTS, clients)

        assert isinstance(clients, list)
        my_idx = clients.index(self.me)
        if current_round == num_rounds:
            # am I the last leg?
            if my_idx == len(clients) - 1:
                # I'm the last leg - the RR is done!
                self.log_info(fl_ctx, "I'm the last leg - got final result")
                self.current_status = StatusReport(
                    last_round=current_round, start_time=start_time, end_time=end_time, final_result="final"
                )
                self.status_change_time = end_time
                return

        # update status
        self.current_status = StatusReport(
            last_round=current_round,
            start_time=start_time,
            end_time=end_time,
        )
        self.status_change_time = end_time

        # send to next leg
        if my_idx < len(clients) - 1:
            next_client = clients[my_idx + 1]
        else:
            next_client = clients[0]

        resp = self.engine.send_aux_request(
            targets=[next_client], topic=RRConstant.TOPIC_LEARN, request=result, timeout=2.0, fl_ctx=fl_ctx
        )

        assert isinstance(resp, dict)
        reply = resp.get(next_client)
        if not isinstance(reply, Shareable):
            self.log_error(fl_ctx, f"failed to send learn request to next client {next_client}")
            self.log_error(fl_ctx, f"reply must be Shareable but got {type(reply)}")
            self._set_learn_error(ReturnCode.EXECUTION_EXCEPTION)
            return

        rc = reply.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"failed to send learn request to next client {next_client}: {rc}")
            self._set_learn_error(rc)
            return

        self.log_info(fl_ctx, f"sent learn request to next client {next_client}")

    def _set_learn_error(self, err: str):
        self.learn_error = err
        self.status_change_time = time.time()

    def _process_learn_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        sender = peer_ctx.get_identity_name()

        # process request from prev client
        self.log_info(fl_ctx, f"got Learn request from {sender}")

        if self.learn_task:
            # should never happen!
            self.log_error(fl_ctx, f"got Learn request from {sender} while I'm still busy!")
            self._set_learn_error(ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        # Am I the starting client?
        if self.is_starting_client:
            # need to start the next round
            current_round = request.get_header(AppConstants.CURRENT_ROUND)
            num_rounds = request.get_header(AppConstants.NUM_ROUNDS)
            if current_round >= num_rounds:
                # should never happen!
                self.log_error(
                    fl_ctx,
                    f"logic error: current round {current_round} >= num rounds {num_rounds} for starting client!",
                )
                self._set_learn_error(ReturnCode.EXECUTION_EXCEPTION)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            # start next round
            self.log_info(fl_ctx, f"Starting new round {current_round+1}")
            request.set_header(AppConstants.CURRENT_ROUND, current_round + 1)

        self.log_info(fl_ctx, f"accepted learn request from {sender}")
        self.learn_task = _LearnTask(task_name=self.train_task_name, task_data=request, fl_ctx=fl_ctx)
        return make_reply(ReturnCode.OK)
