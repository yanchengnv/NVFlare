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
from abc import abstractmethod

from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey, ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.cw_utils import Constant, StatusReport, execution_failure
from nvflare.security.logging import secure_format_traceback


class _LearnTask:
    def __init__(self, task_name: str, task_data: Shareable, fl_ctx: FLContext):
        self.task_name = task_name
        self.task_data = task_data
        self.fl_ctx = fl_ctx
        self.abort_signal = Signal()


class ClientWorkflowExecutor(Executor):
    def __init__(
        self,
        start_task_name=Constant.TASK_NAME_START,
        configure_task_name=Constant.TASK_NAME_START,
        submit_result_task_name=Constant.TASK_NAME_SUBMIT_RESULT,
        learn_task_name=AppConstants.TASK_TRAIN,
        max_status_report_interval: float = 600.0,
        learn_task_check_interval: float = 1.0,
        learn_task_send_timeout: float = 10.0,
        learn_task_abort_timeout: float = 5.0,
        report_status_check_interval: float = 0.5,
        allow_busy_task: bool = False,
    ):
        """
        Constructor of a CWE object.

        Args:
            start_task_name: task name for starting the workflow
            configure_task_name: task name for getting workflow config properties
            submit_result_task_name: task name for submitting the final result
            learn_task_name: name for the Learning Task (LT)
            max_status_report_interval: max interval between status reports to the server
            learn_task_check_interval: interval for checking incoming Learning Task (LT)
            learn_task_send_timeout: timeout for sending the LT to other client(s)
            learn_task_abort_timeout: time to wait for the LT to become stopped after aborting it
            report_status_check_interval: interval for checking and sending status report
            allow_busy_task:
        """
        super().__init__()
        self.start_task_name = start_task_name
        self.configure_task_name = configure_task_name
        self.submit_result_task_name = submit_result_task_name
        self.learn_task_name = learn_task_name
        self.max_status_report_interval = max_status_report_interval
        self.report_status_check_interval = report_status_check_interval
        self.learn_task_abort_timeout = learn_task_abort_timeout
        self.learn_task_check_interval = learn_task_check_interval
        self.learn_task_send_timeout = learn_task_send_timeout
        self.allow_busy_task = allow_busy_task
        self.status_report_thread = threading.Thread(target=self._check_status)
        self.status_report_thread.daemon = True
        self.current_status = StatusReport()
        self.learn_error = None
        self.last_status_report_time = time.time()  # time of last status report to server
        self.status_change_time = 0  # time of last status change
        self.config = None

        self.learn_thread = threading.Thread(target=self._do_learn)
        self.learn_thread.daemon = True
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
        self.best_result = None
        self.best_metric = None
        self.best_round = 0

    def get_config_prop(self, name: str, default=None):
        """
        Get a specified config property.

        Args:
            name: name of the property
            default: default value to return if the property is not defined.

        Returns:

        """
        if not self.config:
            return default
        return self.config.get(name, default)

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
            self.learn_executor = runner.task_table.get(self.learn_task_name)
            if not self.learn_executor:
                self.system_panic(f"no executor for task {self.learn_task_name}", fl_ctx)
                return

            self.engine.register_aux_message_handler(
                topic=Constant.TOPIC_LEARN, message_handle_func=self._process_learn_request
            )

            self.initialize(fl_ctx)

            self.log_info(fl_ctx, "Started learn thread")
            self.learn_thread.start()
            self.status_report_thread.start()
        elif event_type in [EventType.ABORT_TASK, EventType.END_RUN]:
            self.asked_to_stop = True
            task = self.learn_task
            if task:
                task.abort_signal.trigger(True)
            self.finalize(fl_ctx)

    def initialize(self, fl_ctx: FLContext):
        """Called to initialize the executor.

        Args:
            fl_ctx: The FL Context

        Returns: None

        """
        fl_ctx.set_prop(FLContextKey.EXECUTOR, self, private=True, sticky=False)
        self.fire_event(AppEventType.EXECUTOR_INITIALIZED, fl_ctx)

    def finalize(self, fl_ctx: FLContext):
        """Called to finalize the executor.

        Args:
            fl_ctx: the FL Context

        Returns: None

        """
        fl_ctx.set_prop(FLContextKey.EXECUTOR, self, private=True, sticky=False)
        self.fire_event(AppEventType.EXECUTOR_FINALIZED, fl_ctx)

    def process_config(self):
        """This is called to allow the subclass to process config props.

        Returns: None

        """
        pass

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name == self.configure_task_name:
            self.config = shareable[Constant.CONFIG]
            self.process_config()
            return make_reply(ReturnCode.OK)

        if task_name == self.start_task_name:
            self.is_starting_client = True
            return self.start_workflow(shareable, fl_ctx, abort_signal)

        if task_name == self.submit_result_task_name:
            self.log_info(fl_ctx, "Submitting my result")
            result = self.prepare_final_result(fl_ctx)
            if not result:
                self.log_error(fl_ctx, "got request to submit result but I have no result to submit!")
                return make_reply(ReturnCode.BAD_REQUEST_DATA)
            result.set_return_code(ReturnCode.OK)
            return result
        else:
            self.log_error(fl_ctx, f"Could not handle task: {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

    def prepare_final_result(self, fl_ctx: FLContext) -> Shareable:
        """
        This is called to allow the subclass to prepare the final result before submitting it to server.
        If the subclass does not overwrite this method, the current last_result will be returned.

        Args:
            fl_ctx: the FL Context

        Returns: a Shareable object to be returned as the final result.

        """
        return self.last_result

    @abstractmethod
    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        """
        This is called for the subclass to start the workflow.
        This only happens on the starting_client.

        Args:
            shareable: the initial task data (e.g. initial model weights)
            fl_ctx: FL context
            abort_signal: abort signal for task execution

        Returns:

        """
        pass

    def _check_status(self):
        while not self.asked_to_stop:
            must_report = False
            now = time.time()
            if self.status_change_time > self.last_status_report_time:
                # status has changed since last reported:
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
                        topic = Constant.TOPIC_FAILURE
                    else:
                        request = self.current_status.to_shareable()
                        topic = Constant.TOPIC_REPORT_STATUS

                    self.log_info(fl_ctx, f"send status report: {request}")
                    resp = self.engine.send_aux_request(
                        targets=[], topic=topic, request=request, timeout=2.0, fl_ctx=fl_ctx
                    )
                    reply = resp.get("server")
                    if reply and isinstance(reply, Shareable) and reply.get_return_code() == ReturnCode.OK:
                        self.last_status_report_time = now
            time.sleep(self.report_status_check_interval)

    def set_learn_task(self, task_data: Shareable, fl_ctx: FLContext) -> bool:
        task_data.set_header(AppConstants.NUM_ROUNDS, self.get_config_prop(AppConstants.NUM_ROUNDS))
        task = _LearnTask(self.learn_task_name, task_data, fl_ctx)
        current_task = self.learn_task
        if not current_task:
            self.learn_task = task
            return True

        if not self.allow_busy_task:
            return False

        # already has a task!
        self.log_warning(fl_ctx, "already running a task: aborting it")
        assert isinstance(current_task, _LearnTask)
        current_task.abort_signal.trigger(True)
        fl_ctx.set_prop(FLContextKey.TASK_NAME, current_task.task_name)
        self.fire_event(EventType.ABORT_TASK, fl_ctx)

        # monitor until the task is done
        start = time.time()
        while self.learn_task:
            if time.time() - start > self.learn_task_abort_timeout:
                self.log_error(fl_ctx, f"failed to stop the running task after {self.learn_task_abort_timeout} seconds")
                return False
            time.sleep(0.1)

        self.learn_task = task
        return True

    def _do_learn(self):
        while not self.asked_to_stop:
            if self.learn_task:
                t = self.learn_task
                assert isinstance(t, _LearnTask)
                self.logger.info("Got a Learn task")
                try:
                    self.do_learn_task(t.task_name, t.task_data, t.fl_ctx, t.abort_signal)
                except:
                    self.logger.log(f"exception from do_learn_task: {secure_format_traceback()}")
                self.learn_task = None
            time.sleep(self.learn_task_check_interval)

    def update_status(self, status: StatusReport, timestamp: float):
        status.best_metric = self.best_metric
        self.current_status = status
        self.status_change_time = timestamp

    def set_error(self, err: str):
        self.learn_error = err
        self.status_change_time = time.time()

    @abstractmethod
    def do_learn_task(self, name: str, data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        """This is called to do a Learn Task.
        Subclass must implement this method.

        Args:
            name: task name
            data: task data
            fl_ctx: FL context of the task
            abort_signal: abort signal for the task execution

        Returns:

        """
        pass

    def _process_learn_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        try:
            return self._try_process_learn_request(topic, request, fl_ctx)
        except Exception as ex:
            self.log_exception(fl_ctx, f"exception: {ex}")
            self.set_error(ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _try_process_learn_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        sender = peer_ctx.get_identity_name()

        # process request from prev client
        self.log_info(fl_ctx, f"Got Learn request from {sender}")

        if self.learn_task and not self.allow_busy_task:
            # should never happen!
            self.log_error(fl_ctx, f"got Learn request from {sender} while I'm still busy!")
            self.set_error(ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, f"accepted learn request from {sender}")
        self.set_learn_task(task_data=request, fl_ctx=fl_ctx)
        return make_reply(ReturnCode.OK)

    def send_learn_task(self, targets: list, request: Shareable, fl_ctx: FLContext) -> bool:
        self.log_info(fl_ctx, f"sending learn task to clients {targets}")
        request.set_header(AppConstants.NUM_ROUNDS, self.get_config_prop(AppConstants.NUM_ROUNDS))

        resp = self.engine.send_aux_request(
            targets=targets,
            topic=Constant.TOPIC_LEARN,
            request=request,
            timeout=self.learn_task_send_timeout,
            fl_ctx=fl_ctx,
        )

        assert isinstance(resp, dict)
        for t in targets:
            reply = resp.get(t)
            if not isinstance(reply, Shareable):
                self.log_error(fl_ctx, f"failed to send learn request to client {t}")
                self.log_error(fl_ctx, f"reply must be Shareable but got {type(reply)}")
                self.set_error(ReturnCode.EXECUTION_EXCEPTION)
                return False

            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"bad response for learn request from client {t}: {rc}")
                self.set_error(rc)
                return False
        return True

    def execute_train(self, data: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        current_round = data.get_header(AppConstants.CURRENT_ROUND)

        self.log_info(fl_ctx, f"started training round {current_round}")
        try:
            result = self.learn_executor.execute(self.learn_task_name, data, fl_ctx, abort_signal)
        except:
            self.log_exception(fl_ctx, f"trainer exception: {secure_format_traceback()}")
            result = make_reply(ReturnCode.EXECUTION_EXCEPTION)
        self.log_info(fl_ctx, f"finished training round {current_round}")

        # make sure to include cookies in result
        cookie_jar = data.get_cookie_jar()
        result.set_cookie_jar(cookie_jar)
        result.set_header(AppConstants.CURRENT_ROUND, current_round)
        result.add_cookie(AppConstants.CONTRIBUTION_ROUND, current_round)  # to make model selector happy
        return result
