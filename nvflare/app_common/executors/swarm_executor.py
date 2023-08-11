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
import random
import threading
import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.aggregator import Aggregator
from nvflare.app_common.abstract.learnable import Learnable
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.app_event_type import AppEventType
from nvflare.app_common.utils.cw_utils import Constant, StatusReport
from nvflare.security.logging import secure_format_traceback

from .cwe import ClientWorkflowExecutor


class _TrainerStatus:
    def __init__(self, name: str):
        self.name = name
        self.reply_time = None


class Gatherer(FLComponent):
    def __init__(
        self,
        fl_ctx: FLContext,
        for_round: int,
        executor: ClientWorkflowExecutor,
        aggregator: Aggregator,
        all_clients: list,
        trainers: list,
        min_responses_required: int,
        wait_time_after_min_resps_received: float,
        timeout,
    ):
        FLComponent.__init__(self)
        self.fl_ctx = fl_ctx
        self.executor = executor
        self.aggregator = aggregator
        self.all_clients = all_clients
        self.trainers = trainers
        self.for_round = for_round
        self.trainer_statuses = {}
        self.start_time = time.time()
        self.timeout = timeout

        for t in trainers:
            self.trainer_statuses[t] = _TrainerStatus(t)
        if min_responses_required <= 0 or min_responses_required >= len(trainers):
            min_responses_required = len(trainers)
        self.min_responses_required = min_responses_required
        self.wait_time_after_min_resps_received = wait_time_after_min_resps_received
        self.min_resps_received_time = None
        self.lock = threading.Lock()

    def gather(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> Shareable:
        with self.lock:
            try:
                return self._do_gather(client_name, result, fl_ctx)
            except:
                self.log_error(fl_ctx, f"exception gathering: {secure_format_traceback()}")
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def _do_gather(self, client_name: str, result: Shareable, fl_ctx: FLContext) -> Shareable:
        result_round = result.get_header(AppConstants.CURRENT_ROUND)
        ts = self.trainer_statuses.get(client_name)
        if not ts:
            self.log_error(
                fl_ctx, f"received result from {client_name} for round {result_round}, but it is not a trainer"
            )
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if result_round > self.for_round:
            # this should never happen!
            # otherwise it means that the client is sending me result for a round that I couldn't possibly schedule!
            self.log_error(
                fl_ctx,
                f"logic error: received result from {client_name} for round {result_round}, "
                f"which is > gatherer's current round {self.for_round}",
            )
            self.executor.set_error(ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        if result_round < self.for_round:
            # this is a late result for a round that I scheduled in the past.
            # Note: we still accept it!
            self.log_warning(
                fl_ctx,
                f"received late result from {client_name} for round {result_round}, "
                f"which is < gatherer's current round {self.for_round}",
            )

        if result_round == self.for_round:
            # this is the result that I'm waiting for.
            now = time.time()
            ts.reply_time = now
            if not self.min_resps_received_time:
                # see how many responses I have received
                num_resps_received = 0
                for _, ts in self.trainer_statuses.items():
                    if ts.reply_time:
                        num_resps_received += 1
                if num_resps_received >= self.min_responses_required:
                    self.min_resps_received_time = now

        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"Bad result from {client_name} for round {result_round}: {rc}.")
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, self.for_round, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.TRAINING_RESULT, result, private=True, sticky=False)
        self.fire_event(AppEventType.BEFORE_CONTRIBUTION_ACCEPT, fl_ctx)

        accepted = self.aggregator.accept(result, fl_ctx)
        accepted_msg = "ACCEPTED" if accepted else "REJECTED"
        self.log_info(
            fl_ctx, f"Contribution from {client_name} {accepted_msg} by the aggregator at round {result_round}."
        )

        fl_ctx.set_prop(AppConstants.AGGREGATION_ACCEPTED, accepted, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_CONTRIBUTION_ACCEPT, fl_ctx)
        return make_reply(ReturnCode.OK)

    def aggregate(self):
        fl_ctx = self.fl_ctx
        self.log_info(fl_ctx, "Start aggregation.")
        self.fire_event(AppEventType.BEFORE_AGGREGATION, fl_ctx)
        aggr_result = self.aggregator.aggregate(fl_ctx)
        fl_ctx.set_prop(AppConstants.AGGREGATION_RESULT, aggr_result, private=True, sticky=False)
        self.fire_event(AppEventType.AFTER_AGGREGATION, fl_ctx)
        self.log_info(fl_ctx, "End aggregation.")
        return aggr_result

    def is_done(self):
        unfinished = 0
        for c, s in self.trainer_statuses.items():
            if not s.reply_time:
                unfinished += 1
        if unfinished == 0:
            return True

        # timeout?
        now = time.time()
        if self.timeout and now - self.start_time > self.timeout:
            return True

        if (
            self.min_resps_received_time
            and now - self.min_resps_received_time > self.wait_time_after_min_resps_received
        ):
            # received min responses required and waited for long time
            return True


class SwarmExecutor(ClientWorkflowExecutor):
    def __init__(
        self,
        start_task_name=Constant.TASK_NAME_START,
        configure_task_name=Constant.TASK_NAME_CONFIGURE,
        submit_result_task_name=Constant.TASK_NAME_SUBMIT_RESULT,
        learn_task_name=AppConstants.TASK_TRAIN,
        max_status_report_interval: float = 600.0,
        learn_task_check_interval: float = 1.0,
        learn_task_abort_timeout: float = 5.0,
        learn_task_send_timeout: float = 30.0,
        learn_task_timeout=None,
        min_responses_required: int = 1,
        wait_time_after_min_resps_received: float = 10.0,
        aggregator_id=AppConstants.DEFAULT_AGGREGATOR_ID,
        shareable_generator_id=AppConstants.DEFAULT_SHAREABLE_GENERATOR_ID,
    ):
        super().__init__(
            start_task_name=start_task_name,
            configure_task_name=configure_task_name,
            submit_result_task_name=submit_result_task_name,
            learn_task_name=learn_task_name,
            max_status_report_interval=max_status_report_interval,
            learn_task_check_interval=learn_task_check_interval,
            learn_task_send_timeout=learn_task_send_timeout,
            learn_task_abort_timeout=learn_task_abort_timeout,
            allow_busy_task=True,
        )
        self.learn_task_timeout = learn_task_timeout
        self.min_responses_required = min_responses_required
        self.wait_time_after_min_resps_received = wait_time_after_min_resps_received
        self.aggregator_id = aggregator_id
        self.shareable_generator_id = shareable_generator_id
        self.aggregator = None
        self.shareable_gen = None
        self.gatherer = None
        self.gatherer_waiter = threading.Event()
        self.trainers = None
        self.aggrs = None
        self.is_trainer = False
        self.is_aggr = False
        self.last_aggr_round_done = -1

    def process_config(self):
        all_clients = self.get_config_prop(Constant.CLIENTS)

        self.trainers = self.get_config_prop(Constant.TRAIN_CLIENTS)
        if not self.trainers:
            self.trainers = all_clients
        self.is_trainer = self.me in self.trainers

        self.aggrs = self.get_config_prop(Constant.AGGR_CLIENTS)
        if not self.aggrs:
            self.aggrs = all_clients
        self.is_aggr = self.me in self.aggrs

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        super().handle_event(event_type, fl_ctx)
        if event_type == EventType.START_RUN:
            self.aggregator = self.engine.get_component(self.aggregator_id)
            if not isinstance(self.aggregator, Aggregator):
                self.system_panic(
                    f"aggregator {self.aggregator_id} must be an Aggregator but got {type(self.aggregator)}",
                    fl_ctx,
                )
                return

            self.shareable_gen = self.engine.get_component(self.shareable_generator_id)
            if not isinstance(self.shareable_gen, ShareableGenerator):
                self.system_panic(
                    f"Shareable generator {self.shareable_generator_id} must be ShareableGenerator, "
                    f"but got {type(self.shareable_gen)}",
                    fl_ctx,
                )
                return

            self.engine.register_aux_message_handler(
                topic=Constant.TOPIC_RESULT, message_handle_func=self._process_learn_result
            )

            aggr_thread = threading.Thread(target=self._monitor_gather)
            aggr_thread.daemon = True
            aggr_thread.start()
            self.log_info(fl_ctx, "started aggregator thread")
        elif event_type == AppEventType.GLOBAL_BEST_MODEL_AVAILABLE:
            self.best_metric = fl_ctx.get_prop(AppConstants.VALIDATION_RESULT)
            self.best_result = fl_ctx.get_prop(AppConstants.GLOBAL_MODEL)
            self.log_info(fl_ctx, f"Received GLOBAL_BEST_MODEL_AVAILABLE: best metric={self.best_metric}")
            current_round = fl_ctx.get_prop(AppConstants.CURRENT_ROUND)
            self.best_round = current_round
            self.update_status(status=StatusReport(last_round=current_round), timestamp=time.time())

    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        clients = self.get_config_prop(Constant.CLIENTS)
        aggr_clients = self.get_config_prop(Constant.AGGR_CLIENTS, [])
        train_clients = self.get_config_prop(Constant.TRAIN_CLIENTS, [])

        self.log_info(
            fl_ctx, f"Starting Swarm Workflow on clients {clients}, aggrs {aggr_clients}, trainers {train_clients}"
        )

        if not self._scatter(
            task_data=shareable, for_round=self.get_config_prop(Constant.START_ROUND, 0), fl_ctx=fl_ctx
        ):
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

        self.log_info(fl_ctx, "Started Swarm Workflow")
        return make_reply(ReturnCode.OK)

    def _scatter(self, task_data: Shareable, for_round: int, fl_ctx: FLContext) -> bool:
        clients = self.get_config_prop(Constant.CLIENTS)
        aggr_clients = self.get_config_prop(Constant.AGGR_CLIENTS)

        # determine aggr client
        if not aggr_clients:
            aggr_clients = clients
        aggr = random.choice(aggr_clients)

        task_data.set_header(AppConstants.CURRENT_ROUND, for_round)
        task_data.add_cookie(AppConstants.CONTRIBUTION_ROUND, for_round)
        task_data.set_header(Constant.AGGREGATOR, aggr)

        self.log_info(fl_ctx, f"broadcasting learn task of round {for_round} to {clients}; aggr client is {aggr}")
        return self.send_learn_task(targets=clients, request=task_data, fl_ctx=fl_ctx)

    def _monitor_gather(self):
        while True:
            if self.asked_to_stop:
                return

            gatherer = self.gatherer
            if gatherer:
                assert isinstance(gatherer, Gatherer)
                if gatherer.is_done():
                    self.last_aggr_round_done = gatherer.for_round
                    self.gatherer = None
                    self.gatherer_waiter.clear()
                    try:
                        self._end_gather(gatherer)
                    except:
                        self.logger.error(f"exception ending gatherer: {secure_format_traceback()}")
            time.sleep(0.2)

    def _end_gather(self, gatherer: Gatherer):
        fl_ctx = gatherer.fl_ctx
        try:
            aggr_result = gatherer.aggregate()
        except:
            self.log_error(fl_ctx, f"exception in aggregation: {secure_format_traceback()}")
            self.set_error(ReturnCode.EXECUTION_EXCEPTION)
            return

        global_weights = self.shareable_gen.shareable_to_learnable(aggr_result, fl_ctx)
        self.last_result = global_weights
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, global_weights, private=True, sticky=True)

        # are we done with training?
        now = time.time()
        num_rounds_done = gatherer.for_round - self.get_config_prop(Constant.START_ROUND, 0) + 1
        if num_rounds_done >= self.get_config_prop(AppConstants.NUM_ROUNDS):
            self.log_info(fl_ctx, f"Swarm Learning Done: number of rounds completed {num_rounds_done}")
            self.update_status(
                status=StatusReport(
                    last_round=gatherer.for_round, start_time=gatherer.start_time, end_time=now, all_done=True
                ),
                timestamp=now,
            )
            return

        # continue next round
        next_round_data = self.shareable_gen.learnable_to_shareable(global_weights, fl_ctx)
        assert isinstance(next_round_data, Shareable)
        self._scatter(next_round_data, gatherer.for_round + 1, gatherer.fl_ctx)

    def _process_learn_result(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        try:
            peer_ctx = fl_ctx.get_peer_context()
            assert isinstance(peer_ctx, FLContext)
            client_name = peer_ctx.get_identity_name()
            current_round = request.get_header(AppConstants.CURRENT_ROUND)
            self.log_info(fl_ctx, f"got training result from {client_name} for round {current_round}")

            gatherer = self.gatherer
            if not gatherer:
                # this could be from a fast client before I even create the waiter;
                # or from a late client after I already finished gathering.
                if current_round <= self.last_aggr_round_done:
                    # late client case - drop the result
                    self.log_info(fl_ctx, f"dropped result from late {client_name} for round {current_round}")
                    return make_reply(ReturnCode.OK)

                # case of fast client
                # wait until the gatherer is set up.
                self.log_info(fl_ctx, f"got result from {client_name} for round {current_round} before gatherer setup")
                self.gatherer_waiter.wait(self.learn_task_abort_timeout)

            gatherer = self.gatherer
            if not gatherer:
                self.log_error(fl_ctx, f"Still no gatherer after {self.learn_task_abort_timeout} seconds")
                self.log_error(fl_ctx, f"Ignored result from {client_name} for round {current_round} since no gatherer")
                self.set_error(ReturnCode.EXECUTION_EXCEPTION)
                return make_reply(ReturnCode.EXECUTION_EXCEPTION)

            assert isinstance(gatherer, Gatherer)
            if gatherer.for_round != current_round:
                self.log_warning(
                    fl_ctx,
                    f"Got result from {client_name} for round {current_round}, "
                    f"but I'm waiting for round {gatherer.for_round}",
                )
            return gatherer.gather(client_name, request, fl_ctx)
        except Exception as ex:
            self.log_exception(fl_ctx, f"exception: {ex}")
            self.set_error(ReturnCode.EXECUTION_EXCEPTION)
            return make_reply(ReturnCode.EXECUTION_EXCEPTION)

    def do_learn_task(self, name: str, data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        # set status report of starting task
        current_round = data.get_header(AppConstants.CURRENT_ROUND)
        start_time = time.time()
        self.update_status(StatusReport(last_round=current_round, start_time=start_time), start_time)

        aggr = data.get_header(Constant.AGGREGATOR)
        if not aggr:
            self.log_error(fl_ctx, f"missing aggregation client for round {current_round}")
            self.set_error(ReturnCode.EXECUTION_EXCEPTION)
            return

        self.log_info(fl_ctx, f"Round {current_round} started.")
        global_weights = self.shareable_gen.shareable_to_learnable(data, fl_ctx)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, global_weights, private=True, sticky=True)
        fl_ctx.set_prop(AppConstants.CURRENT_ROUND, current_round, private=True, sticky=True)
        self.fire_event(AppEventType.ROUND_STARTED, fl_ctx)

        if self.me == aggr:
            # set up the aggr waiter
            gatherer = self.gatherer
            if gatherer:
                # already waiting for aggregation - should never happen
                self.log_error(
                    fl_ctx,
                    f"logic error: got task for round {current_round} while gathering for round {gatherer.for_round}",
                )
                self.set_error(ReturnCode.EXECUTION_EXCEPTION)
                return

            self.log_info(fl_ctx, f"setting up the gatherer for round {current_round}")
            self.gatherer = Gatherer(
                fl_ctx=fl_ctx,
                all_clients=self.get_config_prop(Constant.CLIENTS),
                trainers=self.trainers,
                for_round=current_round,
                timeout=self.learn_task_timeout,
                min_responses_required=self.min_responses_required,
                wait_time_after_min_resps_received=self.wait_time_after_min_resps_received,
                aggregator=self.aggregator,
                executor=self,
            )
            self.gatherer_waiter.set()

        # execute the task
        if self.is_trainer:
            # update status
            self.update_status(
                status=StatusReport(
                    last_round=current_round,
                    start_time=start_time,
                ),
                timestamp=start_time,
            )

            result = self.execute_train(data, fl_ctx, abort_signal)

            rc = result.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"learn executor failed: {rc}")
                self.set_error(rc)
                return

            # send the result to the aggr
            self.log_info(fl_ctx, f"sending training result to aggregation client {aggr}")
            resp = self.engine.send_aux_request(
                targets=[aggr],
                topic=Constant.TOPIC_RESULT,
                request=result,
                timeout=self.learn_task_send_timeout,
                fl_ctx=fl_ctx,
            )
            reply = resp.get(aggr)
            if not reply:
                self.log_error(fl_ctx, f"failed to receive reply from aggregation client: {aggr}")
                self.set_error(ReturnCode.EXECUTION_EXCEPTION)
                return

            if not isinstance(reply, Shareable):
                self.log_error(
                    fl_ctx, f"bad reply from aggregation client {aggr}: expect Shareable but got {type(reply)}"
                )
                self.set_error(ReturnCode.EXECUTION_EXCEPTION)
                return

            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"bad return code from aggregation client {aggr}: {rc}")
                self.set_error(ReturnCode.EXECUTION_EXCEPTION)
                return

            self.log_info(fl_ctx, f"Finished round {current_round}")

            # update status
            end_time = time.time()
            self.update_status(
                status=StatusReport(
                    last_round=current_round,
                    start_time=start_time,
                    end_time=end_time,
                ),
                timestamp=end_time,
            )

    def prepare_final_result(self, fl_ctx: FLContext) -> Shareable:
        # My last result is a ModelLearnable. Must convert it to Shareable.
        result = None
        if self.best_result:
            self.log_info(fl_ctx, f"I have a best result with metric {self.best_metric}")
            result = self.best_result
        elif self.last_result:
            self.log_info(fl_ctx, f"I have a last result with metric {self.best_metric}")
            result = self.last_result

        if not result:
            self.log_error(fl_ctx, "Hmm, I have no result to submit!")
            return make_reply(ReturnCode.EMPTY_RESULT)

        # the result is a Learnable
        if not isinstance(result, Learnable):
            self.log_error(fl_ctx, f"expect final result to be Learnable, but got {type(result)}")
            return make_reply(ReturnCode.EMPTY_RESULT)

        return self.shareable_gen.learnable_to_shareable(result, fl_ctx)
