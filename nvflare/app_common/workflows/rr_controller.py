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

import time

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.rr_utils import (
    RRConstant,
    RROrder,
    StatusReport,
    learnable_to_shareable,
    shareable_to_learnable,
    status_report_from_shareable,
)


class ClientStatus:
    def __init__(self):
        self.last_report_time = time.time()
        self.num_reports = 0
        self.status = None


class RRController(Controller):
    def __init__(
        self,
        num_rounds: int,
        persistor_id="persistor",
        shareable_generator_id="shareable_generator",
        rr_task_name=RRConstant.TASK_NAME_RR,
        submit_result_task_name=RRConstant.TASK_NAME_SUBMIT_RESULT,
        rr_task_timeout=5,
        submit_result_task_timeout=5,
        task_check_period: float = 0.5,
        job_status_check_interval: float = 2.0,
        starting_client: str = None,
        rr_order: str = RROrder.FIXED,
        max_status_report_interval: float = 3600.0,
    ):
        Controller.__init__(self, task_check_period)
        self.rr_task_name = rr_task_name
        self.submit_result_task_name = submit_result_task_name
        self.rr_task_timeout = rr_task_timeout
        self.submit_result_task_timeout = submit_result_task_timeout
        self.persistor_id = persistor_id
        self.shareable_generator_id = shareable_generator_id
        self.num_rounds = num_rounds
        self.max_status_report_interval = max_status_report_interval
        self.job_status_check_interval = job_status_check_interval
        self.starting_client = starting_client
        self.rr_order = rr_order
        self.persistor = None
        self.shareable_generator = None
        self._learnable = None
        self._client_names = None
        self.client_statuses = {}
        self.rr_started = False
        self.asked_to_stop = False
        self._final_result_type = None
        self._final_result_client = None
        self._last_learnable = None

    def start_controller(self, fl_ctx: FLContext):
        self.log_debug(fl_ctx, "starting controller")
        self.persistor = self._engine.get_component(self.persistor_id)
        if not isinstance(self.persistor, LearnablePersistor):
            self.system_panic(
                f"Persistor {self.persistor_id} must be a Persistor instance, but got {type(self.persistor)}", fl_ctx
            )
            return

        if self.shareable_generator_id:
            self.shareable_generator = self._engine.get_component(self.shareable_generator_id)
            if not isinstance(self.shareable_generator, ShareableGenerator):
                self.system_panic(
                    f"Shareable generator {self.shareable_generator_id} must be a Shareable Generator instance,"
                    f"but got {type(self.shareable_generator)}",
                    fl_ctx,
                )

        self._learnable = self.persistor.load(fl_ctx)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._learnable, private=True, sticky=True)

        all_clients = self._engine.get_clients()
        if len(all_clients) <= 1:
            self.system_panic("Not enough client sites.", fl_ctx)
            return

        self._client_names = [t.name for t in all_clients]
        if self.starting_client and self.starting_client not in self._client_names:
            self.system_panic(f"Configured starting client {self.starting_client} is invalid", fl_ctx)
            return

        for c in self._client_names:
            self.client_statuses[c] = ClientStatus()

        # make sure the starting client is the 1st
        idx = self._client_names.index(self.starting_client)
        if idx != 0:
            self._client_names.pop(idx)
            self._client_names.insert(0, self.starting_client)

        self._engine.register_aux_message_handler(
            topic=RRConstant.TOPIC_REPORT_STATUS, message_handle_func=self._process_status_report
        )

        self._engine.register_aux_message_handler(
            topic=RRConstant.TOPIC_FAILURE, message_handle_func=self._process_failure
        )

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.log_info(fl_ctx, f"Starting RR Control on Clients: {self._client_names}")

        # determine starting client
        if self.starting_client:
            target_name = self.starting_client
        else:
            target_name = self._client_names[0]

        if self.shareable_generator:
            shareable = self.shareable_generator.learnable_to_shareable(self._learnable, fl_ctx)
        else:
            shareable = learnable_to_shareable(self._learnable)

        shareable.set_header(AppConstants.NUM_ROUNDS, self.num_rounds)
        shareable.set_header(RRConstant.ORDER, self.rr_order)
        shareable.set_header(RRConstant.CLIENTS, self._client_names)

        task = Task(
            name=self.rr_task_name,
            data=shareable,
            timeout=self.rr_task_timeout,
            result_received_cb=self._process_rr_start,
        )

        self.log_info(fl_ctx, f"sending RR task {self.rr_task_name} to client {target_name}")

        self.send_and_wait(
            task=task,
            targets=[target_name],
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        if task.completion_status != TaskCompletionStatus.OK:
            self.system_panic(
                f"failed to send RR task {self.rr_task_name} to client {target_name}: {task.completion_status}",
                fl_ctx,
            )
            return

        if not self.rr_started:
            self.system_panic(f"couldn't start RR on clients {target_name}", fl_ctx)
            return

        self.log_info(fl_ctx, f"started RR task {self.rr_task_name} on client {target_name}")
        self.log_info(fl_ctx, "Waiting for clients to finish ...")
        while not abort_signal.triggered and not self.asked_to_stop:
            time.sleep(self.job_status_check_interval)
            self.log_info(fl_ctx, "checking job status ...")
            done = self._check_job_status(fl_ctx)
            if done:
                break

        self.log_info(fl_ctx, "Clients finished RR")

        if self.submit_result_task_name:
            # try to get the final result
            if not self._final_result_client:
                self.log_error(fl_ctx, "Final result not available")
            else:
                shareable = Shareable()
                shareable.set_header(RRConstant.FINAL_RESULT, self._final_result_type)
                task = Task(
                    name=self.submit_result_task_name,
                    data=shareable,
                    timeout=self.submit_result_task_timeout,
                    result_received_cb=self._process_final_result,
                )

                target_name = self._final_result_client
                self.log_info(
                    fl_ctx, f"sending task {self.submit_result_task_name} to client {target_name} for final result"
                )

                self.send_and_wait(
                    task=task,
                    targets=[target_name],
                    fl_ctx=fl_ctx,
                    abort_signal=abort_signal,
                )

                if task.completion_status != TaskCompletionStatus.OK:
                    self.log_error(
                        fl_ctx,
                        f"failed to get final result from client {target_name}: {task.completion_status}",
                    )

        self.log_info(fl_ctx, "RR Control Flow done!")

    def _process_rr_start(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        assert isinstance(result, Shareable)
        if result.get_return_code() == ReturnCode.OK:
            self.log_info(fl_ctx, f"RR started by client {client_task.client.name}")
            self.rr_started = True
        else:
            reason = result.get(RRConstant.REASON, "?")
            self.log_error(fl_ctx, f"client {client_task.client.name} couldn't start RR: {reason}")

    def _process_final_result(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        assert isinstance(result, Shareable)
        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"Got final result from client {client_task.client.name}")
            if self.shareable_generator:
                self._last_learnable = self.shareable_generator.shareable_to_learnable(result, fl_ctx)
            else:
                self._last_learnable = shareable_to_learnable(result)
        else:
            self.log_error(fl_ctx, f"client {client_task.client.name} couldn't submit final reason: {rc}")

    def _check_job_status(self, fl_ctx: FLContext):
        now = time.time()
        for client_name, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)
            assert isinstance(cs.status, StatusReport)
            final_result = cs.status.final_result
            if final_result:
                self.log_info(fl_ctx, f"got final result from client {client_name}")
                self._final_result_client = client_name
                self._final_result_type = final_result
                return True

            if now - cs.last_report_time > self.max_status_report_interval:
                self.system_panic(
                    f"client {client_name} didn't report status for {self.max_status_report_interval} seconds",
                    fl_ctx,
                )
                return True
        return False

    def _update_client_status(self, fl_ctx: FLContext, client_name: str, result: Shareable):
        if client_name not in self.client_statuses:
            self.log_error(fl_ctx, f"received result from unknown client {client_name}!")
            return

        cs = self.client_statuses[client_name]
        assert isinstance(cs, ClientStatus)
        cs.last_report_time = time.time()
        cs.num_reports += 1
        cs.status = status_report_from_shareable(result)

    def _process_status_report(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()
        self.log_info(fl_ctx, f"got status report from client {client_name}")
        self._update_client_status(fl_ctx, client_name=client_name, result=request)
        return make_reply(ReturnCode.OK)

    def _process_failure(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        # client reports that it cannot continue RR
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()
        reason = request.get(RRConstant.REASON, "?")
        self.asked_to_stop = True
        self.system_panic(f"received failure report from client {client_name}: {reason}", fl_ctx)
        return make_reply(ReturnCode.OK)

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        pass

    def stop_controller(self, fl_ctx: FLContext):
        if self._last_learnable:
            self.persistor.save(learnable=self._last_learnable, fl_ctx=fl_ctx)
            self.log_info(fl_ctx, f"Final result saved: {self._last_learnable}")
        self.log_debug(fl_ctx, "controller stopped")
