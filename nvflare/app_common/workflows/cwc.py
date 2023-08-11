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
from abc import abstractmethod
from datetime import datetime

from nvflare.apis.client import Client
from nvflare.apis.controller_spec import TaskCompletionStatus
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.abstract.learnable_persistor import LearnablePersistor
from nvflare.app_common.abstract.shareable_generator import ShareableGenerator
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.cw_utils import (
    Constant,
    StatusReport,
    learnable_to_shareable,
    shareable_to_learnable,
    status_report_from_shareable,
)


class ClientStatus:
    def __init__(self):
        self.ready_time = None
        self.last_report_time = time.time()
        self.last_progress_time = time.time()
        self.num_reports = 0
        self.status = StatusReport()


class ClientWorkflowController(Controller):
    def __init__(
        self,
        num_rounds: int,
        start_round: int = 0,
        persistor_id="persistor",
        shareable_generator_id="shareable_generator",
        configure_task_name=Constant.TASK_NAME_CONFIGURE,
        configure_task_timeout=2,
        start_task_name=Constant.TASK_NAME_START,
        start_task_timeout=5,
        submit_result_task_name=Constant.TASK_NAME_SUBMIT_RESULT,
        submit_result_task_timeout=5,
        task_check_period: float = 0.5,
        job_status_check_interval: float = 2.0,
        starting_client: str = None,
        participating_clients=None,
        max_status_report_interval: float = 3600.0,
        client_ready_timeout: float = 60.0,
        progress_timeout: float = 3600,
    ):
        Controller.__init__(self, task_check_period)
        self.configure_task_name = configure_task_name
        self.configure_task_timeout = configure_task_timeout
        self.start_task_name = start_task_name
        self.start_task_timeout = start_task_timeout
        self.submit_result_task_name = submit_result_task_name
        self.submit_result_task_timeout = submit_result_task_timeout
        self.persistor_id = persistor_id
        self.shareable_generator_id = shareable_generator_id
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.max_status_report_interval = max_status_report_interval
        self.client_ready_timeout = client_ready_timeout
        self.progress_timeout = progress_timeout
        self.job_status_check_interval = job_status_check_interval
        self.starting_client = starting_client
        self.participating_clients = participating_clients
        self.persistor = None
        self.shareable_generator = None
        self._learnable = None
        self.client_statuses = {}  # client name => ClientStatus
        self.cw_started = False
        self.asked_to_stop = False
        self._last_learnable = None

        if num_rounds <= 0:
            raise ValueError(f"invalid num_rounds {num_rounds}: must > 0")

        if participating_clients and len(participating_clients) < 2:
            raise ValueError(f"Not enough participating_clients: must > 1, but got {participating_clients}")

    def start_controller(self, fl_ctx: FLContext):
        self.log_debug(fl_ctx, "starting controller")
        self.persistor = self._engine.get_component(self.persistor_id)
        if not isinstance(self.persistor, LearnablePersistor):
            raise RuntimeError(
                f"Persistor {self.persistor_id} must be a Persistor instance, but got {type(self.persistor)}"
            )

        if self.shareable_generator_id:
            self.shareable_generator = self._engine.get_component(self.shareable_generator_id)
            if not isinstance(self.shareable_generator, ShareableGenerator):
                raise RuntimeError(
                    f"Shareable generator {self.shareable_generator_id} must be a Shareable Generator instance, "
                    f"but got {type(self.shareable_generator)}",
                )

        self._learnable = self.persistor.load(fl_ctx)
        fl_ctx.set_prop(AppConstants.GLOBAL_MODEL, self._learnable, private=True, sticky=True)

        all_clients = self._engine.get_clients()
        if len(all_clients) <= 1:
            raise RuntimeError("Not enough client sites.")

        all_client_names = [t.name for t in all_clients]
        if not self.participating_clients:
            self.participating_clients = all_client_names
        else:
            # make sure participating_clients exist
            for c in self.participating_clients:
                if c not in all_client_names:
                    raise RuntimeError(f"Configured participating client {c} is invalid")

        if not self.starting_client:
            self.starting_client = self.participating_clients[0]
        elif self.starting_client not in self.participating_clients:
            raise RuntimeError(f"Configured starting client {self.starting_client} is invalid")

        for c in self.participating_clients:
            self.client_statuses[c] = ClientStatus()

        self._engine.register_aux_message_handler(
            topic=Constant.TOPIC_REPORT_STATUS, message_handle_func=self._process_status_report
        )

        self._engine.register_aux_message_handler(
            topic=Constant.TOPIC_FAILURE, message_handle_func=self._process_failure
        )

    @abstractmethod
    def prepare_config(self) -> dict:
        pass

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # wait for every client to become ready
        self.log_info(fl_ctx, f"Waiting for clients to be ready: {self.participating_clients}")

        # GET STARTED
        self.log_info(fl_ctx, f"Starting CW Control on Clients: {self.participating_clients}")
        if self.shareable_generator:
            model_shareable = self.shareable_generator.learnable_to_shareable(self._learnable, fl_ctx)
        else:
            model_shareable = learnable_to_shareable(self._learnable)

        learn_config = {
            Constant.CLIENTS: self.participating_clients,
            AppConstants.NUM_ROUNDS: self.num_rounds,
            Constant.START_ROUND: self.start_round,
        }

        extra_config = self.prepare_config()
        if extra_config:
            learn_config.update(extra_config)

        # configure all clients
        shareable = Shareable()
        shareable[Constant.CONFIG] = learn_config

        task = Task(
            name=self.configure_task_name,
            data=shareable,
            timeout=self.configure_task_timeout,
            result_received_cb=self._process_configure_reply,
        )

        self.log_info(fl_ctx, f"sending {self.start_task_name} to configure clients {self.participating_clients}")

        self.broadcast_and_wait(
            task=task,
            targets=self.participating_clients,
            min_responses=len(self.participating_clients),
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        failed_clients = []
        for c, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)
            if not cs.ready_time:
                failed_clients.append(c)

        if failed_clients:
            self.system_panic(
                f"failed to configure clients {failed_clients}",
                fl_ctx,
            )
            return

        self.log_info(fl_ctx, f"successfully configured clients {self.participating_clients}")

        # starting the starting_client
        shareable = model_shareable

        task = Task(
            name=self.start_task_name,
            data=shareable,
            timeout=self.configure_task_timeout,
            result_received_cb=self._process_start_reply,
        )

        self.send_and_wait(
            task=task,
            targets=[self.starting_client],
            fl_ctx=fl_ctx,
            abort_signal=abort_signal,
        )

        if not self.cw_started:
            self.system_panic(
                f"failed to start on client {self.starting_client}",
                fl_ctx,
            )
            return

        self.log_info(fl_ctx, f"started CW on client {self.starting_client}")

        self.log_info(fl_ctx, "Waiting for clients to finish ...")
        while not abort_signal.triggered and not self.asked_to_stop:
            time.sleep(self.job_status_check_interval)
            done = self._check_job_status(fl_ctx)
            if done:
                break

        self.log_info(fl_ctx, "Workflow finished on all clients")

        if self.submit_result_task_name:
            # try to get the final result
            target_name, shareable = self.select_final_result(fl_ctx)
            if target_name:
                task = Task(
                    name=self.submit_result_task_name,
                    data=shareable,
                    timeout=self.submit_result_task_timeout,
                    result_received_cb=self._process_final_result,
                )

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
            else:
                self.log_warning(fl_ctx, "no client selected for final result")

        self.log_info(fl_ctx, "CW Control Flow done!")

    def select_final_result(self, fl_ctx: FLContext) -> (str, Shareable):
        best_client = None
        overall_best_metric = 0.0
        last_client = None
        for c, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)

            s = cs.status
            if s:
                assert isinstance(s, StatusReport)
                if s.all_done:
                    last_client = c

                if s.best_metric:
                    # it contains the best metrics from this client
                    if overall_best_metric < s.best_metric:
                        overall_best_metric = s.best_metric
                        best_client = c

        shareable = Shareable()

        if best_client:
            self.log_info(fl_ctx, f"client {best_client} has best metric {overall_best_metric}")
            shareable.set_header(Constant.RESULT_TYPE, "best")
            return best_client, shareable
        elif last_client:
            self.log_info(fl_ctx, f"no best_client, use last client {last_client}")
            shareable.set_header(Constant.RESULT_TYPE, "last")
            return last_client, shareable

        self.log_error(fl_ctx, "cannot select client for final result")
        return "", None

    def _process_configure_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"successfully configured client {client_name}")
            cs = self.client_statuses.get(client_name)
            if cs:
                assert isinstance(cs, ClientStatus)
                cs.ready_time = time.time()
        else:
            reason = result.get(Constant.REASON, "?")
            self.log_error(fl_ctx, f"client {client_task.client.name} failed to configure: {rc}: {reason}")

    def _process_start_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"workflow started by client {client_name}")
            self.cw_started = True
        else:
            reason = result.get(Constant.REASON, "?")
            self.log_error(fl_ctx, f"client {client_task.client.name} couldn't start workflow: {rc}: {reason}")

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
        overall_last_progress_time = 0
        for client_name, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)
            assert isinstance(cs.status, StatusReport)

            if cs.status.all_done:
                self.log_info(fl_ctx, f"got ALL_DONE from client {client_name}")
                return True

            if now - cs.last_report_time > self.max_status_report_interval:
                self.system_panic(
                    f"client {client_name} didn't report status for {self.max_status_report_interval} seconds",
                    fl_ctx,
                )
                return True

            if overall_last_progress_time < cs.last_progress_time:
                overall_last_progress_time = cs.last_progress_time

        if time.time() - overall_last_progress_time > self.progress_timeout:
            self.system_panic(
                f"the workflow has no progress for {self.progress_timeout} seconds",
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
        now = time.time()
        cs.last_report_time = now
        cs.num_reports += 1
        report = status_report_from_shareable(result)
        if cs.status != report:
            # updated
            cs.status = report
            cs.last_progress_time = now
            started = datetime.fromtimestamp(report.start_time) if report.start_time else False
            ended = datetime.fromtimestamp(report.end_time) if report.end_time else False
            self.log_info(
                fl_ctx,
                f"updated status of client {client_name} on round {report.last_round}: "
                f"started={started}, ended={ended}, metric={report.best_metric}",
            )
        else:
            self.log_info(
                fl_ctx, f"ignored status report from client {client_name} on round {report.last_round}: no change"
            )

    def _process_status_report(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()
        self._update_client_status(fl_ctx, client_name=client_name, result=request)
        return make_reply(ReturnCode.OK)

    def _process_failure(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        # client reports that it cannot continue RR
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()
        reason = request.get(Constant.REASON, "?")
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
        self.log_debug(fl_ctx, "controller stopped")
