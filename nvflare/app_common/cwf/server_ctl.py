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
from datetime import datetime

from nvflare.apis.client import Client
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import ClientTask, Controller, Task
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.cwf.common import Constant, StatusReport, status_report_from_dict, topic_for_end_workflow


class ClientStatus:
    def __init__(self):
        self.ready_time = None
        self.last_report_time = time.time()
        self.last_progress_time = time.time()
        self.num_reports = 0
        self.status = StatusReport()


class ServerSideController(Controller):
    def __init__(
        self,
        num_rounds: int,
        start_round: int = 0,
        configure_task_name=Constant.TASK_NAME_CONFIGURE,
        configure_task_timeout=2,
        end_workflow_timeout=2.0,
        start_task_name=Constant.TASK_NAME_START,
        start_task_timeout=5,
        task_check_period: float = 0.5,
        job_status_check_interval: float = 2.0,
        starting_client: str = None,
        participating_clients=None,
        result_clients=None,
        max_status_report_interval: float = 3600.0,
        client_ready_timeout: float = 60.0,
        progress_timeout: float = 3600,
    ):
        """
        Constructor

        Args:
            num_rounds:
            start_round:
            configure_task_name:
            configure_task_timeout:
            end_workflow_timeout:
            start_task_name:
            start_task_timeout:
            task_check_period:
            job_status_check_interval:
            starting_client:
            participating_clients:
            result_clients: clients to receive final results
            max_status_report_interval:
            client_ready_timeout:
            progress_timeout:
        """
        Controller.__init__(self, task_check_period)
        self.configure_task_name = configure_task_name
        self.configure_task_timeout = configure_task_timeout
        self.start_task_name = start_task_name
        self.start_task_timeout = start_task_timeout
        self.end_workflow_timeout = end_workflow_timeout
        self.num_rounds = num_rounds
        self.start_round = start_round
        self.max_status_report_interval = max_status_report_interval
        self.client_ready_timeout = client_ready_timeout
        self.progress_timeout = progress_timeout
        self.job_status_check_interval = job_status_check_interval
        self.starting_client = starting_client
        self.participating_clients = participating_clients
        self.result_clients = result_clients
        self.client_statuses = {}  # client name => ClientStatus
        self.cw_started = False
        self.asked_to_stop = False
        self.workflow_id = None

        if num_rounds <= 0:
            raise ValueError(f"invalid num_rounds {num_rounds}: must > 0")

        if participating_clients and len(participating_clients) < 2:
            raise ValueError(f"Not enough participating_clients: must > 1, but got {participating_clients}")

    def start_controller(self, fl_ctx: FLContext):
        wf_id = fl_ctx.get_prop(FLContextKey.WORKFLOW)
        self.log_debug(fl_ctx, f"starting controller for workflow {wf_id}")
        if not wf_id:
            raise RuntimeError("workflow ID is missing from FL context")
        self.workflow_id = wf_id

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

        if self.result_clients:
            for c in self.result_clients:
                if c not in self.participating_clients:
                    raise RuntimeError(f"Configured result client {c} is invalid")
        else:
            self.result_clients = []

        for c in self.participating_clients:
            self.client_statuses[c] = ClientStatus()

    def prepare_config(self) -> dict:
        return {}

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        # wait for every client to become ready
        self.log_info(fl_ctx, f"Waiting for clients to be ready: {self.participating_clients}")

        # GET STARTED
        self.log_info(fl_ctx, f"Starting workflow on clients: {self.participating_clients}")

        learn_config = {
            Constant.CLIENTS: self.participating_clients,
            Constant.RESULT_CLIENTS: self.result_clients,
            AppConstants.NUM_ROUNDS: self.num_rounds,
            Constant.START_ROUND: self.start_round,
            FLContextKey.WORKFLOW: self.workflow_id,
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
        shareable = Shareable()
        task = Task(
            name=self.start_task_name,
            data=shareable,
            timeout=self.start_task_timeout,
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

        # ask all clients to end the workflow
        self.log_info(fl_ctx, f"asking all clients to end workflow {self.workflow_id}")
        engine = fl_ctx.get_engine()
        end_wf_request = Shareable()
        resp = engine.send_aux_request(
            targets=self.participating_clients,
            topic=topic_for_end_workflow(self.workflow_id),
            request=end_wf_request,
            timeout=self.end_workflow_timeout,
            fl_ctx=fl_ctx,
        )

        assert isinstance(resp, dict)
        num_errors = 0
        for c in self.participating_clients:
            reply = resp.get(c)
            if not reply:
                self.log_error(fl_ctx, f"not reply from client {c} for ending workflow {self.workflow_id}")
                num_errors += 1
                continue

            assert isinstance(reply, Shareable)
            rc = reply.get_return_code(ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"client {c} failed to end workflow {self.workflow_id}: {rc}")
                num_errors += 1

        if num_errors > 0:
            self.system_panic(f"failed to end workflow {self.workflow_id} on all clients", fl_ctx)

        self.log_info(fl_ctx, f"Workflow {self.workflow_id} done!")

    def process_task_request(self, client: Client, fl_ctx: FLContext):
        self._update_client_status(fl_ctx)
        return super().process_task_request(client, fl_ctx)

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
            error = result.get(Constant.ERROR, "?")
            self.log_error(fl_ctx, f"client {client_task.client.name} failed to configure: {rc}: {error}")

    def _process_start_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"workflow started by client {client_name}")
            self.cw_started = True
        else:
            error = result.get(Constant.ERROR, "?")
            self.log_error(fl_ctx, f"client {client_task.client.name} couldn't start workflow: {rc}: {error}")

    def _check_job_status(self, fl_ctx: FLContext):
        now = time.time()
        overall_last_progress_time = 0.0
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

    def _update_client_status(self, fl_ctx: FLContext):
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)
        client_name = peer_ctx.get_identity_name()
        if client_name not in self.client_statuses:
            self.log_error(fl_ctx, f"received result from unknown client {client_name}!")
            return

        # see whether status is available
        reports = peer_ctx.get_prop(Constant.STATUS_REPORTS)
        if not reports:
            return

        my_report = reports.get(self.workflow_id)
        if not my_report:
            return

        report = status_report_from_dict(my_report)
        cs = self.client_statuses[client_name]
        assert isinstance(cs, ClientStatus)
        now = time.time()
        cs.last_report_time = now
        cs.num_reports += 1

        if report.error:
            self.asked_to_stop = True
            self.system_panic(f"received failure report from client {client_name}: {report.error}", fl_ctx)
            return

        if cs.status != report:
            # updated
            cs.status = report
            cs.last_progress_time = now
            timestamp = datetime.fromtimestamp(report.timestamp) if report.timestamp else False
            self.log_info(
                fl_ctx,
                f"updated status of client {client_name} on round {report.last_round}: "
                f"timestamp={timestamp}, action={report.action}, all_done={report.all_done}",
            )
        else:
            self.log_info(
                fl_ctx, f"ignored status report from client {client_name} at round {report.last_round}: no change"
            )

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        self.log_warning(fl_ctx, f"ignored unknown task {task_name} from client {client.name}")

    def stop_controller(self, fl_ctx: FLContext):
        pass
