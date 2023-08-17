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

from nvflare.apis.fl_context import FLContext
from nvflare.app_common.cwf.common import Constant
from nvflare.app_common.cwf.server_ctl import ServerSideController


class SwarmServerController(ServerSideController):
    def __init__(
        self,
        num_rounds: int,
        start_round: int = 0,
        start_task_name=Constant.TASK_NAME_START,
        start_task_timeout=5,
        configure_task_name=Constant.TASK_NAME_CONFIGURE,
        configure_task_timeout=2,
        task_check_period: float = 0.5,
        job_status_check_interval: float = 2.0,
        participating_clients=None,
        result_clients=None,
        starting_client: str = None,
        max_status_report_interval: float = 3600.0,
        aggr_clients=None,
        train_clients=None,
    ):
        super().__init__(
            num_rounds=num_rounds,
            start_round=start_round,
            start_task_name=start_task_name,
            start_task_timeout=start_task_timeout,
            configure_task_name=configure_task_name,
            configure_task_timeout=configure_task_timeout,
            task_check_period=task_check_period,
            job_status_check_interval=job_status_check_interval,
            participating_clients=participating_clients,
            result_clients=result_clients,
            starting_client=starting_client,
            max_status_report_interval=max_status_report_interval,
        )
        self.aggr_clients = aggr_clients
        self.train_clients = train_clients

    def start_controller(self, fl_ctx: FLContext):
        super().start_controller(fl_ctx)
        if not self.train_clients:
            # every participating client is a train client
            self.train_clients = []
        else:
            for c in self.train_clients:
                if c not in self.participating_clients:
                    raise RuntimeError(f"Config Error: train client {c} is not in participating_clients")

        if not self.aggr_clients:
            # every participating client is an aggr client
            self.aggr_clients = []
        else:
            for c in self.aggr_clients:
                if c not in self.participating_clients:
                    raise RuntimeError(f"Config Error: aggr client {c} is not in participating_clients")

        # make sure every participating client is either training or aggr client
        if self.train_clients and self.aggr_clients:
            # both are explicitly specified
            for c in self.participating_clients:
                if c not in self.train_clients and c not in self.aggr_clients:
                    raise RuntimeError(f"Config Error:  client {c} is neither train client nor aggr client")

    def prepare_config(self):
        return {Constant.AGGR_CLIENTS: self.aggr_clients, Constant.TRAIN_CLIENTS: self.train_clients}
