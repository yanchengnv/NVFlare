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
from nvflare.app_common.xgb.connectors.grpc_server_connector import GrpcServerConnector
from nvflare.app_common.xgb.applets.xgb_server import XGBServerApplet
from nvflare.app_common.tie.controller import TieController
from nvflare.app_common.tie.defs import Constant as TieConstant
from nvflare.fuel.utils.validation_utils import check_object_type

from .defs import Constant


class XGBFedController(TieController):
    def __init__(
        self,
        num_rounds: int,
        configure_task_name=TieConstant.CONFIG_TASK_NAME,
        configure_task_timeout=TieConstant.CONFIG_TASK_TIMEOUT,
        start_task_name=TieConstant.START_TASK_NAME,
        start_task_timeout=TieConstant.START_TASK_TIMEOUT,
        job_status_check_interval: float = TieConstant.JOB_STATUS_CHECK_INTERVAL,
        max_client_op_interval: float = TieConstant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = TieConstant.WORKFLOW_PROGRESS_TIMEOUT,
        client_ranks=None,
        int_client_grpc_options=None,
        in_process=True,
    ):
        TieController.__init__(
            self,
            configure_task_name=configure_task_name,
            configure_task_timeout=configure_task_timeout,
            start_task_name=start_task_name,
            start_task_timeout=start_task_timeout,
            job_status_check_interval=job_status_check_interval,
            max_client_op_interval=max_client_op_interval,
            progress_timeout=progress_timeout,
        )

        if client_ranks:
            check_object_type("client_ranks", client_ranks, dict)

        self.num_rounds = num_rounds
        self.client_ranks = client_ranks
        self.int_client_grpc_options = int_client_grpc_options
        self.in_process = in_process

    def get_connector(self, fl_ctx: FLContext):
        return GrpcServerConnector(
            int_client_grpc_options=self.int_client_grpc_options,
            in_process=self.in_process,
        )

    def get_applet(self, fl_ctx: FLContext):
        return XGBServerApplet()

    def get_client_config_params(self, fl_ctx: FLContext) -> dict:
        config = {}

        # compute client ranks
        if not self.client_ranks:
            # dynamically assign ranks, starting from 0
            # Assumption: all clients are used
            clients = self.participating_clients

            # Sort by client name so rank is consistent
            clients.sort()
            self.client_ranks = {clients[i]: i for i in range(0, len(clients))}
        else:
            # validate ranks - ranks must be unique consecutive integers, starting from 0.
            num_clients = len(self.participating_clients)
            assigned_ranks = {}  # rank => client
            if len(self.client_ranks) != num_clients:
                # either missing client or duplicate client
                raise RuntimeError(
                    f"expecting rank assignments for {self.participating_clients} but got {self.client_ranks}"
                )

            # all clients must have ranks
            for c in self.participating_clients:
                if c not in self.client_ranks:
                    raise RuntimeError(f"missing rank assignment for client '{c}'")

            # check each client's rank
            for c, r in self.client_ranks.items():
                if not isinstance(r, int):
                    raise RuntimeError(f"bad rank assignment {r} for client '{c}': expect int but got {type(r)}")
                if r < 0 or r >= num_clients:
                    raise RuntimeError(f"bad rank assignment {r} for client '{c}': must be 0 to {num_clients - 1}")

                assigned_client = assigned_ranks.get(r)
                if assigned_client:
                    raise RuntimeError(f"rank {r} is assigned to both client '{c}' and '{assigned_client}'")
                assigned_ranks[r] = c

        config[Constant.CONF_KEY_CLIENT_RANKS] = self.client_ranks
        config[Constant.CONF_KEY_NUM_ROUNDS] = self.num_rounds
        return config

    def get_connector_config_params(self, fl_ctx: FLContext) -> dict:
        return {Constant.CONF_KEY_WORLD_SIZE: len(self.participating_clients)}
