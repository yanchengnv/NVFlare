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
from nvflare.app_common.tie.executor import TieExecutor
from nvflare.app_common.xgb.applets.xgb_client import XGBClientApplet
from nvflare.app_common.xgb.connectors.grpc_client_connector import GrpcClientConnector

from .defs import Constant


class FedXGBHistogramExecutor(TieExecutor):
    def __init__(
        self,
        early_stopping_rounds,
        xgb_params: dict,
        data_loader_id: str,
        verbose_eval=False,
        use_gpus=False,
        per_msg_timeout=10.0,
        tx_timeout=100.0,
        model_file_name="model.json",
        metrics_writer_id: str = None,
        in_process=True,
    ):
        TieExecutor.__init__(
            self,
            start_task_name=Constant.START_TASK_NAME,
            configure_task_name=Constant.CONFIG_TASK_NAME,
        )
        self.early_stopping_rounds = early_stopping_rounds
        self.xgb_params = xgb_params
        self.data_loader_id = data_loader_id
        self.verbose_eval = verbose_eval
        self.use_gpus = use_gpus
        self.int_server_grpc_options = None
        self.model_file_name = model_file_name
        self.per_msg_timeout = per_msg_timeout
        self.tx_timeout = tx_timeout
        self.metrics_writer_id = metrics_writer_id
        self.in_process = in_process
        self.config = None

    def get_connector(self, fl_ctx: FLContext):
        return GrpcClientConnector(
            int_server_grpc_options=self.int_server_grpc_options,
            in_process=self.in_process,
            per_msg_timeout=self.per_msg_timeout,
            tx_timeout=self.tx_timeout,
        )

    def get_applet(self, fl_ctx: FLContext):
        return XGBClientApplet(
            data_loader_id=self.data_loader_id,
            early_stopping_rounds=self.early_stopping_rounds,
            xgb_params=self.xgb_params,
            verbose_eval=self.verbose_eval,
            use_gpus=self.use_gpus,
            model_file_name=self.model_file_name,
            metrics_writer_id=self.metrics_writer_id,
        )

    def configure(self, config: dict, fl_ctx: FLContext):
        ranks = config.get(Constant.CONF_KEY_CLIENT_RANKS)
        if not ranks:
            raise RuntimeError(f"missing {Constant.CONF_KEY_CLIENT_RANKS} from config")

        if not isinstance(ranks, dict):
            raise RuntimeError(f"expect config data to be dict but got {ranks}")

        me = fl_ctx.get_identity_name()
        my_rank = ranks.get(me)
        if my_rank is None:
            raise RuntimeError(f"missing rank for me ({me}) in config data")

        self.log_info(fl_ctx, f"got my rank: {my_rank}")
        num_rounds = config.get(Constant.CONF_KEY_NUM_ROUNDS)
        if not num_rounds:
            raise RuntimeError(f"missing {Constant.CONF_KEY_NUM_ROUNDS} from config")

        world_size = len(ranks)
        self.config = {
            Constant.CONF_KEY_RANK: my_rank,
            Constant.CONF_KEY_NUM_ROUNDS: num_rounds,
            Constant.CONF_KEY_WORLD_SIZE: world_size,
        }

    def get_connector_config(self, fl_ctx: FLContext) -> dict:
        return self.config
