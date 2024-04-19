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
from nvflare.app_common.xgb.fed_executor import FedXGBHistogramExecutor
from nvflare.app_common.xgb.mock.mock_client_applet import MockClientApplet


class MockXGBExecutor(FedXGBHistogramExecutor):
    def __init__(
        self,
        per_msg_timeout=2.0,
        tx_timeout=10.0,
        in_process=True,
    ):
        FedXGBHistogramExecutor.__init__(
            self,
            early_stopping_rounds=0,
            xgb_params={},
            data_loader_id="",
            in_process=in_process,
            per_msg_timeout=per_msg_timeout,
            tx_timeout=tx_timeout
        )

    def get_applet(self, fl_ctx: FLContext):
        return MockClientApplet()
