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
from nvflare.app_common.xgb.fed_controller import XGBFedController
from nvflare.app_common.xgb.defs import Constant
from nvflare.app_common.xgb.mock.mock_server_applet import MockServerApplet


class MockXGBController(XGBFedController):
    def __init__(
        self,
        num_rounds: int,
        configure_task_name=Constant.CONFIG_TASK_NAME,
        configure_task_timeout=Constant.CONFIG_TASK_TIMEOUT,
        start_task_name=Constant.START_TASK_NAME,
        start_task_timeout=Constant.START_TASK_TIMEOUT,
        job_status_check_interval: float = Constant.JOB_STATUS_CHECK_INTERVAL,
        max_client_op_interval: float = Constant.MAX_CLIENT_OP_INTERVAL,
        progress_timeout: float = Constant.WORKFLOW_PROGRESS_TIMEOUT,
        client_ranks=None,
        aggr_timeout=10.0,
        in_process=True,
    ):
        XGBFedController.__init__(
            self,
            num_rounds=num_rounds,
            configure_task_name=configure_task_name,
            configure_task_timeout=configure_task_timeout,
            start_task_name=start_task_name,
            start_task_timeout=start_task_timeout,
            job_status_check_interval=job_status_check_interval,
            max_client_op_interval=max_client_op_interval,
            progress_timeout=progress_timeout,
            client_ranks=client_ranks,
            in_process=in_process,
        )
        self.aggr_timeout = aggr_timeout

    def get_applet(self, fl_ctx: FLContext):
        return MockServerApplet(aggr_timeout=self.aggr_timeout)
