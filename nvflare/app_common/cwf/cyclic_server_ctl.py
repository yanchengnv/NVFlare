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
from nvflare.apis.shareable import Shareable
from nvflare.app_common.cwf.common import Constant, RROrder, StatusReport
from nvflare.app_common.cwf.server_ctl import ClientStatus, ServerSideController


class CyclicServerController(ServerSideController):
    def __init__(
        self,
        num_rounds: int,
        persistor_id="persistor",
        shareable_generator_id="shareable_generator",
        start_task_name=Constant.TASK_NAME_START,
        start_task_timeout=5,
        configure_task_name=Constant.TASK_NAME_CONFIGURE,
        configure_task_timeout=2,
        submit_result_task_name=Constant.TASK_NAME_SUBMIT_RESULT,
        submit_result_task_timeout=5,
        task_check_period: float = 0.5,
        job_status_check_interval: float = 2.0,
        participating_clients=None,
        starting_client: str = None,
        max_status_report_interval: float = 3600.0,
        rr_order: str = RROrder.FIXED,
    ):
        super().__init__(
            num_rounds=num_rounds,
            persistor_id=persistor_id,
            shareable_generator_id=shareable_generator_id,
            start_task_name=start_task_name,
            start_task_timeout=start_task_timeout,
            configure_task_name=configure_task_name,
            configure_task_timeout=configure_task_timeout,
            submit_result_task_name=submit_result_task_name,
            submit_result_task_timeout=submit_result_task_timeout,
            task_check_period=task_check_period,
            job_status_check_interval=job_status_check_interval,
            participating_clients=participating_clients,
            starting_client=starting_client,
            max_status_report_interval=max_status_report_interval,
        )
        self.rr_order = rr_order

    def prepare_config(self):
        return {Constant.ORDER: self.rr_order}

    def select_final_result(self, fl_ctx: FLContext) -> (str, Shareable):
        best_client = None
        last_progress_time = 0
        for c, cs in self.client_statuses.items():
            assert isinstance(cs, ClientStatus)
            if cs.last_progress_time and cs.last_progress_time > last_progress_time:
                best_client = c
                last_progress_time = cs.last_progress_time

            s = cs.status
            if s:
                assert isinstance(s, StatusReport)
                if s.all_done:
                    best_client = c
                    break

        if not best_client:
            self.log_error(fl_ctx, "cannot select final result: no client has result")
            return "", None
        shareable = Shareable()
        shareable.set_header(Constant.RESULT_TYPE, "last")
        return best_client, shareable
