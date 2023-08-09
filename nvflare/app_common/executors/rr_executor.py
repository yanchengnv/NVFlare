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
import time

from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal
from nvflare.app_common.app_constant import AppConstants
from nvflare.app_common.utils.cw_utils import Constant, RROrder, StatusReport, rotate_to_front

from .cwe import ClientWorkflowExecutor


class RRExecutor(ClientWorkflowExecutor):
    def __init__(
        self,
        start_task_name=Constant.TASK_NAME_START,
        configure_task_name=Constant.TASK_NAME_CONFIGURE,
        submit_result_task_name=Constant.TASK_NAME_SUBMIT_RESULT,
        train_task_name=AppConstants.TASK_TRAIN,
        max_status_report_interval: float = 600.0,
        task_check_interval: float = 1.0,
        task_abort_timeout: float = 5.0,
    ):
        super().__init__(
            start_task_name=start_task_name,
            configure_task_name=configure_task_name,
            submit_result_task_name=submit_result_task_name,
            train_task_name=train_task_name,
            max_status_report_interval=max_status_report_interval,
            task_check_interval=task_check_interval,
            task_abort_timeout=task_abort_timeout,
            allow_busy_task=False,
        )

    def start_workflow(self, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        clients = self.get_config_prop(Constant.CLIENTS)
        rr_order = self.get_config_prop(Constant.ORDER)
        self.log_info(fl_ctx, f"Starting RR Workflow on clients {clients} with Order {rr_order} ")
        shareable.set_header(AppConstants.CURRENT_ROUND, 1)
        shareable.set_header(Constant.CLIENT_ORDER, clients)
        self.set_learn_task(task_name=self.train_task_name, task_data=shareable, fl_ctx=fl_ctx)
        return make_reply(ReturnCode.OK)

    def do_learn_task(self, name: str, data: Shareable, fl_ctx: FLContext, abort_signal: Signal):
        # set status report of starting task
        current_round = data.get_header(AppConstants.CURRENT_ROUND)
        start_time = time.time()
        self.update_status(StatusReport(last_round=current_round, start_time=start_time), start_time)

        # execute the task
        result = self.execute_train(data, fl_ctx, abort_signal)

        rc = result.get_return_code(ReturnCode.OK)
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"learn executor failed: {rc}")
            self.set_error(rc)
            return

        self.last_result = result

        # see whether we need to send to next leg
        end_time = time.time()
        num_rounds = data.get_header(AppConstants.NUM_ROUNDS)
        current_round = data.get_header(AppConstants.CURRENT_ROUND)
        client_order = data.get_header(Constant.CLIENT_ORDER)

        result.set_header(AppConstants.NUM_ROUNDS, num_rounds)
        result.set_header(AppConstants.CURRENT_ROUND, current_round)
        result.set_header(Constant.CLIENT_ORDER, client_order)

        all_done = False
        assert isinstance(client_order, list)
        my_idx = client_order.index(self.me)
        if current_round == num_rounds:
            # am I the last leg?
            if my_idx == len(client_order) - 1:
                # I'm the last leg - the RR is done!
                self.log_info(fl_ctx, "I'm the last leg - got final result")
                all_done = True

        # update status
        self.update_status(
            status=StatusReport(
                last_round=current_round,
                start_time=start_time,
                end_time=end_time,
                all_done=all_done,
            ),
            timestamp=end_time,
        )

        if all_done:
            return

        # send to next leg
        if my_idx < len(client_order) - 1:
            next_client = client_order[my_idx + 1]
        else:
            next_client = client_order[0]

        sent = self.send_learn_task(
            targets=[next_client],
            request=result,
            timeout=2.0,
            fl_ctx=fl_ctx,
        )
        if sent:
            self.log_info(fl_ctx, f"sent learn request to next client {next_client}")

    def prepare_learn_task(self, data: Shareable, fl_ctx: FLContext):
        peer_ctx = fl_ctx.get_peer_context()
        assert isinstance(peer_ctx, FLContext)

        # Am I the starting client?
        if self.is_starting_client:
            # need to start the next round
            current_round = data.get_header(AppConstants.CURRENT_ROUND)
            num_rounds = data.get_header(AppConstants.NUM_ROUNDS)
            if current_round >= num_rounds:
                # should never happen!
                self.log_error(
                    fl_ctx,
                    f"logic error: current round {current_round} >= num rounds {num_rounds} for starting client!",
                )
                raise RuntimeError("logic error")

            # start next round
            next_round = current_round + 1
            clients = self.get_config_prop(Constant.CLIENTS)
            rr_order = self.get_config_prop(Constant.ORDER)
            if rr_order == RROrder.RANDOM:
                random.shuffle(clients)
                rotate_to_front(self.me, clients)
                data.set_header(Constant.CLIENT_ORDER, clients)

            self.log_info(fl_ctx, f"Starting new round {next_round} on clients: {clients}")
            data.set_header(AppConstants.CURRENT_ROUND, next_round)
