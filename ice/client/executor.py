# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from ice.defs import CONFIG_TASK_NAME, REQUEST_TOPIC, PropKey
from ice.utils import dispatch_request
from nvflare.apis.event_type import EventType
from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable, make_reply
from nvflare.apis.signal import Signal


class IceExecutor(Executor):
    def __init__(self):
        Executor.__init__(self)
        self.register_event_handler(EventType.START_RUN, self._handle_start_run)

    def _handle_start_run(self, event_type: str, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        engine.register_aux_message_handler(
            topic=REQUEST_TOPIC,
            message_handle_func=self._handle_server_request,
        )

    def _handle_server_request(self, topic: str, request: Shareable, fl_ctx: FLContext) -> Shareable:
        req_topic = request.get_header(PropKey.TOPIC)
        req_data = request.get(PropKey.DATA)
        result = dispatch_request(self, req_topic, req_data, None, fl_ctx)
        return Shareable({PropKey.DATA: result})

    def _process_config(self, config_data) -> bool:
        # place holder for processing config data
        self.logger.info(f"config data from server: {config_data}")
        return True

    def execute(self, task_name: str, shareable: Shareable, fl_ctx: FLContext, abort_signal: Signal) -> Shareable:
        if task_name != CONFIG_TASK_NAME:
            self.log_error(fl_ctx, f"received unsupported task {task_name}")
            return make_reply(ReturnCode.TASK_UNKNOWN)

        # the only way to get abort_signal is through a task
        config_data = shareable.get(PropKey.DATA)
        ok = self._process_config(config_data)
        if ok:
            return make_reply(ReturnCode.OK)
        else:
            return make_reply(ReturnCode.ERROR)
