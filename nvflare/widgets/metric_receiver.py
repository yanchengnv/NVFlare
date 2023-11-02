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

from nvflare.apis.dxo import DXO
from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.utils.constants import PipeChannelName
from nvflare.fuel.utils.pipe.pipe import Message, Pipe
from nvflare.fuel.utils.pipe.pipe_handler import PipeHandler
from nvflare.widgets.widget import Widget


class MetricReceiver(Widget):
    def __init__(
        self,
        pipe_id: str,
        read_interval=0.1,
        heartbeat_interval=5.0,
        heartbeat_timeout=30.0,
        pipe_channel_name=PipeChannelName.METRIC,
    ):
        super().__init__()
        self.pipe_id = pipe_id
        self.read_interval = read_interval
        self.heartbeat_interval = heartbeat_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.pipe_channel_name = pipe_channel_name
        self.pipe = None
        self.pipe_handler = None

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            pipe = engine.get_component(self.pipe_id)
            if not isinstance(pipe, Pipe):
                self.log_error(fl_ctx, f"component {self.pipe_id} must be Pipe but got {type(pipe)}")
                self.system_panic(f"bad component {self.pipe_id}", fl_ctx)
                return
            self.pipe = pipe
            self.pipe_handler = PipeHandler(
                pipe=self.pipe,
                read_interval=self.read_interval,
                heartbeat_interval=self.heartbeat_interval,
                heartbeat_timeout=self.heartbeat_timeout,
            )
            self.pipe_handler.set_status_cb(self._pipe_status_cb)
            self.pipe_handler.set_message_cb(self._pipe_msg_cb)
            self.pipe.open(self.pipe_channel_name)
            self.pipe_handler.start()
        elif event_type == EventType.END_RUN:
            self.log_info(fl_ctx, "Stopping pipe handler")
            if self.pipe_handler:
                self.pipe_handler.notify_end("end_of_job")
                self.pipe_handler.stop()

    def _pipe_status_cb(self, msg: Message):
        self.logger.info(f"{self.pipe_channel_name} pipe status changed to {msg.topic}")
        self.pipe_handler.stop()

    def _pipe_msg_cb(self, msg: Message):
        if not isinstance(msg.data, DXO):
            self.logger.error(f"bad metric data: expect DXO but got {type(msg.data)}")
        self.logger.info(f"received metric record: {msg.topic}: {msg.data.data}")
