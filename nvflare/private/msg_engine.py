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
import threading
from typing import List

from nvflare.apis.aux_spec import AuxMessenger
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.rm import RMEngine
from nvflare.apis.shareable import Shareable
from nvflare.apis.streaming import ConsumerFactory, ObjectProducer, StreamableEngine, StreamContext

from .rm_runner import ReliableMessenger
from .stream_runner import ObjectStreamer


class MessagingEngine(StreamableEngine, RMEngine, FLComponent):
    def __init__(self, messenger: AuxMessenger):
        FLComponent.__init__(self)
        self.messenger = messenger
        self._lock = threading.Lock()

        # We do not immediately create ObjectStreamer and ReliableMessenger here since they need to
        # register aux CBs with the AuxMessenger, but the AuxMessenger may not be ready now.
        # Instead, we will create them later when needed.
        self.streamer = None
        self.reliable_messenger = None

    def _open_streamer(self):
        with self._lock:
            if not self.streamer:
                self.streamer = ObjectStreamer(self.messenger)

    def _open_reliable_messenger(self):
        self.logger.info(f"trying to open reliable messenger for {id(self)}")
        with self._lock:
            if not self.reliable_messenger:
                self.reliable_messenger = ReliableMessenger(self.messenger)
                self.logger.info(f"reliable_messenger is opened for engine {id(self)}, aux {id(self.messenger)}")
            else:
                self.logger.info(f"reliable messenger for {id(self)} is already opened")

    def register_reliable_request_handler(self, channel: str, topic: str, handler_f, **handler_kwargs):
        self._open_reliable_messenger()
        self.reliable_messenger.register_request_handler(channel, topic, handler_f, **handler_kwargs)

    def send_reliable_request(
        self,
        target: str,
        channel: str,
        topic: str,
        request: Shareable,
        per_msg_timeout: float,
        tx_timeout: float,
        fl_ctx: FLContext,
        optional=False,
        secure=False,
    ) -> Shareable:
        self._open_reliable_messenger()
        self.logger.info(f"trying to send RM: {target=} {channel=} {topic=} {per_msg_timeout=} {optional=} {secure=}")
        return self.reliable_messenger.send_request(
            target, channel, topic, request, per_msg_timeout, tx_timeout, fl_ctx, secure, optional
        )

    def shutdown_reliable_messenger(self):
        with self._lock:
            if self.reliable_messenger:
                self.reliable_messenger.shutdown()

    def register_stream_processing(
        self,
        channel: str,
        topic: str,
        factory: ConsumerFactory,
        stream_done_cb=None,
        **cb_kwargs,
    ):
        self._open_streamer()
        return self.streamer.register_stream_processing(channel, topic, factory, stream_done_cb, **cb_kwargs)

    def stream_objects(
        self,
        channel: str,
        topic: str,
        stream_ctx: StreamContext,
        targets: List[str],
        producer: ObjectProducer,
        fl_ctx: FLContext,
        optional=False,
        secure=False,
    ):
        self._open_streamer()
        return self.streamer.stream(
            channel,
            topic,
            stream_ctx,
            targets,
            producer,
            fl_ctx,
            secure=secure,
            optional=optional,
        )

    def shutdown_streamer(self):
        with self._lock:
            if self.streamer:
                self.streamer.shutdown()

    def shutdown_messaging(self):
        self.shutdown_streamer()
        self.shutdown_reliable_messenger()
