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
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable, make_reply
from nvflare.apis.streaming import StreamContext
from nvflare.app_common.streamers.streamer_base import StreamerBase
from nvflare.security.logging import secure_format_exception

RETRIEVER_TX_ID = "_rtr_tx_id_"

_SHORT_WAIT = 0.1


class _Waiter(threading.Event):
    def __init__(self):
        super().__init__()
        self.result = None
        self.last_status_time = time.time()

    def set_result(self, rc: str, data: Any):
        self.result = (rc, data)


class ObjectRetriever(FLComponent, ABC):
    def __init__(
        self,
        topic: str = None,
    ):
        FLComponent.__init__(self)
        class_name = self.__class__.__name__
        if not topic:
            topic = class_name
        self.topic = topic
        self.stream_channel = class_name
        self.tx_table = {}

    @abstractmethod
    def register_stream_processing(
        self,
        channel: str,
        topic: str,
        fl_ctx: FLContext,
        status_cb,
        **cb_kwargs,
    ):
        """Object requester side, which will receive data stream.
        This is called to register the status_cb for received stream.

        Args:
            channel: stream channel
            topic: stream topic
            fl_ctx: FLContext object
            status_cb: the stream status callback to be registered
            **cb_kwargs: kwargs to be passed to the CB

        Returns:

        """
        pass

    @abstractmethod
    def validate_request(self, request: Shareable, fl_ctx: FLContext) -> (str, Any):
        """Object sending side. Called to validate the received retrieval request.

        Args:
            request: the request to be validated
            fl_ctx: FLContext object

        Returns: tuple of (ReturnCode, Validation Data)
        This method should do as much as possible so that the do_stream method won't be called if any error
        is detected (the do_stream method is called in a separate thread).
        The validation data produced by this method will be passed to the do_stream method.

        """
        pass

    @abstractmethod
    def do_stream(
        self,
        target: str,
        request: Shareable,
        fl_ctx: FLContext,
        stream_ctx: StreamContext,
        validation_data: Any,
    ) -> Any:
        """Object sending side. Called to stream data to the requesting side.

        Args:
            target: the requesting site to stream to
            request: the object retrieval request
            fl_ctx: a FLContext object
            stream_ctx: stream context data
            validation_data: the validation data produced by the validate_request method.

        Returns: Any object

        """
        pass

    @abstractmethod
    def get_result(self, stream_ctx: StreamContext) -> (str, Any):
        """Object requesting side, which is also the stream receiving side.
        Called to get the result of the streaming.

        Args:
            stream_ctx: StreamContext object

        Returns: tuple of (ReturnCode, Result Object)

        """
        pass

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            self.register_stream_processing(
                fl_ctx=fl_ctx,
                channel=self.stream_channel,
                topic=self.topic,
                status_cb=self._handle_stream_status,
            )
            engine.register_aux_message_handler(topic=self.topic, message_handle_func=self._handle_request)

    def retrieve(self, from_site: str, fl_ctx: FLContext, progress_timeout: float, **obj_attrs) -> (str, Any):
        """Retrieve an object from a specified site.

        Args:
            from_site: the site to retrieve the object from
            fl_ctx: a FLContext object
            progress_timeout: max number of seconds between received data segments
            **obj_attrs: attributes of the object to be retrieved

        Returns: tuple of (ReturnCode, Retrieved Data)

        """
        engine = fl_ctx.get_engine()
        waiter = _Waiter()
        tx_id = str(uuid.uuid4())
        self.tx_table[tx_id] = waiter
        self.log_info(fl_ctx, f"set waiter for Rtr {tx_id}")

        request = Shareable({RETRIEVER_TX_ID: tx_id})
        if obj_attrs:
            request.update(obj_attrs)

        reply = engine.send_aux_request(
            targets=[from_site], request=request, topic=self.topic, fl_ctx=fl_ctx, timeout=progress_timeout
        )
        # the reply is a dict keyed with site names!
        reply = reply.get(from_site)

        # now the reply is a Shareable object
        if not isinstance(reply, Shareable):
            self.log_error(fl_ctx, f"bad reply from site {from_site}: expect Shareable but got {type(reply)}")
            return ReturnCode.EXECUTION_EXCEPTION, None

        rc = reply.get_return_code()
        if rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"retrieval request rejected by site {from_site}: {rc}")
            return rc, None

        # wait for result until either the result is received or progress timed out
        rc = ReturnCode.OK
        abort_signal = fl_ctx.get_run_abort_signal()
        while True:
            # wait a short time so that we can check other conditions
            if not waiter.wait(_SHORT_WAIT):
                # see whether we have any progress
                if time.time() - waiter.last_status_time > progress_timeout:
                    # no progress for too long
                    self.log_error(fl_ctx, f"no progress for {progress_timeout} seconds")
                    rc = ReturnCode.TIMEOUT
                    break

                if abort_signal and abort_signal.triggered:
                    rc = ReturnCode.TASK_ABORTED
                    break
            else:
                # result available!
                break

        self.tx_table.pop(tx_id, None)
        self.log_info(fl_ctx, f"popped waiter for RTR {tx_id}")
        if waiter.result:
            # If the waiter already got result, we return it.
            # Note that due to racing condition, it is possible that the waiter still got the result
            # even after we determined the streaming is timed out!
            return waiter.result
        else:
            return rc, None

    def _handle_stream_status(self, stream_ctx: StreamContext, fl_ctx: FLContext):
        # On stream receiving side, which is also the requesting side
        tx_id = stream_ctx.get(RETRIEVER_TX_ID)
        waiter = self.tx_table.get(tx_id)
        if not waiter:
            self.log_error(fl_ctx, f"late stream data {tx_id=} after timed out")
            return

        waiter.last_status_time = time.time()
        if not StreamerBase.stream_is_done(stream_ctx):
            self.log_info(fl_ctx, f"streaming not done yet for RTR {tx_id}")
            return

        try:
            rc, result = self.get_result(stream_ctx)
        except Exception as ex:
            self.log_error(fl_ctx, f"Exception when get_result: {secure_format_exception(ex)}")
            rc = ReturnCode.EXECUTION_EXCEPTION
            result = None

        waiter.result = (rc, result)
        waiter.set()
        self.log_info(fl_ctx, f"got result for RTR {tx_id}: {waiter.result}")

    def _handle_request(self, topic, request: Shareable, fl_ctx: FLContext) -> Shareable:
        # On request receiving side, which is also stream sending side.
        tx_id = request.get(RETRIEVER_TX_ID)
        if not tx_id:
            self.log_error(fl_ctx, f"bad request '{topic}': missing {RETRIEVER_TX_ID}")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        peer_ctx = fl_ctx.get_peer_context()
        if not peer_ctx:
            self.log_error(fl_ctx, f"bad request '{topic}': missing peer context")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        if not isinstance(peer_ctx, FLContext):
            self.log_error(fl_ctx, f"bad request '{topic}': bad peer context ({type(peer_ctx)})")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        peer = peer_ctx.get_identity_name()
        if not peer:
            self.log_error(fl_ctx, f"bad request '{topic}': missing peer name")
            return make_reply(ReturnCode.BAD_REQUEST_DATA)

        # validate the request before starting stream
        rc, validated_data = self.validate_request(request, fl_ctx)
        if rc and rc != ReturnCode.OK:
            self.log_error(fl_ctx, f"bad request '{topic}': failed validation ({rc})")
            return make_reply(rc)

        # start the streaming in a separate thread so that we can respond to the requestor.
        self.log_info(fl_ctx, "About to start streaming ...")
        t = threading.Thread(target=self._do_stream, args=(request, fl_ctx, validated_data), daemon=True)
        t.start()
        return make_reply(ReturnCode.OK)

    def _do_stream(self, request: Shareable, fl_ctx: FLContext, validated_data: Any):
        # On request receiving side, which is also stream sending side.
        tx_id = request.get(RETRIEVER_TX_ID)
        self.log_info(fl_ctx, f"Started streaming for RTR Request {tx_id}")

        stream_ctx = {RETRIEVER_TX_ID: tx_id}
        peer_ctx = fl_ctx.get_peer_context()
        peer = peer_ctx.get_identity_name()
        try:
            # start streaming object to the peer
            result = self.do_stream(peer, request, fl_ctx, stream_ctx, validated_data)
            self.log_info(fl_ctx, f"finished streaming for RTR {tx_id}: {result=}")
        except Exception as ex:
            self.log_error(fl_ctx, f"streaming exception occurred: {secure_format_exception(ex)}")
