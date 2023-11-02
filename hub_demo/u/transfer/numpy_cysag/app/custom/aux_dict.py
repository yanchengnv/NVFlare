import time

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import ReturnCode
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReservedHeaderKey, Shareable, make_reply

request_data = {
    "data_0": None,
    "data_1": True,
    "data_2": 0,
    "data_3": 1.0,
    "data_4": "foobar",
    "data_5": b"foobar",
    "data_6": bytearray(b"foobar"),
    "data_7": memoryview(b"foobar"),
    "data_8": ["foo", "bar"],
    "data_9": {"foo": 1, "bar": 2},
}


class AuxDictSender(FLComponent):
    def __init__(self):
        FLComponent.__init__(self)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type in [EventType.END_RUN, EventType.BEFORE_TASK_EXECUTION]:
            self._send_request(fl_ctx)

    def _send_request(self, fl_ctx: FLContext):
        self.logger.info("sending aux dict to server ...")

        request = Shareable()
        request["data"] = request_data

        engine = fl_ctx.get_engine()
        reply = engine.send_aux_request(targets=[], topic="dict_test", request=request, timeout=5.0, fl_ctx=fl_ctx)
        self.log_info(fl_ctx, f"@@@ server reply:{reply} at {time.ctime()}")


class AuxDictReceiver(FLComponent):
    def __init__(self):
        FLComponent.__init__(self)

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.START_RUN:
            engine = fl_ctx.get_engine()
            engine.register_aux_message_handler(topic="dict_test", message_handle_func=self._process_req)

    def _process_req(self, topic: str, request: Shareable, fl_ctx: FLContext):
        self.log_info(
            fl_ctx,
            f"@@@ AuxHandler got the send_result_test message !!! topic:{request.get_header(ReservedHeaderKey.TOPIC)}",
        )
        self.log_info(fl_ctx, f"@@@ Data in the aux message: {request.get('data')}")
        data = request.get("data")
        if data != request_data:
            self.log_error(fl_ctx, f"req data mismatch: {request_data} != {data}")
            raise RuntimeError(
                f"""
                            Server received corrupted data from client.
                            Original: {request_data}
                            Received: {data}
                        """
            )
        return make_reply(ReturnCode.OK)
