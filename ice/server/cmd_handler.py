from abc import abstractmethod
from typing import List, Optional, Union

from ice.defs import REQUEST_TOPIC, PropKey
from ice.req_handler import RequestHandler
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import ReturnCode, Shareable


class CmdHandler(RequestHandler):

    def __init__(self, supported_commands: Union[str, List[str]], request_timeout: float = 2.0):
        RequestHandler.__init__(self, supported_commands)
        self.request_timeout = request_timeout

    @abstractmethod
    def prepare_request(self, cmd: str, data: dict, fl_ctx: FLContext) -> dict:
        pass

    @abstractmethod
    def process_replies(self, cmd: str, cmd_data: dict, replies: dict, fl_ctx: FLContext) -> Optional[dict]:
        pass

    def handle_request(self, cmd: str, data: dict, fl_ctx: FLContext):
        # get request data
        req = self.prepare_request(cmd, data, fl_ctx)
        shareable = Shareable({PropKey.DATA: req})
        shareable.set_header(PropKey.TOPIC, cmd)
        engine = fl_ctx.get_engine()
        replies = engine.send_aux_request(
            targets=[], topic=REQUEST_TOPIC, request=shareable, timeout=self.request_timeout, fl_ctx=fl_ctx
        )

        # replies contain a Shareable for each site
        assert isinstance(replies, dict)
        replies_to_process = {}
        for client_name, reply in replies.items():
            assert isinstance(reply, Shareable)
            rc = reply.get_return_code(default=ReturnCode.OK)
            if rc != ReturnCode.OK:
                self.log_error(fl_ctx, f"error from client {client_name}: {rc}")
                replies_to_process[client_name] = None
            else:
                replies_to_process[client_name] = reply.get(PropKey.DATA)

        return self.process_replies(cmd, data, replies_to_process, fl_ctx)
