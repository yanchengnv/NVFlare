from typing import Any, List, Optional, Union

from ice.server.cmd_handler import CmdHandler
from nvflare.apis.fl_context import FLContext


class Relay(CmdHandler):

    def __init__(self, supported_topics: Union[str, List[str]], request_timeout: float = 2.0):
        CmdHandler.__init__(self, supported_topics, request_timeout)

    def prepare_request(self, cmd: str, data: dict, fl_ctx: FLContext) -> Any:
        return data

    def process_replies(self, cmd: str, cmd_data: dict, replies: dict, fl_ctx: FLContext) -> Optional[dict]:
        return replies
