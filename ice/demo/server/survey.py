from ice.server.cmd_handler import CmdHandler
from nvflare.apis.fl_context import FLContext


class Survey(CmdHandler):

    def __init__(self):
        CmdHandler.__init__(self, "survey")

    def prepare_request(self, cmd: str, data: dict, fl_ctx: FLContext):
        return {"aspect": "age"}

    def process_replies(self, cmd: str, cmd_data: dict, replies: dict, fl_ctx: FLContext):
        total = 0
        count = 0
        for c, r in replies.items():
            if not r:
                replies[c] = "no result"
            else:
                total += r
                count += 1
        if count > 0:
            replies["avg"] = total / count
        return replies
