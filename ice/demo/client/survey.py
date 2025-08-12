import random

from ice.req_handler import RequestHandler
from nvflare.apis.fl_context import FLContext


class Survey(RequestHandler):

    def __init__(self):
        RequestHandler.__init__(self, "survey")

    def handle_request(self, topic: str, data, fl_ctx: FLContext):
        self.log_debug(fl_ctx, f"got server request {topic=} {data=}")
        assert isinstance(data, dict)
        return random.randint(0, 200)
