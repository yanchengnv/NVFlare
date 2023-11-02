from nvflare.apis.filter import Filter
from nvflare.apis.fl_context import FLContext
from nvflare.apis.fl_exception import UnsafeJobError
from nvflare.apis.shareable import Shareable


class UnsafeJobDetector(Filter):
    def process(self, shareable: Shareable, fl_ctx: FLContext) -> Shareable:
        raise UnsafeJobError("this task is unsafe")
