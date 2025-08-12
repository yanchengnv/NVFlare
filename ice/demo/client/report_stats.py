import os.path
import uuid

from ice.client.subproc import Subproc
from ice.defs import PropKey, StatusCode
from ice.utils import make_file_refs
from nvflare.apis.fl_context import FLContext

RESULT_FILE = "ICE_RESULT_FILE"


class ReportStats(Subproc):

    def __init__(self):
        Subproc.__init__(self, supported_topics="report.stats")

    def get_command(self, topic: str, req_data, fl_ctx: FLContext) -> (str, str, dict):
        dest_file = f"/tmp/stats_{uuid.uuid4()}.json"
        cmd = "python -m ice.demo.client.run_stats"
        env = {RESULT_FILE: dest_file}

        # use fl_ctx to keep contextual data
        # do not keep such info in self - because it is a singleton
        fl_ctx.set_prop(RESULT_FILE, dest_file, private=True, sticky=False)
        return cmd, ".", env

    def process_result(self, process_rc: int, topic: str, req_data, fl_ctx: FLContext):
        dest_file = fl_ctx.get_prop(RESULT_FILE)
        if not os.path.exists(dest_file):
            # the subproc failed to create result
            return {PropKey.STATUS: StatusCode.ERROR, PropKey.DETAIL: f"no result file: {process_rc=}"}

        # return result file back
        # send the files back to admin - testing
        refs = make_file_refs({"stats": dest_file}, fl_ctx, 10.0)
        return {
            PropKey.STATUS: StatusCode.OK,
            PropKey.DATA: {
                "type": "age group stats",
                PropKey.ATTACHMENTS: refs,
            },
        }
