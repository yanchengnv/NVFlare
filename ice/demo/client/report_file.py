import os

from ice.defs import PropKey, StatusCode
from ice.req_handler import RequestHandler
from ice.utils import make_file_refs, process_file_refs
from nvflare.apis.fl_context import FLContext


class ReportFile(RequestHandler):

    def __init__(self, file_download_timeout=2.0):
        RequestHandler.__init__(self, "report.file")
        self.file_download_timeout = file_download_timeout

    def handle_request(self, topic: str, data, fl_ctx: FLContext):
        self.log_debug(fl_ctx, f"got server request {topic=} {data=}")
        assert isinstance(data, dict)
        file_refs = data.get(PropKey.ATTACHMENTS)

        if not isinstance(file_refs, dict):
            self.log_error(fl_ctx, f"bad file refs in request: expect dict but got {type(file_refs)}")
            return {
                PropKey.STATUS: StatusCode.ERROR,
                PropKey.DETAIL: "bad file refs",
            }

        process_file_refs(file_refs, fl_ctx, self.file_download_timeout)

        req_data = data.get(PropKey.DATA)
        self.log_info(fl_ctx, f"got request data: {req_data}")

        # send the files back to admin - testing
        reply_files = {}
        for file_name, ref in file_refs.items():
            reply_files[file_name] = ref[PropKey.FILE_LOCATION]
        reply_refs = make_file_refs(reply_files, fl_ctx, self.file_download_timeout, self._delete_temp_files)
        return {
            PropKey.STATUS: StatusCode.OK,
            PropKey.DATA: "received",
            PropKey.ATTACHMENTS: reply_refs,
        }

    def _delete_temp_files(self, tx_id, file_names):
        for f in file_names:
            os.remove(f)
            self.logger.info(f"deleted temp file {f}")
