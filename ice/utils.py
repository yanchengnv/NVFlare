from typing import Any, Dict

from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.fuel.f3.streaming.file_downloader import FileDownloader

from .defs import EventType, PropKey


def dispatch_request(comp: FLComponent, topic: str, data: dict, abort_signal, ctx: FLContext):
    ctx.set_prop(PropKey.TOPIC, topic, private=True, sticky=False)
    ctx.set_prop(PropKey.DATA, data, private=True, sticky=False)
    if abort_signal:
        ctx.set_prop(FLContextKey.RUN_ABORT_SIGNAL, abort_signal, private=True, sticky=True)
    comp.fire_event(EventType.REQUEST_RECEIVED, ctx)

    exceptions = ctx.get_prop(FLContextKey.EXCEPTIONS)
    if exceptions:
        comp.log_error(ctx, f"exception encountered executing the command {topic}")
        return None

    return ctx.get_prop(PropKey.RESULT)


def make_file_refs(files: Dict[str, str], fl_ctx: FLContext, download_timeout: float, timeout_cb=None, **cb_kwargs):
    engine = fl_ctx.get_engine()
    cell = engine.get_cell()
    return make_file_refs_for_cell(files, cell, download_timeout, timeout_cb, **cb_kwargs)


def make_file_refs_for_cell(files: Dict[str, str], cell, download_timeout: float, timeout_cb=None, **cb_kwargs):
    fqcn = cell.get_fqcn()
    tx_id = FileDownloader.new_transaction(
        cell,
        download_timeout,
        timeout_cb=timeout_cb,
        **cb_kwargs,
    )
    refs = {}
    for file_name, file_path in files.items():
        ref_id = FileDownloader.add_file(tx_id, file_path)
        refs[file_name] = {
            PropKey.FILE_REF_ID: ref_id,
            PropKey.FQCN: fqcn,
        }
    return refs


def process_file_refs(file_refs: Dict[str, Dict], fl_ctx: FLContext, download_timeout: float):
    engine = fl_ctx.get_engine()
    cell = engine.get_cell()
    download_files(cell, file_refs, download_timeout)


def download_files(cell, d: Any, timeout):
    if isinstance(d, list):
        for v in d:
            download_files(cell, v, timeout)
    elif isinstance(d, dict):
        if PropKey.FILE_REF_ID in d and PropKey.FQCN in d:
            # this is a file ref
            err, location = FileDownloader.download_file(
                cell=cell,
                from_fqcn=d[PropKey.FQCN],
                ref_id=d[PropKey.FILE_REF_ID],
                per_request_timeout=timeout,
            )
            if err:
                d[PropKey.FILE_ERROR] = err
            else:
                d[PropKey.FILE_LOCATION] = location
        else:
            for v in d.values():
                download_files(cell, v, timeout)
