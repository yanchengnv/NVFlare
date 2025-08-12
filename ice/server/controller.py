import time

from ice.defs import CONFIG_TASK_NAME, PropKey, StatusCode
from ice.utils import dispatch_request
from nvflare.apis.controller_spec import Client, ClientTask, Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.impl.controller import Controller
from nvflare.apis.shareable import ReturnCode, Shareable
from nvflare.apis.signal import Signal


class IceController(Controller):
    def __init__(
        self,
        cmd_timeout: float = 2.0,
        config_data: dict = None,
        config_task_timeout: int = 2,
    ):
        Controller.__init__(self)
        if not config_data:
            config_data = {}

        self.cmd_timeout = cmd_timeout
        self.app_done = False
        self.abort_signal = None
        self.config_data = config_data
        self.config_task_timeout = config_task_timeout
        self.client_statuses = {}  # client name => bool
        self.ready = False

    def start_controller(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()

        engine.register_app_command(
            topic="hello",
            cmd_func=self._handle_hello,
        )

        engine.register_app_command(
            topic="bye",
            cmd_func=self._handle_bye,
        )

        engine.register_app_command(
            topic="*",
            cmd_func=self._handle_command,
        )

    def stop_controller(self, fl_ctx: FLContext):
        self.app_done = True

    def _handle_command(self, topic: str, data, fl_ctx: FLContext):
        if not self.ready:
            return {PropKey.STATUS: StatusCode.NOT_READY}

        if not isinstance(data, dict):
            return {PropKey.STATUS: StatusCode.ERROR, PropKey.DETAIL: f"command data must be dict but got {type(data)}"}

        engine = fl_ctx.get_engine()
        with engine.new_context() as ctx:
            assert isinstance(ctx, FLContext)
            result = dispatch_request(self, topic, data, self.abort_signal, ctx)
            if result is None:
                self.log_error(fl_ctx, f"no result for cmd {topic}")
                return {PropKey.STATUS: StatusCode.NO_REPLY}
            elif not isinstance(result, dict):
                self.log_error(
                    fl_ctx, f"processing error for cmd {topic}: expect result to be dict but got {type(result)}"
                )
                return {PropKey.STATUS: StatusCode.ERROR}
            elif PropKey.STATUS not in result:
                result[PropKey.STATUS] = StatusCode.OK
            return result

    def _handle_bye(self, topic: str, data, fl_ctx: FLContext) -> dict:
        self.app_done = True
        return {PropKey.STATUS: StatusCode.OK}

    def _handle_hello(self, topic: str, data, fl_ctx: FLContext):
        if self.ready:
            return {PropKey.STATUS: StatusCode.OK}
        else:
            return {PropKey.STATUS: StatusCode.NOT_READY}

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        self.abort_signal = abort_signal

        # configure all sites
        task = Task(
            name=CONFIG_TASK_NAME,
            data=Shareable({PropKey.DATA: self.config_data}),
            timeout=self.config_task_timeout,
            result_received_cb=self._process_config_reply,
        )

        engine = fl_ctx.get_engine()
        all_clients = engine.get_clients()
        num_clients = len(all_clients)
        for c in all_clients:
            assert isinstance(c, Client)
            self.client_statuses[c.name] = False

        start_time = time.time()
        self.broadcast_and_wait(
            task=task,
            min_responses=num_clients,
            abort_signal=abort_signal,
            fl_ctx=fl_ctx,
        )
        time_taken = time.time() - start_time
        self.log_info(fl_ctx, f"client configuration took {time_taken} seconds")

        failed_clients = []
        for c, ok in self.client_statuses.items():
            if not ok:
                failed_clients.append(c)

        if failed_clients:
            self.system_panic(
                f"failed to configure clients {failed_clients}",
                fl_ctx,
            )
            return

        self.log_info(fl_ctx, f"successfully configured clients {self.client_statuses.keys()}")
        self.ready = True

        # wait until the job is aborted or stopped by app command
        while not abort_signal.triggered and not self.app_done:
            time.sleep(1.0)

    def _process_config_reply(self, client_task: ClientTask, fl_ctx: FLContext):
        result = client_task.result
        client_name = client_task.client.name

        rc = result.get_return_code()
        if rc == ReturnCode.OK:
            self.log_info(fl_ctx, f"successfully configured client {client_name}")
            self.client_statuses[client_name] = True
        else:
            self.log_error(fl_ctx, f"client {client_task.client.name} failed to configure: {rc}")
            self.client_statuses[client_name] = False

    def process_result_of_unknown_task(
        self, client: Client, task_name: str, client_task_id: str, result: Shareable, fl_ctx: FLContext
    ):
        # since we don't assign any tasks, we should never receive any task results
        self.log_error(fl_ctx, f"received result of unknown task {task_name}")
