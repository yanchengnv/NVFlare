from ice.client.executor import IceExecutor
from ice.req_handler import RequestHandler
from ice.server.controller import IceController
from nvflare.job_config.api import FedJob


class ServerConfig:

    def __init__(
        self,
        cmd_timeout: float = 2.0,
        config_data: dict = None,
        config_task_timeout: int = 2,
    ):
        self.cmd_timeout = cmd_timeout
        self.config_data = config_data
        self.config_task_timeout = config_task_timeout


class IceJob(FedJob):

    def __init__(
        self,
        name: str,
        server_config: ServerConfig,
        min_clients: int = 1,
    ):
        FedJob.__init__(self, name, min_clients)
        controller = IceController(
            server_config.cmd_timeout, server_config.config_data, server_config.config_task_timeout
        )
        self.to_server(controller)

        executor = IceExecutor()
        self.to_clients(executor)

    def add_server_handler(self, handler: RequestHandler):
        if not isinstance(handler, RequestHandler):
            raise ValueError(f"handler must be RequestHandler but got {type(handler)}")
        self.to_server(handler)

    def add_client_handler(self, handler: RequestHandler):
        if not isinstance(handler, RequestHandler):
            raise ValueError(f"handler must be RequestHandler but got {type(handler)}")
        self.to_clients(handler)
