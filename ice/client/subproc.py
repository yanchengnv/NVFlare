import os
import shlex
import subprocess
from abc import abstractmethod
from typing import Any, List, Union

from ice.req_handler import RequestHandler
from nvflare.apis.fl_context import FLContext


class Subproc(RequestHandler):

    def __init__(self, supported_topics: Union[str, List[str]], process_timeout=10.0):
        RequestHandler.__init__(self, supported_topics)
        self.process_timeout = process_timeout

    @abstractmethod
    def get_command(self, topic: str, req_data, fl_ctx: FLContext) -> (str, str, dict):
        """Get the command for popen the subprocess

        Returns: tuple of: command to be run, current work dir, extra env vars

        """
        pass

    @abstractmethod
    def process_result(self, process_rc: int, topic: str, req_data, fl_ctx: FLContext):
        pass

    def handle_request(self, topic: str, data: Any, fl_ctx: FLContext) -> Any:
        cmd, cwd, cmd_env = self.get_command(topic, data, fl_ctx)
        command_seq = shlex.split(cmd)

        env = os.environ.copy()
        if cmd_env:
            env.update(cmd_env)

        self.log_info(fl_ctx, f"run process command: {cmd}")
        process = subprocess.Popen(command_seq, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd, env=env)
        rc = process.wait(self.process_timeout)
        return self.process_result(rc, topic, data, fl_ctx)
