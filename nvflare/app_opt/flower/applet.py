# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import os.path
import threading
import time

import tomli
import tomli_w

from nvflare.apis.fl_context import FLContext
from nvflare.apis.workspace import Workspace
from nvflare.app_common.tie.applet import Applet
from nvflare.app_common.tie.cli_applet import CLIApplet
from nvflare.app_common.tie.defs import Constant as TieConstant
from nvflare.app_common.tie.process_mgr import CommandDescriptor, ProcessManager, run_command, start_process
from nvflare.app_opt.flower.defs import Constant
from nvflare.fuel.utils.grpc_utils import create_channel
from nvflare.security.logging import secure_format_exception


class FlowerClientApplet(CLIApplet):
    def __init__(self, extra_env: dict = None):
        """Constructor of FlowerClientApplet, which extends CLIApplet."""
        CLIApplet.__init__(self)
        self.extra_env = extra_env

    def get_command(self, ctx: dict) -> CommandDescriptor:
        """Implementation of the get_command method required by the super class CLIApplet.
        It returns the CLI command for starting Flower's client app, as well as the full path of the log file
        for the client app.

        Args:
            ctx: the applet run context

        Returns: CLI command for starting client app and name of log file.

        """
        superlink_addr = ctx.get(Constant.APP_CTX_SUPERLINK_ADDR)
        clientapp_api_addr = ctx.get(Constant.APP_CTX_CLIENTAPP_API_ADDR)
        fl_ctx = ctx.get(Constant.APP_CTX_FL_CONTEXT)
        if not isinstance(fl_ctx, FLContext):
            self.logger.error(f"expect APP_CTX_FL_CONTEXT to be FLContext but got {type(fl_ctx)}")
            raise RuntimeError("invalid FLContext")

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        if not isinstance(ws, Workspace):
            self.logger.error(f"expect workspace to be Workspace but got {type(ws)}")
            raise RuntimeError("invalid workspace")

        job_id = fl_ctx.get_job_id()
        app_dir = ws.get_app_dir(job_id)

        """ Example:
        flower-supernode --insecure --grpc-adapter
        --superlink 127.0.0.1:9092
        --clientappio-api-address 127.0.0.1:9094
        """

        cmd = (
            f"flower-supernode --insecure --grpc-adapter "
            f"--superlink {superlink_addr} "
            f"--clientappio-api-address {clientapp_api_addr}"
        )

        # use app_dir as the cwd for flower's client app.
        # this is necessary for client_api to be used with the flower client app for metrics logging
        # client_api expects config info from the "config" folder in the cwd!
        self.logger.info(f"starting flower client app: {cmd}")
        return CommandDescriptor(
            cmd=cmd, cwd=app_dir, env=self.extra_env, log_file_name="client_app_log.txt", stdout_msg_prefix="FLWR-CA"
        )


class FlowerServerApplet(Applet):
    def __init__(
        self,
        database: str,
        superlink_ready_timeout: float,
        superlink_grace_period=1.0,
    ):
        """Constructor of FlowerServerApplet.

        Args:
            database: database spec to be used by the server app
            superlink_ready_timeout: how long to wait for the superlink process to become ready
            superlink_grace_period: how long to wait before stopping superlink after stopping the app
        """
        Applet.__init__(self)
        self._superlink_process_mgr = None
        self.database = database
        self.superlink_ready_timeout = superlink_ready_timeout
        self.superlink_grace_period = superlink_grace_period
        self.run_id = None
        self.last_check_status = None
        self.last_check_time = None
        self.last_check_stopped = False
        self.flower_app_dir = None
        self.flower_run_finished = False
        self.flower_run_stopped = False  # have we issued 'flwr stop'?
        self.flower_run_rc = None
        self._start_error = False
        self.stop_lock = threading.Lock()

    def _start_process(self, name: str, cmd_desc: CommandDescriptor, fl_ctx: FLContext) -> ProcessManager:
        self.logger.info(f"starting {name}: {cmd_desc.cmd}")
        try:
            return start_process(cmd_desc, fl_ctx)
        except Exception as ex:
            self.logger.error(f"exception starting applet: {secure_format_exception(ex)}")
            self._start_error = True

    def _modify_flower_app_config(self, exec_api_addr: str):
        """Currently the exec-api-address must be specified in pyproject.toml to be able to submit to the
        superlink with "flwr run" command.

        Args:
            exec_api_addr:

        Returns:

        """
        config_file = os.path.join(self.flower_app_dir, "pyproject.toml")
        if not os.path.isfile(config_file):
            raise RuntimeError(f"invalid flower app: missing {config_file}")

        with open(config_file, mode="rb") as fp:
            config = tomli.load(fp)

        # add or modify address
        tool = config.get("tool")
        if not tool:
            tool = {}
            config["tool"] = tool

        flwr = tool.get("flwr")
        if not flwr:
            flwr = {}
            tool["flwr"] = flwr

        fed = flwr.get("federations")
        if not fed:
            fed = {}
            flwr["federations"] = fed

        default_mode = fed.get("default")
        if not default_mode:
            default_mode = "local-poc"
            fed["default"] = default_mode

        mode_config = fed.get(default_mode)
        if not mode_config:
            mode_config = {}
            fed[default_mode] = mode_config

        mode_config["address"] = exec_api_addr
        mode_config["insecure"] = True

        # recreate the app config
        with open(config_file, mode="wb") as fp:
            tomli_w.dump(config, fp)

    def start(self, app_ctx: dict):
        """Start the applet.

        Flower requires two processes for server application:
            superlink: this process is responsible for client communication
            server_app: this process performs server side of training.

        We start the superlink first, and wait for it to become ready, then start the server app.
        Each process will have its own log file in the job's run dir. The superlink's log file is named
        "superlink_log.txt". The server app's log file is named "server_app_log.txt".

        Args:
            app_ctx: the run context of the applet.

        Returns:

        """
        # try to start superlink first
        serverapp_api_addr = app_ctx.get(Constant.APP_CTX_SERVERAPP_API_ADDR)
        fleet_api_addr = app_ctx.get(Constant.APP_CTX_FLEET_API_ADDR)
        exec_api_addr = app_ctx.get(Constant.APP_CTX_EXEC_API_ADDR)
        fl_ctx = app_ctx.get(Constant.APP_CTX_FL_CONTEXT)
        if not isinstance(fl_ctx, FLContext):
            self.logger.error(f"expect APP_CTX_FL_CONTEXT to be FLContext but got {type(fl_ctx)}")
            raise RuntimeError("invalid FLContext")

        engine = fl_ctx.get_engine()
        ws = engine.get_workspace()
        if not isinstance(ws, Workspace):
            self.logger.error(f"expect workspace to be Workspace but got {type(ws)}")
            raise RuntimeError("invalid workspace")

        custom_dir = ws.get_app_custom_dir(fl_ctx.get_job_id())
        self.flower_app_dir = custom_dir

        self._modify_flower_app_config(exec_api_addr)

        db_arg = ""
        if self.database:
            db_arg = f"--database {self.database}"

        """ Example:
        flower-superlink --insecure --fleet-api-type grpc-adapter
        --serverappio-api-address 127.0.0.1:9091
        --fleet-api-address 127.0.0.1:9092
        --exec-api-address 127.0.0.1:9093
        """
        superlink_cmd = (
            f"flower-superlink --insecure --fleet-api-type grpc-adapter {db_arg} "
            f"--serverappio-api-address {serverapp_api_addr} "
            f"--fleet-api-address {fleet_api_addr}  "
            f"--exec-api-address {exec_api_addr}"
        )

        cmd_desc = CommandDescriptor(cmd=superlink_cmd, log_file_name="superlink_log.txt", stdout_msg_prefix="FLWR-SL")

        self._superlink_process_mgr = self._start_process(name="superlink", cmd_desc=cmd_desc, fl_ctx=fl_ctx)
        if not self._superlink_process_mgr:
            raise RuntimeError("cannot start superlink process")

        # wait until superlink's fleet_api_addr is ready before starting server app
        # note: the server app will connect to serverapp_api_addr, not fleet_api_addr
        start_time = time.time()
        create_channel(
            server_addr=fleet_api_addr,
            grpc_options=None,
            ready_timeout=self.superlink_ready_timeout,
            test_only=True,
        )
        self.logger.info(f"superlink is ready for server app in {time.time() - start_time} seconds")

        # submitting the server app using "flwr run" command
        # flwr_run_cmd = f"flwr run --format json -c 'address={exec_api_addr}' {custom_dir}"
        flwr_run_cmd = f"flwr run --format json {self.flower_app_dir}"
        run_info = self._run_flower_command(flwr_run_cmd)
        run_id = run_info.get("run-id")
        if not run_id:
            raise RuntimeError(f"invalid result from command '{flwr_run_cmd}': missing run-id")

        self.logger.info(f"submitted Flower App and got run id {run_id}")
        self.run_id = run_id

    def _run_flower_command(self, command: str):
        self.logger.info(f"running flower command: {command}")
        cmd_desc = CommandDescriptor(cmd=command)
        reply = run_command(cmd_desc)
        if not isinstance(reply, str):
            raise RuntimeError(f"failed to run command '{command}': expect reply to be str but got {type(reply)}")

        self.logger.info(f"flower command {command}: {reply=}")
        # the reply must be a json str
        try:
            result = json.loads(reply)
        except Exception as ex:
            err = f"invalid result from command '{command}': {secure_format_exception(ex)}"
            self.logger.error(err)
            raise RuntimeError(err)

        if not isinstance(result, dict):
            err = f"invalid result from command '{command}': expect dict but got {type(result)}"
            self.logger.error(err)
            raise RuntimeError(err)

        success = result.get("success", False)
        if not success:
            err = f"failed command '{command}': {success=}"
            self.logger.error(err)
            raise RuntimeError(err)

        self.logger.info(f"result of {command}: {result}")
        return result

    @staticmethod
    def _stop_process(p: ProcessManager) -> int:
        if not p:
            # nothing to stop
            return 0
        else:
            return p.stop()

    def stop(self, timeout=0.0) -> int:
        """Stop the server applet's superlink.

        Args:
            timeout: how long to wait before forcefully stopping (kill) the process.

        Note: we always stop the process immediately - do not wait for the process to stop itself.

        Returns:

        """
        with self.stop_lock:
            if self.run_id and not self.flower_run_stopped and not self.flower_run_finished:
                # stop the server app
                # we may not be able to issue 'flwr stop' more than once!
                self.flower_run_stopped = True
                flwr_stop_cmd = f"flwr stop --format json {self.run_id} {self.flower_app_dir}"
                try:
                    self._run_flower_command(flwr_stop_cmd)
                except Exception as ex:
                    # ignore exception
                    self.logger.error(f"exception running '{flwr_stop_cmd}': {secure_format_exception(ex)}")

                # wait a while to let superlink and supernodes gracefully stop the app
                time.sleep(self.superlink_grace_period)

            # stop the superlink
            self._stop_process(self._superlink_process_mgr)
            self._superlink_process_mgr = None
            return 0

    @staticmethod
    def _is_process_stopped(p: ProcessManager):
        if p:
            return_code = p.poll()
            if return_code is None:
                return False, 0
            else:
                return True, return_code
        else:
            return True, 0

    def _check_flower_run_status(self):
        # check whether the app is finished
        flwr_ls_cmd = f"flwr ls --format json {self.flower_app_dir}"
        try:
            run_info = self._run_flower_command(flwr_ls_cmd)
        except Exception as ex:
            self.logger.error(f"exception running '{flwr_ls_cmd}': {secure_format_exception(ex)}")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        runs = run_info.get("runs")
        if not runs:
            # the app is no longer there
            return True, 0

        if not isinstance(runs, list):
            self.logger.error(f"invalid result from command '{flwr_ls_cmd}': expect run list but got {type(runs)}")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        run = runs[0]
        if not isinstance(run, dict):
            self.logger.error(f"invalid result from command '{flwr_ls_cmd}': expect run to be dict but got {type(run)}")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        status = run.get("status")
        if not status:
            self.logger.error(f"invalid result from command '{flwr_ls_cmd}': missing status from {run}")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        if not isinstance(status, str):
            self.logger.error(f"invalid result from command '{flwr_ls_cmd}': bad status value '{status}'")
            return True, TieConstant.EXIT_CODE_FATAL_ERROR

        if status.startswith("finished"):
            self.logger.info(f"Flower Run {self.run_id} finished: {status=}")
            self.flower_run_finished = True
            if status.endswith("completed"):
                rc = 0
            else:
                rc = TieConstant.EXIT_CODE_FAILED
            self.flower_run_rc = rc
            return True, rc
        else:
            return False, 0

    def is_stopped(self) -> (bool, int):
        """Check whether the server applet is already stopped

        Returns: a tuple of: whether the applet is stopped, exit code if stopped.

        Note: if either superlink or server app is stopped, we treat the applet as stopped.

        """
        if self._start_error:
            return True, TieConstant.EXIT_CODE_CANT_START

        superlink_stopped, superlink_rc = self._is_process_stopped(self._superlink_process_mgr)
        if superlink_stopped:
            self._superlink_process_mgr = None
            return True, superlink_rc

        if self.flower_run_finished:
            return True, self.flower_run_rc

        with self.stop_lock:
            if not self.last_check_time or time.time() - self.last_check_time >= 2.0:
                self.last_check_stopped, self.last_check_status = self._check_flower_run_status()
                self.last_check_time = time.time()

        return self.last_check_stopped, self.last_check_status
