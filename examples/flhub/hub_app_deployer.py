# Copyright (c) 2021-2022, NVIDIA CORPORATION.  All rights reserved.
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
import os
import shutil

from nvflare.apis.app_deployer_spec import AppDeployerSpec, FLContext
from nvflare.apis.fl_constant import WorkspaceConstants, SystemComponents
from nvflare.apis.job_def import JobMetaKey
from nvflare.apis.job_def_manager_spec import JobDefManagerSpec
from nvflare.apis.workspace import Workspace
from nvflare.fuel.utils.dict_utils import update_components
from nvflare.apis.utils.job_utils import load_job_def

# TBD: this is a violation: private stuff should not be exposed to examples
from nvflare.private.fed.utils.app_deployer import AppDeployer
from nvflare.private.fed.server.job_meta_validator import JobMetaValidator


class HubAppDeployer(AppDeployerSpec):

    def __init__(self):
        AppDeployerSpec.__init__(self)

    def deploy(self,
               workspace: Workspace,
               job_id: str,
               job_meta: dict,
               app_name: str,
               app_data: bytes,
               fl_ctx: FLContext) -> str:
        # step 1: deploy the T1 app into the workspace
        deployer = AppDeployer()
        err = deployer.deploy(workspace, job_id, job_meta, app_name, app_data, fl_ctx)
        if err:
            return err

        server_app_config_path = workspace.get_server_app_config_file_path(job_id)
        if not os.path.exists(server_app_config_path):
            return f"missing {server_app_config_path}"

        # step 2: make a copy of the app for T2
        t1_run_dir = workspace.get_run_dir(job_id)
        t2_job_id = job_id + "_t2"
        t2_run_dir = workspace.get_run_dir(t2_job_id)
        shutil.copytree(t1_run_dir, t2_run_dir)

        # step 3: modify the T1 client's config_fed_client.json to use HubExecutor
        # simply use t1_config_fed_client.json in the site folder
        t1_client_app_config_path = workspace.get_file_path_in_site_config(
            "t1_" + WorkspaceConstants.CLIENT_APP_CONFIG)

        if not os.path.exists(t1_client_app_config_path):
            return f"missing {t1_client_app_config_path}"

        shutil.copyfile(t1_client_app_config_path,
                        workspace.get_client_app_config_file_path(job_id))

        # step 4: modify T2 server's config_fed_server.json to use HubShareableGenerator
        t2_server_app_config_path = workspace.get_server_app_config_file_path(t2_job_id)
        if not os.path.exists(t2_server_app_config_path):
            return f"missing {t2_server_app_config_path}"

        t2_server_component_file = workspace.get_file_path_in_site_config(
            "t2_server_components.json")

        if not os.path.exists(t2_server_component_file):
            return f"missing {t2_server_component_file}"

        with open(t2_server_app_config_path) as file:
            t2_server_app_config_dict = json.load(file)

        with open(t2_server_component_file) as file:
            t2_server_component_dict = json.load(file)

        # update components in the server's config with changed components
        # This will replace shareable_generator with the one defined in t2_server_components.json
        update_components(target_dict=t2_server_app_config_dict, from_dict=t2_server_component_dict)

        # recreate T2's server app config file
        with open(t2_server_app_config_path, "w") as f:
            json.dump(t2_server_app_config_dict, f, indent=4)

        # step 5: submit T2 app (as a job) to T1's job store
        t2_job_def = load_job_def(
            from_path=workspace.get_root_dir(),
            def_name=t2_job_id
        )

        job_validator = JobMetaValidator()
        valid, error, meta = job_validator.validate(t2_job_id, t2_job_def)
        if not valid:
            return f"invalid T2 job def: {error}"

        # make sure meta contains the right job ID
        t2_jid = meta.get(JobMetaKey.JOB_ID.value, None)
        if not t2_jid:
            return "missing Job ID from T2 meta!"

        if job_id != t2_jid:
            return f"T2 Job ID {t2_jid} != T1 Job ID {job_id}"

        engine = fl_ctx.get_engine()
        job_manager = engine.get_component(SystemComponents.JOB_MANAGER)
        assert isinstance(job_manager, JobDefManagerSpec)
        job_manager.create(meta, t2_job_def, fl_ctx)

        # step 6: remove the temporary job def for T2
        shutil.rmtree(t2_run_dir)
        return ""
