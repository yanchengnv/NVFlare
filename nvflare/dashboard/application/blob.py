# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import io
import os
import subprocess
import tempfile

from nvflare.lighter.constants import PropKey
from nvflare.lighter.entity import Project as ProvProject
from nvflare.lighter.impl.aws import AWSBuilder
from nvflare.lighter.impl.azure import AzureBuilder
from nvflare.lighter.impl.cert import CertBuilder
from nvflare.lighter.impl.signature import SignatureBuilder
from nvflare.lighter.impl.static_file import StaticFileBuilder
from nvflare.lighter.impl.workspace import WorkspaceBuilder
from nvflare.lighter.provisioner import Provisioner

from .cert import deserialize_ca_key
from .models import Client, Project, User
from .store import Store, inc_dl


class DummyLogger:
    """This dummy logger is used to suppress all log messages generated by the Provisioner, except for errors.
    We print error messages to stdout.
    """

    def info(self, msg: str):
        pass

    def error(self, msg: str):
        print(f"ERROR: {msg}")

    def debug(self, msg: str):
        pass

    def warning(self, msg: str):
        pass


def _get_provisioner(root_dir: str, scheme, docker_image=None):
    overseer_agent = {
        "path": "nvflare.ha.dummy_overseer_agent.DummyOverseerAgent",
        "overseer_exists": False,
        "args": {"sp_end_point": "server:8002:8003"},
    }
    builders = [
        WorkspaceBuilder(),
        StaticFileBuilder(
            config_folder="config",
            scheme=scheme,
            docker_image=docker_image,
            overseer_agent=overseer_agent,
        ),
        AWSBuilder(),
        AzureBuilder(),
        CertBuilder(),
        SignatureBuilder(),
    ]
    return Provisioner(root_dir, builders)


def gen_server_blob(key):
    return _gen_kit(key)


def _gen_kit(download_key, prepare_target_cb=None, **cb_kwargs):
    u = Store.get_user(1)
    super_user = u.get("user")
    fl_port = 8002
    admin_port = 8003
    with tempfile.TemporaryDirectory() as tmp_dir:
        project = Project.query.first()
        scheme = project.scheme if hasattr(project, "scheme") else "grpc"
        docker_image = project.app_location.split(" ")[-1] if project.app_location else "nvflare/nvflare"
        provisioner = _get_provisioner(tmp_dir, scheme, docker_image)

        # the root key is protected by password
        root_pri_key = deserialize_ca_key(project.root_key)

        # need to serialize without password
        # serialized_root_private_key = serialize_pri_key(root_pri_key)

        prov_project = ProvProject(
            project.short_name,
            project.description,
            props={
                "api_version": 3,
            },
            root_private_key=root_pri_key,
            serialized_root_cert=project.root_cert,
        )

        # use org of superuser
        org = super_user.get("organization", "nvflare")
        server_name = project.server1
        server = prov_project.set_server(
            name=server_name,
            org=org,
            props={
                "fed_learn_port": fl_port,
                "admin_port": admin_port,
                "default_host": server_name,
            },
        )

        target = server
        if prepare_target_cb is not None:
            target = prepare_target_cb(prov_project, **cb_kwargs)

        ctx = provisioner.provision(prov_project, logger=DummyLogger())
        result_dir = ctx.get_result_location()
        ent_dir = os.path.join(result_dir, target.name)
        run_args = ["zip", "-rq", "-P", download_key, "tmp.zip", "."]
        subprocess.run(run_args, cwd=ent_dir)
        fileobj = io.BytesIO()
        with open(os.path.join(ent_dir, "tmp.zip"), "rb") as fo:
            fileobj.write(fo.read())
        fileobj.seek(0)
    return fileobj, f"{target.name}.zip"


def gen_client_blob(key, id):
    return _gen_kit(key, _prepare_client, client_id=id)


def _prepare_client(prov_project: ProvProject, client_id):
    client = Client.query.get(client_id)
    inc_dl(Client, client_id)
    return prov_project.add_client(name=client.name, org=client.organization.name, props={})


def gen_user_blob(key, id):
    return _gen_kit(key, _prepare_user, user_id=id)


def _prepare_user(prov_project: ProvProject, user_id):
    user = User.query.get(user_id)
    inc_dl(User, user_id)
    admin = prov_project.add_admin(name=user.email, org=user.organization.name, props={PropKey.ROLE: user.role.name})
    return admin
