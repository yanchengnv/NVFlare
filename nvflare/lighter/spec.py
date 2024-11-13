# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

import os
import shutil
import traceback
from abc import ABC
from typing import List

from nvflare.apis.utils.format_check import name_check


class PropKey:
    CONN_SECURITY = "connection_security"
    CUSTOM_CA_CERT = "custom_ca_cert"
    SSL_MODE = "ssl_mode"


class SSLMode:
    ONE_WAY = "one_way"
    TWO_WAY = "two_way"


class ConnSecurity:
    INSECURE = "insecure"
    SECURE = "secure"


class ConfigEntity:
    def __init__(self, props):
        self.props = props
        self.ssl_mode = None
        self.conn_security = None
        self.custom_ca_cert = None

        # validate properties
        conn_security = self.get_prop(PropKey.CONN_SECURITY)
        if conn_security:
            valid_values = [ConnSecurity.SECURE, ConnSecurity.INSECURE]
            if conn_security not in valid_values:
                raise ValueError(
                    f"invalid value for {PropKey.CONN_SECURITY}: {conn_security}. Must be one of {valid_values}"
                )
        self.conn_security = conn_security

        ssl_mode = self.get_prop(PropKey.SSL_MODE)
        if ssl_mode:
            valid_values = [SSLMode.TWO_WAY, SSLMode.ONE_WAY]
            if ssl_mode not in valid_values:
                raise ValueError(f"invalid value for {PropKey.SSL_MODE}: {ssl_mode}. Must be one of {valid_values}")
        self.ssl_mode = ssl_mode

        custom_ca_cert = self.get_prop(PropKey.CUSTOM_CA_CERT)
        if custom_ca_cert:
            if not os.path.isfile(custom_ca_cert):
                raise ValueError(f"specified {PropKey.CUSTOM_CA_CERT} {custom_ca_cert} is not a valid file.")
            _, file_extension = os.path.splitext(custom_ca_cert)
            if file_extension != ".pem":
                raise ValueError(f"specified {PropKey.CUSTOM_CA_CERT} {custom_ca_cert} must have '.pem' extension.")
        self.custom_ca_cert = custom_ca_cert

    def get_prop(self, key: str, default=None):
        return self.props.get(key, default)


class Participant(ConfigEntity):
    def __init__(self, type: str, name: str, org: str, enable_byoc: bool = False, *args, **kwargs):
        """Class to represent a participant.

        Each participant communicates to other participant.  Therefore, each participant has its
        own name, type, organization it belongs to, rules and other information.

        Args:
            type (str): server, client, admin or other string that builders can handle
            name (str): system-wide unique name
            org (str): system-wide unique organization
            enable_byoc (bool, optional): whether this participant allows byoc codes to be loaded. Defaults to False.

        Raises:
            ValueError: if name or org is not compliant with characters or format specification.
        """
        ConfigEntity.__init__(self, kwargs)
        err, reason = name_check(name, type)
        if err:
            raise ValueError(reason)
        err, reason = name_check(org, "org")
        if err:
            raise ValueError(reason)
        self.type = type
        self.name = name
        self.org = org
        self.subject = name
        self.enable_byoc = enable_byoc


class Project(ConfigEntity):
    def __init__(self, name: str, description: str, participants: List[Participant], config: dict):
        """A container class to hold information about this FL project.

        This class only holds information.  It does not drive the workflow.

        Args:
            name (str): the project name
            description (str): brief description on this name
            participants (List[Participant]): All the participants that will join this project
            config: the whole config dict of the project

        Raises:
            ValueError: when duplicate name found in participants list
        """
        ConfigEntity.__init__(self, config)
        self.name = name
        all_names = list()
        for p in participants:
            if p.name in all_names:
                raise ValueError(f"Unable to add a duplicate name {p.name} into this project.")
            else:
                all_names.append(p.name)
        self.description = description
        self.participants = participants

    def get_participants_by_type(self, type, first_only=True):
        found = list()
        for p in self.participants:
            if p.type == type:
                if first_only:
                    return p
                else:
                    found.append(p)
        return found


class Builder(ABC):
    def initialize(self, ctx: dict):
        pass

    def build(self, project: Project, ctx: dict):
        pass

    def finalize(self, ctx: dict):
        pass

    def get_wip_dir(self, ctx: dict):
        return ctx.get("wip_dir")

    def get_ws_dir(self, participate: Participant, ctx: dict):
        return os.path.join(self.get_wip_dir(ctx), participate.name)

    def get_kit_dir(self, participant: Participant, ctx: dict):
        return os.path.join(self.get_ws_dir(participant, ctx), "startup")

    def get_transfer_dir(self, participant: Participant, ctx: dict):
        return os.path.join(self.get_ws_dir(participant, ctx), "transfer")

    def get_local_dir(self, participant: Participant, ctx: dict):
        return os.path.join(self.get_ws_dir(participant, ctx), "local")

    def get_state_dir(self, ctx: dict):
        return ctx.get("state_dir")

    def get_resources_dir(self, ctx: dict):
        return ctx.get("resources_dir")


class Provisioner(object):
    def __init__(self, root_dir: str, builders: List[Builder]):
        """Workflow class that drive the provision process.

        Provisioner's tasks:

            - Maintain the provision workspace folder structure;
            - Invoke Builders to generate the content of each startup kit

        ROOT_WORKSPACE Folder Structure::

            root_workspace_dir_name: this is the root of the workspace
                project_dir_name: the root dir of the project, could be named after the project
                    resources: stores resource files (templates, configs, etc.) of the Provisioner and Builders
                    prod: stores the current set of startup kits (production)
                        participate_dir: stores content files generated by builders
                    wip: stores the set of startup kits to be created (WIP)
                        participate_dir: stores content files generated by builders
                    state: stores the persistent state of the Builders

        Args:
            root_dir (str): the directory path to hold all generated or intermediate folders
            builders (List[Builder]): all builders that will be called to build the content
        """
        self.root_dir = root_dir
        self.builders = builders
        self.ctx = None

    def _make_dir(self, dirs):
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

    def _prepare_workspace(self, ctx):
        workspace = ctx.get("workspace")
        wip_dir = os.path.join(workspace, "wip")
        state_dir = os.path.join(workspace, "state")
        resources_dir = os.path.join(workspace, "resources")
        ctx.update(dict(wip_dir=wip_dir, state_dir=state_dir, resources_dir=resources_dir))
        dirs = [workspace, resources_dir, wip_dir, state_dir]
        self._make_dir(dirs)

    def provision(self, project: Project):
        # ctx = {"workspace": os.path.join(self.root_dir, project.name), "project": project}
        workspace = os.path.join(self.root_dir, project.name)

        # project is more static information while ctx is dynamic
        ctx = {
            "workspace": workspace,
            "project": project,
        }
        self._prepare_workspace(ctx)
        try:
            for b in self.builders:
                b.initialize(ctx)

            # call builders!
            for b in self.builders:
                b.build(project, ctx)

            for b in self.builders[::-1]:
                b.finalize(ctx)

        except Exception as ex:
            prod_dir = ctx.get("current_prod_dir")
            if prod_dir:
                shutil.rmtree(prod_dir)
            print("Exception raised during provision.  Incomplete prod_n folder removed.")
            traceback.print_exc()
        finally:
            wip_dir = ctx.get("wip_dir")
            if wip_dir:
                shutil.rmtree(wip_dir)
        return ctx
