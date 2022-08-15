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

import logging
import pickle
import json
import os
import sys
from logging.handlers import RotatingFileHandler
from multiprocessing.connection import Listener
from typing import List

from nvflare.apis.fl_constant import WorkspaceConstants, SiteType
from nvflare.apis.fl_context import FLContext
from nvflare.apis.app_validation import AppValidator
from nvflare.fuel.sec.security_content_service import LoadResult, SecurityContentService
from nvflare.fuel.sec.authz import AuthorizationService
from nvflare.fuel.sec.audit import AuditService
from nvflare.private.defs import SSLConstants
from nvflare.private.fed.protos.federated_pb2 import ModelData
from nvflare.private.fed.utils.numproto import bytes_to_proto
from nvflare.security.security import EmptyAuthorizer, FLAuthorizer

from .app_authz import AppAuthzService


def shareable_to_modeldata(shareable, fl_ctx):
    # make_init_proto message
    model_data = ModelData()  # create an empty message

    model_data.params["data"].CopyFrom(make_shareable_data(shareable))

    context_data = make_context_data(fl_ctx)
    model_data.params["fl_context"].CopyFrom(context_data)
    return model_data


def make_shareable_data(shareable):
    return bytes_to_proto(shareable.to_bytes())


def make_context_data(fl_ctx):
    shared_fl_ctx = FLContext()
    shared_fl_ctx.set_public_props(fl_ctx.get_all_public_props())
    props = pickle.dumps(shared_fl_ctx)
    context_data = bytes_to_proto(props)
    return context_data


def add_logfile_handler(log_file):
    root_logger = logging.getLogger()
    main_handler = root_logger.handlers[0]
    file_handler = RotatingFileHandler(log_file, maxBytes=20 * 1024 * 1024, backupCount=10)
    file_handler.setLevel(main_handler.level)
    file_handler.setFormatter(main_handler.formatter)
    root_logger.addHandler(file_handler)


def listen_command(listen_port, engine, execute_func, logger):
    conn = None
    listener = None
    try:
        address = ("localhost", listen_port)
        listener = Listener(address, authkey="client process secret password".encode())
        conn = listener.accept()

        execute_func(conn, engine)

    except Exception as e:
        logger.exception(f"Could not create the listener for this process on port: {listen_port}: {e}.", exc_info=True)
    finally:
        if conn:
            conn.close()
        if listener:
            listener.close()


def _check_secure_content(site_type: str) -> List[str]:
    """To check the security contents.

    Args:
        site_type (str): "server" or "client"

    Returns:
        A list of insecure content.
    """
    if site_type == SiteType.SERVER:
        config_file_name = WorkspaceConstants.SERVER_STARTUP_CONFIG
    else:
        config_file_name = WorkspaceConstants.CLIENT_STARTUP_CONFIG

    insecure_list = []
    data, sig = SecurityContentService.load_json(config_file_name)
    if sig != LoadResult.OK:
        insecure_list.append(config_file_name)

    sites_to_check = data["servers"] if site_type == SiteType.SERVER else [data["client"]]

    for site in sites_to_check:
        for filename in [SSLConstants.CERT, SSLConstants.PRIVATE_KEY, SSLConstants.ROOT_CERT]:
            content, sig = SecurityContentService.load_content(site.get(filename))
            if sig != LoadResult.OK:
                insecure_list.append(site.get(filename))

    if WorkspaceConstants.AUTHORIZATION_CONFIG in SecurityContentService.security_content_manager.signature:
        data, sig = SecurityContentService.load_json(WorkspaceConstants.AUTHORIZATION_CONFIG)
        if sig != LoadResult.OK:
            insecure_list.append(WorkspaceConstants.AUTHORIZATION_CONFIG)

    return insecure_list


def security_init(secure_train: bool,
                  site_org: str,
                  workspace_dir: str,
                  app_validator: AppValidator,
                  site_type: str):
    """To check the security content if running in security mode.

    Args:
       secure_train (bool): if run in secure mode or not.
       site_org: organization of the site
       workspace_dir (str): the workspace to check.
       app_validator: app validator for application validation
       site_type (str): server or client. fed_client.json or fed_server.json
    """
    # initialize the SecurityContentService.
    # must do this before initializing other services since it may be needed by them!
    startup_dir = os.path.join(workspace_dir, WorkspaceConstants.STARTUP_FOLDER_NAME)
    SecurityContentService.initialize(content_folder=startup_dir)

    if secure_train:
        insecure_list = _check_secure_content(site_type=site_type)
        if len(insecure_list):
            print("The following files are not secure content.")
            for item in insecure_list:
                print(item)
            sys.exit(1)

    # initialize the AuditService, which is used by command processing.
    # The Audit Service can be used in other places as well.
    audit_file_name = os.path.join(workspace_dir, WorkspaceConstants.AUDIT_LOG)
    AuditService.initialize(audit_file_name)

    if app_validator:
        AppAuthzService.initialize(app_validator)

    # Initialize the AuthorizationService. It is used by command authorization
    # We use FLAuthorizer for policy processing.
    # AuthorizationService depends on SecurityContentService to read authorization policy file.
    authorizer = None
    if secure_train:
        site_dir = os.path.join(workspace_dir, WorkspaceConstants.SITE_FOLDER_NAME)
        if os.path.exists(site_dir):
            policy_config = os.path.join(site_dir, WorkspaceConstants.AUTHORIZATION_CONFIG)
            if os.path.exists(policy_config):
                policy_config = json.load(open(policy_config, "rt"))
                authorizer = FLAuthorizer(site_org, policy_config)

    if not authorizer:
        authorizer = EmptyAuthorizer()

    _, err = AuthorizationService.initialize(authorizer)

    if err:
        print("AuthorizationService error: {}".format(err))
        sys.exit(1)
