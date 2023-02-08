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
import sys

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_component import FLComponent
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

from .attestation_helper import AttestationHelper


_KEY_PARTICIPANTS = "cc.participants"


class AttestationManager(FLComponent):

    def __init__(
            self,
            verifiers: list):
        FLComponent.__init__(self)
        self.site_name = None
        self.helper = None
        self.verifiers = verifiers
        self.tokens = {}    # used by the Server to keep tokens of all clients
        self.valid_participants = {}  # do not validate valid participants repeatedly

    def _prepare_token_for_login(self, fl_ctx: FLContext):
        # client side
        my_token = self.helper.get_token()
        fl_ctx.set_prop(key=FLContextKey.CLIENT_TOKEN, value=my_token, sticky=False, private=True)

    def _add_client_token(self, fl_ctx: FLContext):
        # server side
        token = fl_ctx.get_prop(key=FLContextKey.CLIENT_TOKEN, default=None)
        client_name = fl_ctx.get_prop(key=FLContextKey.CLIENT_NAME, default="")
        self.tokens[client_name] = token

    def _prepare_for_attestation(self, fl_ctx: FLContext) -> str:
        # both server and client sides
        self.site_name = fl_ctx.get_identity_name()
        self.helper = AttestationHelper(
            site_name=self.site_name,
            verifiers=self.verifiers
        )
        ok = self.helper.prepare()
        if not ok:
            return "failed to attest"
        self.tokens[self.site_name] = self.helper.get_token()
        return ""

    def _block_job(self, reason: str, fl_ctx: FLContext):
        job_id = fl_ctx.get_prop(FLContextKey.CURRENT_JOB_ID, "")
        self.log_error(fl_ctx, f"Job {job_id} is blocked: {reason}")
        fl_ctx.set_prop(
            key=FLContextKey.JOB_BLOCK_REASON,
            value=reason
        )

    def _check_participants_for_client(self, fl_ctx: FLContext) -> str:
        # Client side
        resource_specs = fl_ctx.get_prop(FLContextKey.CLIENT_RESOURCE_SPECS, None)
        if not resource_specs:
            return f"missing '{FLContextKey.CLIENT_RESOURCE_SPECS}' prop in fl_ctx"

        if not isinstance(resource_specs, dict):
            return f"bad '{FLContextKey.CLIENT_RESOURCE_SPECS}' prop in fl_ctx: "\
                f"must be a dict but got {type(resource_specs)}"

        participants = resource_specs.get(_KEY_PARTICIPANTS, None)
        if not participants:
            return f"bad '{FLContextKey.CLIENT_RESOURCE_SPECS}' prop in fl_ctx: "\
                f"missing '{_KEY_PARTICIPANTS}'"

        if not isinstance(participants, dict):
            return f"bad '{FLContextKey.CLIENT_RESOURCE_SPECS}' prop in fl_ctx: " \
                   f"invalid '{_KEY_PARTICIPANTS}' - expect dict but got {type(participants)}"

        participants_to_validate = {}
        for p, t in participants.items():
            if p not in self.valid_participants:
                participants_to_validate[p] = t

        if not participants_to_validate:
            return ""

        return self._validate_participants(participants_to_validate)

    def _validate_participants(self, participants) -> str:
        err = ""
        participants.pop(self.site_name, None)
        result = self.helper.validate_participants(participants)
        assert isinstance(result, dict)
        for p, ok in result.items():
            if ok:
                self.valid_participants[p] = True
            else:
                err = f"participant {p} does not meet CC requirement"
        return err

    def _check_participants_for_server(self, fl_ctx: FLContext) -> str:
        participants = fl_ctx.get_prop(FLContextKey.JOB_PARTICIPANTS)
        if not participants:
            return f"missing '{FLContextKey.JOB_PARTICIPANTS}' prop in fl_ctx"

        if not isinstance(participants, list):
            return f"bad value for {FLContextKey.JOB_PARTICIPANTS} in fl_ctx: expect list bot got {type(participants)}"

        participant_tokens = {}
        for p in participants:
            assert isinstance(p, str)
            if p not in self.tokens:
                return f"no token available for participant {p}"
            participant_tokens[p] = self.tokens[p]

        err = self._validate_participants(participant_tokens)
        if err:
            return err

        participant_tokens[self.site_name] = self.tokens[self.site_name]
        resource_specs = fl_ctx.get_prop(FLContextKey.CLIENT_RESOURCE_SPECS, None)
        if resource_specs is None:
            return f"missing '{FLContextKey.CLIENT_RESOURCE_SPECS}' prop in fl_ctx"

        # add "participants" to each client's resource spec so the client side can validate
        for client_name, spec in resource_specs:
            if not isinstance(spec, dict):
                return f"bad resource spec for client {client_name}: expect dict but got {type(spec)}"
            spec[_KEY_PARTICIPANTS] = participant_tokens
        return ""

    def handle_event(self, event_type: str, fl_ctx: FLContext):
        if event_type == EventType.SYSTEM_BOOTSTRAP:
            try:
                err = self._prepare_for_attestation(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in attestation preparation")
                err = "exception occurred"
            if err:
                self.log_critical(fl_ctx, err, fire_event=False)
                sys.exit(-1)
        elif event_type == EventType.BEFORE_CLIENT_REGISTER:
            # On client side
            self._prepare_token_for_login(fl_ctx)
        elif event_type == EventType.CLIENT_REGISTERED:
            # Server side
            self._add_client_token(fl_ctx)
        elif event_type == EventType.BEFORE_CHECK_RESOURCE_MANAGER:
            # Client side
            try:
                err = self._check_participants_for_client(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in validating participants")
                err = "exception occurred"
            if err:
                self._block_job(err, fl_ctx)
        elif event_type == EventType.BEFORE_CHECK_CLIENT_RESOURCES:
            # Server side
            try:
                err = self._check_participants_for_server(fl_ctx)
            except:
                self.log_exception(fl_ctx, "exception in validating participants")
                err = "exception occurred"
            if err:
                self._block_job(err, fl_ctx)
