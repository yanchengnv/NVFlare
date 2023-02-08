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
import os.path
from typing import Dict
from .attestation import Attestation, Environment, Devices


class VerifierProp:

    DEVICES = "devices"   # GPU, CPU, etc.
    ENV = "env"
    URL = "url"
    APPRAISAL_POLICY_FILE = "appraisal_policy_file"
    RESULT_POLICY_FILE = "result_policy_file"


class Device:

    GPU = "gpu"
    CPU = "cpu"
    NIC = "nic"
    OS = "os"
    DPU = "dpu"

    mapping = {
        GPU: Devices.GPU,
        CPU: Devices.CPU,
        NIC: Devices.NIC,
        OS: Devices.OS,
        DPU: Devices.DPU
    }


class Env:

    TEST = "test"
    LOCAL = "local"
    AZURE = "azure"
    GCP = "gcp"

    mapping = {
        TEST: Environment.TEST,
        LOCAL: Environment.LOCAL,
        AZURE: Environment.AZURE,
        GCP: Environment.GCP
    }


class AttestationHelper(object):

    def __init__(
            self,
            site_name: str,
            verifiers: list
    ):
        """Create an AttestationHelper instance

        Args:
            site_name: name of the site
            verifiers: dict that specifies verifiers to be used
        """
        self.site_name = site_name
        self.verifiers = verifiers
        attestation = Attestation()
        attestation.set_name(site_name)
        self.attestation = attestation
        self.token = None
        for v in verifiers:
            assert isinstance(v, dict)
            url = None
            env = None
            devices = 0
            appraisal_policy_file = None
            result_policy_file = None
            for prop, value in v.items():
                if prop == VerifierProp.URL:
                    url = value
                elif prop == VerifierProp.ENV:
                    env = Env.mapping.get(value)
                elif prop == VerifierProp.DEVICES:
                    assert isinstance(v, list)
                    for d in value:
                        dv = Device.mapping.get(d)
                        if not dv:
                            raise ValueError(f"invalid device '{d}'")
                        devices += dv
                elif prop == VerifierProp.APPRAISAL_POLICY_FILE:
                    appraisal_policy_file = value
                elif prop == VerifierProp.RESULT_POLICY_FILE:
                    result_policy_file = value
            if not env:
                raise ValueError("Environment is not specified for verifier")
            if not devices:
                raise ValueError("Devices is not specified for verifier")
            if not url:
                raise ValueError("Url is not specified for verifier")
            if not appraisal_policy_file:
                raise ValueError("Appraisal policy file is not specified for verifier")
            if not os.path.exists(appraisal_policy_file):
                raise ValueError(f"Appraisal policy file '{appraisal_policy_file}' does not exist")
            if not result_policy_file:
                raise ValueError("Result policy file is not specified for verifier")
            if not os.path.exists(result_policy_file):
                raise ValueError(f"Result policy file '{result_policy_file}' does not exist")
            attestation.add_verifier(devices, env, url, appraisal_policy_file, result_policy_file)

    def reset_participant(self, participant_name: str):
        pass

    def prepare(self) -> bool:
        """Prepare for attestation process

        Returns: error if any
        """
        ok = self.attestation.attest()
        if ok:
            self.token = self.attestation.get_token(self.site_name)
        return ok

    def get_token(self):
        return self.token

    def validate_participants(
            self,
            participants: Dict[str, str]) -> dict[str, bool]:
        """Validate CC policies of specified participants against the requirement policy of the site.

        Args:
            participants: dict of participant name => token

        Returns: dict of participant name => bool

        """
        if not participants:
            return {}
        return self.attestation.validate_tokens(participants)
