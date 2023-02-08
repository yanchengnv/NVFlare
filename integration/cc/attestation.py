#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 02:05:00 2023

@author: pbialecki
@author: michaelo
"""

from enum import IntFlag
from enum import IntEnum
from typing import Dict

import secrets


class Devices(IntFlag):
    CPU = 1
    GPU = 2
    NIC = 4
    OS = 8
    DPU = 16


class Environment(IntEnum):
    TEST = 1
    LOCAL = 2
    AZURE = 3
    GCP = 4


class Attestation(object):
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Attestation, cls).__new__(cls)
            cls._name = ""
            cls._nonceServer = ""
            cls._verifiers = []
            cls._evidencePolicies = {}
            cls._resultsPolicies = {}
            cls._tokens = {}
        return cls._instance

    @classmethod
    def set_name(cls, name: str) -> None:
        cls._name = name

    @classmethod
    def get_name(cls) -> str:
        return cls._name

    @classmethod
    def set_nonce_server(cls, url: str) -> None:
        cls._nonceServer = url

    @classmethod
    def get_nonce_server(cls) -> str:
        return cls._nonceServer
#
#   Piotr added the set / add which is problably more consistent with Python methodology
#   set - sets one.  add - adds another.
#   Not doing it that way.  
#   set_verifier will add and thus there is only once call
#
#    @classmethod
#    def set_verifier(cls, dev: Devices, env: Environment, url: str) -> None:
#        lst = [dev, env, url]
#        cls._verifiers = lst

#   This was Piotr's add_verifier call
#    def add_verifier(cls, dev: Devices, env: Environment, url: str) -> None:
    @classmethod
    def set_verifier(cls, dev: Devices, env: Environment, url: str) -> None:
        lst = [dev, env, url]
        cls._verifiers.append(lst)

    @classmethod
    def add_verifier(cls, dev: Devices, env: Environment, url: str, policy_4_evidence: str, policy_4_results: str):
        pass

    @classmethod
    def get_verifiers(cls) -> list:
        return cls._verifiers

    @classmethod
    def set_appraisal_policy_for_attestation_results(cls, name: str, policy: str) -> None:
        if name == "":
            entry = {cls.get_name(): policy}
        else:
            entry = {name: policy}
        cls._resultsPolicies.update(entry)

    @classmethod
    def get_appraisal_policy_for_attestation_results(cls, name: str) -> str:
        if name == "":
            return cls._resultsPolicies[cls.get_name()]
        else:
            return cls._resultsPolicies[name]

    @classmethod
    def set_appraisal_policy_for_evidence(cls, name: str, policy: str) -> None:
        if name == "":
            entry = {cls.get_name(): policy}
        else:
            entry = {name: policy}
        cls._evidencePolicies.update(entry)

    @classmethod
    def get_appraisal_policy_for_evidence(cls, name: str) -> str:
        if name == "":
            return cls._evidencePolicies[cls.get_name()]
        else:
            return cls._evidencePolicies[name]

    @classmethod
    def send_appraisal_policy_for_evidence(cls) -> bool:
        return True

    @classmethod
    def attest(cls) -> bool:
        # this should consist of doing the following things
        # Nonce _generateNonce()
        # Evidence generateEvidence(nonce)
        #   Retrieve quote from vTPM (locally)
        # Token verifyEvidence(evidence)
        #   Evidence -> verifier, validated against policy, returns token
        # Status provideEvidence(token)
        #   Token -> relying party, returns Status
        # cls.token = ""
        return True

    @classmethod
    def set_token(cls, name: str, token: str) -> None:
        entry = {name: token}
        cls._tokens.update(entry)

    @classmethod
    def get_token(cls, name: str) -> str:
        # check for no parameter
        if name == "":
            return cls._tokens[cls.get_name()]
        else:
            return cls._tokens[name]

    @classmethod
    def _validate_token_internal(cls, token: str) -> bool:
        if token == "":
            return False
        else:
            return True

    @classmethod
    def validate_tokens(cls, tokens: Dict[str, str]) -> Dict[str, bool]:
        pass

    @classmethod
    def validate_token(cls, x=None) :
        if x == None or x == "": 
            name = cls.get_name()
            if name == "":
                return False
            else:
                token = cls._tokens[name]
                return cls._validate_token_internal(token)

        elif isinstance(x,str):
            return True

        elif isinstance(x,list):
            return False

        elif isinstance(x,dict):
            retdict = {}
            for name in x:
                if (name != ""):
                    token = x[name]
                    if (token != ""):
                        retdict[name] = cls._validate_token_internal(token)
                    else:
                        retdict[name] = False
            return retdict
        else:
            return False

    @classmethod
    def _generate_nonce(cls) -> str:
        # Check for the nonce server AND the name.  If one is missing, generate a local nonce
        if cls._nonceServer != "" and cls._name != "" :
            # probably should only do this if name and url are non-null
            # make call to url to get nonce
            return "0xdeadbeefdeadbeefdeadbeefdeadbeef"
        else:
            # create nonce locally - 256 bits total
            nonceStr = "0x" + secrets.token_hex(16)
            return nonceStr

