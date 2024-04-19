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
from typing import Any

import phe

from nvflare.app_common.xgb.he.cipher import Cipher, ClearText


class PheCipher(Cipher):

    def __init__(self):
        self.public_key = None
        self.private_key = None

    def generate_keys(self, key_length: int):
        self.public_key, self.private_key = phe.paillier.generate_paillier_keypair(n_length=key_length)

    def get_public_key_str(self) -> str:
        return phe.util.int_to_base64(self.public_key.n)

    def set_public_key_str(self, public_key_str: str):
        self.public_key = phe.paillier.PaillierPublicKey(n=phe.util.base64_to_int(public_key_str))

    def encode_cipher_text(self, cipher_text: Any) -> Any:
        if not isinstance(cipher_text, phe.paillier.EncryptedNumber):
            raise TypeError(f"Invalid type {type(cipher_text)}")

        return phe.util.int_to_base64(cipher_text.ciphertext()), cipher_text.exponent

    def decode_cipher_text(self, encoded_cipher_text: Any) -> Any:
        cipher_str, exp = encoded_cipher_text
        return phe.paillier.EncryptedNumber(
            self.public_key, ciphertext=phe.util.base64_to_int(cipher_str), exponent=exp)

    def encrypt(self, clear_text: ClearText) -> Any:
        return self.public_key.encrypt(clear_text)

    def decrypt(self, cipher_text: Any) -> ClearText:
        return self.private_key.decrpty(cipher_text)

    def add(self, a: Any, b: Any) -> Any:
        return a + b
