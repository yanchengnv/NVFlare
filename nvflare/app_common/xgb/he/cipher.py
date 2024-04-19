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
from abc import ABC, abstractmethod
from typing import Any, Union

ClearText = Union[float, int]


class Cipher(ABC):
    """An abstract class for Homomorphic Encryption operations"""
    @abstractmethod
    def generate_keys(self, key_length: int):
        pass

    @abstractmethod
    def get_public_key_str(self) -> str:
        pass

    @abstractmethod
    def set_public_key_str(self, public_key_str: str):
        pass

    @abstractmethod
    def encode_cipher_text(self, cipher_text: Any) -> Any:
        pass

    @abstractmethod
    def decode_cipher_text(self, encoded_cipher_text : Any) -> Any:
        pass

    @abstractmethod
    def encrypt(self, clear_text: ClearText) -> Any:
        pass

    @abstractmethod
    def decrypt(self, cipher_text: Any) -> ClearText:
        pass

    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        pass
