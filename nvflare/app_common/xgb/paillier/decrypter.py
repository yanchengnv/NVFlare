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

import concurrent.futures

from .util import compute_chunk_size


class Decrypter:
    def __init__(self, private_key, max_workers=10):
        self.max_workers = max_workers
        self.private_key = private_key
        self.exe = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    def decrypt(self, encrypted_number_groups):
        """
        Encrypt a list of clear text numbers

        Args:
            encrypted_number_groups: list of lists of encrypted numbers to be decrypted

        Returns: list of lists of decrypted numbers

        """
        # print(f"decrypting {len(encrypted_number_groups)} number groups")
        items = []

        for g in encrypted_number_groups:
            items.append(
                (
                    self.private_key,
                    g,
                )
            )

        chunk_size = compute_chunk_size(len(items), self.max_workers)
        results = self.exe.map(_do_decrypt, items, chunksize=chunk_size)
        rl = []
        for r in results:
            rl.append(r)
        return rl


def _do_decrypt(item):
    # t = time.time()
    private_key, numbers = item
    ev = []
    for v in numbers:
        if isinstance(v, int):
            d = v
        else:
            d = private_key.decrypt(v)
        ev.append(d)
    # print(f"decrypted {len(numbers)} numbers in {time.time()-t} secs")
    return ev
