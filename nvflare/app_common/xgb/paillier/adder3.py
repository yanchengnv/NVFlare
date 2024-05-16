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

from nvflare.app_common.xgb.aggr import Aggregator

from .util import encode_encrypted_numbers_to_str


class Adder:
    def __init__(self, max_workers=10):
        self.num_workers = max_workers
        self.exe = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers)

    def add(self, encrypted_numbers, features, sample_groups=None, encode_sum=True):
        """

        Args:
            encrypted_numbers: list of encrypted numbers (combined gh), one for each sample
            features: list of tuples of (feature_id, mask, num_bins), one for each feature.
                    size of mask = size of encrypted_numbers: there is a bin number for each sample
                    num_bins specifies the number of bins for the feature
            sample_groups: list of sample groups, each group is a tuple of (group_id, id_list)
                    group_id is the group id, id_list is a list of sample IDs for which the add will be applied to
            encode_sum: if true, encode the sum into a JSON string

        Returns: list of tuples of (feature_id, group_id, sum), sum is the result of adding encrypted values of
            samples in the group for the feature.

        """
        feature_table = {}
        g_table = {}
        items = []

        for f in features:
            fid, mask, num_bins = f
            feature_table[fid] = (mask, num_bins)
            if not sample_groups:
                items.append((fid, 0))
            else:
                for g in sample_groups:
                    gid, sample_id_list = g
                    g_table[gid] = sample_id_list
                    items.append((fid, gid))

        # chunk items into worker_items
        num_items_per_worker = int(len(items) / self.num_workers)
        remaining = len(items) % self.num_workers

        worker_items = []
        start_idx = 0
        for i in range(self.num_workers):
            # determine number of items to be assigned to worker i
            n = num_items_per_worker
            if i < remaining:
                n += 1
            if n == 0:
                # this only happens when the number of items < number of workers
                break

            wi = items[start_idx:start_idx+n]
            start_idx += n
            ft = {}
            gt = {}

            # determine content of the feature table (ft) and gid table (gt) to be used for this worker
            for it in wi:
                fid, gid = it
                ft[fid] = feature_table[fid]
                if gid > 0:
                    gt[gid] = g_table[gid]
            worker_items.append((encode_sum, ft, gt, encrypted_numbers, wi))

        print(f"num items: {len(items)}")
        print(f"num worker items: {len(worker_items)}")
        results = self.exe.map(_do_add, worker_items)
        rl = []
        for r in results:
            rl.extend(r)
        return rl


def _do_add(item):
    encode_sum, feature_table, g_table, encrypted_numbers, wi = item
    result = []

    for it in wi:
        fid, gid = it
        aggr = Aggregator()
        mask, num_bins = feature_table[fid]
        sample_id_list = g_table.get(gid, None)
        bins = aggr.aggregate(
            gh_values=encrypted_numbers,
            sample_bin_assignment=mask,
            num_bins=num_bins,
            sample_ids=sample_id_list,
        )

        if encode_sum:
            sums = encode_encrypted_numbers_to_str(bins)
        else:
            sums = bins

        result.append((fid, gid, sums))

    return result
