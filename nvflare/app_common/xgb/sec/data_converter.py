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
from typing import List


class FeatureContext:
    def __init__(self, feature_id, sample_bins, num_bins: int):
        self.feature_id = feature_id
        self.num_bins = num_bins  # how many bins this feature has
        self.sample_bins = sample_bins  # sample/bin assignment; normalized to [0 .. num_bins-1]


class AggregationContext:
    def __init__(self, features: List[FeatureContext], sample_groups: List[List[int]]):
        self.features = features
        self.sample_groups = sample_groups


class DataConverter:
    def decode_gh_pairs(self, buffer: bytes) -> (List[int], List[int]):
        """Decode the buffer to extract (g, h) pairs.

        Args:
            buffer: the buffer to be decoded

        Returns: if the buffer contains (g, h) pairs, return a tuple of (g_numbers, h_numbers);
            otherwise, return None

        """
        pass

    def decode_aggregation_context(self, buffer: bytes) -> AggregationContext:
        """Decode the buffer to extract aggregation context info

        Args:
            buffer: buffer to be decoded

        Returns: if the buffer contains aggregation context, return an AggregationContext object;
            otherwise, return None

        """
        pass

    def encode_all_gather_v(self, entries: List[bytes]) -> bytes:
        """Encode entries into a buffer following MPI's AllGatherV structure

        Args:
            entries: to be encoded

        Returns: a buffer of bytes

        """
        pass

    def encode_aggregation_result(self, aggr_result) -> bytes:
        """Encode an individual rank's aggr result to a buffer based on XGB data structure

        Args:
            aggr_result: list of tuples of (fid, gid, (G_list, H_list))

        Returns: a buffer of bytes

        """
        pass
