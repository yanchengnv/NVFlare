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
from nvflare.fuel.utils.import_utils import optional_import

tf, tf_ok = optional_import(module="tensorflow")
if tf_ok:
    from nvflare.app_opt.tf.model_persistor import TFModelPersistor


class Wrap:
    def __init__(self, model):
        self.model = model

    def add_to_fed_job(self, job, ctx):
        """This method is required by Job API.

        Args:
            job: the Job object to add to
            ctx: Job Context

        Returns:

        """
        if tf_ok and isinstance(self.model, tf.keras.Model):  # if model, create a TF persistor
            component = TFModelPersistor(model=self.model)
            persistor_id = job.generate_tracked_component_id(base_id="persistor", ctx=ctx)
            job.add_component(comp_id=persistor_id, obj=component, ctx=ctx)
