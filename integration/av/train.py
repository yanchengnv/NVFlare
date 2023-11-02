# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import logging
import random

from nvflare.client.flare_agent import RC, AgentClosed, FlareAgentWithCellPipe, FlareAgent
from nvflare.apis.dxo import DXO, MetaKey

NUMPY_KEY = "numpy_key"


def main():

    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", "-w", type=str, help="workspace folder", required=False, default=".")
    parser.add_argument("--site_name", "-s", type=str, help="flare site name", required=True)
    parser.add_argument("--agent_id", "-a", type=str, help="agent id", required=True)

    args = parser.parse_args()

    agent = FlareAgentWithCellPipe(
        agent_id=args.agent_id,
        site_name=args.site_name,
        root_url="grpc://server:8002",
        secure_mode=True,
        workspace_dir=args.workspace,
        submit_result_timeout=2.0,
        has_metrics=True,
    )

    agent.start()

    while True:
        print("getting task ...")
        try:
            task = agent.get_task()
        except AgentClosed:
            print("agent closed - exit")
            break

        print(f"got task: {task}")
        dxo = task.dxo
        assert isinstance(dxo, DXO)
        rc, meta, result = train(dxo.meta, dxo.data, agent)
        result_dxo = DXO(
            data_kind=dxo.data_kind,
            meta=meta,
            data=result
        )
        submitted = agent.submit_result(result=result_dxo, rc=rc)
        print(f"result submitted: {submitted}")

    agent.stop()


def train(meta, model, agent: FlareAgent):
    current_round = meta.get(MetaKey.CURRENT_ROUND)
    total_rounds = meta.get(MetaKey.TOTAL_ROUNDS)

    # Ensure that data is of type weights. Extract model data
    np_data = model

    # Display properties.
    print(f"Model: \n{np_data}")
    print(f"Current Round: {current_round}")
    print(f"Total Rounds: {total_rounds}")

    # Doing some dummy training.
    if np_data:
        if NUMPY_KEY in np_data:
            np_data[NUMPY_KEY] += 1.0
        else:
            print("error: numpy_key not found in model.")
            return RC.BAD_TASK_DATA, None, None
    else:
        print("No model weights found in shareable.")
        return RC.BAD_TASK_DATA, None, None

    # Save local numpy model
    agent.log_metric(DXO(data_kind="metric", data={"loss": random.random(), "round": current_round}))
    agent.log_metric(DXO(data_kind="metric", data={"accuracy": random.random()}))
    print(f"Model after training: {np_data}")

    # Prepare a DXO for our updated model. Create shareable and return
    return RC.OK, {MetaKey.NUM_STEPS_CURRENT_ROUND: 1}, np_data


if __name__ == "__main__":
    main()
