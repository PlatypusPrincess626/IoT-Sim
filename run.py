import os
import argparse
from time import time, sleep
from typing import Optional, Tuple
import csv
import datetime

# custom dependencies
from UAV_IoT_Sim import UAV_IoT_Sim

import os
import tensorflow as tf

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-name",
        type=str,
        default="uav-iot-sim-test",
        help="The project name (for loggers) to store results."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="test",
        help="The scene for the project to take place."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Q-Learning",
        help="ML model to use for testing."
    )
    parser.add_argument(
        "--life",
        type=int,
        default=720,
        help="Maximum number of steps in the episode."
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100_000,
        help="Maximum number of steps for training."
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10_000,
        help="Frequency of agent evaluation."
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="Episodes to evaluate each evaluation period."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=10_000,
        help="Dimensions of square (dim x dim) area."
    )
    parser.add_argument(
        "--uavs",
        type=int,
        default=1,
        help="The number of UAVs on site."
    )
    parser.add_argument(
        "--clusterheads",
        type=int,
        default=5,
        help="The number of clusterheads."
    )
    parser.add_argument(
        "--sensors",
        type=int,
        default=50,
        help="The number of sensors spread across the environment."
    )
    parser.add_argument(
        "--disable_wandb",
        type=bool,
        default=False,
        help="Activate wandb."
    )
    return parser.parse_args()


def test_env(
        env: object,
        env_str: str,
        total_steps: int,
        eval_frequency: int,
        eval_episodes: int,
        policy_path: str,
):
    start_time = time()
    env.reset()
    CH_Metrics = [[0, 0] for _ in range(env.num_ch)]
    CH_Power = []
    CH_Data = []

    env.reset()

    for i in range(total_steps):
        state = env.step()

        ch: int
        for ch in range(len(CH_Metrics)):
            CH_Metrics[ch][0] += state[ch + 1][2]
            CH_Metrics[ch][1] = state[ch + 1][1]

        CH_Power.append([CH_Metrics[0][1], CH_Metrics[1][1], CH_Metrics[2][1], \
                        CH_Metrics[3][1], CH_Metrics[4][1]])
        CH_Data.append([CH_Metrics[0][0], CH_Metrics[1][0], CH_Metrics[2][0], \
                        CH_Metrics[3][0], CH_Metrics[4][0]])

    curr_date_time = datetime.datetime.now()

    filename = "age_metrics_" + curr_date_time.strftime("%d") + "_" + curr_date_time.strftime("%m") + ".csv"
    open(filename, 'x')
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|')
        csvwriter.writerows(CH_Power)

    filename = "data_metrics_" + curr_date_time.strftime("%d") + "_" + curr_date_time.strftime("%m") + ".csv"
    open(filename, 'x')
    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|')
        csvwriter.writerows(CH_Data)


def run_experiment(args):
    env_str = args.env
    print("Creating Evironment")
    env = UAV_IoT_Sim.make_env(scene=env_str, num_sensors=50, num_ch=5, max_num_steps=720)

    policy_save_dir = os.path.join(
        os.getcwd(), "policies", args.project_name
    )
    os.makedirs(policy_save_dir, exist_ok=True)
    policy_path = os.path.join(
        policy_save_dir,
        f"model={args.model}"
    )

    print("Beginning Training")
    test_env(
        env,
        args.env,
        args.steps,
        args.eval_frequency,
        args.eval_episodes,
        policy_path,
    )


if __name__ == "__main__":
    run_experiment(get_args())
