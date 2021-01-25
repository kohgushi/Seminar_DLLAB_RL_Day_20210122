# -*- coding: utf-8 -*-
import argparse
import os
import sys

import ray
from ray.rllib import train
from ray import tune

from cartpolecustom_env import create_env
from utils import callbacks


DEFAULT_RAY_ADDRESS = 'localhost:6379'
ENV_NAME = "CartPoleCustom-v0"


if __name__ == "__main__":
    train_parser = train.create_parser()
    args = train_parser.parse_args()
    args.config["callbacks"] = {"on_train_result": callbacks.on_train_result}
    
    env_name = args.env
        
    # 環境の登録
    tune.register_env(env_name, create_env)

    # Start (connect to) Ray cluster 
    ray.init(address=DEFAULT_RAY_ADDRESS)
    
    # Run training task using tune.run
    tune.run(
        run_or_experiment=args.run,
        config=dict(args.config, env=args.env),
        stop=args.stop,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_at_end=args.checkpoint_at_end,
        local_dir=args.local_dir
    )
