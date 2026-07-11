from __future__ import annotations

import argparse

import ray
from ray import tune

from xnerf.training.train import run_training


def trainable(config: dict) -> None:
    metrics = run_training(config.get("config_path", "config.yaml"))
    tune.report(best_val_loss=metrics["best_val_loss"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Ray launcher for X-NERF++ experiments")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--gpus-per-trial", type=float, default=1.0)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    ray.init(ignore_reinit_error=True)
    tuner = tune.Tuner(
        tune.with_resources(trainable, {"cpu": 2, "gpu": args.gpus_per_trial}),
        tune_config=tune.TuneConfig(num_samples=args.num_samples),
        param_space={"config_path": args.config},
    )
    tuner.fit()


if __name__ == "__main__":
    main()
