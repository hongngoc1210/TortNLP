"""Train the Final V2 architecture end-to-end in a single phase.

Unlike ``main.py``, this entry point builds the complete Final V2 model once
and trains every stage in one continuous optimization run.  It deliberately
does not create, load, or transfer a phase-1 checkpoint.

Example:
    python src/main_v2.py --config config/final_v2_single_phase.yaml
"""

from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Final V2 end-to-end in one phase.",
    )
    parser.add_argument(
        "--config",
        default="config/final_v2_single_phase.yaml",
        help="Resolved single-phase training config.",
    )
    parser.add_argument(
        "--run-name",
        default="final_v2_single_phase",
        help=(
            "Output subdirectory name. It is created below "
            "system.save_dir from the config."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Keep the CLI lightweight (for example, ``--help`` does not need torch).
    # Training dependencies are imported only after argument parsing succeeds.
    from ablation_main import load_yaml, run_experiment

    cfg = load_yaml(args.config)
    run_experiment(args.run_name, cfg)


if __name__ == "__main__":
    main()
