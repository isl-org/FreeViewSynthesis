import argparse
import itertools
import sys

import create_data_pw

sys.path.append("..")
import config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--scales", nargs="+", type=float, default=[0.25])
    parser.add_argument(
        "-d",
        "--datasets",
        nargs="+",
        type=str,
        default=config.tat_train_sets + config.tat_eval_sets,
    )
    args = parser.parse_args()

    for dset, scale in itertools.product(args.datasets, args.scales):
        dense_dir = config.tat_root / dset / "dense"
        print(f"create_data_pw for {dense_dir}")
        create_data_pw.run(dense_dir, scale, dm_write_vis=True)
