import argparse
import enum
import os

import numpy as np
import pandas as pd
from matplotlib.pyplot import axis

from ncp import datasets, models, tools
from ncp.datasets.uci import UCIDataset, get_num_epochs

""" Compile the uci results into a digestible csv """


def main(args):

    load_results = lambda x: tools.load_results(
        os.path.join(args.logdir + "/" + args.dataset, x) + "-*/*.npz"
    )

    if args.dataset is None:
        datasets_to_run = [ds.value for ds in UCIDataset]
    else:
        assert args.dataset in [ds.value for ds in UCIDataset]
        datasets_to_run = [args.dataset]

    _dfs = []
    for dataset_to_run in datasets_to_run:
        args.dataset = dataset_to_run
        dataset = datasets.load_numpy_dataset(datasets.UCI_DATASETS_PATH / args.dataset)

        results = [
            ("BBB+NCP", load_results("bbb_ncp")),
            # ("ODC+NCP", load_results("det_mix_ncp")),
            ("BBB", load_results("bbb")),
            ("Det", load_results("det")),
        ]

        def get_metrics(i: int, scl: float) -> np.array:
            return np.array(
                [
                    [
                        (np.stack(results[i][1]["test_likelihoods"]) / scl).mean(
                            axis=0
                        )[-1]
                    ],
                    [
                        (np.stack(results[i][1]["test_distances"]) / scl).mean(axis=0)[
                            -1
                        ]
                    ],
                    [
                        (np.stack(results[i][1]["test_likelihoods"]) / scl).std(axis=0)[
                            -1
                        ]
                    ],
                    [(np.stack(results[i][1]["test_distances"]) / scl).std(axis=0)[-1]],
                ]
            )

        kinds = ["mean", "std"]
        metrics = ["test_expected_log_likelihood↑", "test_mean_fit_rmse↓"]
        iterables = [[f"uci_{dataset_to_run}"], kinds, metrics]
        columns = ["BBB+NCP", "BBB", "Det"]

        _df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                iterables, names=["experiment", "kind", "metric"]
            ),
            columns=columns,
        )
        for i, c in enumerate(columns):
            _df.loc[:, c] = get_metrics(i, dataset.target_scale)

        _dfs.append(_df)
    df = pd.concat(_dfs)
    print(df)

    df.query("kind == 'mean'").to_csv(f"{args.logdir}/uci_benchmarks.csv")
    df.query("kind == 'std'").to_csv(f"{args.logdir}/uci_benchmarks_std.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument(
        "--dataset", default=None, choices=[ds.value for ds in datasets.UCIDataset]
    )
    args = parser.parse_args()
    args.logdir = os.path.expanduser(args.logdir)
    main(args)
