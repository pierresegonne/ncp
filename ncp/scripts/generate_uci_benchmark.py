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

    is_shifted = "shifted_split" in args.logdir

    if args.dataset is None:
        datasets_to_run = [ds.value for ds in UCIDataset]
    else:
        assert args.dataset in [ds.value for ds in UCIDataset]
        datasets_to_run = [args.dataset]

    _dfs = []
    for dataset_to_run in datasets_to_run:
        args.dataset = dataset_to_run

        if not os.path.isdir(args.logdir + "/" + args.dataset):
            print(f"`{args.logdir + '/' + args.dataset}` not in logdir")
            continue

        dataset = datasets.load_numpy_dataset(datasets.UCI_DATASETS_PATH / args.dataset)

        results = [
            ("BBB+NCP", load_results("bbb_ncp")),
            # ("ODC+NCP", load_results("det_mix_ncp")),
            ("BBB", load_results("bbb")),
            ("Det", load_results("det")),
        ]

        def get_metrics(i: int, scl: float) -> np.array:
            # scl includes correction back to standardised variables
            # see likelihood = scipy.stats.norm(mean, std).logpdf(target) - np.log(target_scale)
            # l157 in training_regression.py
            return np.array(
                [
                    [
                        (
                            np.stack(results[i][1]["test_likelihoods"]) + np.log(scl)
                        ).mean(axis=0)[-1],
                        (np.stack(results[i][1]["test_likelihoods"]) + np.log(scl)).std(
                            axis=0
                        )[-1],
                    ],
                    [
                        (np.stack(results[i][1]["test_distances"]) / scl).mean(axis=0)[
                            -1
                        ],
                        (np.stack(results[i][1]["test_distances"]) / scl).std(axis=0)[
                            -1
                        ],
                    ],
                ]
            )

        experiments = [f"uci_{dataset_to_run}{'_shifted' if is_shifted else ''}"]
        metrics = ["test_expected_log_likelihood↑", "test_mean_fit_rmse↓"]
        index_iterables = [experiments, metrics]
        methods = ["BBB+NCP", "BBB", "Det"]
        kinds = ["mean", "std"]
        columns_iterables = [methods, kinds]

        _df = pd.DataFrame(
            index=pd.MultiIndex.from_product(
                index_iterables, names=["experiment_name", "metric"]
            ),
            columns=pd.MultiIndex.from_product(columns_iterables),
        )
        for i, m in enumerate(methods):
            map_method_to_dir = {"BBB+NCP": "bbb_ncp-0", "BBB": "bbb-0", "Det": "det-0"}
            if not os.path.isdir(
                args.logdir + "/" + args.dataset + "/" + map_method_to_dir[m]
            ):
                print(f"`{args.logdir + '/' + args.dataset + '/' + m}` not in logdir")
                continue

            _df.loc[:, m] = get_metrics(i, dataset.target_scale)

        _dfs.append(_df)
    df = pd.concat(_dfs)
    print(df)

    df.to_csv(f"{args.logdir}/uci_benchmarks{'_shifted' if is_shifted else ''}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)
    parser.add_argument(
        "--dataset", default=None, choices=[ds.value for ds in datasets.UCIDataset]
    )
    args = parser.parse_args()
    args.logdir = os.path.expanduser(args.logdir)
    main(args)
