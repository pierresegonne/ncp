import argparse
import itertools
import os
import warnings

import matplotlib as mpl

from ncp.datasets.uci import UCIDataset

mpl.use("Agg")
import matplotlib.pyplot as plt
import ruamel.yaml as yaml
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from ncp import datasets, models, tools


# TODO - check config
def default_schedule(model):
    config = tools.AttrDict()
    config.num_epochs = 1000
    _range = range(0, config.num_epochs + 1, 50)
    config.eval_after_epochs = _range
    config.log_after_epochs = _range
    config.visualize_after_epochs = _range
    config.batch_size = 16
    config.filetype = "pdf"
    config.record_tensorboard = False
    if model == "det":
        config.has_uncertainty = False
    return config


# TODO - check config
def default_config(model):
    config = tools.AttrDict()
    config.num_inputs = 1  # This must be overriden based on the uci experiment
    config.layer_sizes = [200, 200]  # [200, 200]  # [50, 50]
    if model == "bbb":
        config.divergence_scale = 0.1
    if model == "bbb_ncp":
        config.noise_std = 0.5
        config.ncp_scale = 0.1
        config.divergence_scale = 0
        config.ood_std_prior = 0.1
        config.center_at_target = True
    if model == "det_mix_ncp":
        config.noise_std = 0.5
        config.center_at_target = True
    config.learning_rate = 3e-4
    config.weight_std = 0.1
    config.clip_gradient = 1.0
    return config


def plot_results(args):
    load_results = lambda x: tools.load_results(
        os.path.join(args.logdir + "/" + args.dataset, x) + "-*/*.npz"
    )
    results = [
        ("BBB+NCP", load_results("bbb_ncp")),
        ("ODC+NCP", load_results("det_mix_ncp")),
        ("BBB", load_results("bbb")),
        ("Det", load_results("det")),
    ]
    tools.pretty_print_results(results)
    fig, ax = plt.subplots(ncols=4, figsize=(8, 2))
    for a in ax:
        a.xaxis.set_major_locator(plt.MaxNLocator(5))
        a.yaxis.set_major_locator(plt.MaxNLocator(5))
    tools.plot_distance(ax[0], results, "train_distances", {})
    ax[0].set_xlabel("Epochs")
    ax[0].set_title("Train RMSE")
    tools.plot_likelihood(ax[1], results, "train_likelihoods", {})
    ax[1].set_xlabel("Epochs")
    ax[1].set_title("Train NLPD")
    tools.plot_distance(ax[2], results, "test_distances", {})
    ax[2].set_xlabel("Epochs")
    ax[2].set_title("Test RMSE")
    tools.plot_likelihood(ax[3], results, "test_likelihoods", {})
    ax[3].set_xlabel("Epochs")
    ax[3].set_title("Test NLPD")
    ax[3].legend(frameon=False, labelspacing=0.2, borderpad=0)
    fig.tight_layout(pad=0, w_pad=0.5)
    filename = os.path.join(args.logdir, "results.pdf")
    fig.savefig(filename)


def main(args):
    if args.replot:
        plot_results(args)
        return
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # TensorFlow.
    # NOTE
    # Here we define the models
    # We only want to experiment against *_ncp
    models_ = [
        # ("bbb", models.bbb.define_graph),
        ("det", models.det.define_graph),
        # ("bbb_ncp", models.bbb_ncp.define_graph),
        # ('det_mix_ncp', models.det_mix_ncp.define_graph),
    ]
    if args.dataset is None:
        datasets_to_run = [ds.value for ds in UCIDataset]
    else:
        assert args.dataset in [ds.value for ds in UCIDataset]
        datasets_to_run = [args.dataset]
    for dataset_to_run in datasets_to_run:
        dataset = datasets.load_numpy_dataset(
            str(datasets.UCI_DATASETS_PATH / dataset_to_run) + "/"
        )
        experiments = itertools.product(range(args.seeds), models_)
        for seed, (model, define_graph) in experiments:
            schedule = globals()[args.schedule](model)
            config = globals()[args.config](model)
            # Override num_inputs based on dataset
            config.num_inputs = dataset.train.inputs.shape[1]
            logdir = os.path.join(
                f"{args.logdir}/{dataset_to_run}", "{}-{}".format(model, seed)
            )
            tf.gfile.MakeDirs(logdir)
            if os.path.exists(os.path.join(logdir, "metrics.npz")):
                if args.resume:
                    continue
                raise RuntimeError("The log directory is not empty.")
            with open(os.path.join(logdir, "schedule.yaml"), "w") as file_:
                yaml.dump(schedule.copy(), file_)
            with open(os.path.join(logdir, "config.yaml"), "w") as file_:
                yaml.dump(config.copy(), file_)
            message = "\n{0}\n# Model {1} seed {2}\n{0}"
            print(message.format("#" * 79, model, seed))
            tf.reset_default_graph()
            tf.set_random_seed(seed)
            graph = define_graph(config)
            tools.run_experiment(logdir, graph, dataset, **schedule, seed=seed)
            plot_results(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", default="default_schedule")
    parser.add_argument("--config", default="default_config")
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--replot", action="store_true", default=False)
    parser.add_argument(
        "--dataset", default=None, choices=[ds.value for ds in datasets.UCIDataset]
    )
    args = parser.parse_args()
    args.logdir = os.path.expanduser(args.logdir)
    main(args)
