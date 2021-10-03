import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from ncp.tools.plotting import plot_prediction


def get_latest_epoch(dir: str) -> int:
    paths = glob(os.path.join(dir, "model_*.ckpt.meta"))
    n_epochs = [int(p.replace(".ckpt.meta", "").split("model_")[-1]) for p in paths]
    return max(n_epochs)


def generate_aistats_plot(
    logdir,
    graph,
    dataset,
    num_epochs,
    batch_size,
    has_uncertainty=True,
    filetype="pdf",
    seed=0,
    **kwargs,
):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        # sess.run(tf.global_variables_initializer())
        epoch = get_latest_epoch(logdir)
        saver.restore(sess, os.path.join(logdir, f"model_{epoch}.ckpt"))
        means, noises, uncertainties = [], [], []
        for index in range(0, len(dataset.domain), batch_size):
            mean, noise, uncertainty = sess.run(
                [graph.data_mean, graph.data_noise, graph.data_uncertainty],
                {graph.inputs: dataset.domain[index : index + batch_size]},
            )
            means.append(mean)
            noises.append(noise)
            uncertainties.append(uncertainty)
        mean = np.concatenate(means, 0)
        noise = np.concatenate(noises, 0)
        std = np.sqrt(noise ** 2 + uncertainty ** 2) if has_uncertainty else noise
        visibles = np.array([i for i in range(len(dataset.train.inputs))])
        not_visible = np.ones(len(dataset.train.targets), dtype=bool)
        not_visible[visibles] = False
        # TODO test that we get the right mean and noise here
        index = 0
        fig, ax = plt.subplots()
        ax.scatter(
            dataset.test.inputs[:, index],
            dataset.test.targets[:, 0],
            c="#dddddd",
            lw=0,
            s=3,
        )
        ax.scatter(
            dataset.train.inputs[not_visible, index],
            dataset.train.targets[not_visible, 0],
            c="#dddddd",
            lw=0,
            s=3,
        )
        plot_prediction(ax, dataset.domain[:, index], mean[:, 0], std[:, 0])
        ax.scatter(
            dataset.train.inputs[visibles, index],
            dataset.train.targets[visibles, 0],
            c="#000000",
            lw=0,
            s=4,
        )
        plt.savefig("tmp.png")
