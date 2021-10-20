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


def save_outputs_for_aistats_plot(
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
        domain = np.linspace(-4, 14, 10000)[:, None]
        for index in range(0, len(domain), batch_size):
            mean, noise, uncertainty = sess.run(
                [graph.data_mean, graph.data_noise, graph.data_uncertainty],
                {graph.inputs: domain[index : index + batch_size]},
            )
            means.append(mean)
            noises.append(noise)
            uncertainties.append(uncertainty)
        mean = np.concatenate(means, 0)
        noise = np.concatenate(noises, 0)
        if has_uncertainty:
            uncertainty = np.concatenate(uncertainties, 0)
        std = np.sqrt(noise ** 2 + uncertainty ** 2) if has_uncertainty else noise
        outputs_for_aistats_plot = {
            "inputs": domain[:, 0],
            "mean": mean[:, 0],
            "std": std[:, 0],
        }
        np.savez_compressed(
            os.path.join(logdir, "outputs_for_aistats_plot.npz"),
            **outputs_for_aistats_plot,
        )
