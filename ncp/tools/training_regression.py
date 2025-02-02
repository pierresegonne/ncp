import collections
import os
import sys

import numpy as np
import scipy.stats
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from ncp.tools import attrdict, plotting

# NOTE
# Use the NOTE marker to indicate changes


def run_experiment(
    logdir,
    graph,
    dataset,
    num_epochs,
    eval_after_epochs,
    log_after_epochs,
    visualize_after_epochs,
    batch_size,
    has_uncertainty=True,
    filetype="pdf",
    seed=0,
    record_tensorboard=True,
    save_model=False,
):
    logdir = os.path.expanduser(logdir)
    tf.gfile.MakeDirs(logdir)
    random = np.random.RandomState(seed)
    metrics = attrdict.AttrDict(
        # NOTE
        epochs=[],
        train_likelihoods=[],
        train_distances=[],
        test_likelihoods=[],
        test_distances=[],
        test_variances=[],
        test_samples_distances=[],
    )
    # NOTE
    visibles = [i for i in range(len(dataset.train.inputs))]

    # NOTE
    # Tensorboard support
    if record_tensorboard:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name, var)

        merged = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    if save_model:
        saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # NOTE
        if record_tensorboard:
            writer = tf.summary.FileWriter("./graphs", sess.graph)
        for epoch in range(num_epochs):
            visible = np.array(visibles)

            # Shuffle, batch, drop remainder.
            indices = random.permutation(np.arange(len(visible)))
            N = len(visible)
            for index in range(0, N, batch_size):
                current = visible[indices[index : index + batch_size]]
                sess.run(
                    graph.optimize,
                    {
                        graph.inputs: dataset.train.inputs[current],
                        graph.targets: dataset.train.targets[current],
                        graph.num_visible: len(visible),
                    },
                )
                # NOTE
                if record_tensorboard:
                    summary = sess.run(merged)
                    writer.add_summary(summary, epoch)

            if epoch in eval_after_epochs:
                target_scale = dataset.get("target_scale", 1)
                # NOTE
                metrics.epochs.append(epoch)
                likelihood, distance, _, _ = evaluate_model(
                    sess,
                    graph,
                    has_uncertainty,
                    dataset.train.inputs[visible],
                    dataset.train.targets[visible],
                    target_scale,
                )
                metrics.train_likelihoods.append(likelihood)
                metrics.train_distances.append(distance)
                test_inputs = dataset.test.inputs
                test_targets = dataset.test.targets
                likelihood, distance, variance, samples_distance = evaluate_model(
                    sess,
                    graph,
                    has_uncertainty,
                    test_inputs,
                    test_targets,
                    target_scale,
                )
                metrics.test_likelihoods.append(likelihood)
                metrics.test_distances.append(distance)
                metrics.test_variances.append(variance)
                metrics.test_samples_distances.append(samples_distance)

            if epoch in log_after_epochs:
                print(
                    "Epoch",
                    epoch,
                    "train nlpd {:.2f}".format(-metrics.train_likelihoods[-1]),
                    "train rmse {:.2f}".format(metrics.train_distances[-1]),
                    "test nlpd {:.2f}".format(-metrics.test_likelihoods[-1]),
                    "test rmse {:.2f}".format(metrics.test_distances[-1]),
                )
                sys.stdout.flush()
                if save_model:
                    # Save the variables to disk.
                    saver.save(sess, os.path.join(logdir, f"model_{epoch}.ckpt"))

            # FIXME
            if epoch in visualize_after_epochs:
                filename = os.path.join(logdir, "epoch-{}.{}".format(epoch, filetype))
                plotting.visualize_model(
                    filename, sess, graph, has_uncertainty, dataset, visibles
                )

    metrics = {key: np.array(value) for key, value in metrics.items()}
    np.savez_compressed(os.path.join(logdir, "metrics.npz"), **metrics)
    return metrics


def evaluate_model(
    sess, graph, has_uncertainty, inputs, targets, target_scale, batch_size=100
):
    likelihoods, squared_distances, squared_variances, squared_samples_distances = (
        [],
        [],
        [],
        [],
    )
    for index in range(0, len(inputs), batch_size):
        target = targets[index : index + batch_size]
        mean, noise, uncertainty = sess.run(
            [graph.data_mean, graph.data_noise, graph.data_uncertainty],
            {graph.inputs: inputs[index : index + batch_size]},
        )
        squared_distances.append((target_scale * (target - mean)) ** 2)
        if has_uncertainty:
            std = np.sqrt(noise ** 2 + uncertainty ** 2 + 1e-8)
        else:
            std = noise
        squared_variances.append((std ** 2 - (target - mean) ** 2) ** 2)
        samples = np.random.normal(loc=mean, scale=std)
        squared_samples_distances.append((target_scale * (target - samples)) ** 2)
        # Subtracting the log target scale is equivalent to evaluting the
        # log-probability of the unnormalized targets under the scaled predicted
        # mean and standard deviation.
        # likelihood = scipy.stats.norm(
        #     target_scale * mean, target_scale * std).logpdf(
        #         target_scale * target)
        likelihood = scipy.stats.norm(mean, std).logpdf(target) - np.log(target_scale)
        likelihoods.append(likelihood)
    likelihood = np.concatenate(likelihoods, 0).sum(1).mean(0)
    distance = np.sqrt(np.concatenate(squared_distances, 0).sum(1).mean(0))
    variance = np.sqrt(np.concatenate(squared_variances, 0).sum(1).mean(0))
    samples_distance = np.sqrt(
        np.concatenate(squared_samples_distances, 0).sum(1).mean(0)
    )
    return likelihood, distance, variance, samples_distance


def load_results(pattern):
    results = collections.defaultdict(list)
    for filepath in tf.gfile.Glob(pattern):
        metrics = np.load(filepath)
        for key in metrics.keys():
            results[key].append(metrics[key])
    for key, value in results.items():
        results[key] = np.array(value)
    return attrdict.AttrDict(results)
