# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from ncp import tools


def load_numpy_dataset(path_directory: Path, train_amount=None, test_amount=None):
    random = np.random.RandomState(0)
    # Train
    filepath = path_directory / "-train-inputs.npy"
    with tf.gfile.Open(filepath, "rb") as file_:
        train_inputs = np.load(file_).astype(np.float32)
    filepath = path_directory / "-train-targets.npy"
    with tf.gfile.Open(filepath, "rb") as file_:
        train_targets = np.load(file_).astype(np.float32)
    # Test
    filepath = path_directory / "-test-inputs.npy"
    with tf.gfile.Open(filepath, "rb") as file_:
        test_inputs = np.load(file_).astype(np.float32)
    filepath = path_directory / "-test-targets.npy"
    with tf.gfile.Open(filepath, "rb") as file_:
        test_targets = np.load(file_).astype(np.float32)

    if train_amount:
        train_indices = random.permutation(len(train_inputs))[:train_amount]
        train_inputs = train_inputs[train_indices]
        train_targets = train_targets[train_indices]
    if test_amount:
        test_amount = random.permutation(len(test_inputs))[:test_amount]
        test_inputs = test_inputs[test_amount]
        test_targets = test_targets[test_amount]

    domain = test_inputs[::10]  # Subsample inputs for visualization.
    mean = train_inputs.mean(0)[None]
    std = train_inputs.std(0)[None] + 1e-6
    train_inputs = (train_inputs - mean) / std
    test_inputs = (test_inputs - mean) / std
    domain = (domain - mean) / std
    mean = train_targets.mean(0)[None]
    std = train_targets.std(0)[None] + 1e-6
    train_targets = (train_targets - mean) / std
    test_targets = (test_targets - mean) / std
    train = tools.AttrDict(inputs=train_inputs, targets=train_targets)
    test = tools.AttrDict(inputs=test_inputs, targets=test_targets)
    return tools.AttrDict(domain=domain, train=train, test=test, target_scale=std)


def load_numpy_dataset_shifted_split(
    path_directory: Path, dim_idx: int
) -> tools.AttrDict:
    """
    Additionnal split test samples are added to already existing test samples
    That's not ideal but that's the way it's done in sggm
    """
    dataset = load_numpy_dataset(path_directory)
    assert dim_idx >= 0
    assert dim_idx < dataset.train.inputs.shape[1]
    train_inputs, train_targets = dataset.train.inputs, dataset.train.targets
    test_inputs, test_targets = dataset.test.inputs, dataset.test.targets

    dim_col_sorted_indices = np.argsort(train_inputs[:, dim_idx])
    N = dim_col_sorted_indices.shape[0]
    split_indices = dim_col_sorted_indices[int(N / 3) : int((2 * N) / 3)]
    non_split_indices = np.concatenate(
        (
            dim_col_sorted_indices[: int(N / 3)],
            dim_col_sorted_indices[int((2 * N) / 3) :],
        )
    )

    test_inputs = np.concatenate((test_inputs, train_inputs[split_indices]), axis=0)
    test_targets = np.concatenate((test_targets, train_targets[split_indices]), axis=0)
    train_inputs = train_inputs[non_split_indices]
    train_targets = train_targets[non_split_indices]

    domain = test_inputs[::10]  # Subsample inputs for visualization.
    train = tools.AttrDict(inputs=train_inputs, targets=train_targets)
    test = tools.AttrDict(inputs=test_inputs, targets=test_targets)
    return tools.AttrDict(
        domain=domain, train=train, test=test, target_scale=dataset.target_scale
    )
