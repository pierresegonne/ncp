import pathlib

import matplotlib.pyplot as plt
import numpy as np

from ncp import datasets


def main():
    save_fn = "outputs_for_aistats_plot.npz"
    root_path = pathlib.Path(__file__).parent.parent.parent.resolve()

    normal_ood_path = root_path / "logs_toy_plot" / "ours"
    more_ood_path = root_path / "logs_toy_plot_more_ood" / "ours"

    outputs = {}
    outputs["bbb"] = np.load(normal_ood_path / "bbb-0" / save_fn)
    outputs["bbb_ncp_normal_ood"] = np.load(normal_ood_path / "bbb_ncp-0" / save_fn)
    outputs["bbb_ncp_more_ood"] = np.load(more_ood_path / "bbb_ncp-0" / save_fn)

    dataset = datasets.generate_toy_ours_dataset()

    fig, ax = plt.subplots()

    ax.plot(dataset.train.inputs, dataset.train.targets, "o", color="black", alpha=0.7)
    ax.plot(dataset.test.inputs, dataset.test.targets, "o", color="navy", alpha=0.9)

    colors = ["darkred", "gold", "forestgreen"]
    for i, run in enumerate(["bbb", "bbb_ncp_normal_ood", "bbb_ncp_more_ood"]):
        ax.plot(outputs[run]["inputs"], outputs[run]["mean"], color=colors[i])

    plt.show()


if __name__ == "__main__":
    main()
