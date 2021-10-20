import pathlib

import matplotlib.pyplot as plt
import numpy as np

from ncp import datasets

plt.rcParams.update(
    {
        "text.usetex": True,
    }
)


def dewhiten_x(x):
    return x * 3.0126251862413524 + 5.04480454393157


def dewhiten_y(y):
    return y * 4.147721555561024 + 0.864837661287082


def data_mean(x):
    return x * np.sin(x)


def data_std(x):
    return np.abs(0.3 * np.sqrt(1 + x * x))


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

    x_plot = np.linspace(-4, 14, 1000)
    y_plot = data_mean(x_plot)
    y_plot_mstd = y_plot - 1.96 * data_std(x_plot)
    y_plot_pstd = y_plot + 1.96 * data_std(x_plot)

    aspect_ratio = 2.35 / 1
    colour_navy_blue = (3 / 255, 15 / 255, 79 / 255)
    fig, ax = plt.subplots(figsize=(2 * aspect_ratio, 2))

    ax.set_facecolor("#F7F8F6")
    ax.grid(True, color="white")
    ax.set_xlim([-4, 14])
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel(r"$x$", fontsize=14, labelpad=2)
    ax.set_ylim(-15, 15)
    ax.set_yticks([-10, 0, 10])
    ax.set_ylabel(r"$y$", fontsize=14, labelpad=1)

    ax.plot(
        x_plot,
        y_plot,
        color="black",
        linestyle="dashed",
        linewidth=1,
        label=r"$\mathrm{Truth}$",
    )
    ax.plot(x_plot, y_plot_mstd, color="black", linestyle="dotted", linewidth=0.5)
    ax.plot(x_plot, y_plot_pstd, color="black", linestyle="dotted", linewidth=0.5)
    ax.plot(
        dewhiten_x(dataset.train.inputs),
        dewhiten_y(dataset.train.targets),
        "o",
        markersize=2.5,
        markerfacecolor=(*colour_navy_blue, 0.6),
        markeredgewidth=1,
        markeredgecolor=(*colour_navy_blue, 0.1),
        zorder=5,
    )

    colors = ["forestgreen", "gold", "darkred"]
    labels = [
        r"$\mathrm{BBB}$",
        r"$\mathrm{BBB\!+\!NCP}$",
        r"$\mathrm{BBB\!+\!NCP^*}$",
    ]
    for i, run in enumerate(["bbb", "bbb_ncp_normal_ood", "bbb_ncp_more_ood"]):
        ax.plot(
            dewhiten_x(outputs[run]["inputs"]),
            dewhiten_y(outputs[run]["mean"]),
            color=colors[i],
            label=labels[i],
            alpha=0.95,
            linewidth=2,
            zorder=5,
        )

    ax.legend(
        loc="upper left",
        # bbox_to_anchor=(-0.05, 1.55),
        ncol=4,
        edgecolor="black",
        handlelength=1.2,
        fancybox=False,
        columnspacing=0.85,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
