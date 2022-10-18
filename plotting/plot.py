import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set colors
color_map = {
    "CSC": "b",
    "Scratch": "m",
    "Supervised": "g",
    "PointContrast": "r",
    "CME": "c",
    "DepthContrast": "y",
}


def plot(data_dir, title, save_name):
    # Collect all data from folder with name
    data_dict = {}
    for filename in (Path.cwd() / data_dir).iterdir():
        data_dict[filename.stem] = np.loadtxt(
            filename,
            delimiter=",",
            skiprows=1,
        )

    # Plot translation errors
    fig, (ax) = plt.subplots(1, 1, sharex=True, figsize=(8, 10))

    max_steps = 10000
    ax.set_title(title)
    ax.set_ylabel("Validation (mIOU)")
    ax.set_xlabel("Steps")
    for name, data in data_dict.items():
        select = data[:, 1] < max_steps
        ax.plot(data[select, 1], data[select, 2], label=name, color=color_map[name])

    fig.set_figwidth(7)
    fig.set_figheight(3)
    fig.subplots_adjust(bottom=0.2)  # or whatever
    plt.legend()
    # plt.show()
    plt.savefig(save_name + ".pdf", format="pdf")


if __name__ == "__main__":

    plot("plotting/S3DIS", "S3DIS Semantic Segmentation", "s3dis_semantic")
    plot(
        "plotting/scannet/0.05",
        "Scannet (data limited 5%) - Semantic Segmentation",
        "scannet_0.05_semantic",
    )
    plot("plotting/scannet/1.00", "Scannet - Semantic Segmentation", "scannet_semantic")

    waithere = 1
