from typing import List, Optional

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch


def plot_confusion_matrix(output: torch.Tensor, class_names: Optional[List[str]] = None):

    if class_names is None:
        class_names = [str(i) for i in range(output.shape[0])]

    N = output.shape[0]
    fig = plt.figure(figsize=(N + 2, N))
    ax = plt.subplot()
    output = output / (output.sum(dim=1)[:, None] + 1e-9)
    sns.heatmap(
        (100 * output).cpu().numpy().astype(np.uint8),
        vmin=0.0,
        vmax=100.0,
        annot=True,
        ax=ax,
        fmt="2d"
    )

    # labels, title and ticks
    ax.set_xlabel('Output', fontsize=20)
    ax.xaxis.set_label_position('bottom')
    plt.xticks(rotation=90)
    ax.xaxis.set_ticklabels(class_names, fontsize=10)
    ax.xaxis.tick_bottom()

    ax.set_ylabel('Target', fontsize=20)
    ax.yaxis.set_ticklabels(class_names, fontsize=10)
    plt.yticks(rotation=0)

    return fig, ax
