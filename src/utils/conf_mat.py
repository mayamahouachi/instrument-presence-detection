import matplotlib.pyplot as plt
from torch import Tensor


def plot_confusion_matrix(confmat: Tensor, class_names=None):
    confmat_np = confmat.cpu().numpy()

    fig = plt.figure(figsize=(6, 6))
    plt.imshow(confmat_np)
    plt.colorbar()

    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    if class_names is not None:
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.yticks(range(len(class_names)), class_names)

    plt.tight_layout()
    return fig
