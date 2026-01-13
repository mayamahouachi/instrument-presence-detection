"""Encode the three-flags output into one-integer class."""

import torch


def to_one_integer_class(y: torch.Tensor):
    """Transform into one integer class.

    :param y: binary tensor of shape (B, stem_nb)
    :return: the encoded tensor of shape (B,)
    """
    _, stem_nb = y.shape
    binary_exp = (2 ** torch.arange(stem_nb).flip([0]))[torch.newaxis, :].to(y.device)
    return torch.sum(y * binary_exp, dim=1)


def to_binary(y: torch.Tensor, nb_stems: int):
    """Reverse of the `to_one_integer_class` function.

    :param y: tensor of shape (B,)
    :return: the binary flag tensor of shape (B, nb_stems)
    """

    bits = (y.unsqueeze(1) >> torch.arange(nb_stems - 1, -1, -1, device=y.device)) & 1
    return bits
