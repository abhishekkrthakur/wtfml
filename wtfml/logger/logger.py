"""
__author__: Abhishek Thakur
"""

import logging
import sys
import warnings
import torch.nn as nn

TRAINABLE_LAYERS = (nn.Conv2d, nn.BatchNorm2d, nn.Linear, nn.ConvTranspose2d)


def logger(name=None):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    logger_fn = logging.getLogger(name=name)
    return logger_fn

def log_gradients(model, batch=None, min_check=1e-10, max_check=10):
    """
    Function to raise warning for vanishing/exploding gradients while training.

    Args:
        model (nn.Module): Model used for training
        batch (batch number, optional): Mini batch number. Defaults to None.
        min_check (, optional): Minumum accepted gradient value. Defaults to 1e-10.
        max_check (int, optional): Maximum accepted gradient value. Defaults to 10.
    """
    for i, layer in enumerate(model.modules()):
        if isinstance(layer, TRAINABLE_LAYERS):
            # Weights
            if layer.weight.grad is not None:
                min_grad = abs(layer.bias.grad).min().item()
                max_grad = abs(layer.bias.grad).max().item()

                # Log if vanishing/exploding
                if min_weight <= min_check:
                    warnings.warn(
                        "Weight Gradient vanishing during Batch - {} at layer ({} - {})."
                        "Minimum weight at layer - {}".format(batch, i, layer._get_name(), min_weight),
                        category=RuntimeWarning,
                    )
                if max_weight >= max_check:
                    warnings.warn(
                        "Weight Gradient exploding during Batch - {} at layer ({} - {})."
                        "Minimum weight at layer - {}".format(batch, i, layer._get_name(), min_weight),
                        category=RuntimeWarning,
                    )

            # Bias
            if layer.bias.grad is not None:
                min_grad = abs(layer.bias.grad).min().item()
                max_grad = abs(layer.bias.grad).max().item()

                # Log if vanishing/exploding
                if min_bias <= min_check:
                    warnings.warn(
                        "Bias Gradient vanishing during Batch - {} at layer ({} - {})."
                        "Minimum bias at layer - {}".format(batch, i, layer._get_name(), min_bias),
                        category=RuntimeWarning,
                    )
                if max_bias >= max_check:
                    warnings.warn(
                        "Bias Gradient exploding during Batch - {} at layer ({} - {})."
                        "Minimum bias at layer - {}".format(batch, i, layer._get_name(), min_bias),
                        category=RuntimeWarning,
                    )
