"""Attention U‑Net implementation for DeepSSFP
================================================
Adds attention gates on skip connections as described in:
Oktay et al., "Attention U-Net: Learning Where to Look for the Pancreas" (2018).

Usage
-----
from attention_unet import attention_unet_model
model = attention_unet_model(height, width, in_channels, out_channels)

The model follows the same input/output convention as the existing `models.unet_model`,
so you can replace it transparently in `deepssfp.train`.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Conv2D, Conv2DTranspose, MaxPooling2D, BatchNormalization, Activation,
    Add, Multiply, concatenate, Dropout
)
from tensorflow.keras.models import Model

# -------------------------------------------------
# Building blocks
# -------------------------------------------------

def conv_block(x, filters, kernel_size=(3, 3), dropout_rate=0.1, name_prefix="conv"):
    """Two‑layer conv block with BN & ReLU."""
    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal",
               name=f"{name_prefix}_1")(x)
    x = Activation("relu", name=f"{name_prefix}_act1")(x)
    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer="he_normal",
               name=f"{name_prefix}_2")(x)
    x = Activation("relu", name=f"{name_prefix}_act2")(x)
    x = BatchNormalization(name=f"{name_prefix}_bn")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate, name=f"{name_prefix}_drop")(x)
    return x


def attention_gate(x, g, inter_filters, name_prefix="att"):
    """Attention gate that filters the skip connection *x* using gating signal *g*."""
    theta_x = Conv2D(inter_filters, (1, 1), padding="same", name=f"{name_prefix}_theta_x")(x)
    phi_g = Conv2D(inter_filters, (1, 1), padding="same", name=f"{name_prefix}_phi_g")(g)
    add_xg = Add(name=f"{name_prefix}_add")([theta_x, phi_g])
    act_xg = Activation("relu", name=f"{name_prefix}_relu")(add_xg)
    psi = Conv2D(1, (1, 1), padding="same", name=f"{name_prefix}_psi")(act_xg)
    sigm_xg = Activation("sigmoid", name=f"{name_prefix}_sigm")(psi)
    attn_x = Multiply(name=f"{name_prefix}_mul")([x, sigm_xg])
    return attn_x


def attention_unet_model(HEIGHT: int, WIDTH: int, CHANNELS: int, NUM_OUTPUTS: int,
                          filters=(32, 64, 128, 256, 512), dropout_rate=0.1) -> Model:
    """Create Attention U‑Net model.

    Parameters
    ----------
    HEIGHT, WIDTH : int
        Spatial dimensions of the input.
    CHANNELS : int
        Number of input channels.
    NUM_OUTPUTS : int
        Number of output channels.
    filters : tuple
        Filter sizes for each encoder depth.
    dropout_rate : float
        Dropout rate applied after each conv block.
    """
    # Encoder
    inputs = keras.Input(shape=(HEIGHT, WIDTH, CHANNELS), name="img")
    x = inputs
    skips = []
    for i, f in enumerate(filters[:-1]):
        x = conv_block(x, f, dropout_rate=dropout_rate, name_prefix=f"enc{i}")
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=2, padding="same", name=f"enc{i}_pool")(x)

    # Bottleneck
    x = conv_block(x, filters[-1], dropout_rate=dropout_rate, name_prefix="bottleneck")

    # Decoder with attention gates
    for i, f in enumerate(reversed(filters[:-1])):
        idx = len(filters) - 2 - i  # to index skips
        x = Conv2DTranspose(f, (3, 3), strides=2, padding="same",
                            kernel_initializer="he_normal", name=f"dec{i}_up")(x)
        # Apply attention to corresponding skip connection
        attn_skip = attention_gate(skips[idx], x, inter_filters=f // 2, name_prefix=f"dec{i}_att")
        x = concatenate([x, attn_skip], axis=-1, name=f"dec{i}_concat")
        x = conv_block(x, f, dropout_rate=dropout_rate, name_prefix=f"dec{i}")

    outputs = Conv2D(NUM_OUTPUTS, (1, 1), padding="same", activation=None, name="output")(x)
    model = Model(inputs=inputs, outputs=outputs, name="attention_unet")
    return model


# Convenience alias to mirror existing naming pattern
attention_unet = attention_unet_model
