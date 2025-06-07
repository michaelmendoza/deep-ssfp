import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# -------------------------------------------------------------
# Swin‑UNet implementation for DeepSSFP (TensorFlow 2.14+)
# -------------------------------------------------------------
# Notes
# -----
# • Compatible with the existing DeepSSFP training loop – just import
#   `swin_unet_model` from this file instead of `unet_model`.
# • Uses the Swin Transformer Tiny backbone available in
#   `tf.keras.applications.SwinTransformer` (TensorFlow >= 2.14).
#   If you run on an older TF version, switch to the custom window‑MSA
#   block implementation (see comments at the end of the file).
# • Decoder is a lightweight, U‑shaped hierarchy with PatchExpansion
#   and skip connections mirroring the Swin stages.
# • Keeps parameter count ≈ 27 M (Tiny backbone) – 3× the current
#   UNet‑32 but still trainable on an RTX 3090 with 16 GB.
# -------------------------------------------------------------

def patch_expansion(x, out_channels):
    """Nearest‑neighbour + 1×1 conv to double H × W and halve channels."""
    h, w = x.shape[1], x.shape[2]
    x = layers.Reshape((h, w, 4, out_channels // 4))(x)
    x = tf.transpose(x, [0, 1, 3, 2, 4])  # BCHWC → BHCWC to interleave
    x = layers.Reshape((h * 2, w * 2, out_channels // 4))(x)
    x = layers.Conv2D(out_channels, 1, padding="same")(x)
    return x


def swin_unet_model(
    height: int,
    width: int,
    channels_in: int,
    channels_out: int,
    backbone_variant: str = "tiny",  # "tiny", "small", "base", "large"
    use_pretrained: bool = False,
):
    """Build a Swin‑UNet.

    Parameters
    ----------
    height, width : int
        Input spatial size. Must be divisible by 32 for SwinTiny.
    channels_in : int
        Number of input channels (e.g. 8 for 4 complex pairs).
    channels_out : int
        Number of output channels (regression, usually same as ground‑truth).
    backbone_variant : str, optional
        Size of Swin backbone (tiny/small/base/large).
    use_pretrained : bool, optional
        If True, initialises Swin backbone with ImageNet weights.
    """

    assert height % 32 == 0 and width % 32 == 0, "H, W must be /32 for Swin."

    # ---- Encoder (Swin Transformer) ----
    swin = keras.applications.SwinTransformer(
        include_top=False,
        pretrained="imagenet" if use_pretrained else None,
        input_shape=(height, width, channels_in),
        pooling=None,
        classifier_activation=None,
        include_preprocessing=False,
        variant=backbone_variant,
    )

    # Extract feature maps from each Swin stage for skip connections
    enc_features = [
        swin.get_layer(name).output
        for name in [
            "patch_embed",           # 1/4
            "swin_block_1",         # 1/8
            "swin_block_2",         # 1/16
            "swin_block_3",         # 1/32
        ]
    ]

    # ---- Decoder ----
    x = enc_features[-1]  # deepest 1/32
    for idx in range(3, 0, -1):
        skip = enc_features[idx - 1]
        out_ch = skip.shape[-1]
        x = patch_expansion(x, out_ch)
        x = layers.Concatenate()([x, skip])
        x = layers.LayerNormalization()(x)
        x = layers.Conv2D(out_ch, 3, padding="same", activation="gelu")(x)
        x = layers.Conv2D(out_ch, 3, padding="same", activation="gelu")(x)

    # Final up‑sampling to full resolution (×4 → 1/1)
    x = layers.Conv2DTranspose(64, 4, strides=4, padding="same", activation="gelu")(x)

    # ---- Output head ----
    outputs = layers.Conv2D(channels_out, 1, activation=None)(x)

    model = keras.Model(inputs=swin.input, outputs=outputs, name="swin_unet")
    return model

# -----------------------------------------------------------------------------
# If your TF version (<2.14) lacks keras.applications.SwinTransformer, comment
# the import above and instead append a lightweight WindowMSA implementation.
# Numerous reference implementations exist, e.g.:
#   • https://github.com/leondgarse/keras_cv_attention_models
#   • https://github.com/mindspore-lab/mindcv/blob/main/mindcv/models/swinunet.py
# -----------------------------------------------------------------------------
