"""Download pretrained ResNet-50d weights from timm into the custom HuggingFace model.

Usage:
    python load_pretrained.py            # saves to ./custom-resnet50d
    python load_pretrained.py -o path    # saves to custom path
"""

import argparse

import timm

from configuration_resnet import ResnetConfig
from modeling_resnet import ResnetModel


def load_weights_into_custom_model(
    timm_model_name: str = "resnet50d",
    output_dir: str = "custom-resnet50d",
):
    """Create a custom ResnetModel filled with pretrained timm weights.

    Args:
        timm_model_name:  Name of the timm model to load weights from.
        output_dir:       Directory where the HF-compatible checkpoint is saved.
    """
    # Build the custom model using the default config (resnet50d architecture)
    config = ResnetConfig()
    model = ResnetModel(config)

    # Download the pretrained timm model (downloads ~98 MB on first run)
    print(f"Downloading pretrained weights from timm: {timm_model_name} ...")
    pretrained_model = timm.create_model(timm_model_name, pretrained=True)

    # Load the timm state dict into the wrapped ResNet
    missing, unexpected = model.model.load_state_dict(
        pretrained_model.state_dict(), strict=False
    )
    if missing:
        print(f"  WARNING – missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        print(f"  WARNING – unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")

    # Save in HuggingFace format so it can be reloaded with from_pretrained()
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    print(f"Saved custom pretrained model to: {output_dir}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load timm ResNet-50d weights into custom HF model")
    parser.add_argument("-o", "--output-dir", default="custom-resnet50d", help="Output directory")
    parser.add_argument("-m", "--timm-model", default="resnet50d", help="timm model name")
    args = parser.parse_args()

    load_weights_into_custom_model(
        timm_model_name=args.timm_model,
        output_dir=args.output_dir,
    )
