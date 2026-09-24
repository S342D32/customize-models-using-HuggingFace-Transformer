from transformers import PreTrainedModel
from timm.models.resnet import ResNet, BasicBlock, Bottleneck
from configuration_resnet import ResnetConfig

BLOCK_MAPPING = {
    "basic": BasicBlock,
    "bottleneck": Bottleneck,
}


class ResnetModel(PreTrainedModel):
    config_class = ResnetConfig

    def __init__(self, config):
        super().__init__(config)

        self.model = ResNet(
            block=BLOCK_MAPPING[config.block_type],
            layers=config.layers,
            num_classes=config.num_classes,
            in_chans=config.input_channels,
            cardinality=config.cardinality,
            base_width=config.base_width,
            stem_width=config.stem_width,
            stem_type=config.stem_type,
            avg_down=config.avg_down,
        )

    def forward(self, x):
        return self.model(x)

    def forward_features(self, x, pool: bool = True):
        """Extract features before the classification head.

        Args:
            x:    Input tensor of shape (B, C, H, W).
            pool: If True  → returns global-average-pooled vector (B, 2048).
                  If False → returns spatial feature map  (B, 2048, H', W').
        """
        features = self.model.forward_features(x)   # (B, 2048, H', W')
        if pool:
            features = self.model.forward_head(features, pre_logits=True)  # (B, 2048)
        return features

  