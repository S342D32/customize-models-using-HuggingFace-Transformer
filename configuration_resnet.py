from transformers import PretrainedConfig


class ResnetConfig(PretrainedConfig):
    model_type = "resnet"

    def __init__(
        self,
        block_type="bottleneck",
        layers=None,
        num_classes=1000,
        input_channels=3,
        cardinality=1,
        base_width=64,
        stem_width=32,
        stem_type="deep",
        avg_down=True,
        **kwargs,
    ):
        if layers is None:
            layers = [3, 4, 6, 3]
        super().__init__(**kwargs)
        self.block_type = block_type
        self.layers = layers
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.cardinality = cardinality
        self.base_width = base_width
        self.stem_width = stem_width
        self.stem_type = stem_type
        self.avg_down = avg_down




    