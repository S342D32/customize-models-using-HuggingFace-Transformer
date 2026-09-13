r"""Comprehensive test suite for the custom ResNet model.

Covers:
  1. Configuration   - ResnetConfig defaults, customisation, serialisation
  2. Model creation  - ResnetModel instantiation, architecture, type checks
  3. Forward pass    - output shapes, multi-sample batches, different configs
  4. Save / Load     - HuggingFace save_pretrained / from_pretrained round-trip
  5. Pre-trained     - loading timm resnet50d weights, output equivalence
  6. Integration     - end-to-end workflow via load_pretrained module

Run:
    .venv\Scripts\pytest test_custom_resnet.py -v
    .venv\Scripts\pytest test_custom_resnet.py -v -m "not network"
"""

import pytest
import torch
import timm

from transformers import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

from configuration_resnet import ResnetConfig
from modeling_resnet import ResnetModel, BLOCK_MAPPING


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config():
    """Default ResNet-50 config (matches timm resnet50d)."""
    return ResnetConfig()


@pytest.fixture
def model(config):
    """Custom ResNet model in eval mode."""
    m = ResnetModel(config)
    m.eval()
    return m


@pytest.fixture
def dummy_input():
    """A single ImageNet-style sample: (1, 3, 224, 224)."""
    torch.manual_seed(123)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def dummy_batch():
    """A mini-batch: (4, 3, 224, 224)."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 224, 224)


def _has_network():
    """Return True if a quick connection to the HF Hub succeeds."""
    try:
        import urllib.request
        urllib.request.urlopen("https://huggingface.co", timeout=5)
        return True
    except Exception:
        return False


NETWORK_AVAILABLE = _has_network()


# ---------------------------------------------------------------------------
# 1. Config tests
# ---------------------------------------------------------------------------

class TestResnetConfig:
    """Tests for ResnetConfig."""

    def test_is_pretrained_config(self, config):
        assert isinstance(config, PretrainedConfig)

    def test_model_type(self, config):
        assert config.model_type == "resnet"

    def test_default_values(self, config):
        """Defaults should match the timm resnet50d architecture."""
        assert config.block_type == "bottleneck"
        assert config.layers == [3, 4, 6, 3]
        assert config.num_classes == 1000
        assert config.input_channels == 3
        assert config.cardinality == 1
        assert config.base_width == 64
        assert config.stem_width == 32
        assert config.stem_type == "deep"
        assert config.avg_down is True

    def test_custom_values(self):
        """User overrides should propagate to the config attributes."""
        cfg = ResnetConfig(
            block_type="basic",
            layers=[1, 2, 3, 4],
            num_classes=10,
            input_channels=1,
            cardinality=2,
            base_width=128,
            stem_width=64,
            stem_type="default",
            avg_down=False,
        )
        assert cfg.block_type == "basic"
        assert cfg.layers == [1, 2, 3, 4]
        assert cfg.num_classes == 10
        assert cfg.input_channels == 1
        assert cfg.cardinality == 2
        assert cfg.base_width == 128
        assert cfg.stem_width == 64
        assert cfg.stem_type == "default"
        assert cfg.avg_down is False

    def test_save_and_load_pretrained(self, tmp_path):
        """Config should round-trip through save_pretrained / from_pretrained."""
        cfg = ResnetConfig(num_classes=42, layers=[2, 2, 2, 2])
        cfg.save_pretrained(str(tmp_path))
        loaded = ResnetConfig.from_pretrained(str(tmp_path))
        assert loaded.model_type == "resnet"
        assert loaded.num_classes == 42
        assert loaded.layers == [2, 2, 2, 2]
        assert loaded.block_type == "bottleneck"

    def test_config_to_json_string(self, config):
        """Config should serialise to a valid JSON string."""
        json_str = config.to_json_string()
        assert isinstance(json_str, str)
        assert '"model_type": "resnet"' in json_str


# ---------------------------------------------------------------------------
# 2. Model creation tests
# ---------------------------------------------------------------------------

class TestResnetModel:
    """Tests for ResnetModel."""

    def test_is_pretrained_model(self, model):
        """Model should inherit from PreTrainedModel."""
        assert isinstance(model, PreTrainedModel)

    def test_config_class(self, model):
        assert model.config_class is ResnetConfig
        assert isinstance(model.config, ResnetConfig)

    def test_block_mapping(self):
        """BLOCK_MAPPING should map names to timm block classes."""
        from timm.models.resnet import BasicBlock, Bottleneck
        assert BLOCK_MAPPING["basic"] is BasicBlock
        assert BLOCK_MAPPING["bottleneck"] is Bottleneck

    def test_state_dict_not_empty(self, model):
        """Model should have non-trivial parameters after init."""
        sd = model.state_dict()
        assert len(sd) > 50

    def test_state_dict_has_expected_prefixes(self, model):
        """State-dict keys should reference stem, layers, and fc."""
        keys = [str(k) for k in model.state_dict().keys()]
        joined = " ".join(keys)
        assert "conv1" in joined
        assert any(f"layer{i}" in joined for i in range(1, 5))
        assert "fc" in joined


# ---------------------------------------------------------------------------
# 3. Forward pass tests
# ---------------------------------------------------------------------------

class TestForwardPass:
    """Tests for ResnetModel.forward."""

    def test_default_output_shape(self, model, dummy_input):
        """Forward of (1, 3, 224, 224) should give (1, 1000)."""
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (1, 1000)

    def test_batch_output_shape(self, model, dummy_batch):
        """Forward of (4, 3, 224, 224) should give (4, 1000)."""
        with torch.no_grad():
            out = model(dummy_batch)
        assert out.shape == (4, 1000)

    def test_custom_num_classes(self, dummy_input):
        """Output channels should track num_classes."""
        cfg = ResnetConfig(num_classes=10)
        m = ResnetModel(cfg)
        m.eval()
        with torch.no_grad():
            out = m(dummy_input)
        assert out.shape == (1, 10)

    def test_different_input_sizes(self):
        """Model should handle non-standard spatial sizes."""
        cfg = ResnetConfig()
        m = ResnetModel(cfg)
        m.eval()
        for hw in [(32, 32), (64, 64), (128, 128), (160, 160)]:
            x = torch.randn(1, 3, hw[0], hw[1])
            with torch.no_grad():
                out = m(x)
            assert out.shape == (1, 1000), f"Failed for input size {hw}"

    def test_deterministic(self, model, dummy_input):
        """Same input in eval mode should always give the same output."""
        with torch.no_grad():
            out1 = model(dummy_input)
            out2 = model(dummy_input)
        assert torch.allclose(out1, out2)



# ---------------------------------------------------------------------------
# 4. Save / Load tests
# ---------------------------------------------------------------------------

class TestSaveLoad:
    """Tests for HuggingFace save_pretrained / from_pretrained."""

    def test_config_only_save_load(self, config, tmp_path):
        """Config should round-trip through HF serialisation."""
        config.save_pretrained(str(tmp_path))
        loaded = ResnetConfig.from_pretrained(str(tmp_path))
        assert loaded.num_classes == config.num_classes
        assert loaded.layers == config.layers

    def test_model_save_load(self, model, tmp_path, dummy_input):
        """Saved model reloaded with from_pretrained must match."""
        with torch.no_grad():
            original = model(dummy_input)

        model.save_pretrained(str(tmp_path))
        loaded = ResnetModel.from_pretrained(str(tmp_path))
        loaded.eval()

        with torch.no_grad():
            recovered = loaded(dummy_input)
        assert torch.allclose(original, recovered, atol=1e-5)

    def test_state_dict_keys_preserved(self, model, tmp_path):
        """Reloaded model should have identical state-dict keys."""
        model.save_pretrained(str(tmp_path))
        loaded = ResnetModel.from_pretrained(str(tmp_path))
        assert set(model.state_dict().keys()) == set(loaded.state_dict().keys())


# ---------------------------------------------------------------------------
# 5. Pre-trained weight tests (network)
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestPretrainedWeights:
    """Tests that download pretrained weights from the HF Hub.

    Automatically skipped when there is no network access.
    """

    def test_architecture_matches_timm(self):
        """Custom model (default config) should have identical keys to resnet50d."""
        if not NETWORK_AVAILABLE:
            pytest.skip("Network access required")
        timm_model = timm.create_model("resnet50d", pretrained=False)
        custom = ResnetModel(ResnetConfig())
        custom_keys = set(custom.model.state_dict().keys())
        timm_keys = set(timm_model.state_dict().keys())
        assert custom_keys == timm_keys

    def test_load_pretrained_weights(self, dummy_input):
        """Loading timm resnet50d weights should make the custom model match."""
        if not NETWORK_AVAILABLE:
            pytest.skip("Network access required")

        model = ResnetModel(ResnetConfig())
        model.eval()
        timm_model = timm.create_model("resnet50d", pretrained=True)
        timm_model.eval()

        # Verify keys match before loading
        assert set(model.model.state_dict().keys()) == set(timm_model.state_dict().keys())

        model.model.load_state_dict(timm_model.state_dict())

        with torch.no_grad():
            custom_out = model(dummy_input)
            timm_out = timm_model(dummy_input)
        assert torch.allclose(custom_out, timm_out, atol=1e-4)

    def test_save_load_with_pretrained(self, tmp_path, dummy_input):
        """Pretrained weights should survive a save -> load round-trip."""
        if not NETWORK_AVAILABLE:
            pytest.skip("Network access required")

        model = ResnetModel(ResnetConfig())
        pretrained = timm.create_model("resnet50d", pretrained=True)
        pretrained.eval()
        model.model.load_state_dict(pretrained.state_dict())
        model.eval()

        model.save_pretrained(str(tmp_path))
        loaded = ResnetModel.from_pretrained(str(tmp_path))
        loaded.eval()

        with torch.no_grad():
            orig = model(dummy_input)
            recovered = loaded(dummy_input)
        assert torch.allclose(orig, recovered, atol=1e-5)


# ---------------------------------------------------------------------------
# 6. Integration test
# ---------------------------------------------------------------------------

@pytest.mark.network
class TestIntegration:
    """End-to-end workflow test."""

    def test_load_pretrained_function(self, tmp_path):
        """The load_pretrained helper should produce a working checkpoint."""
        if not NETWORK_AVAILABLE:
            pytest.skip("Network access required")

        from load_pretrained import load_weights_into_custom_model

        model = load_weights_into_custom_model(
            timm_model_name="resnet50d",
            output_dir=str(tmp_path),
        )
        model.eval()

        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 1000)

        reloaded = ResnetModel.from_pretrained(str(tmp_path))
        reloaded.eval()
        with torch.no_grad():
            out2 = reloaded(x)
        assert torch.allclose(out, out2, atol=1e-5)


# ---------------------------------------------------------------------------
# Allow running directly:  python test_custom_resnet.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

