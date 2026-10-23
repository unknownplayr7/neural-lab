from PIL import Image
import pytest
import torch
from torchvision import transforms

from neural_lab.data import HuggingFaceImageStream, image_transforms, make_loaders


def test_stream_adapter_reads_only_the_requested_window():
    samples = [
        {"img": Image.new("RGB", (32, 32), color=(i, 0, 0)), "label": i}
        for i in range(5)
    ]
    dataset = HuggingFaceImageStream(
        "unused/test", "train", "img", "label", transforms.ToTensor(),
        limit=2, offset=1, stream_factory=lambda: samples,
    )
    rows = list(dataset)
    assert len(rows) == 2
    assert [label for _, label in rows] == [1, 2]
    assert rows[0][0].shape == (3, 32, 32)


def test_stream_adapter_rejects_invalid_limits():
    with pytest.raises(ValueError, match="limit"):
        HuggingFaceImageStream(
            "unused/test", "train", "img", "label", transforms.ToTensor(),
            limit=0, stream_factory=lambda: [],
        )


def test_transforms_return_normalized_rgb_tensor():
    image = Image.new("RGB", (32, 32), color=(128, 128, 128))
    tensor = image_transforms(augment=False, dataset_name="cifar10")(image)
    assert tensor.shape == (3, 32, 32)
    assert tensor.dtype == torch.float32


def test_loader_configuration_errors_before_network_access():
    with pytest.raises(ValueError, match="dataset_name"):
        make_loaders(dataset_name="not-a-dataset")
    with pytest.raises(ValueError, match="batch_size"):
        make_loaders(batch_size=0)
