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



def _real_huggingface_stream(records):
    from datasets import Features, Image as HFImage, IterableDataset, Value

    def generate():
        yield from records

    return IterableDataset.from_generator(
        generate,
        features=Features({"image": HFImage(), "label": Value("int64")}),
    )


def test_real_huggingface_iterable_dataset_supports_offset_limit_and_batches():
    from torch.utils.data import DataLoader

    records = [
        {"image": Image.new("RGB", (4, 4), (index, 0, 0)), "label": index}
        for index in range(7)
    ]
    stream = _real_huggingface_stream(records)
    dataset = HuggingFaceImageStream(
        "unused/real-fixture", "train", "image", "label", transforms.ToTensor(),
        limit=3, offset=2, shuffle=False, stream_factory=lambda: stream,
    )
    batches = list(DataLoader(dataset, batch_size=2))
    labels = torch.cat([batch_labels for _images, batch_labels in batches]).tolist()
    expected = [
        row["label"]
        for row in _real_huggingface_stream(records).shuffle(seed=42, buffer_size=10_000).skip(2).take(3)
    ]
    assert labels == expected
    assert len(labels) == 3
    assert [len(batch_labels) for _images, batch_labels in batches] == [2, 1]


def test_real_huggingface_iterable_dataset_handles_empty_and_exhausted_streams():
    empty_stream = _real_huggingface_stream([])
    empty_dataset = HuggingFaceImageStream(
        "unused/empty-fixture", "train", "image", "label", transforms.ToTensor(),
        limit=4, stream_factory=lambda: empty_stream,
    )
    assert list(empty_dataset) == []

    short_stream = _real_huggingface_stream([
        {"image": Image.new("RGB", (4, 4)), "label": index}
        for index in range(2)
    ])
    exhausted_dataset = HuggingFaceImageStream(
        "unused/short-fixture", "train", "image", "label", transforms.ToTensor(),
        limit=5, offset=10, stream_factory=lambda: short_stream,
    )
    assert list(exhausted_dataset) == []
