"""Streamlit front end for Neural Lab."""
from __future__ import annotations

import io
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageOps
from torch import nn

from neural_lab.data import dataset_stats, evaluation_loader_config, make_loaders
from neural_lab.model import CIFAR10_CLASSES, CifarCNN, available_device, save_checkpoint, validate_checkpoint_payload
from neural_lab.training import evaluate_classifier, run_epoch, seed_everything

st.set_page_config(page_title="Neural Lab", layout="wide")
if platform.system() != "Darwin":
    st.error("Neural Lab currently supports macOS only.")
    st.stop()
st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 1.8rem; padding-bottom: 3rem;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25);
       border-radius: 18px; background: linear-gradient(120deg, rgba(92,124,250,.13), rgba(32,201,151,.08));}
.muted {color: var(--text-color); opacity: .72;}
div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.22); padding: .8rem 1rem; border-radius: 12px;}
</style>
<div class="hero">
  <h1>Neural Lab</h1>
  <p>Train it. Watch it learn. Find out where it gets confused.</p>
  <p class="muted">A hands-on CIFAR-10 and CIFAR-100 image-classification workbench powered by PyTorch.</p>
</div>
""", unsafe_allow_html=True)

if "model" not in st.session_state:
    st.session_state.model = None
if "history" not in st.session_state:
    st.session_state.history = []
if "last_eval" not in st.session_state:
    st.session_state.last_eval = None
if "class_names" not in st.session_state:
    st.session_state.class_names = CIFAR10_CLASSES
if "model_dataset" not in st.session_state:
    st.session_state.model_dataset = None
if "optimizer" not in st.session_state:
    st.session_state.optimizer = None
if "current_epoch" not in st.session_state:
    st.session_state.current_epoch = 0
if "run_config" not in st.session_state:
    st.session_state.run_config = None


def device_options() -> list[str]:
    options = ["auto", "cpu"]
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        options.append("mps")
    return options


@st.cache_resource(show_spinner=False)
def get_loaders(dataset_name: str, batch_size: int, train_limit: int, validation_limit: int, test_limit: int, augment: bool, seed: int):
    return make_loaders(
        dataset_name=dataset_name, batch_size=batch_size,
        train_limit=train_limit, validation_limit=validation_limit,
        test_limit=test_limit, augment=augment, num_workers=0, seed=seed,
    )


def prepare_image(image: Image.Image, dataset_name: str = "cifar10") -> torch.Tensor:
    image = ImageOps.exif_transpose(image).convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    mean, std = dataset_stats(dataset_name)
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)


def show_history(history: list[dict[str, float]]) -> None:
    if not history:
        st.info("No training history yet. Start a run to populate the charts.")
        return
    rows = [{k: v for k, v in row.items() if k != "seconds"} for row in history]
    st.dataframe(rows, use_container_width=True, hide_index=True)
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], marker="o", label="Train")
        ax.plot([r["epoch"] for r in history], [r["val_loss"] for r in history], marker="o", label="Validation")
        ax.set(title="Loss by epoch", xlabel="Epoch", ylabel="Cross-entropy loss")
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.plot([r["epoch"] for r in history], [100*r["train_accuracy"] for r in history], marker="o", label="Train")
        ax.plot([r["epoch"] for r in history], [100*r["val_accuracy"] for r in history], marker="o", label="Validation")
        ax.set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy (%)", ylim=(0, 100))
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


with st.sidebar:
    st.header("Experiment setup")
    dataset_choice = st.selectbox("Dataset", ["CIFAR-10", "CIFAR-100"], index=0)
    dataset_key = "cifar100" if dataset_choice == "CIFAR-100" else "cifar10"
    device_choice = st.selectbox("Compute device", device_options(), index=0, help="Auto prefers Apple Silicon MPS, then CPU.")
    epochs = st.slider("Epochs", min_value=1, max_value=30, value=5)
    batch_size = st.select_slider("Batch size", options=[16, 32, 64, 128, 256], value=64)
    learning_rate = st.select_slider("Learning rate", options=[0.0001, 0.0003, 0.001, 0.003, 0.01], value=0.001, format_func=lambda x: f"{x:g}")
    train_limit = st.select_slider("Training examples", options=[1000, 2500, 5000, 10000, 20000, 40000], value=10000, help="Streaming caps examples read per epoch; 40,000 leaves room for validation.")
    validation_limit = st.select_slider("Validation examples", options=[1000, 2500, 5000], value=2500)
    test_limit = st.select_slider("Held-out test examples", options=[1000, 2500, 5000, 10000], value=5000)
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    augment = st.toggle("Random crop + flip augmentation", value=True)
    st.caption("Designed for macOS. Images stream from Hugging Face as needed; full dataset archives are not downloaded.")

try:
    device = available_device(device_choice)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

tab_train, tab_evaluate, tab_predict, tab_about = st.tabs(["Train", "Evaluate", "Predict", "Model & notes"])

with tab_train:
    st.subheader("Train a convolutional neural network")
    c1, c2, c3 = st.columns(3)
    c1.metric("Device", str(device).upper())
    c2.metric("Training examples / epoch", f"{train_limit:,}")
    c3.metric("Batch size", batch_size)
    st.write("Train from random initialization. Validation examples are separate from the held-out test split.")
    st.caption("Use **Train next epoch** to pause between epochs, or **Train remaining epochs** to run to the selected target. Saving a checkpoint preserves optimizer state and history for resuming.")

    if st.button("Start / reset run", type="primary", use_container_width=True, key="start-run"):
        seed_everything(int(seed))
        config = {
            "dataset_name": dataset_key, "batch_size": int(batch_size),
            "train_limit": int(train_limit), "validation_limit": int(validation_limit),
            "test_limit": int(test_limit), "augment": bool(augment), "seed": int(seed),
        }
        try:
            with st.spinner("Connecting to Hugging Face and preparing streaming data…"):
                _train_loader, _validation_loader, _test_loader, class_names = get_loaders(**config)
            model = CifarCNN(num_classes=len(class_names)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
            st.session_state.model = model
            st.session_state.optimizer = optimizer
            st.session_state.class_names = class_names
            st.session_state.model_dataset = dataset_key
            st.session_state.run_config = {**config, "learning_rate": float(learning_rate)}
            st.session_state.current_epoch = 0
            st.session_state.history = []
            st.session_state.last_eval = None
            st.success(f"New {dataset_choice} run initialized. Train the first epoch when ready.")
        except Exception as exc:
            st.error(f"Could not initialize training: {exc}")
            st.exception(exc)

    if st.session_state.model is None or st.session_state.run_config is None:
        st.info("Start a new run or load a checkpoint to begin training.")
    else:
        st.write(f"**Active run:** {st.session_state.model_dataset.upper()} · epoch {st.session_state.current_epoch}/{epochs}")
        if st.session_state.current_epoch >= epochs:
            st.success("Target epoch count reached. Increase the Epochs setting to continue training.")
        can_train = st.session_state.current_epoch < epochs
        next_col, all_col = st.columns(2)
        with next_col:
            train_next = st.button("Train next epoch", disabled=not can_train, use_container_width=True, key="train-next")
        with all_col:
            train_remaining = st.button("Train remaining epochs", disabled=not can_train, use_container_width=True, key="train-remaining")

        if train_next or train_remaining:
            config = st.session_state.run_config
            try:
                train_loader, validation_loader, _test_loader, _class_names = get_loaders(
                    config["dataset_name"], config["batch_size"], config["train_limit"],
                    config["validation_limit"], config["test_limit"], config["augment"], config["seed"]
                )
                model = st.session_state.model.to(device)
                optimizer = st.session_state.optimizer
                for optimizer_state in optimizer.state.values():
                    for state_key, state_value in optimizer_state.items():
                        if torch.is_tensor(state_value):
                            optimizer_state[state_key] = state_value.to(device)
                criterion = nn.CrossEntropyLoss()
                total_steps = 1 if train_next else epochs - st.session_state.current_epoch
                progress = st.progress(0, text="Preparing epoch…")
                batch_progress = st.progress(0, text="Waiting for first batch…")
                live = st.empty()

                for step in range(total_steps):
                    epoch_number = st.session_state.current_epoch + 1

                    def batch_callback(batch_index: int, batch_count: int, rolling_loss: float) -> None:
                        batch_progress.progress(
                            batch_index / max(batch_count, 1),
                            text=f"Epoch {epoch_number} · batch {batch_index}/{batch_count} · running loss {rolling_loss:.3f}",
                        )

                    if hasattr(train_loader.dataset, "set_epoch"):
                        train_loader.dataset.set_epoch(epoch_number - 1)
                    train_metrics = run_epoch(
                        model, train_loader, criterion, device, optimizer=optimizer, on_batch=batch_callback
                    )
                    validation_metrics = run_epoch(model, validation_loader, criterion, device)
                    record = {
                        "epoch": float(epoch_number),
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "val_loss": validation_metrics["loss"],
                        "val_accuracy": validation_metrics["accuracy"],
                        "seconds": train_metrics["seconds"],
                    }
                    st.session_state.history = [*st.session_state.history, record]
                    st.session_state.current_epoch = epoch_number
                    progress.progress((step + 1) / total_steps, text=f"Completed epoch {epoch_number}/{epochs}")
                    with live.container():
                        a, b, c, d = st.columns(4)
                        a.metric("Train loss", f"{train_metrics['loss']:.3f}")
                        b.metric("Train accuracy", f"{train_metrics['accuracy']:.1%}")
                        c.metric("Validation loss", f"{validation_metrics['loss']:.3f}")
                        d.metric("Validation accuracy", f"{validation_metrics['accuracy']:.1%}")
                        show_history(st.session_state.history)
                batch_progress.empty()
                st.success(f"Training paused at epoch {st.session_state.current_epoch}. You can resume with the next epoch.")
            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.exception(exc)

    show_history(st.session_state.history)
    if st.session_state.model is not None:
        st.divider()
        st.subheader("Save a resumable checkpoint")
        checkpoint_name = st.text_input("Checkpoint filename", value="cifar-model.pt", help="Saved inside the local checkpoints folder.")
        if st.button("Save checkpoint", key="save-checkpoint"):
            try:
                latest = st.session_state.history[-1] if st.session_state.history else {}
                if Path(checkpoint_name).name != checkpoint_name or Path(checkpoint_name).suffix.lower() not in {".pt", ".pth"}:
                    raise ValueError("Enter a filename ending in .pt or .pth, without folders.")
                saved = save_checkpoint(
                    Path("checkpoints") / checkpoint_name, st.session_state.model,
                    epoch=int(st.session_state.current_epoch),
                    accuracy=latest.get("val_accuracy"),
                    class_names=tuple(st.session_state.class_names),
                    optimizer=st.session_state.optimizer,
                    history=st.session_state.history,
                    run_config=st.session_state.run_config,
                )
                st.success(f"Saved resumable checkpoint to {saved}")
            except (OSError, ValueError) as exc:
                st.error(f"Could not save checkpoint: {exc}")

with tab_evaluate:
    st.subheader("Held-out evaluation")
    st.write(f"Evaluation uses the held-out {st.session_state.model_dataset or dataset_key} test split and does not update model weights.")
    if st.session_state.model is None:
        st.info("Train a model or load a checkpoint before evaluating.")
    else:
        if st.button("Evaluate model", type="primary"):
            try:
                with st.spinner("Evaluating…"):
                    config = st.session_state.run_config
                    _, _, test_loader, _class_names = get_loaders(
                        **evaluation_loader_config(config)
                    )
                    st.session_state.last_eval = evaluate_classifier(st.session_state.model.to(device), test_loader, device, num_classes=len(st.session_state.class_names))
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")
        metrics = st.session_state.last_eval
        if metrics:
            a, b, c = st.columns(3)
            a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
            b.metric("Test loss", f"{metrics['loss']:.3f}")
            c.metric("Images evaluated", f"{int(metrics['examples']):,}")
            st.caption("Performance is an estimate on the selected test subset, not a guarantee on arbitrary photos.")
            matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
            fig, ax = plt.subplots(figsize=(8, 6))
            image_plot = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
            ax.set(title="Confusion matrix", xlabel="Predicted class", ylabel="True class",
                   xticks=range(len(st.session_state.class_names)), yticks=range(len(st.session_state.class_names)))
            tick_step = 1 if len(st.session_state.class_names) <= 20 else 5
            ticks = list(range(0, len(st.session_state.class_names), tick_step))
            ax.set_xticks(ticks, labels=[st.session_state.class_names[i] for i in ticks], rotation=45, ha="right")
            ax.set_yticks(ticks, labels=[st.session_state.class_names[i] for i in ticks])
            fig.colorbar(image_plot, ax=ax, fraction=0.046, pad=0.04)
            threshold = matrix.max() / 2 if matrix.size else 0
            if matrix.shape[0] <= 20:
                for row in range(matrix.shape[0]):
                    for col in range(matrix.shape[1]):
                        ax.text(col, row, str(matrix[row, col]), ha="center", va="center",
                                color="white" if matrix[row, col] > threshold else "black", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            class_rows = []
            for row in metrics["per_class"]:
                class_rows.append({"Class": st.session_state.class_names[row["class_index"]],
                                   "Accuracy": row["accuracy"], "Support": row["support"]})
            st.subheader("Per-class accuracy")
            st.dataframe(class_rows, use_container_width=True, hide_index=True)

            mistakes = metrics.get("misclassified", [])
            if mistakes:
                st.subheader("Examples the model got wrong")
                mean, std = dataset_stats(st.session_state.model_dataset or dataset_key)
                columns = 4
                rows = (len(mistakes) + columns - 1) // columns
                fig, axes = plt.subplots(rows, columns, figsize=(12, 3.0 * rows))
                axes = np.asarray(axes).reshape(-1)
                for axis, mistake in zip(axes, mistakes):
                    pixels = mistake["image"].permute(1, 2, 0).numpy()
                    pixels = np.clip(pixels * np.asarray(std) + np.asarray(mean), 0, 1)
                    axis.imshow(pixels)
                    true_name = st.session_state.class_names[mistake["true_label"]]
                    predicted_name = st.session_state.class_names[mistake["predicted_label"]]
                    axis.set_title("True: {}\nPred: {} ({:.0%})".format(true_name, predicted_name, mistake["confidence"]), fontsize=9)
                    axis.axis("off")
                for axis in axes[len(mistakes):]:
                    axis.axis("off")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.divider()
    st.subheader("Load checkpoint")
    uploaded_checkpoint = st.file_uploader("Checkpoint (.pt or .pth)", type=["pt", "pth"], key="checkpoint-upload")
    if uploaded_checkpoint is not None and st.button("Load checkpoint"):
        try:
            checkpoint_bytes = uploaded_checkpoint.getvalue()
            if len(checkpoint_bytes) > 25 * 1024 * 1024:
                raise ValueError("Checkpoint exceeds the 25 MiB upload limit.")
            payload_bytes = io.BytesIO(checkpoint_bytes)
            payload = torch.load(payload_bytes, map_location="cpu", weights_only=True)
            validate_checkpoint_payload(payload)
            checkpoint_classes = tuple(payload["class_names"])
            if len(checkpoint_classes) not in (10, 100) or not all(isinstance(name, str) and name for name in checkpoint_classes):
                raise ValueError("Checkpoint class metadata is invalid.")
            model = CifarCNN(num_classes=len(checkpoint_classes))
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model = model.to(device)
            model_dataset = "cifar100" if len(checkpoint_classes) > 10 else "cifar10"
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=1e-4)
            optimizer_state = payload.get("optimizer_state_dict")
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
                restored_epoch = int(payload.get("epoch", 0))
                restored_history = payload.get("history", [])
            else:
                restored_epoch = 0
                restored_history = []
            st.session_state.model = model
            st.session_state.optimizer = optimizer
            st.session_state.class_names = checkpoint_classes
            st.session_state.model_dataset = model_dataset
            st.session_state.current_epoch = restored_epoch
            st.session_state.history = restored_history
            saved_config = payload.get("run_config") or {}
            st.session_state.run_config = {
                "dataset_name": model_dataset,
                "batch_size": int(saved_config.get("batch_size", batch_size)),
                "train_limit": int(saved_config.get("train_limit", train_limit)),
                "validation_limit": int(saved_config.get("validation_limit", validation_limit)),
                "test_limit": int(saved_config.get("test_limit", test_limit)),
                "augment": bool(saved_config.get("augment", augment)),
                "seed": int(saved_config.get("seed", seed)),
                "learning_rate": float(saved_config.get("learning_rate", learning_rate)),
            }
            st.session_state.last_eval = None
            if optimizer_state:
                st.success("Resumed checkpoint with optimizer state and history at epoch {}.".format(restored_epoch))
            else:
                st.success("Loaded model weights. No optimizer state was saved, so training starts at epoch 0.")
        except Exception as exc:
            st.error(f"Could not load checkpoint: {exc}")

with tab_predict:
    st.subheader("Try an image")
    st.write("Upload a picture to see which class the selected model considers most likely.")
    st.warning("This model only knows the selected dataset’s classes. A confident answer for an unrelated object can still be wrong.")
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"], key="image-upload")
    if uploaded_image is not None:
        try:
            image_bytes = uploaded_image.getvalue()
            if len(image_bytes) > 10 * 1024 * 1024:
                raise ValueError("Image exceeds the 10 MiB upload limit.")
            with Image.open(io.BytesIO(image_bytes)) as source_image:
                source_image.verify()
            image = Image.open(io.BytesIO(image_bytes))
            if image.width * image.height > 20_000_000:
                raise ValueError("Image dimensions exceed the 20-megapixel limit.")
            image = ImageOps.exif_transpose(image).convert("RGB")
            st.image(image, caption="Uploaded image", width=280)
            if st.session_state.model is None:
                st.info("Train a model or load a checkpoint first.")
            elif st.button("Classify image", type="primary"):
                tensor = prepare_image(image, st.session_state.model_dataset or dataset_key).to(device)
                model = st.session_state.model.to(device)
                model.eval()
                with torch.inference_mode():
                    probabilities = torch.softmax(model(tensor)[0], dim=0).cpu().numpy()
                top_indices = np.argsort(probabilities)[::-1][:5]
                best = int(top_indices[0])
                st.metric("Top prediction", st.session_state.class_names[best], f"{probabilities[best]:.1%} model confidence")
                st.bar_chart({st.session_state.class_names[int(i)]: float(probabilities[int(i)]) for i in top_indices})
                st.dataframe([{"Class": st.session_state.class_names[int(i)], "Probability": float(probabilities[int(i)])} for i in top_indices],
                             use_container_width=True, hide_index=True)
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            st.error(f"Could not read that image: {exc}")

with tab_about:
    st.subheader("What is being trained?")
    st.info("Neural Lab is a macOS-only project. Apple Silicon uses MPS acceleration when available; Intel Macs use CPU. Windows and Linux are not supported.")
    st.markdown("""
**Architecture:** convolution, batch normalization, ReLU blocks, max-pooling, dropout,
adaptive average pooling, and a linear output layer sized to the selected dataset.

**Loss:** cross-entropy. **Optimizer:** AdamW with weight decay.
**Preprocessing:** RGB conversion, 32×32 resize, and dataset-specific channel normalization.
Training augmentation uses random crop and horizontal flip. CIFAR-10 and CIFAR-100 stream from Hugging Face instead of storing full dataset archives locally.

### What the metrics mean
- **Training accuracy** measures performance on examples used for gradient updates.
- **Validation accuracy** guides training decisions. **Test accuracy** is measured separately in the Evaluate tab; repeatedly tuning against a test split can make it less independent.
- **Loss** measures confidence-weighted classification error. Lower is generally better.

### Device notes
Auto-selection prefers Apple Silicon MPS, then CPU. MPS speed and behavior depend on installed PyTorch/macOS versions; CPU is the fallback.
""")
    st.code("Input:  N × 3 × 32 × 32\nOutput: N × class_count logits", language="text")
    st.caption("Neural Lab is an educational experiment, not a general-purpose object detector.")
