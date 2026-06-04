import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs") / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from data import load_datasets
from models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Treino das CNNs do OrbitalWatch.")
    parser.add_argument(
        "--data-dir",
        default="dataset_eurosat",
        help="Pasta com subpastas train, val e test.",
    )
    parser.add_argument("--model", choices=["simple_cnn", "deep_cnn"], required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--verbose", type=int, default=2)
    return parser.parse_args()


def plot_history(history, output_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="treino")
    axes[0].plot(history.history["val_accuracy"], label="validação")
    axes[0].set_title("Acurácia")
    axes[0].set_xlabel("Época")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="treino")
    axes[1].plot(history.history["val_loss"], label="validação")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Época")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def collect_predictions(model, dataset):
    y_true = []
    y_pred = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(probabilities, axis=1))

    return np.array(y_true), np.array(y_pred)


def compute_weights_from_directory(data_dir: str, class_names: list[str]):
    labels = []
    train_dir = Path(data_dir) / "train"

    for class_index, class_name in enumerate(class_names):
        class_dir = train_dir / class_name
        image_count = len([path for path in class_dir.iterdir() if path.is_file()])
        labels.extend([class_index] * image_count)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=np.array(labels),
    )
    return {index: float(weight) for index, weight in enumerate(weights)}


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(42)

    image_size = (args.image_size, args.image_size)
    input_shape = (args.image_size, args.image_size, 3)
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    models_dir.mkdir(exist_ok=True)
    outputs_dir.mkdir(exist_ok=True)

    train_ds, val_ds, test_ds, class_names = load_datasets(
        args.data_dir,
        image_size=image_size,
        batch_size=args.batch_size,
    )

    model = build_model(args.model, input_shape=input_shape, num_classes=len(class_names))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=models_dir / f"{args.model}_best.keras",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
        ),
    ]
    class_weight = None if args.no_class_weights else compute_weights_from_directory(args.data_dir, class_names)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=args.verbose,
    )

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    model.save(models_dir / f"{args.model}_final.keras")

    plot_history(history, outputs_dir / f"{args.model}_accuracy_loss.png")

    y_true, y_pred = collect_predictions(model, test_ds)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", xticks_rotation=30)
    fig.tight_layout()
    fig.savefig(outputs_dir / f"{args.model}_confusion_matrix.png", dpi=160)
    plt.close(fig)

    metrics = {
        "model": args.model,
        "classes": class_names,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_weight": class_weight,
        "classification_report": report,
    }
    with open(outputs_dir / f"{args.model}_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)
    with open(models_dir / f"{args.model}_classes.json", "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2, ensure_ascii=False)

    print(f"Modelo: {args.model}")
    print(f"Classes: {class_names}")
    print(f"Acurácia no teste: {test_accuracy:.4f}")
    print(f"Loss no teste: {test_loss:.4f}")


if __name__ == "__main__":
    main()
