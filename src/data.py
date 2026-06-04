from pathlib import Path

import tensorflow as tf


AUTOTUNE = tf.data.AUTOTUNE


def load_datasets(data_dir: str, image_size: tuple[int, int], batch_size: int):
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    for directory in (train_dir, val_dir, test_dir):
        if not directory.exists():
            raise FileNotFoundError(
                f"Pasta não encontrada: {directory}. Esperado train/, val/ e test/."
            )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    class_names = train_ds.class_names

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip([]),
        ],
        name="data_augmentation",
    )

    train_ds = train_ds.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=AUTOTUNE,
    )

    return (
        train_ds.prefetch(AUTOTUNE),
        val_ds.prefetch(AUTOTUNE),
        test_ds.prefetch(AUTOTUNE),
        class_names,
    )
