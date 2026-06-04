import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Classificar uma imagem satelital.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--classes-path", default=None)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Nomes das classes na ordem do treino.",
    )
    return parser.parse_args()


def load_image(image_path: str, image_size: int):
    image = tf.keras.utils.load_img(image_path, target_size=(image_size, image_size))
    array = tf.keras.utils.img_to_array(image)
    return np.expand_dims(array, axis=0)


def main():
    args = parse_args()
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: {image_path}")

    model = tf.keras.models.load_model(model_path)
    if args.classes_path:
        with open(args.classes_path, encoding="utf-8") as file:
            class_names = json.load(file)
    elif args.classes:
        class_names = args.classes
    else:
        class_names = [f"classe_{index}" for index in range(model.output_shape[-1])]

    image = load_image(str(image_path), args.image_size)
    probabilities = model.predict(image, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print(f"Classe prevista: {class_names[predicted_index]}")
    print(f"Confiança: {probabilities[predicted_index] * 100:.2f}%")
    print("Probabilidades:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"- {class_name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()
