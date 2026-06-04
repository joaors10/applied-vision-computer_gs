from pathlib import Path
from shutil import copy2
from sklearn.model_selection import train_test_split

SOURCE_DIR = Path("dataset_eurosat/2750")
OUTPUT_DIR = Path("dataset_eurosat")

TRAIN_SIZE = 0.70
SEED = 42

def create_split():
    classes = [d for d in SOURCE_DIR.iterdir() if d.is_dir()]

    for split in ["train", "val", "test"]:
        (OUTPUT_DIR / split).mkdir(exist_ok=True)

    for class_dir in classes:
        images = list(class_dir.glob("*"))

        train_files, temp_files = train_test_split(
            images,
            train_size=TRAIN_SIZE,
            random_state=SEED,
            shuffle=True,
        )

        val_files, test_files = train_test_split(
            temp_files,
            test_size=0.50,
            random_state=SEED,
            shuffle=True,
        )

        split_map = {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }

        for split_name, files in split_map.items():
            target_dir = OUTPUT_DIR / split_name / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)

            for file in files:
                copy2(file, target_dir / file.name)

        print(
            f"{class_dir.name}: "
            f"train={len(train_files)} "
            f"val={len(val_files)} "
            f"test={len(test_files)}"
        )

if __name__ == "__main__":
    create_split()