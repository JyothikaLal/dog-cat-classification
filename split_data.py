import os
import shutil
import random
from pathlib import Path

# Configurations
SOURCE_DIR = "/home/hitech/projects/dog-cat-classification/PetImages"
OUTPUT_DIR = "training_dataset"  # You can change this
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test

def create_split_dirs(output_dir, class_names):
    for split in ['train', 'val', 'test']:
        for class_name in class_names:
            Path(os.path.join(output_dir, split, class_name)).mkdir(parents=True, exist_ok=True)

def split_data(source_dir, output_dir, split_ratios):
    class_names = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    create_split_dirs(output_dir, class_names)

    for class_name in class_names:
        class_path = os.path.join(source_dir, class_name)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(images)

        total = len(images)
        train_end = int(split_ratios[0] * total)
        val_end = train_end + int(split_ratios[1] * total)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split, files in splits.items():
            for file in files:
                src_path = os.path.join(class_path, file)
                dst_path = os.path.join(output_dir, split, class_name, file)
                shutil.copy2(src_path, dst_path)

    print(f"Dataset split complete. Output stored in '{output_dir}'.")

if __name__ == "__main__":
    random.seed(42)  
    split_data(SOURCE_DIR, OUTPUT_DIR, SPLIT_RATIOS)
