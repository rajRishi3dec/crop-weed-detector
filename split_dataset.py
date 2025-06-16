import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(all_path, train_path, val_path, test_size=0.2):
    images = os.listdir(all_path)
    if len(images) == 0:
        print(f"⚠️ No images found in: {all_path}")
        return
    
    # Ensure target directories exist
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    train_images, val_images = train_test_split(images, test_size=test_size)

    for image in train_images:
        shutil.move(os.path.join(all_path, image), os.path.join(train_path, image))

    for image in val_images:
        shutil.move(os.path.join(all_path, image), os.path.join(val_path, image))

    print(f"✅ Split {len(images)} images from {all_path} -> Train: {len(train_images)}, Val: {len(val_images)}")

# Define paths
all_crops_path = "all/crop"
all_weeds_path = "all/weed"
train_crops_path = "dataset/train/crop"
train_weeds_path = "dataset/train/weed"
val_crops_path = "dataset/val/crop"
val_weeds_path = "dataset/val/weed"

# Run split
split_data(all_crops_path, train_crops_path, val_crops_path)
split_data(all_weeds_path, train_weeds_path, val_weeds_path)
