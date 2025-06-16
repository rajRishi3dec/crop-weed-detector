import os

# Paths to your dataset
train_dir = 'dataset/train'
val_dir = 'dataset/val'

# Function to count images
def count_images(directory):
    for class_name in os.listdir(directory):
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            num_images = len(os.listdir(class_path))
            print(f"📁 {class_name}: {num_images} images")

print("🔎 Images in Train folder:")
count_images(train_dir)

print("\n🔎 Images in Validation folder:")
count_images(val_dir)
