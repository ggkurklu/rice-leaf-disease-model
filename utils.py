import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def get_image_stats(dataset_path, class_names):
    """
    Computes image counts and paths per class.
    Args:
        dataset_path (str): Path to dataset directory.
        class_names (list): List of class names (subdirectory names).
    Returns:
        dict: Image counts per class.
        dict: Image paths per class.
    """
    image_counts = {}
    image_paths = {}

    for cls in class_names:
        class_dir = os.path.join(dataset_path, cls)
        images = glob.glob(os.path.join(class_dir, "*.[jJ][pP][gG]"))
        image_counts[cls] = len(images)
        image_paths[cls] = images

    return image_counts, image_paths

def plot_class_distribution(image_counts, palette="Set2"):
    """
    Plots a bar chart of image counts per class.
    Args:
        image_counts (dict): Dictionary of {class_name: image_count}.
        palette (str): Seaborn color palette.
    """
    plt.figure(figsize=(8, 5))
    sns.barplot(
        x=list(image_counts.keys()), 
        y=list(image_counts.values()), 
        palette=palette
    )
    plt.title("Image Count per Class")
    plt.ylabel("Number of Images")
    plt.xlabel("Class")
    plt.xticks(rotation=15)
    plt.show()

def show_sample_images(image_paths, class_names, num_images=3):
    """
    Displays sample images from each class.
    Args:
        image_paths (dict): Dictionary of {class_name: list_of_image_paths}.
        class_names (list): List of class names.
        num_images (int): Number of samples to display per class.
    """
    plt.figure(figsize=(15, 6))
    for idx, cls in enumerate(class_names):
        for i in range(num_images):
            img_path = image_paths[cls][i]
            img = Image.open(img_path)
            plt.subplot(len(class_names), num_images, idx * num_images + i + 1)
            plt.imshow(img)
            plt.title(cls)
            plt.axis("off")
    plt.tight_layout()
    plt.show()