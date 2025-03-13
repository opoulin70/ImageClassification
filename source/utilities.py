import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image


def resize_image(image, size=(64, 64)):
    """"Resize an image from a tensor"""
    pil_image = transforms.ToPILImage()(image)  # Convert tensor to PIL image
    pil_image = pil_image.resize(size, Image.BICUBIC)  # Resize with bicubic interpolation
    return transforms.ToTensor()(pil_image)  # Convert back to tensor


def _show_sample_images(train_loader, size=(64, 64)):
    """"Load and show a few sample images from the dataset"""
    # Get a batch of images
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    show_images(images)


def show_images(images, size=(64, 64)):
    """Show images in a matplotlib grid"""
    images_copy = images.detach().cpu()  # Ensure the tensor is detached and moved to CPU
    resized_images = []  # List to hold resized images

    # Unnormalize the image (from [-1, 1] to [0, 1])
    # Assumes CIFAR-10 normalization with mean=0.5, std=0.5
    for i in range(len(images_copy)):
        img = images_copy[i] / 2 + 0.5  # Unnormalize the image
        img_resized = resize_image(img, size)  # Resize each image
        resized_images.append(img_resized)  # Add resized image to the list

    # Convert list of resized images back to a tensor
    resized_images_tensor = torch.stack(resized_images)

    # Create a grid from the resized images
    grid_image = torchvision.utils.make_grid(resized_images_tensor)

    np_img = grid_image.numpy()  # Convert the tensor to a NumPy array
    plt.imshow(np.transpose(np_img, (1, 2, 0)))  # Convert from (C, H, W) to (H, W, C)
    plt.show()


def list_possible_values(data: dict | set) -> list:
    """Returns the keys if `data` is a dictionary, or the values if `data` is a set."""
    if isinstance(data, dict):
        return list(data.keys())
    elif isinstance(data, set):
        return list(data)
    else:
        raise TypeError("Input must be a dictionary or a set.")
