import torch
import torch.optim as optim
import torch.nn as nn
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from data_loader import load_data
from model import load_model


def train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device='cpu'):
    model.to(device)
    start_time = time.time()

    print(f"Starting training on device : {device}")
    for epoch in range(num_epochs):  # Loop over the dataset multiple times
        epoch_start_time = time.time()
        running_loss = 0.0
        for inputs, labels in train_loader:
            # Move data to GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}, "
              f"Loss: {running_loss / len(train_loader)}, "
              f"Time spent: {time.time() - epoch_start_time:.2f}")

        evaluate_model(model, test_loader, device=device)

    print(f"Finished training in {time.time() - start_time:.2f} seconds")
    torch.cuda.empty_cache()


def evaluate_model(model, test_loader, device='cpu'):
    model.eval()  # Set the model to evaluation mode
    correct = 0  # Count of correct predictions
    total = 0  # Total number of test samples

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU if available

            # Forward pass: compute outputs (predictions) from inputs
            outputs = model(inputs)

            # Get predicted classes (the index with the highest score)
            _, predicted = torch.max(outputs, 1)

            # Update total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()  # Count correct predictions

    # Calculate accuracy percentage
    accuracy = (correct / total) * 100
    print(f"Accuracy of the model on the test images: {accuracy:.2f}%")


def visualize_predictions(model, test_loader, num_images=4, device='cpu'):
    model.eval()  # Set the model to evaluation mode

    dataiter = iter(test_loader)  # Get an iterator for the test set
    images, labels = next(dataiter)  # Get a batch of images and labels

    # Make predictions
    outputs = model(images.to(device))  # Forward pass
    _, predicted = torch.max(outputs, 1)  # Get predicted class (highest score)

    # Display the images with predicted and true labels
    show_images(images)
    for i in range(num_images):
        # Display predicted and true labels
        print(f"Predicted: {predicted[i].item()}, True label: {labels[i].item()}")


# Function to resize an image from a tensor
def resize_image(image, size=(64, 64)):
    pil_image = transforms.ToPILImage()(image)  # Convert tensor to PIL image
    pil_image = pil_image.resize(size, Image.BICUBIC)  # Resize with bicubic interpolation
    return transforms.ToTensor()(pil_image)  # Convert back to tensor


# Function to show images in a matplotlib grid
def show_images(images, size=(64, 64)):
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


# Load and show a few sample images from the dataset
def show_sample_images(train_loader, size=(64, 64)):
    # Get a batch of images
    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    show_images(images)


def main():
    matplotlib.use('TkAgg')  # Set the backend for interactive plotting
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    train_loader, test_loader = load_data(batch_size=4)

    # Show a few sample images from dataset
    show_sample_images(train_loader)

    # Load model and move it to GPU if available
    model = load_model(num_classes=10).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Stochastic Gradient Descent

    # Train model
    try:
        train_model(model, train_loader, test_loader, criterion, optimizer, num_epochs=10, device=device)
    except RuntimeError as e:
        print(f"Runtime error during training: {e}")
        if "out of memory" in str(e):
            print("Try reducing the batch size or using a smaller model.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    visualize_predictions(model, test_loader, device=device)


if __name__ == '__main__':
    main()
