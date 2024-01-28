import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import layers

from src.model.model import MNISTCNNClassifier

classifier = MNISTCNNClassifier()
weights_path = "C:/Users/Administrator/PycharmProjects/MNIST-CNN-demo/weights/mnist_cnn_weights_2.h5"
classifier.load_weights(weights_path)
classifier.model.summary()

test_images = classifier.x_test[:10]
outputs = classifier.intermediate_model.predict(test_images)
matrices = [output for output in outputs]


def save_batch_layer_visualizations(outputs, image_index, layer_index, save_path):
    # Select the output for a specific image in the specified layer
    layer_output = outputs[layer_index][image_index]

    # Get the number of filters in the layer
    num_filters = layer_output.shape[-1]

    # Determine the grid size
    grid_size = int(np.ceil(np.sqrt(num_filters)))

    # Create a figure to hold the grid
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))

    # Plot each filter's activations
    for i in range(grid_size ** 2):
        ax = axes[i // grid_size, i % grid_size]
        ax.set_xticks([])
        ax.set_yticks([])
        if i < num_filters:
            ax.imshow(layer_output[:, :, i], cmap='viridis')

    # Remove empty subplots
    for i in range(num_filters, grid_size ** 2):
        if i < axes.flatten().shape[0]:
            fig.delaxes(axes.flatten()[i])

    # Save the figure
    plt.savefig(f"{save_path}/Layer{layer_index + 1}_Image{image_index + 1}_visualization.png")
    plt.close(fig)


img_dir = f"C:/Users/Administrator/PycharmProjects/MNIST-CNN-demo/images/results_images"
for i in range(10):  # For each image
    for j in range(len(outputs)):  # For each layer
        print(i, j)
        save_batch_layer_visualizations(outputs, image_index=i, layer_index=j, save_path=img_dir)

import cv2
import os

save_directory = "path/to/your/save/directory"  # Replace with your desired save path
size = (500, 500)  # Size to which each image will be resized

for x in range(10):  # Loop through the first 10 images
    images_to_concat = []

    # Resize and add the input image to the list
    input_image = cv2.resize(test_images[x], size)
    images_to_concat.append(input_image * 255)

    # Loop through the layers and add each feature map image
    for l in [1, 2, 3]:
        img_path = f"/images/results_images/Layer{l}_Image{x + 1}_visualization.png"
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print(img.shape)
        resized_img = cv2.resize(img, size)
        images_to_concat.append(resized_img)

    if images_to_concat:
        # Concatenate all images horizontally
        concatted_img = np.hstack(images_to_concat)

        # Save the concatenated image
        save_path = os.path.join(save_directory, f"/images/concatted_img/Combined_Image_{x + 1}.png")
        cv2.imwrite(save_path, concatted_img)

print("Images saved successfully.")


########################################################################################################################
########################################################################################################################
########################################################################################################################


def apply_pooling(image, pool_type='max', pool_size=(2, 2), strides=(2, 2)):
    if pool_type == 'max':
        pooling_layer = layers.MaxPooling2D(pool_size=pool_size, strides=strides)
    elif pool_type == 'average':
        pooling_layer = layers.AveragePooling2D(pool_size=pool_size, strides=strides)
    else:
        raise ValueError("Invalid pooling type. Choose 'max' or 'average'.")

    # Add batch and channel dimensions
    image_expanded = np.expand_dims(image, axis=[0, -1])
    pooled_image = pooling_layer(image_expanded)

    pooled_image_2d = pooled_image[0, :, :, 0]  # Remove batch and channel dimensions
    return np.array(pooled_image_2d).astype(np.float64)


# Select an image
c = 0
for img in test_images:
    print(c)
    print("Original shape:", img.shape, "Type:", img.dtype)

    original_image = img.squeeze()
    # original_image = original_image.astype(np.float32)
    print("Original shape:", original_image.shape, "Type:", original_image.dtype)

    # Apply pooling operations
    max_pooled_image = apply_pooling(original_image, pool_type='max')
    average_pooled_image = apply_pooling(original_image, pool_type='average')

    print("Max pooled shape:", max_pooled_image.shape, "Type:", max_pooled_image.dtype)
    print("Average pooled shape:", average_pooled_image.shape, "Type:", average_pooled_image.dtype)

    # Resize images to the same size for concatenation
    # size = original_image.shape  # Keep the original size
    size = (500, 500)
    print(original_image.shape)
    original_resized = cv2.resize(original_image, size)

    max_pooled_resized = cv2.resize(max_pooled_image, size) * 255
    average_pooled_resized = cv2.resize(average_pooled_image, size) * 255

    # Concatenate images horizontally
    combined_image = np.hstack([original_resized * 255, max_pooled_resized, average_pooled_resized])

    # Save the concatenated image
    save_path = "/images/pooling"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    c += 1
    cv2.imwrite(os.path.join(save_path, f"Combined_Pooling_Image_{c}.png"), combined_image)

    print("Image saved successfully.")
