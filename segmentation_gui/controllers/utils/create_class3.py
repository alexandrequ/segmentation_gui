import cv2
import numpy as np
import os

# Define the input and output folder paths
input_folder = (
    "data/dataset_2/masks_augmented"  # Replace with the path to your input folder
)
output_folder = "data/dataset_2/masks_augmented_class3"  # Replace with the path to your output folder

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # Load the grayscale image
        image_path = os.path.join(input_folder, filename)
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Convert the mask to np.uint8
        mask_class1 = (mask == 1).astype(np.uint8)

        # Create a copy of the mask to store the result
        result = np.zeros_like(mask)

        # Find the contours of class 1
        contours, _ = cv2.findContours(
            mask_class1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw the contours on the result image and set them to class 3 with thickness 5 pixels
        cv2.drawContours(mask, contours, -1, 3, thickness=4)

        # Perform dilation specifically for class 3 to increase its thickness by 5 pixels
        # kernel = np.ones((5, 5), np.uint8)
        # result = cv2.dilate(result, kernel, iterations=1)

        # Create an empty BGR image
        height, width = result.shape
        bgr_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Define the color mapping
        color_mapping = {
            0: (0, 0, 0),  # BGR color for class 0
            1: (255, 0, 0),  # BGR color for class 1
            2: (0, 255, 0),  # BGR color for class 2
            3: (0, 0, 255),  # BGR color for class 3
        }

        # Iterate through the result and assign the color to the contours
        for class_id, color in color_mapping.items():
            bgr_image[mask == class_id] = color

        # Save the resulting merged mask
        merged_mask_path = os.path.join(output_folder, filename)
        cv2.imwrite(merged_mask_path, mask)

        # # Optionally, you can also save the BGR image for visualization in the same loop
        # bgr_image_path = os.path.join(output_folder, filename.replace('.png', '_bgr.png'))
        # cv2.imwrite(bgr_image_path, bgr_image)
