import cv2
import numpy as np
import os


def process_mask(mask):
    # Create a copy of the mask to store the result
    result = mask.copy()

    result[(mask == 5)] = 1
    result[(mask == 4)] = 1
    result[(mask == 3)] = 2

    return result


def keep_largest_contour(mask, class_id):
    class_mask = (mask == class_id).astype(np.uint8)

    contours, _ = cv2.findContours(
        class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Keep only the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(class_mask)
        cv2.drawContours(mask, [largest_contour], -1, class_id, thickness=cv2.FILLED)

    return mask


# Define the input and output folder paths
input_folder = "data/sample_masks_2"  # Replace with the path to your input folder
output_folder = (
    "data/sample_masks_2_class_color"  # Replace with the path to your output folder
)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over all files in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # Load the grayscale image
        image_path = os.path.join(input_folder, filename)
        mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Process the mask
        result = process_mask(mask)

        # Keep only the largest contour for class 1 and 2
        result1 = keep_largest_contour(result, class_id=1)
        result2 = keep_largest_contour(result, class_id=3)
        result = result1 | result2

        # Create an empty BGR image
        height, width = result.shape
        bgr_image = np.zeros((height, width, 3), dtype=np.uint8)

        # Define the color mapping
        color_mapping = {
            0: (0, 0, 0),  # BGR color for class 0 (Background)
            1: (255, 0, 0),  # BGR color for class 1
            2: (0, 255, 0),  # BGR color for class 2
            3: (0, 0, 255),  # BGR color for class 3
        }

        mask_path = os.path.join(input_folder, filename)
        cv2.imwrite(mask_path, result)
        # Iterate through the result and assign the color to the contours
        for class_id, color in color_mapping.items():
            bgr_image[result == class_id] = color

        # Save the resulting colored mask
        colored_mask_path = os.path.join(
            output_folder, filename.replace(".png", "_colored.png")
        )
        cv2.imwrite(colored_mask_path, bgr_image)
