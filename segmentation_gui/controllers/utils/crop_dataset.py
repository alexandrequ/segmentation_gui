import cv2
import numpy as np
import os


# Function to find bounding box around a class in the mask
def find_bbox(mask, class_id):
    class_mask = (mask == class_id).astype(np.uint8)
    contours, _ = cv2.findContours(
        class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        # Find the bounding box around the contour
        x, y, w, h = cv2.boundingRect(contours[0])
        return x, y, w, h
    else:
        return None


# Function to find the leftmost element among all classes
def find_leftmost_all_classes(mask):
    # Combine all class masks
    all_classes_mask = (mask > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        all_classes_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if contours:
        leftmost_contour = min(contours, key=lambda c: cv2.boundingRect(c)[0])
        x, _, _, _ = cv2.boundingRect(leftmost_contour)
        return x
    else:
        return None


# Function to crop frame and mask
def crop_images(mask_path, frame_path, bgr_path, output_dir, class_ids):
    # Load grayscale mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask_bgr = cv2.imread(bgr_path)

    # Load frame
    frame = cv2.imread(frame_path)

    # Find the leftmost element among all classes
    leftmost_x = find_leftmost_all_classes(mask)

    # Initialize variables for the longest vertical height and corresponding class
    max_vertical_height = 0
    max_vertical_height_bbox = None

    # Iterate over specified class IDs
    for class_id in class_ids:
        bbox = find_bbox(mask, class_id)
        if bbox is not None:
            # Check if the height is greater than the current maximum
            if bbox[3] + 60 > max_vertical_height:
                max_vertical_height = bbox[3]
                max_vertical_height_bbox = bbox

    # Crop the frame and masks based on the class bounding box
    if max_vertical_height_bbox is not None and leftmost_x is not None:
        x, y, w, h = max_vertical_height_bbox

        # Calculate the new width and height for a square crop with a margin
        new_size = (
            max_vertical_height + 2 * 30
        )  # Add a margin of 30 pixels on each side

        # Calculate the top-left corner for cropping with a margin
        crop_x = max(0, int(leftmost_x - new_size / 2))
        crop_y = max(0, y - 30)

        # Crop the frame
        cropped_frame = frame[crop_y : crop_y + new_size, crop_x : crop_x + new_size]

        # Crop the grayscale mask
        cropped_mask = mask[crop_y : crop_y + new_size, crop_x : crop_x + new_size]

        # Crop the BGR mask if it exists with a margin
        if mask_bgr is not None:
            cropped_mask_bgr = mask_bgr[
                crop_y : crop_y + new_size, crop_x : crop_x + new_size
            ]
            # Create a mask for the cropped region and fill it with black pixels (class 0)
            mask_black_bgr = np.zeros_like(cropped_mask_bgr)
            mask_black_bgr[:, :] = 0
            # Set the relevant region to the cropped mask
            mask_black_bgr[
                : cropped_mask_bgr.shape[0], : cropped_mask_bgr.shape[1]
            ] = cropped_mask_bgr
        else:
            cropped_mask_bgr = None
            mask_black_bgr = None

        # cv2.imwrite(frame_filename, cropped_frame)
        # cv2.imwrite(mask_filename, cropped_mask)

        # if cropped_mask_bgr is not None:
        #     cv2.imwrite(mask_bgr_filename, cropped_mask_bgr)

        # Resize images to 512x512
        resized_frame = cv2.resize(cropped_frame, (512, 512))
        resized_mask = cv2.resize(cropped_mask, (512, 512))

        output_folder_masks = "data/dataset_2/masks_2_crop"
        output_folder_frames = "data/dataset_2/frames_2_crop"
        output_folder_bgr = "data/dataset_2/bgr_2_crop"

        cv2.imwrite(
            os.path.join(output_folder_frames, os.path.basename(frame_path)),
            resized_frame,
        )
        cv2.imwrite(
            os.path.join(output_folder_masks, os.path.basename(mask_path)), resized_mask
        )

        if cropped_mask_bgr is not None:
            resized_mask_bgr = cv2.resize(cropped_mask_bgr, (512, 512))
            cv2.imwrite(
                os.path.join(output_folder_bgr, os.path.basename(mask_path)),
                resized_mask_bgr,
            )


def process_folders(frames_folder, masks_folder, bgr_folder, output_folder, class_ids):
    frame_files = os.listdir(frames_folder)

    for frame_file in frame_files:
        # Check if the file has a .png extension
        if not frame_file.lower().endswith(".png"):
            continue

        frame_path = os.path.join(frames_folder, frame_file)
        mask_path = os.path.join(
            masks_folder, frame_file
        )  # Assuming corresponding masks have the same filenames
        bgr_path = os.path.join(bgr_folder, frame_file)

        if os.path.isfile(frame_path) and os.path.isfile(mask_path):
            # Process each frame-mask pair
            crop_images(mask_path, frame_path, bgr_path, output_folder, class_ids)


# Example usage
frames_folder = "data/dataset_2/sample_frames_2"
masks_folder = "data/dataset_2/sample_masks_2"
bgr_folder = "data/dataset_2/bgr_large"
output_folder = "data/dataset_2/sample_masks_2_crop"

class_ids = [1, 2, 3]  # Replace with the class IDs you are interested in

process_folders(frames_folder, masks_folder, bgr_folder, output_folder, class_ids)
print("ok")
