import cv2
import os
from imgaug import augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapOnImage
import random

# Image augmentation pipeline
augmentation_seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Flipud(0.5),  # vertical flips
        iaa.Affine(rotate=(-45, 45)),  # random rotations
        iaa.SomeOf(
            (0, 5),
            [
                iaa.GaussianBlur(
                    (0, 3.0)
                ),  # blur images with a sigma between 0 and 3.0
                iaa.AdditiveGaussianNoise(
                    scale=(0, 0.05 * 255)
                ),  # add Gaussian noise to images
                iaa.Multiply((0.5, 1.5)),  # multiply pixel values
                iaa.LinearContrast(
                    (0.5, 2.0), per_channel=0.5
                ),  # improve or worsen the contrast
                iaa.MultiplySaturation((0.5, 1.5)),  # multiply saturation
                iaa.AddToHueAndSaturation(
                    value=(-10, 10), per_channel=True
                ),  # change hue and saturation
                iaa.AddToBrightness((-30, 30)),  # change brightness
            ],
        ),
    ],
    random_order=True,
)  # Randomize the order of augmentations

# # Color mapping
# color_mapping = {
#     0: (0, 0, 0),    # BGR color for class 0
#     1: (255, 0, 0),  # BGR color for class 1
#     2: (0, 255, 0),  # BGR color for class 2
#     3: (0, 0, 255)   # BGR color for class 3
# }


# Function to augment and preprocess images
def augment_and_preprocess(image, mask):
    # Ensure mask has the required shape (H, W, 1)
    mask = SegmentationMapOnImage(mask, shape=image.shape[:-1] + (1,))

    # Augment the image and mask
    augmented = augmentation_seq(image=image, segmentation_maps=mask)

    # Extract augmented image and mask
    augmented_image = augmented[0]  # Access the first (and only) element
    augmented_mask = augmented[1].arr

    return augmented_image, augmented_mask


def process_folders(frames_folder, masks_folder, output_folder, num_variations=5):
    frame_files = [f for f in os.listdir(frames_folder) if f.lower().endswith(".png")]
    random.shuffle(frame_files)

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        mask_path = os.path.join(
            masks_folder, frame_file
        )  # Assuming corresponding masks have the same filenames

        # Check if images are not None
        frame = cv2.imread(frame_path)
        mask = cv2.imread(mask_path)

        if frame is not None and mask is not None:
            for variation in range(num_variations):
                # Augment and preprocess images
                augmented_frame, augmented_mask = augment_and_preprocess(frame, mask)

                # Save augmented images
                output_folder_frames = "data/dataset_2/frames_augmented"
                output_folder_masks = "data/dataset_2/masks_augmented"
                output_folder_bgr = "data/dataset_2/bgr_augmented"
                os.makedirs(output_folder_frames, exist_ok=True)
                os.makedirs(output_folder_masks, exist_ok=True)
                os.makedirs(output_folder_bgr, exist_ok=True)

                augmented_frame_filename = os.path.join(
                    output_folder_frames,
                    f"{os.path.basename(frame_path).split('.')[0]}_{variation}.png",
                )
                augmented_mask_filename = os.path.join(
                    output_folder_masks,
                    f"{os.path.basename(mask_path).split('.')[0]}_{variation}.png",
                )
                bgr_mask_filename = os.path.join(
                    output_folder_bgr,
                    f"{os.path.basename(mask_path).split('.')[0]}_{variation}.png",
                )

                cv2.imwrite(augmented_frame_filename, augmented_frame)
                cv2.imwrite(augmented_mask_filename, augmented_mask)

                # # Apply color mapping to the augmented mask
                # colored_mask = np.zeros_like(augmented_mask, dtype=np.uint8)
                # for class_id, color in color_mapping.items():
                #     colored_mask[augmented_mask[:, :, 0] == class_id] = color

                # cv2.imwrite(bgr_mask_filename, colored_mask)

                print(augmented_frame_filename)


# Example usage
frames_folder = "data/dataset_2/frames_2_crop"
masks_folder = "data/dataset_2/bgr_2_crop"
output_folder = "data/augmented_dataset"

process_folders(frames_folder, masks_folder, output_folder, num_variations=5)
print("Augmentation and preprocessing with color variation complete.")
