import cv2
import numpy as np
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# --------------------------------
# LOAD SAM2 MODEL
# --------------------------------

checkpoint = "sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)
print("Loading SAM2...")

sam2_model = build_sam2(
    model_cfg,
    checkpoint,
    device=device
)

predictor = SAM2ImagePredictor(sam2_model)


# --------------------------------
# FUNCTION FOR SAM2 SEGMENTATION
# --------------------------------

def segment_person(image_path, mask_output, boundary_output):

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load {image_path}"
        )

    # Convert BGR to RGB for SAM2
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    predictor.set_image(image_rgb)

    height, width = image.shape[:2]

    # Bounding box around the main person
    box = np.array([
        int(width * 0.15),
        int(height * 0.05),
        int(width * 0.85),
        height - 5
    ])

    # Run SAM2
    with torch.inference_mode():

        masks, scores, _ = predictor.predict(
            box=box,
            multimask_output=True
        )

    # Select best mask
    best_mask = masks[np.argmax(scores)]

    # Convert mask to visible image
    mask_image = (
        best_mask * 255
    ).astype(np.uint8)

    # Save mask
    cv2.imwrite(
        mask_output,
        mask_image
    )

    # Find contours
    contours, _ = cv2.findContours(
        mask_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise RuntimeError(
            f"No contour found for {image_path}"
        )

    # Largest contour should be the person
    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    boundary_image = image.copy()

    # Draw human boundary
    cv2.drawContours(
        boundary_image,
        [largest_contour],
        -1,
        (0, 255, 0),
        4
    )

    # Save result
    cv2.imwrite(
        boundary_output,
        boundary_image
    )

    print(f"Finished: {image_path}")


# --------------------------------
# RGB IMAGE
# --------------------------------

segment_person(
    "static/images/person_rgb.jpeg",
    "static/results/module4_rgb_sam2_mask.jpg",
    "static/results/module4_rgb_sam2_boundary.jpg"
)


# --------------------------------
# THERMAL IMAGE
# --------------------------------

segment_person(
    "static/images/person_thermal.png",
    "static/results/module4_thermal_sam2_mask.jpg",
    "static/results/module4_thermal_sam2_boundary.jpg"
)


print("All SAM2 comparisons completed!")