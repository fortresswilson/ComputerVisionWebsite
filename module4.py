import cv2

# Load the RGB image
image = cv2.imread("static/images/person_rgb.jpeg")

# Check that the image loaded correctly
if image is None:
    print("Image could not be loaded.")
else:
    print("Image loaded successfully!")
    
    import numpy as np

# Create an empty mask for GrabCut
mask = np.zeros(image.shape[:2], np.uint8)

# Temporary arrays used by GrabCut
background_model = np.zeros((1, 65), np.float64)
foreground_model = np.zeros((1, 65), np.float64)

# Tell GrabCut that the person is roughly inside this rectangle
height, width = image.shape[:2]

rectangle = (
    150,          # start farther from the left
    100,          # start farther from the top
    width - 300,  # narrower rectangle
    height - 100
)

# Separate foreground from background
cv2.grabCut(
    image,
    mask,
    rectangle,
    background_model,
    foreground_model,
    5,
    cv2.GC_INIT_WITH_RECT
)
# Mark the outer edges of the image as definite background
mask[:50, :] = cv2.GC_BGD
mask[-5:, :] = cv2.GC_BGD
mask[:, :50] = cv2.GC_BGD
mask[:, -50:] = cv2.GC_BGD

# Run GrabCut again using the improved mask
cv2.grabCut(
    image,
    mask,
    None,
    background_model,
    foreground_model,
    5,
    cv2.GC_INIT_WITH_MASK
)
# Convert GrabCut result into a black-and-white mask
person_mask = np.where(
    (mask == 2) | (mask == 0),
    0,
    1
).astype("uint8")

# Convert the mask to a visible image
mask_image = person_mask * 255

# Save the mask
cv2.imwrite("static/results/module4_rgb_mask.jpg", mask_image)

print("RGB mask saved successfully!")
# Find the outer contours of the person
contours, _ = cv2.findContours(
    mask_image,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Select the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Make a copy of the original image
boundary_image = image.copy()

# Draw the person's boundary
cv2.drawContours(
    boundary_image,
    [largest_contour],
    -1,
    (0, 255, 0),
    4
)

# Save the final boundary image
cv2.imwrite(
    "static/results/module4_rgb_boundary.jpg",
    boundary_image
)

print("Human boundary detected successfully!")