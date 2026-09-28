import cv2
import numpy as np

# Load thermal image
image = cv2.imread("static/images/person_thermal.png")

if image is None:
    raise FileNotFoundError("Could not load person_thermal.png")

print("Thermal image loaded successfully!")

# Convert thermal image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur slightly to reduce small details/noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Separate warmer/brighter regions from cooler background
_, mask = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Clean small gaps and noise
kernel = np.ones((7, 7), np.uint8)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

# Find external boundaries
contours, _ = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

if not contours:
    raise RuntimeError("No foreground boundary was detected.")

# Choose largest region, which should be the person
largest_contour = max(contours, key=cv2.contourArea)

# Draw boundary on original thermal image
boundary_image = image.copy()

cv2.drawContours(
    boundary_image,
    [largest_contour],
    -1,
    (0, 255, 0),
    4
)

# Save results
cv2.imwrite(
    "static/results/module4_thermal_mask.jpg",
    mask
)

cv2.imwrite(
    "static/results/module4_thermal_boundary.jpg",
    boundary_image
)

print("Thermal mask saved successfully!")
print("Thermal human boundary detected successfully!")