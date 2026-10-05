import cv2
import numpy as np
import os

image_folder = "static/images/module6"
output_folder = "static/results"

os.makedirs(output_folder, exist_ok=True)

image_paths = [
    os.path.join(image_folder, "book_view1.jpg"),
    os.path.join(image_folder, "book_view2.jpg"),
    os.path.join(image_folder, "book_view3.jpg"),
    os.path.join(image_folder, "book_view4.jpg")
]

images = []

for path in image_paths:
    image = cv2.imread(path)

    if image is None:
        print(f"ERROR: Could not load {path}")
        exit()

    images.append(image)

print("All four images loaded successfully!")


def select_corners(image, view_number):

    selected_points = []
    display = image.copy()

    def click_point(event, x, y, flags, param):

        if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:

            selected_points.append((x, y))

            cv2.circle(
                display,
                (x, y),
                15,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                display,
                str(len(selected_points)),
                (x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3
            )

            cv2.imshow(
                f"View {view_number}",
                display
            )

    print(f"\nVIEW {view_number}")
    print("Click the book corners in this order:")
    print("1. Top-left")
    print("2. Top-right")
    print("3. Bottom-right")
    print("4. Bottom-left")
    print("Then press ENTER.")

    window_name = f"View {view_number}"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.imshow(
        window_name,
        display
    )

    cv2.setMouseCallback(
        window_name,
        click_point
    )

    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == 13 or key == 10:
            break

    cv2.destroyWindow(window_name)

    if len(selected_points) != 4:
        print("ERROR: You must select exactly four corners.")
        exit()

    print(f"Selected corners for View {view_number}:")

    for number, point in enumerate(
        selected_points,
        start=1
    ):
        print(f"Point {number}: {point}")

    return np.float32(selected_points)


# Select corners in all four images
all_corners = []

for i, image in enumerate(images):

    corners = select_corners(
        image,
        i + 1
    )

    all_corners.append(corners)

    # Draw the selected boundary
    boundary_image = image.copy()

    cv2.polylines(
        boundary_image,
        [np.int32(corners)],
        True,
        (0, 255, 0),
        8
    )

    cv2.imwrite(
        os.path.join(
            output_folder,
            f"module6_boundary_view{i + 1}.jpg"
        ),
        boundary_image
    )


# View 1 is the reference
reference_points = all_corners[0]

# Calculate homography from View 1 to Views 2, 3 and 4
for i in range(1, 4):

    destination_points = all_corners[i]

    H, mask = cv2.findHomography(
        reference_points,
        destination_points
    )

    print(
        f"\nHomography Matrix: "
        f"View 1 -> View {i + 1}"
    )

    print(H)

    # Predict the View i corners using the homography
    predicted_points = cv2.perspectiveTransform(
        reference_points.reshape(-1, 1, 2),
        H
    ).reshape(-1, 2)

    print(
        f"Reconstructed points in View {i + 1}:"
    )

    for number, point in enumerate(
        predicted_points,
        start=1
    ):
        print(
            f"Point {number}: "
            f"({point[0]:.2f}, {point[1]:.2f})"
        )

    # Compare predicted points with actual selected points
    errors = np.linalg.norm(
        predicted_points - destination_points,
        axis=1
    )

    average_error = np.mean(errors)

    print(
        f"Average reconstruction error: "
        f"{average_error:.4f} pixels"
    )

print("\nStructure from Motion processing completed!")