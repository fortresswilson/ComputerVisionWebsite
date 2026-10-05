import cv2
import numpy as np
import os

# Videos to process
videos = [
    ("static/videos/video1.mov", "static/results/video1_optical_flow.mp4"),
    ("static/videos/video2.MOV", "static/results/video2_optical_flow.mp4")
]

for input_path, output_path in videos:

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print(f"Could not open {input_path}")
        continue

    # Read first frame
    ret, first_frame = cap.read()

    if not ret:
        print(f"Could not read {input_path}")
        continue

    # Convert first frame to grayscale
    previous_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width, height)
    )

    print(f"Processing {input_path}...")

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # Convert current frame to grayscale
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            previous_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        # Draw motion arrows
        step = 25

        for y in range(0, height, step):
            for x in range(0, width, step):

                dx, dy = flow[y, x]

                end_x = int(x + dx * 3)
                end_y = int(y + dy * 3)

                cv2.arrowedLine(
                    frame,
                    (x, y),
                    (end_x, end_y),
                    (0, 255, 0),
                    1,
                    tipLength=0.3
                )

        writer.write(frame)

        # Current frame becomes previous frame
        previous_gray = current_gray

    cap.release()
    writer.release()

    print(f"Finished: {output_path}")

print("All optical flow videos completed!")