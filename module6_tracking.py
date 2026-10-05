import cv2
import numpy as np
import os

# Make sure results folder exists
os.makedirs("static/results", exist_ok=True)

videos = [
    ("static/videos/video1.mov", "video1"),
    ("static/videos/video2.MOV", "video2")
]

for video_path, name in videos:

    print(f"\nProcessing {name}...")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Could not open {video_path}")
        continue

    # Move to a point a few seconds into the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_frame = int(fps * 5)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    # Read two consecutive frames
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()

    cap.release()

    if not ret1 or not ret2:
        print("Could not read consecutive frames.")
        continue

    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect strong feature points in Frame 1
    points1 = cv2.goodFeaturesToTrack(
        gray1,
        maxCorners=100,
        qualityLevel=0.01,
        minDistance=20,
        blockSize=7
    )

    if points1 is None:
        print("No feature points detected.")
        continue

    # Track the points from Frame 1 to Frame 2
    points2, status, error = cv2.calcOpticalFlowPyrLK(
        gray1,
        gray2,
        points1,
        None
    )

    # Keep successfully tracked points
    good_new = points2[status == 1]
    good_old = points1[status == 1]

    # Find the point that moved the most
    movements = np.linalg.norm(good_new - good_old, axis=1)
    index = np.argmax(movements)

    old_point = good_old[index]
    new_point = good_new[index]

    x1, y1 = old_point
    x2, y2 = new_point

    # Calculate displacement
    dx = x2 - x1
    dy = y2 - y1

    distance = np.sqrt(dx**2 + dy**2)

    print(f"Frame 1 point: ({x1:.2f}, {y1:.2f})")
    print(f"Frame 2 point: ({x2:.2f}, {y2:.2f})")
    print(f"Horizontal displacement dx = {dx:.2f} pixels")
    print(f"Vertical displacement dy = {dy:.2f} pixels")
    print(f"Total displacement = {distance:.2f} pixels")

    # Draw the tracked point on Frame 1
    frame1_result = frame1.copy()

    cv2.circle(
        frame1_result,
        (int(x1), int(y1)),
        10,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame1_result,
        f"Point: ({x1:.1f}, {y1:.1f})",
        (int(x1) + 15, int(y1) - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # Draw movement on Frame 2
    frame2_result = frame2.copy()

    cv2.circle(
        frame2_result,
        (int(x2), int(y2)),
        10,
        (0, 255, 0),
        -1
    )

    cv2.arrowedLine(
        frame2_result,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        (0, 255, 0),
        4
    )

    cv2.putText(
        frame2_result,
        f"Point: ({x2:.1f}, {y2:.1f})",
        (int(x2) + 15, int(y2) - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Save evidence images
    cv2.imwrite(
        f"static/results/{name}_tracking_frame1.jpg",
        frame1_result
    )

    cv2.imwrite(
        f"static/results/{name}_tracking_frame2.jpg",
        frame2_result
    )

    print(f"{name} tracking images saved.")

print("\nTracking analysis completed!")