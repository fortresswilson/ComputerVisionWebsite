# Computer Vision Coursework

This repository contains my Computer Vision coursework implementations. The assignments are combined into a Flask web application so that each module can be accessed and demonstrated through the same website.

## Module 2 - Camera Measurement System

Module 2 focuses on camera calibration and perspective projection for estimating the real-world dimensions of an object.

The implementation includes:

- Camera calibration parameters
- Perspective projection calculations
- Real-world object dimension estimation
- Validation measurements
- Error statistics
- Perspective projection theory

## Module 3 - Image Blurring and Filtering

Module 3 implements image blurring using an averaging filter and compares two filtering approaches:

- Spatial-domain convolution
- Fourier-domain filtering

The web application allows an image to be uploaded and a filter size to be selected. The same averaging filter is then applied using both methods.

The resulting images are displayed alongside a difference image and numerical comparison to validate the relationship between spatial convolution and multiplication in the Fourier domain.

## Technologies Used

- Python
- Flask
- NumPy
- OpenCV
- HTML
- CSS

## Running the Application

Install the required Python packages:

```bash
python3 -m pip install -r requirements.txt

Start the Flask application:

```bash
python3 app.py
```

Then open the local address displayed in your browser:

`http://127.0.0.1:5000`

## Website Pages

- `/` - Home page
- `/module2` - Camera Measurement System
- `/module3` - Image Blurring and Filtering
## Module 4 – Human Boundary Detection

### Description
Module 4 performs human boundary detection on RGB and thermal images.

For the RGB image, a classical computer vision approach using OpenCV GrabCut and contour detection is used. The result is compared with SAM2 segmentation.

For the thermal image, intensity thresholding, morphological filtering, and contour detection are used to detect the human boundary. The result is also compared with SAM2.

### Files
- `module4.py` – RGB human boundary detection using OpenCV.
- `module4_thermal.py` – Thermal image human boundary detection.
- `sam2_compare.py` – SAM2 comparison.
- `templates/module4.html` – Displays Module 4 results on the website.

### How to Run

Run the RGB boundary detection:

```bash
python3 module4.py

## Module 6 – Motion and Structure from Motion

### Description
Module 6 explores motion estimation and structure from motion using videos and images.

For Part A, two videos containing motion were used to compute optical flow and track points between consecutive frames.

For Part B, four images of a composition book were captured from different viewpoints. Feature matching and homography were used to estimate and reconstruct the boundary of the planar object.

### Files
- `module6_optical_flow.py` – Computes and visualizes optical flow for the two videos.
- `module6_tracking.py` – Tracks feature points between consecutive video frames.
- `module6_sfm.py` – Performs feature matching, homography estimation, and boundary reconstruction from four viewpoints.
- `templates/module6.html` – Displays the Module 6 experiment and results on the website.

### How to Run

Run the optical flow program:

```bash
python3 module6_optical_flow.py
```

Run the tracking program:

```bash
python3 module6_tracking.py
```

Run the structure-from-motion program:

```bash
python3 module6_sfm.py
```

To view the Module 6 webpage, start the Flask application:

```bash
python3 app.py
```

Then open:

`http://127.0.0.1:5000/module6`