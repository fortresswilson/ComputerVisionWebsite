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
