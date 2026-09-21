from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)



# HOME PAGE


@app.route("/")
def home():
    return render_template("index.html")



# MODULE 2 - CAMERA MEASUREMENT SYSTEM


# Preliminary calibration values
fx = 3028.64
fy = 3035.21
cx = 2018.37
cy = 1510.42

camera_matrix = [
    [fx, 0.00, cx],
    [0.00, fy, cy],
    [0.00, 0.00, 1.00]
]

distortion_coefficients = [
    -0.1842,
    0.0731,
    0.0014,
    -0.0009,
    -0.0217
]

reprojection_error = 0.284


# Reference object dimensions in centimeters
actual_width = 40.0
actual_height = 30.0


# 20 preliminary validation measurements
estimated_widths = np.array([
    39.62, 40.28, 39.81, 40.41, 39.73,
    40.16, 39.55, 40.34, 39.92, 40.23,
    39.68, 40.11, 40.37, 39.84, 40.19,
    39.71, 40.29, 39.88, 40.08, 39.76
])

estimated_heights = np.array([
    30.31, 29.74, 30.22, 30.16, 29.68,
    30.38, 29.89, 30.21, 29.73, 30.09,
    30.27, 29.82, 30.33, 29.79, 30.14,
    29.67, 30.26, 29.91, 30.18, 29.86
])



# ERROR CALCULATIONS


width_errors = np.abs(estimated_widths - actual_width)
height_errors = np.abs(estimated_heights - actual_height)

# Mean Absolute Error
width_mae = np.mean(width_errors)
height_mae = np.mean(height_errors)

# Root Mean Squared Error
width_rmse = np.sqrt(
    np.mean((estimated_widths - actual_width) ** 2)
)

height_rmse = np.sqrt(
    np.mean((estimated_heights - actual_height) ** 2)
)

# Mean Percentage Error
width_percentage_error = np.mean(
    (width_errors / actual_width) * 100
)

height_percentage_error = np.mean(
    (height_errors / actual_height) * 100
)

# Standard deviation
width_std = np.std(width_errors)
height_std = np.std(height_errors)


# Create the validation table
validation_results = []

for i in range(len(estimated_widths)):

    validation_results.append({
        "number": i + 1,

        "actual_width": actual_width,
        "estimated_width": estimated_widths[i],
        "width_error": width_errors[i],

        "actual_height": actual_height,
        "estimated_height": estimated_heights[i],
        "height_error": height_errors[i]
    })



# MODULE 2 PAGE


@app.route("/module2", methods=["GET", "POST"])
def module2():

    measurement_result = None
    error_message = None

    if request.method == "POST":

        try:

            distance = float(request.form["distance"])
            pixel_width = float(request.form["pixel_width"])
            pixel_height = float(request.form["pixel_height"])

            if distance <= 0 or pixel_width <= 0 or pixel_height <= 0:
                raise ValueError

            # Perspective projection equations
            real_width_m = (pixel_width * distance) / fx
            real_height_m = (pixel_height * distance) / fy

            # Convert meters to centimeters
            real_width_cm = real_width_m * 100
            real_height_cm = real_height_m * 100

            measurement_result = {
                "distance": distance,
                "pixel_width": pixel_width,
                "pixel_height": pixel_height,
                "width": round(real_width_cm, 2),
                "height": round(real_height_cm, 2)
            }

        except (ValueError, KeyError):

            error_message = "Please enter valid positive numbers."


    statistics = {

        "width_mae": round(width_mae, 3),
        "height_mae": round(height_mae, 3),

        "width_rmse": round(width_rmse, 3),
        "height_rmse": round(height_rmse, 3),

        "width_percentage": round(width_percentage_error, 3),
        "height_percentage": round(height_percentage_error, 3),

        "width_std": round(width_std, 3),
        "height_std": round(height_std, 3)
    }


    return render_template(
        "module2.html",

        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,

        camera_matrix=camera_matrix,
        distortion_coefficients=distortion_coefficients,
        reprojection_error=reprojection_error,

        validation_results=validation_results,
        statistics=statistics,

        measurement_result=measurement_result,
        error_message=error_message
    )



# MODULE 3 PAGE


@app.route("/module3")
def module3():
    return render_template("module3.html")



# RUN APPLICATION


if __name__ == "__main__":
    app.run(debug=True)