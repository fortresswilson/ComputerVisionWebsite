from flask import Flask, render_template, request, url_for
import numpy as np
import cv2
import os
import uuid

app = Flask(__name__)
# Folder used to store Module 3 images
RESULT_FOLDER = os.path.join("static", "results")

# Create the folder automatically if it does not exist
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config["RESULT_FOLDER"] = RESULT_FOLDER


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


# MODULE 3 - IMAGE BLURRING


@app.route("/module3", methods=["GET", "POST"])
def module3():

    results = None
    error_message = None

    if request.method == "POST":

        try:

            
            # GET UPLOADED IMAGE
            

            uploaded_file = request.files.get("image")

            if uploaded_file is None or uploaded_file.filename == "":
                raise ValueError("Please select an image.")

            # Get selected filter size
            kernel_size = int(request.form.get("kernel_size", 5))

            # Kernel size must be odd
           
            if kernel_size not in [5, 11, 21, 31]:
                raise ValueError("Please select a valid filter size.")


           
            # READ IMAGE
          

            file_bytes = np.frombuffer(
                uploaded_file.read(),
                np.uint8
            )

            image = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            if image is None:
                raise ValueError("The uploaded file is not a valid image.")


         
            # CREATE BLUR KERNEL
           

            kernel = np.ones(
                (kernel_size, kernel_size),
                dtype=np.float32
            )

            kernel = kernel / (kernel_size * kernel_size)


           
            # SPATIAL DOMAIN FILTERING
           

            spatial_result = cv2.filter2D(
                image,
                -1,
                kernel,
                borderType=cv2.BORDER_CONSTANT
            )


            # FOURIER DOMAIN FILTERING
           

            image_float = image.astype(np.float32)

            height, width, channels = image_float.shape

            # Full convolution size
            full_height = height + kernel_size - 1
            full_width = width + kernel_size - 1

            frequency_channels = []

            for channel_number in range(channels):

                channel = image_float[:, :, channel_number]

                # Fourier transform of image
                image_fft = np.fft.fft2(
                    channel,
                    s=(full_height, full_width)
                )

                # Fourier transform of kernel
                kernel_fft = np.fft.fft2(
                    kernel,
                    s=(full_height, full_width)
                )

                # Multiplication in frequency domain
                multiplied = image_fft * kernel_fft

                # Convert back to spatial domain
                full_result = np.fft.ifft2(
                    multiplied
                ).real

                # Crop result so it matches filter2D output
                offset = kernel_size // 2

                cropped = full_result[
                    offset:offset + height,
                    offset:offset + width
                ]

                frequency_channels.append(cropped)


            frequency_result_float = np.stack(
                frequency_channels,
                axis=2
            )

            frequency_result = np.clip(
                frequency_result_float,
                0,
                255
            ).astype(np.uint8)


           
            # COMPARE BOTH METHODS
           

            difference = cv2.absdiff(
                spatial_result,
                frequency_result
            )

            mean_difference = float(
                np.mean(difference)
            )

            max_difference = int(
                np.max(difference)
            )


          
            # SAVE IMAGES
            

            unique_id = uuid.uuid4().hex

            original_name = f"{unique_id}_original.jpg"
            spatial_name = f"{unique_id}_spatial.jpg"
            frequency_name = f"{unique_id}_frequency.jpg"
            difference_name = f"{unique_id}_difference.jpg"


            cv2.imwrite(
                os.path.join(
                    app.config["RESULT_FOLDER"],
                    original_name
                ),
                image
            )

            cv2.imwrite(
                os.path.join(
                    app.config["RESULT_FOLDER"],
                    spatial_name
                ),
                spatial_result
            )

            cv2.imwrite(
                os.path.join(
                    app.config["RESULT_FOLDER"],
                    frequency_name
                ),
                frequency_result
            )


            # Amplify the difference only for visualization.
            # This does NOT affect the numerical comparison.
            visible_difference = cv2.convertScaleAbs(
                difference,
                alpha=10
            )

            cv2.imwrite(
                os.path.join(
                    app.config["RESULT_FOLDER"],
                    difference_name
                ),
                visible_difference
            )


         
            # SEND RESULTS TO WEBPAGE
            

            results = {

                "original":
                    f"results/{original_name}",

                "spatial":
                    f"results/{spatial_name}",

                "frequency":
                    f"results/{frequency_name}",

                "difference":
                    f"results/{difference_name}",

                "kernel_size":
                    kernel_size,

                "mean_difference":
                    round(mean_difference, 4),

                "max_difference":
                    max_difference
            }


        except ValueError as error:

            error_message = str(error)


        except Exception as error:

            error_message = (
                "An error occurred while processing the image: "
                + str(error)
            )


    return render_template(
        "module3.html",
        results=results,
        error_message=error_message
    )

     # MODULE 4 PAGE

@app.route("/module4")
def module4():
    return render_template("module4.html")

# MODULE 6 PAGE

@app.route("/module6")
def module6():
    return render_template("module6.html")

# RUN APPLICATION


if __name__ == "__main__":
    app.run(debug=True)