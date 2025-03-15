import cv2
import numpy as np
import pandas as pd
import time
from picamera2 import Picamera2
from picamera2.utils import Preview
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report

# Function to convert pixels to cm based on dpi
def pixel_to_cm(pixels, dpi=300):
    inches = pixels / dpi
    return inches * 2.54

# Function to get size class of mangosteen based on diameter
def get_size_class(diameter):
    if diameter > 6.2:
        return 1
    elif 5.9 <= diameter <= 6.2:
        return 2
    elif 5.3 <= diameter <= 5.8:
        return 3
    elif 4.6 <= diameter <= 5.2:
        return 4
    elif 3.8 <= diameter <= 4.5:
        return 5
    else:
        return 0

# Function to process image and predict size class
def process_image(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
    contours1, _ = cv2.findContours(binary_image_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours1:
        largest_contour1 = max(contours1, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour1)
        diameter_cm = pixel_to_cm(w)
        size_class = get_size_class(diameter_cm)
        return size_class, diameter_cm
    else:
        print("No contours found.")
        return None, None

# KNN Model
def train_knn_model():
    # Load dataset and preprocess
    data = pd.read_csv("mangosteen_data.csv")  # Update with the actual path to your CSV

    # Split dataset into features (R, G, B) and target (Class)
    x = data[['R', 'G', 'B']]
    y = data['Class']

    # Scale features
    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(x)

    # Split into train and test
    train_x, test_x, train_y, test_y = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(train_x, train_y)

    # Evaluate the model
    predictions = knn.predict(test_x)
    accuracy = accuracy_score(test_y, predictions)
    print(f"KNN Model Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(test_y, predictions))

    return knn, scaler

# Function to capture image from camera
def capture_image(picam2):
    frame = picam2.capture_array()  # Capture image from the camera
    return frame

# Setup the Pi camera
picam2 = Picamera2()
picam2.start_preview(Preview.NULL)  # Disable preview window
time.sleep(2)  # Allow camera to warm up

# Train the KNN model
knn, scaler = train_knn_model()

# Real-time processing loop
while True:
    # Capture image from camera
    image = capture_image(picam2)

    # Process image to detect size and classify
    size_class, diameter = process_image(image)

    if size_class is not None:
        # Show predicted size class on the image
        cv2.putText(image, f"Size Class: {size_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Diameter: {diameter:.2f} cm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Predict class using KNN
        color_features = image[0, 0]  # Take the color of the first pixel for simplicity (or you could extract features from more pixels)
        color_scaled = scaler.transform([color_features])
        predicted_class = knn.predict(color_scaled)

        # Display predicted class on the image
        cv2.putText(image, f"Predicted Class: {predicted_class[0]}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Show the image with results
        cv2.imshow("Processed Image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Close all OpenCV windows
cv2.destroyAllWindows()

# Stop the camera preview
picam2.stop_preview()
