import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import warnings
from picamera2 import Picamera2
from picamera2.utils import Preview

warnings.filterwarnings("ignore")

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

# Function to classify color based on RGB ranges
color_ranges = {
    'Black': {'R': (15, 40), 'G': (15, 60), 'B': (25, 80)},
    'Green': {'R': (20, 90), 'G': (45, 195), 'B': (75, 220)},
    'Red': {'R': (20, 70), 'G': (20, 90), 'B': (25, 110)}
}

def classify_color(r, g, b):
    for color, ranges in color_ranges.items():
        if ranges['R'][0] <= r <= ranges['R'][1] and ranges['G'][0] <= g <= ranges['G'][1] and ranges['B'][0] <= b <= ranges['B'][1]:
            return color
    return "Unknown"

# Load dataset and preprocess for KNN
file_path = "mangosteen_data.csv"  # Update path for your dataset
data = pd.read_csv(file_path)

# Convert color channels to numeric values
for channel in ['R', 'G', 'B']:
    data[channel] = pd.to_numeric(data[channel], errors='coerce').fillna(0)

# Apply color classification
data['Computed_Class'] = data.apply(lambda row: classify_color(row['R'], row['G'], row['B']), axis=1)

# Map classes to numbers
color_mapping = {'Red': 10, 'Black': 20, 'Green': 30, 'Unknown': 10}
data['Class'] = data['Computed_Class'].map(color_mapping)

# Split dataset into train/test
x = data[['R', 'G', 'B']]
y = data['Class']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# Scale data
scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_x_scaled, train_y)

# Evaluate the model
predictions = knn.predict(test_x_scaled)
accuracy = accuracy_score(test_y, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display classification report
print(classification_report(test_y, predictions, target_names=['Red', 'Black', 'Green'], zero_division=0))

# Display confusion matrix
cm = confusion_matrix(test_y, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Red', 'Black', 'Green'])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

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

# Function to capture image from camera
def capture_image():
    picam2 = Picamera2()
    picam2.start_preview(Preview.NULL)  # Disable preview window
    time.sleep(2)  # Allow camera to warm up
    frame = picam2.capture_array()  # Capture image
    picam2.stop_preview()

    return frame

# Real-time processing loop
while True:
    # Capture image from camera
    image = capture_image()

    # Process image
    size_class, diameter = process_image(image)

    if size_class is not None:
        print(f"Predicted Size Class: {size_class}, Diameter: {diameter:.2f} cm")

        # Display image with predicted size class
        cv2.putText(image, f"Size Class: {size_class}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Diameter: {diameter:.2f} cm", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Processed Image", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

# Close all OpenCV windows
cv2.destroyAllWindows()
