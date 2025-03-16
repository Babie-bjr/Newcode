import cv2
import numpy as np
from picamera2 import Picamera2
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Function to convert pixels to cm based on dpi
def pixel_to_cm(pixels, dpi=300):
    inches = pixels / dpi
    return inches * 2.54

# Function to get size class based on diameter
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

# Function to classify color
def classify_color(r, g, b):
    color_ranges = {
        'Black': {'R': (15, 40), 'G': (15, 60), 'B': (25, 80)},
        'Green': {'R': (20, 90), 'G': (45, 195), 'B': (75, 220)},
        'Red': {'R': (20, 70), 'G': (20, 90), 'B': (25, 110)}
    }
    for color, ranges in color_ranges.items():
        if ranges['R'][0] <= r <= ranges['R'][1] and ranges['G'][0] <= g <= ranges['G'][1] and ranges['B'][0] <= b <= ranges['B'][1]:
            return color
    return "Unknown"

# Initialize camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

# Capture image
image_path = "mangosteen.jpg"
picam2.capture_file(image_path)
picam2.stop()

# Read and process image
img = cv2.imread(image_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
    if cv2.contourArea(contour) > 500:
        (x, y, w, h) = cv2.boundingRect(contour)
        diameter = pixel_to_cm(max(w, h))
        size_class = get_size_class(diameter)

        roi = img[y:y+h, x:x+w]
        avg_color = np.mean(roi, axis=(0, 1))
        r, g, b = avg_color.astype(int)
        color_class = classify_color(r, g, b)

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        label = f"Size: {size_class}, Color: {color_class}"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Show processed image
cv2.imshow("Processed Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

