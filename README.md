# mangosteen_processing
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Function to convert pixels to cm (Adjust DPI based on Raspberry Pi camera)
def pixel_to_cm(pixels, dpi=300):  # Adjust DPI according to your camera
    inches = pixels / dpi
    return inches * 2.54

# Function to classify size based on diameter
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

# Set input folder
input_folder = '/home/pi/mangosteen_dataset'  # Change this to your folder
filename = 'sample_image.jpg'  # Replace with your actual image name

# Read image from local storage
image_path = os.path.join(input_folder, filename)
original_image = cv2.imread(image_path)

if original_image is None:
    print(f"Error: Could not load image from {image_path}.")
    exit()

# Convert to grayscale and apply threshold
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

# Morphological processing to clean noise
kernel = np.ones((5, 5), np.uint8)
binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

# Find contours
contours1, _ = cv2.findContours(binary_image_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_image = original_image.copy()

if contours1:
    # Find largest contour
    largest_contour1 = max(contours1, key=cv2.contourArea)
    cv2.drawContours(contours_image, [largest_contour1], -1, (0, 255, 0), 2)

    # Get bounding rectangle
    x, y, w, h = cv2.boundingRect(largest_contour1)
    center_y = y + h // 2
    leftmost = (x, center_y)
    rightmost = (x + w, center_y)

    # Draw diameter line
    cv2.line(contours_image, leftmost, rightmost, (0, 0, 255), 2)
    cv2.line(contours_image, (x, center_y - 10), (x, center_y + 10), (0, 0, 255), 2)
    cv2.line(contours_image, (x + w, center_y - 10), (x + w, center_y + 10), (0, 0, 255), 2)

    # Convert width from pixels to cm
    diameter_cm = pixel_to_cm(w)
    size_class = get_size_class(diameter_cm)

# Convert binary image to RGB for Matplotlib display
binary_image_cleaned_no_text = cv2.cvtColor(binary_image_cleaned, cv2.COLOR_GRAY2RGB)

# Draw diameter on cleaned binary image
binary_image_cleaned = cv2.cvtColor(binary_image_cleaned, cv2.COLOR_GRAY2BGR)
if contours1:
    cv2.line(binary_image_cleaned, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 255), 8)
    cv2.putText(binary_image_cleaned, f"Diameter X: {diameter_cm:.2f} cm", 
                (x + 10, y + h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3, cv2.LINE_AA)

# Define RGB measurement box
image_with_box = original_image.copy()
box_size = int(min(w, h) * 0.3)
box_x = x + int(w / 2 - box_size / 2)
box_y = y + int(h / 2 - box_size / 2)
cv2.rectangle(image_with_box, (box_x, box_y), (box_x + box_size, box_y + box_size), (255, 0, 0), 2)

# Define sampling points within the box
points = [
    (box_x + int(box_size * 0.2), box_y + int(box_size * 0.2)),
    (box_x + int(box_size * 0.8), box_y + int(box_size * 0.2)),
    (box_x + int(box_size * 0.2), box_y + int(box_size * 0.8)),
    (box_x + int(box_size * 0.8), box_y + int(box_size * 0.8)),
    (box_x + int(box_size * 0.5), box_y + int(box_size * 0.5))
]

point_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
colors = [tuple(original_image[p[1], p[0]]) for p in points]

# Draw sampling points
for p, c in zip(points, point_colors):
    cv2.circle(image_with_box, p, 5, c, -1)

# Calculate average color
avg_color = np.mean(colors, axis=0).astype(int)

# Print results
print(f"Average RGB: {tuple(avg_color)}")
print(f"Diameter X: {diameter_cm:.2f} cm")

# ---------------------- Machine Learning Part -----------------------

# เปลี่ยนพาธของไฟล์ให้ถูกต้อง
file_path = "/home/pi/Documents/sampled_colors_300.csv"  # แก้ไขพาธให้ตรงกับที่เก็บไฟล์ในเครื่อง Raspberry Pi

# โหลดข้อมูลจากไฟล์เดียว
data = pd.read_csv(file_path)

# แปลงค่าหมวดหมู่เป็นตัวเลข
data['R'] = pd.to_numeric(data['R'], errors='coerce').fillna(0)
data['G'] = pd.to_numeric(data['G'], errors='coerce').fillna(0)
data['B'] = pd.to_numeric(data['B'], errors='coerce').fillna(0)
data['Class'] = data['Class'].replace({'Red': 10, 'Black': 20, 'Green': 30})

# แบ่งข้อมูลเป็น Train (80%), Test (10%), Validation (10%) โดยใช้ stratify เพื่อรักษาสัดส่วนของคลาส
x = data[['R', 'G', 'B']]
y = data['Class']

# แบ่งข้อมูลเป็น Train 80% และ Test + Validation 20% แล้วแบ่ง Test และ Validation อย่างละ 50%
train_x, temp_x, train_y, temp_y = train_test_split(x, y, train_size=0.8, random_state=30, stratify=y)
test_x, validation_x, test_y, validation_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=30, stratify=temp_y)

# ตรวจสอบขนาดของชุดข้อมูล
print(f'Training Data: {len(train_x)} samples')
print(f'Test Data: {len(test_x)} samples')
print(f'Validation Data: {len(validation_x)} samples')

# Standardize ข้อมูล
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)
validation_x_scaled = scaler.transform(validation_x)

# ตั้งค่า K เป็น 5
k = 5

# Train โมเดลด้วย K ที่เลือก
knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
knn.fit(train_x_scaled, train_y)

# ทำ Cross Validation (ใช้ 5-fold)
cv_scores = cross_val_score(knn, train_x_scaled, train_y, cv=5, scoring='accuracy')

# แสดงผลลัพธ์ของ Cross Validation
print(f'Cross Validation Scores: {cv_scores}')
print(f'Mean Accuracy from Cross Validation: {cv_scores.mean() * 100:.2f}%')

# ทดสอบโมเดลบน Test Set
y_pred_test = knn.predict(test_x_scaled)
test_accuracy = accuracy_score(test_y, y_pred_test)
print(f'Accuracy on Test Data: {test_accuracy * 100:.2f}%')

# แสดงรายละเอียดของ Recall, Precision, F1-Score สำหรับ Test Set
print("\nClassification Report on Test Data:")
print(classification_report(test_y, y_pred_test, target_names=['Red', 'Black', 'Green']))

# ทดสอบโมเดลบน Validation Set
y_pred_validation = knn.predict(validation_x_scaled)
validation_accuracy = accuracy_score(validation_y, y_pred_validation)
print(f'Accuracy on Validation Data: {validation_accuracy * 100:.2f}%')

# แสดงรายละเอียดของ Recall, Precision, F1-Score สำหรับ Validation Set
print("\nClassification Report on Validation Data:")
print(classification_report(validation_y, y_pred_validation, target_names=['Red', 'Black', 'Green']))

# แสดง Confusion Matrix สำหรับ Validation Data
cm = confusion_matrix(validation_y, y_pred_validation)

# คำนวณ Accuracy จาก Conf
