import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score

# พาธโฟลเดอร์และไฟล์
image_folder = "images/"
output_csv = "data/output.csv"
dataset_path = "data/mangosteen_colors.csv"

# ฟังก์ชันแปลงพิกเซลเป็นเซนติเมตร
def pixel_to_cm(pixels, dpi=300):
    inches = pixels / dpi
    return inches * 2.54

# ฟังก์ชันกำหนดคลาสขนาด
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

# ฟังก์ชันจำแนกสีจากค่า RGB
color_ranges = {
    'Black': {'R': (15, 40), 'G': (15, 60), 'B': (25, 80)},
    'Green': {'R': (20, 90), 'G': (45, 195), 'B': (75, 220)},
    'Red': {'R': (20, 70), 'G': (20, 90), 'B': (25, 110)}
}

def classify_color(r, g, b):
    for color, ranges in color_ranges.items():
        if ranges['R'][0] <= r <= ranges['R'][1] and \
           ranges['G'][0] <= g <= ranges['G'][1] and \
           ranges['B'][0] <= b <= ranges['B'][1]:
            return color
    return "Unknown"

# ฟังก์ชันเทรนโมเดล KNN
def train_knn_model(data):
    x = data[['R', 'G', 'B']]
    y = data['Class']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    best_k = max(range(1, 21), key=lambda k: cross_val_score(KNeighborsClassifier(n_neighbors=k), x_train_scaled, y_train, cv=10).mean())

    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(x_train_scaled, y_train)

    predictions = knn.predict(x_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    return knn, scaler

# โหลดข้อมูลสีจากไฟล์ CSV และเทรนโมเดล
data = pd.read_csv(dataset_path)
knn_model, scaler = train_knn_model(data)

# ประมวลผลภาพทั้งหมดในโฟลเดอร์
results = []
for image_file in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_file)
    
    # โหลดภาพ
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: ไม่สามารถโหลดภาพ {image_path}")
        continue

    # แปลงเป็นขาวดำ และหาขอบเขต
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((5, 5), np.uint8)
    binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # หา Contours
    contours, _ = cv2.findContours(binary_image_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Warning: ไม่พบวัตถุในภาพ {image_file}")
        continue

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    diameter_cm = pixel_to_cm(w)
    size_class = get_size_class(diameter_cm)

    # อ่านค่า RGB ที่ตำแหน่งตรงกลางของวัตถุ
    center_x, center_y = x + w // 2, y + h // 2
    b, g, r = original_image[center_y, center_x]
    predicted_class = classify_color(r, g, b)

    # เก็บข้อมูล
    results.append([image_file, diameter_cm, size_class, r, g, b, predicted_class])

    # แสดงผลภาพพร้อมข้อมูล
    plt.figure(figsize=(5, 5))
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.title(f"{image_file} | ขนาด: {diameter_cm:.2f} cm | สี: {predicted_class}")
    plt.axis("off")
    plt.show()

# บันทึกข้อมูลลง CSV
df = pd.DataFrame(results, columns=["Filename", "Diameter (cm)", "Size Class", "R", "G", "B", "Predicted Class"])
df.to_csv(output_csv, index=False)
print(f"ผลลัพธ์ถูกบันทึกลงไฟล์ {output_csv}")
