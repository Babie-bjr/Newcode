import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from picamera2 import Picamera2
from time import sleep
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# เปิดกล้องและถ่ายภาพ
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

# รอให้กล้องปรับแสง
sleep(2)

# แสดงภาพจากกล้องสด
while True:
    frame = picam2.capture_array()  # Capture a frame
    cv2.imshow("Live Camera Feed", frame)  # แสดงผลภาพสดจากกล้อง
    
    # รอการกดปุ่ม 'q' เพื่อหยุด
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# หยุดกล้องและปิดหน้าต่างการแสดงผล
picam2.stop()
cv2.destroyAllWindows()

# ถ่ายภาพและบันทึกเป็นไฟล์
image_path = "mangosteen.jpg"
picam2.capture_file(image_path)

# โหลดภาพ
original_image = cv2.imread(image_path)
if original_image is None:
    print("Error: ไม่สามารถโหลดภาพได้")
    exit()

# ฟังก์ชันแปลงพิกเซลเป็นเซนติเมตร
def pixel_to_cm(pixels, dpi=300):
    return (pixels / dpi) * 2.54

# ฟังก์ชันกำหนดคลาสขนาดมังคุด
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

# แปลงภาพเป็นขาวดำ และประมวลผลคอนทัวร์
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
_, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((5, 5), np.uint8)
binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
contours, _ = cv2.findContours(binary_image_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# วัดขนาดมังคุด
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    diameter_cm = pixel_to_cm(w)
    size_class = get_size_class(diameter_cm)

    # คำนวณค่าเฉลี่ยสีจากพื้นที่ตรงกลาง
    box_size = int(min(w, h) * 0.3)
    box_x = x + int(w / 2 - box_size / 2)
    box_y = y + int(h / 2 - box_size / 2)
    roi = original_image[box_y:box_y + box_size, box_x:box_x + box_size]
    avg_color = cv2.mean(roi)[:3]

    print(f"ค่าเฉลี่ยสี RGB: {avg_color}")
    print(f"Diameter: {diameter_cm:.2f} cm")
    print(f"Size Class: {size_class}")

# โหลดข้อมูลสีมังคุดจากไฟล์ CSV
file_path = "mangosteen_data.csv"
data = pd.read_csv(file_path)

# แปลงค่าหมวดหมู่เป็นตัวเลข
for channel in ['R', 'G', 'B']:
    data[channel] = pd.to_numeric(data[channel], errors='coerce').fillna(0)

# ฟังก์ชันจำแนกสี
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

# คำนวณคลาสของมังคุด
data['Computed_Class'] = data.apply(lambda row: classify_color(row['R'], row['G'], row['B']), axis=1)
color_mapping = {'Red': 10, 'Black': 20, 'Green': 30, 'Unknown': 10}
data['Class'] = data['Computed_Class'].map(color_mapping)

# แบ่งข้อมูลสำหรับ Train/Test
x = data[['R', 'G', 'B']]
y = data['Class']
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# ค้นหาค่า K ที่เหมาะสม
k_list = list(range(1, 51))
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), x_scaled, y, cv=5, scoring='accuracy').mean() for k in k_list]
best_k = k_list[np.argmax(cv_scores)]
print(f"Best K value: {best_k}")

# เทรนโมเดล KNN
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_scaled, y)

# ทำนายสีของมังคุดจากค่าเฉลี่ยสี
sample_scaled = scaler.transform([avg_color])
predicted_class = knn.predict(sample_scaled)[0]

# แสดงผล
print(f"Predicted Mangosteen Color Class: {predicted_class}")

# แสดงภาพ
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)), plt.title("Original Image"), plt.axis("off")
plt.subplot(1, 2, 2), plt.imshow(binary_image_cleaned, cmap='gray'), plt.title("Binary Image"), plt.axis("off")
plt.show()
