import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

# ฟังก์ชันแปลงพิกเซลเป็นเซนติเมตร
def pixel_to_cm(pixels, dpi=300):
    inches = pixels / dpi
    return inches * 2.54  # แปลงนิ้วเป็นเซนติเมตร

# ฟังก์ชันกำหนดรหัสขนาดตามเส้นผ่าศูนย์กลาง
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
        return 0  # เส้นผ่าศูนย์กลางไม่อยู่ในช่วงที่กำหนด

# กำหนดโฟลเดอร์ที่เก็บภาพ
folder_path = "images/"  # เปลี่ยนเป็นชื่อโฟลเดอร์ที่ใช้จริง
output_csv = "mangosteen_data.csv"

# สร้าง DataFrame เก็บผลลัพธ์
data_list = []

# วนลูปอ่านทุกไฟล์ภาพในโฟลเดอร์
for filename in os.listdir(folder_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):  # ตรวจสอบว่าเป็นไฟล์ภาพ
        file_path = os.path.join(folder_path, filename)
        original_image = cv2.imread(file_path)

        # ตรวจสอบว่าโหลดภาพสำเร็จหรือไม่
        if original_image is None:
            print(f"Error: Could not load {filename}")
            continue

        # แปลงเป็น Grayscale และทำ Threshold
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

        # ทำความสะอาด Noise
        kernel = np.ones((5, 5), np.uint8)
        binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # หา Contours
        contours, _ = cv2.findContours(binary_image_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # หา contour ที่มีพื้นที่มากที่สุด
            largest_contour = max(contours, key=cv2.contourArea)

            # ใช้ bounding box หาเส้นผ่านศูนย์กลาง
            x, y, w, h = cv2.boundingRect(largest_contour)

            # แปลงพิกเซลเป็นเซนติเมตร
            diameter_cm = pixel_to_cm(w)

            # หารหัสขนาด
            size_class = get_size_class(diameter_cm)

            # ครอปภาพบริเวณผลไม้
            padding = 20
            x1, y1 = max(x - padding, 0), max(y - padding, 0)
            x2, y2 = min(x + w + padding, original_image.shape[1]), min(y + h + padding, original_image.shape[0])
            cropped_image = original_image[y1:y2, x1:x2]

            # คำนวณค่าเฉลี่ย RGB บริเวณผลไม้
            avg_color_per_row = np.mean(cropped_image, axis=0)
            avg_color = np.mean(avg_color_per_row, axis=0)  # ค่าเฉลี่ย (B, G, R)
            avg_r, avg_g, avg_b = avg_color[2], avg_color[1], avg_color[0]  # แปลงเป็น (R, G, B)

            # เก็บข้อมูลใน List
            data_list.append([filename, avg_r, avg_g, avg_b, diameter_cm, size_class])

            print(f"Processed: {filename} - Size Class {size_class}, Diameter {diameter_cm:.2f} cm")

# แปลงข้อมูลเป็น DataFrame และบันทึก CSV
df = pd.DataFrame(data_list, columns=["Filename", "R", "G", "B", "Diameter (cm)", "Size Class"])
df.to_csv(output_csv, index=False)

print(f"\n Data saved to {output_csv}")
