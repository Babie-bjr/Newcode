import cv2
import numpy as np
import csv
import os

# ฟังก์ชันแปลงพิกเซลเป็นเซนติเมตร
def pixel_to_cm(pixels, dpi=300):
    inches = pixels / dpi
    return inches * 2.54  # แปลงนิ้วเป็นเซนติเมตร

# ฟังก์ชันกำหนดรหัสขนาด
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
        return 0  # ขนาดไม่อยู่ในช่วงที่กำหนด

# ฟังก์ชันกำหนดคลาสสี (Red = 10, Green = 20, Black = 30)
def get_color_class(rgb):
    r, g, b = rgb

    # ค่ามาตรฐานของแต่ละคลาส
    colors = {
        10: (255, 0, 0),   # Red
        20: (0, 255, 0),   # Green
        30: (0, 0, 0)      # Black
    }

    # คำนวณว่าใกล้สีไหนมากที่สุด (ใช้ Euclidean Distance)
    closest_class = min(colors, key=lambda c: np.linalg.norm(np.array(rgb) - np.array(colors[c])))

    return closest_class  # คืนค่าคลาสที่ใกล้ที่สุด (10, 20, 30 เท่านั้น)

# กำหนดโฟลเดอร์ที่เก็บรูป 360 รูป
image_folder = "mangosteen_dataset"  # เปลี่ยนเป็นโฟลเดอร์ของคุณ
output_csv = "output_data/mangosteen_results.csv"
os.makedirs("output_data", exist_ok=True)  # สร้างโฟลเดอร์ถ้ายังไม่มี

# เปิดไฟล์ CSV เพื่อบันทึกผลลัพธ์ทั้งหมด
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Image", "Diameter (cm)", "Size Class", "R", "G", "B", "คลาส"])  # หัวตาราง

    # อ่านไฟล์ภาพทั้งหมดในโฟลเดอร์
    image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

    # เช็คว่ามีภาพในโฟลเดอร์หรือไม่
    if not image_files:
        print("Error: ไม่พบไฟล์ภาพในโฟลเดอร์")
        exit()

    # วนลูปรันทุกภาพ
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        original_image = cv2.imread(image_path)

        if original_image is None:
            print(f"Error: ไม่สามารถโหลดภาพ {image_file}")
            continue  # ข้ามภาพนี้แล้วทำภาพถัดไป

        # แปลงเป็น Grayscale และทำ Threshold
        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)

        # ทำความสะอาด Noise
        kernel = np.ones((5, 5), np.uint8)
        binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # หา Contours
        contours, _ = cv2.findContours(binary_image_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # หา contour ที่ใหญ่ที่สุด
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)

            # คำนวณเส้นผ่านศูนย์กลาง (เป็น cm)
            diameter_cm = pixel_to_cm(w)
            size_class = get_size_class(diameter_cm)

            # ครอปบริเวณผลไม้
            padding = 20
            x1, y1 = max(x - padding, 0), max(y - padding, 0)
            x2, y2 = min(x + w + padding, original_image.shape[1]), min(y + h + padding, original_image.shape[0])
            cropped_image = original_image[y1:y2, x1:x2]

            # คำนวณค่าเฉลี่ย RGB
            avg_color = np.mean(cropped_image, axis=(0, 1)).astype(int)  # ค่า RGB
            color_class = get_color_class(avg_color)  # จะได้แค่ค่า 10, 20, 30 เท่านั้น

            # บันทึกข้อมูลลง CSV
            writer.writerow([image_file, f"{diameter_cm:.2f}", size_class, avg_color[2], avg_color[1], avg_color[0], color_class])

            print(f"✅ {image_file}: เส้นผ่านศูนย์กลาง ≈ {diameter_cm:.2f} cm, รหัสขนาด {size_class}, R={avg_color[2]}, G={avg_color[1]}, B={avg_color[0]}, คลาส {color_class}")

print(f"✅ ประมวลผลภาพทั้งหมด ({len(image_files)} ภาพ) เสร็จสิ้น!")
print(f"📂 ข้อมูลบันทึกลงใน: {output_csv}")
