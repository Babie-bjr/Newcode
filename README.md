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
    if r > g and r > b:
        return 10  # Red
    elif g > r and g > b:
        return 20  # Green
    elif r < 50 and g < 50 and b < 50:
        return 30  # Black
    return 0  # ไม่เข้ากลุ่ม

# กำหนดโฟลเดอร์ภาพ
image_folder = "mangosteen_dataset"  # เปลี่ยนเป็นโฟลเดอร์ที่มี 360 รูป
output_dir = "output_data"
os.makedirs(output_dir, exist_ok=True)

# ไฟล์ CSV สำหรับบันทึกข้อมูล
size_file = os.path.join(output_dir, "size_data.csv")
color_file = os.path.join(output_dir, "color_data.csv")

# เปิดไฟล์ CSV เพื่อบันทึกข้อมูล
with open(size_file, mode="w", newline="") as size_csv, open(color_file, mode="w", newline="") as color_csv:
    size_writer = csv.writer(size_csv)
    color_writer = csv.writer(color_csv)

    # เขียนหัวข้อของไฟล์ CSV
    size_writer.writerow(["Image", "Diameter (cm)", "Size Class"])
    color_writer.writerow(["Image", "R", "G", "B", "คลาส"])

    # อ่านไฟล์ภาพทั้งหมดในโฟลเดอร์
    image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

    # เช็คว่ามีภาพหรือไม่
    if not image_files:
        print("Error: ไม่พบไฟล์ภาพในโฟลเดอร์")
        exit()

    # วนลูปรันทุกภาพ
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        original_image = cv2.imread(image_path)

        # ตรวจสอบว่าโหลดภาพสำเร็จหรือไม่
        if original_image is None:
            print(f"Error: ไม่สามารถโหลดภาพ {image_path}")
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
            color_class = get_color_class(avg_color)

            # บันทึกข้อมูลขนาดลง CSV
            size_writer.writerow([image_file, f"{diameter_cm:.2f}", size_class])

            # บันทึกข้อมูลสีลง CSV
            color_writer.writerow([image_file, avg_color[2], avg_color[1], avg_color[0], color_class])  # OpenCV ใช้ BGR

            print(f"✅ {image_file}: เส้นผ่านศูนย์กลาง ≈ {diameter_cm:.2f} cm, รหัสขนาด {size_class}, สี {avg_color} (คลาส {color_class})")

print(f"✅ ประมวลผลภาพทั้งหมด ({len(image_files)} ภาพ) เสร็จสิ้น!")
print(f"📂 ข้อมูลบันทึกลงใน: {size_file} และ {color_file}")
