import csv
import os

# อ่านข้อมูลจากไฟล์ CSV ที่มีข้อมูลอยู่
data = []
with open('input_data.csv', mode='r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        data.append({'R': int(row['R']), 'G': int(row['G']), 'B': int(row['B']), 'Class': int(row['Class'])})

# สร้างโฟลเดอร์ใหม่สำหรับเก็บไฟล์
output_folder = "output_classes"
os.makedirs(output_folder, exist_ok=True)

# แยกข้อมูลตาม Class
classes = {10: [], 20: [], 30: []}

# แยกข้อมูลใน 'data' ตาม Class
for entry in data:
    classes[entry['Class']].append(entry)

# บันทึกไฟล์ CSV สำหรับแต่ละ Class โดยมีข้อมูล 100 ข้อมูลต่อ Class
for class_value, entries in classes.items():
    # ตรวจสอบว่ามีข้อมูลเพียงพอหรือไม่ และจำกัดให้เท่ากับ 100
    entries = entries[:100]
    
    # สร้างไฟล์ CSV สำหรับแต่ละ Class
    filename = f"{output_folder}/class_{class_value}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['R', 'G', 'B', 'Class'])
        writer.writeheader()
        writer.writerows(entries)

    print(f"บันทึกข้อมูล Class {class_value} ในไฟล์ {filename}")
                if red_count >= max_per_class and green_count >= max_per_class and black_count >= max_per_class:
                    break

print(f"✅ ประมวลผลภาพเสร็จสิ้น! (Red: {red_count}, Green: {green_count}, Black: {black_count})")
print(f"📂 ข้อมูลขนาดบันทึกที่: {size_csv}")
print(f"📂 ข้อมูลสีบันทึกที่: {color_csv}")
