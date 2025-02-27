import pandas as pd
import os

# ตั้งค่าพาธไฟล์ CSV บน Raspberry Pi
csv_path = "/home/pc/size_RGB3-1Class/output_data"
output_file = "/home/pc/size_RGB1/output_data/combined_data.csv"

file1 = pd.read_csv("/home/pc/size_RGB3-1Class/output_data/class_10.csv")
file2 = pd.read_csv("/home/pc/size_RGB3-1Class/output_data/class_20.csv")
file3 = pd.read_csv("/home/pc/size_RGB3-1Class/output_data/class_30.csv")

combined_data = pd.concat([file1, file2, file3], ignore_index=True)
combined_data.to_csv(output_file, index=False)
print(f"บันทึกข้อมูลที่ผสมกันแล้วในไฟล์ {output_file}")
print("Data 3 file save name 'combined_data.csv'.")
