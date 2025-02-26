from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ใช้โหมด headless ถ้าไม่มี GUI
matplotlib.use('Agg')

# ตั้งค่าพาธไฟล์ CSV บน Raspberry Pi
file_path = "/home/pi/dataset/shuffled_sampled_colors.csv"

# โหลดข้อมูลจากไฟล์ CSV
data = pd.read_csv(file_path)

# แปลงค่าหมวดหมู่เป็นตัวเลข
data['R'] = pd.to_numeric(data['R'], errors='coerce').fillna(0)
data['G'] = pd.to_numeric(data['G'], errors='coerce').fillna(0)
data['B'] = pd.to_numeric(data['B'], errors='coerce').fillna(0)
data['Class'] = data['Class'].replace({'Red': 10, 'Black': 20, 'Green': 30})

# แบ่งข้อมูลเป็น Train (80%), Test (10%), Validation (10%)
x = data[['R', 'G', 'B']]
y = data['Class']
train_x, temp_x, train_y, temp_y = train_test_split(x, y, train_size=0.8, random_state=30, stratify=y)
test_x, validation_x, test_y, validation_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=30, stratify=temp_y)

# Standardize ข้อมูล
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)
validation_x_scaled = scaler.transform(validation_x)

# ตั้งค่า K
k = 3

# Train โมเดล
knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
knn.fit(train_x_scaled, train_y)

# ทดสอบโมเดลบน Test Set
y_pred_test = knn.predict(test_x_scaled)
test_accuracy = accuracy_score(test_y, y_pred_test)
print(f'Accuracy on Test Data: {test_accuracy * 100:.2f}%')

# แสดงผล Confusion Matrix (บันทึกเป็นไฟล์แทนการแสดง GUI)
cm = confusion_matrix(test_y, y_pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Red', 'Black', 'Green'])
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix on Test Data\nAccuracy: {test_accuracy * 100:.2f}%')
plt.savefig("/home/pi/results/confusion_matrix.png")  # บันทึกไฟล์รูปแทนการแสดงผล
plt.close()

# ทำนายจากข้อมูลที่คำนวณขนาดจากภาพแล้ว
def predict_color(r, g, b):
    sample = np.array([[r, g, b]])  # ใส่ค่า RGB ที่ได้จากการคำนวณ
    sample_scaled = scaler.transform(sample)
    prediction = knn.predict(sample_scaled)
    class_map = {10: "Red", 20: "Black", 30: "Green"}
    return class_map.get(prediction[0], "Unknown")

# ตัวอย่างการใช้ฟังก์ชันทำนาย
r, g, b = 120, 50, 200  # แทนค่าที่ได้จากภาพ
predicted_class = predict_color(r, g, b)
print(f'Predicted Class for (R={r}, G={g}, B={b}): {predicted_class}')
