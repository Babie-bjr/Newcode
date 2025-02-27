from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# โหลดข้อมูลจากไฟล์ CSV
file_path = "/content/drive/MyDrive/Dataset /mangosteen_colors_225.csv"
data = pd.read_csv(file_path)

# แปลงค่าหมวดหมู่เป็นตัวเลข
data['R'] = pd.to_numeric(data['R'], errors='coerce').fillna(0)
data['G'] = pd.to_numeric(data['G'], errors='coerce').fillna(0)
data['B'] = pd.to_numeric(data['B'], errors='coerce').fillna(0)
data['Class'] = data['Class'].replace({'Red': 10, 'Black': 20, 'Green': 30})

# แสดงการกระจายของข้อมูล RGB
def plot_data_distribution(data):
    sns.pairplot(data, hue='Class', palette='bright')
    plt.show()

plot_data_distribution(data)

# แบ่งข้อมูลเป็น Train 80% และ Test + Validation 20% (Test 10% / Validation 10%)
x = data[['R', 'G', 'B']]
y = data['Class']
train_x, temp_x, train_y, temp_y = train_test_split(x, y, train_size=0.75, random_state=42, stratify=y)
test_x, validation_x, test_y, validation_y = train_test_split(temp_x, temp_y, test_size=0.4, random_state=42, stratify=temp_y)

# Standardize ข้อมูล
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)
validation_x_scaled = scaler.transform(validation_x)

# ฟังก์ชันสำหรับ Train KNN Model
def train_knn(k):
    knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
    knn.fit(train_x_scaled, train_y)
    return knn

# ค้นหาค่า K ที่ดีที่สุด (จาก 1 ถึง 20)
def find_best_k():
    k_values = range(1, 21)
    scores = []
    for k in k_values:
        knn = train_knn(k)
        score = cross_val_score(knn, train_x_scaled, train_y, cv=5, scoring='accuracy').mean()
        scores.append(score)

    best_k = k_values[np.argmax(scores)]
    print(f'Best K: {best_k}, Accuracy: {max(scores) * 100:.2f}%')

    # พล็อตกราฟ Accuracy vs. K
    plt.figure(figsize=(8,5))
    plt.plot(k_values, scores, marker='o', linestyle='dashed', color='b')
    plt.xlabel('Number of Neighbors (K)')
    plt.ylabel('Cross Validation Accuracy')
    plt.title('Accuracy vs. K')
    plt.show()

    return best_k

# หาค่า K ที่ดีที่สุด
optimal_k = find_best_k()

# Train โมเดลด้วย K ที่ดีที่สุด
knn = train_knn(optimal_k)

# ฟังก์ชันสำหรับประเมินโมเดล
def evaluate_model(knn, x_scaled, y_true, dataset_name="Test"):
    y_pred = knn.predict(x_scaled)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy on {dataset_name} Data: {accuracy * 100:.2f}%')

    # แสดง Classification Report
    print(f"\nClassification Report on {dataset_name} Data:")
    print(classification_report(y_true, y_pred, target_names=['Red', 'Black', 'Green']))

    # แสดง Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Red', 'Black', 'Green'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix on {dataset_name} Data\nAccuracy: {accuracy * 100:.2f}%')
    plt.show()

    return accuracy

# ประเมินผลลัพธ์
test_accuracy = evaluate_model(knn, test_x_scaled, test_y, dataset_name="Test")
validation_accuracy = evaluate_model(knn, validation_x_scaled, validation_y, dataset_name="Validation")

# บันทึกผลลัพธ์ลงไฟล์
results = pd.DataFrame({
    'Dataset': ['Test', 'Validation'],
    'Accuracy': [test_accuracy * 100, validation_accuracy * 100]
})
