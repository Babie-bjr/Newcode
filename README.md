import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
from google.colab import drive
import warnings
import cv2

warnings.filterwarnings("ignore")

# Mount Google Drive
drive.mount('/content/drive')

# โหลดข้อมูลจากไฟล์ CSV
file_path = "/content/drive/MyDrive/Project/(2)New_MangosteenData.csv"
data = pd.read_csv(file_path)

# แปลงค่าหมวดหมู่เป็นตัวเลข
for channel in ['R', 'G', 'B']:
    data[channel] = pd.to_numeric(data[channel], errors='coerce').fillna(0)

# **ปรับค่าช่วงสีสำหรับการจัดกลุ่ม**
color_ranges = {
    'Black': {'R': (15, 40), 'G': (15, 60), 'B': (25, 80)},  # ปรับช่วงสี Black
    'Green': {'R': (20, 90), 'G': (45, 195), 'B': (75, 220)},  # ปรับช่วงสี Green
    'Red': {'R': (20, 70), 'G': (20, 90), 'B': (25, 110)}   # สีแดง
}


def classify_color(r, g, b):
    for color, ranges in color_ranges.items():
        if ranges['R'][0] <= r <= ranges['R'][1] and ranges['G'][0] <= g <= ranges['G'][1] and ranges['B'][0] <= b <= ranges['B'][1]:
            return color
    return "Unknown"

def evaluate_model(model, X, y, dataset_name="Test"):
    """Evaluates the model and returns accuracy, predictions, true labels, and features."""
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    return accuracy, predictions, y, X

def print_evaluation_results(y_true, y_pred, dataset_name):
    """Prints classification report and confusion matrix."""
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nClassification Report on {dataset_name} Data:")
    print(classification_report(y_true, y_pred, target_names=['Red', 'Black', 'Green'], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=sorted(y.unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Red', 'Black', 'Green'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix on {dataset_name} Data\nAccuracy: {accuracy * 100:.2f}%')
    plt.show()

# คำนวณคลาสของมังคุด
data['Computed_Class'] = data.apply(lambda row: classify_color(row['R'], row['G'], row['B']), axis=1)

color_mapping = {'Red': 'Class B', 'Black': 'Class C', 'Green': 'Class A', 'Unknown': 'Class B'}
data['Class'] = data['Computed_Class'].map(color_mapping)

# แบ่งข้อมูลสำหรับ Train/Validation/Test โดยใช้ StratifiedKFold
x = data[['R', 'G', 'B']]
y = data['Class']

# 1. หาค่า K ที่ดีที่สุดโดยใช้ nested cross-validation บนข้อมูลทั้งหมด
x = data[['R', 'G', 'B']]
y = data['Class']

# Initialize the scaler
scaler = MinMaxScaler()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # ใช้ StratifiedKFold
for train_index, test_index in skf.split(x, y):
    train_x, test_x = x.iloc[train_index], x.iloc[test_index]
    train_y, test_y = y.iloc[train_index], y.iloc[test_index]

    # Scale the training data within the loop
    train_x_scaled = scaler.fit_transform(train_x)  

    break  # ใช้แค่ fold แรก

# ค้นหาค่า K ที่เหมาะสม **ปรับปรุงส่วนนี้**
k_list = list(range(1, 51))  # ทดสอบค่า K ตั้งแต่ 1 ถึง 50
cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    # Use the scaled training data here
    scores = cross_val_score(knn, train_x_scaled, train_y, cv=5, scoring='accuracy')  # ใช้ 5-fold cross-validation  
    cv_scores.append(scores.mean())

# คำนวณคลาสของมังคุด
data['Computed_Class'] = data.apply(lambda row: classify_color(row['R'], row['G'], row['B']), axis=1)

color_mapping = {'Red': 'Class B', 'Black': 'Class C', 'Green': 'Class A', 'Unknown': 'Class B'}  # แมป "Unknown" กับ "Red" (Class B)
data['Class'] = data['Computed_Class'].map(color_mapping)

# แบ่งข้อมูลสำหรับ Train/Validation/Test โดยใช้ StratifiedKFold
x = data[['R', 'G', 'B']]
y = data['Class']

# 1. หาค่า K ที่ดีที่สุดโดยใช้ nested cross-validation บนข้อมูลทั้งหมด
x = data[['R', 'G', 'B']]
y = data['Class']
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)  # ปรับสเกลข้อมูลทั้งหมด

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # ใช้ StratifiedKFold
for train_index, test_index in skf.split(x, y):
    train_x, test_x = x.iloc[train_index], x.iloc[test_index]
    train_y, test_y = y.iloc[train_index], y.iloc[test_index]
    break  # ใช้แค่ fold แรก

# แสดงการกระจายตัวของข้อมูลสีแบบ 3 มิติ
def plot_3d_color_distribution(data):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = {'Red': 'r', 'Green': 'g', 'Black': 'k'}

    for color, color_code in colors.items():
        subset = data[data['Computed_Class'] == color]
        ax.scatter(subset['R'], subset['G'], subset['B'], c=color_code, label=color, alpha=0.6)

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_title("3D Color Distribution of Mangosteen")
    ax.legend()
    plt.show()

plot_3d_color_distribution(data)

# **ประมวลผลภาพก่อน KNN**

# Process image (example)
img_path = '/content/drive/MyDrive/Project/Dataset/235.jpg'  # Replace with actual image path
img = cv2.imread(img_path)

# Convert to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize for visualization (optional)
img_resized = cv2.resize(img_rgb, (256, 256))

# Show image
plt.imshow(img_resized)
plt.title("Processed Image")
plt.show()

# Process pixels (example)
pixels = img_resized.reshape((-1, 3))  # Flatten the image for RGB values
r, g, b = pixels[0]  # Get RGB values of the first pixel as an example

# Classify the color of the pixel
color = classify_color(r, g, b)
print(f"The color of the pixel is classified as: {color}")

# *** KNN Model: Training and Evaluation ***
# Split data into train, validation, and test sets
train_x, temp_x, train_y, temp_y = train_test_split(x, y, test_size=0.3, random_state=42)
validation_x, test_x, validation_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

# Scale the data
scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)
validation_x_scaled = scaler.transform(validation_x)
test_x_scaled = scaler.transform(test_x)

# Train KNN model with the best K
knn = KNeighborsClassifier(n_neighbors=5)  # Using K=5 as an example
knn.fit(train_x_scaled, train_y)

# Test the model
predictions = knn.predict(test_x_scaled)
accuracy = accuracy_score(test_y, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Print Evaluation Results
print_evaluation_results(test_y, predictions, "Test")
