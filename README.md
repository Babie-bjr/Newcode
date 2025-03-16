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
warnings.filterwarnings("ignore")

drive.mount('/content/drive')

# โหลดข้อมูลจากไฟล์ CSV
file_path = "/content/drive/MyDrive/Dataset /(2)New_MangosteenData.csv"
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

def plot_color_distribution_with_prediction(data, sample_r, sample_g, sample_b, predicted_class):
    """Plots the color distribution with the prediction for a given sample."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = {'Red': 'r', 'Green': 'g', 'Black': 'k'}

    for color, color_code in colors.items():
        subset = data[data['Computed_Class'] == color]
        ax.scatter(subset['R'], subset['G'], subset['B'], c=color_code, label=color, alpha=0.6)

    ax.set_xlabel("Red")
    ax.set_ylabel("Green")
    ax.set_zlabel("Blue")
    ax.set_title(f"3D Color Distribution with Prediction: {predicted_class}")
    ax.scatter(sample_r, sample_g, sample_b, c='yellow', marker='*', s=200, label='Sample')
    ax.legend()
    plt.show()


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
# แปลงคลาสเป็นตัวเลข รวม "Unknown" เป็น "Red"
color_mapping = {'Red': 10, 'Black': 20, 'Green': 30, 'Unknown': 10}  # แมป "Unknown" กับ "Red" (10)
data['Class'] = data['Computed_Class'].map(color_mapping)

# แบ่งข้อมูลสำหรับ Train/Test **โดยใช้ StratifiedKFold**
x = data[['R', 'G', 'B']]
y = data['Class']

# กำหนด index ของข้อมูล Train, Validation, Test เอง
train_index = data.index[:210] # ตัวอย่าง: ใช้ 800 แถวแรกเป็น Train data
validation_index = data.index[210:255] # ตัวอย่าง: ใช้ 200 แถวถัดไปเป็น Validation data
test_index = data.index[255:] # ตัวอย่าง: ใช้ส่วนที่เหลือเป็น Test data

# แบ่งข้อมูลตาม index ที่กำหนด
train_x, train_y = x.loc[train_index], y.loc[train_index]
validation_x, validation_y = x.loc[validation_index], y.loc[validation_index]
test_x, test_y = x.loc[test_index], y.loc[test_index]

# Scale the data
scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x) # Now train_x is defined before use
validation_x_scaled = scaler.transform(validation_x)
test_x_scaled = scaler.transform(test_x)

# ค้นหาค่า K ที่เหมาะสม **ปรับปรุงส่วนนี้**
k_list = list(range(1, 51))  # ทดสอบค่า K ตั้งแต่ 1 ถึง 50
cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, train_x_scaled, train_y, cv=5, scoring='accuracy')  # ใช้ 5-fold cross-validation
    cv_scores.append(scores.mean())

    # ค้นหาค่า K ที่เหมาะสม **ปรับปรุงส่วนนี้**
param_grid = {'n_neighbors': list(range(1, 51)), 'weights': ['uniform', 'distance']}
grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid.fit(train_x_scaled, train_y)

# ค้นหาค่า K ที่เหมาะสม
k_list = list(range(1, 51))
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), train_x_scaled, train_y, cv=10, scoring='accuracy').mean() for k in k_list]

# หาค่า K ที่ดีที่สุด
best_k = k_list[np.argmax(cv_scores)]
print(f"Best K value: {best_k}")

# Plot the accuracy scores for different K values **เพิ่มส่วนนี้**
plt.plot(k_list, cv_scores)
plt.xlabel("K value")
plt.ylabel("Cross-validation accuracy")
plt.title("KNN accuracy vs. K value")
plt.show()

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
x_scaled = scaler.fit_transform(x) # ปรับสเกลข้อมูลทั้งหมด

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

# Split data into train, validation, and test sets
train_x, temp_x, train_y, temp_y = train_test_split(x, y, test_size=0.3, random_state=42)
validation_x, test_x, validation_y, test_y = train_test_split(temp_x, temp_y, test_size=0.5, random_state=42)

# Scale the data
scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)
validation_x_scaled = scaler.transform(validation_x)
test_x_scaled = scaler.transform(test_x)

# แบ่งข้อมูลสำหรับ Train/Test
x = data[['R', 'G', 'B']]
y = data['Class']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

# ใช้ MinMaxScaler
scaler = MinMaxScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# ค้นหาค่า K ที่เหมาะสม
k_list = list(range(1, 21))
cv_scores = [cross_val_score(KNeighborsClassifier(n_neighbors=k), train_x_scaled, train_y, cv=10, scoring='accuracy').mean() for k in k_list]

# Train โมเดล KNN โดยใช้ best_k
knn = KNeighborsClassifier(n_neighbors=best_k)  # ใช้ best_k ที่คำนวณได้
knn.fit(train_x_scaled, train_y)

# ทดสอบโมเดล
predictions = knn.predict(test_x_scaled)
accuracy = accuracy_score(test_y, predictions)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# สร้าง DataFrame สำหรับผลการทำนาย
prediction_df = pd.DataFrame({'Image Number': test_x.index, # <--- Insert this line here
                               'R': test_x['R'],
                               'G': test_x['G'],
                               'B': test_x['B'],
                               'Actual_Class': test_y,
                               'Predicted_Class': predictions})
prediction_df = prediction_df.sort_values(by=['Image Number'])

# แปลงค่าตัวเลขกลับเป็นชื่อสี (update this part)
prediction_df['Actual_Color'] = prediction_df['Actual_Class'] #.map({v: k for k, v in color_mapping.items()})
prediction_df['Predicted_Color'] = prediction_df['Predicted_Class'] #.map({v: k for k, v in color_mapping.items()})

# บันทึกผลการทำนายลงในไฟล์ CSV
file_path_to_save = "/content/drive/MyDrive/Dataset /directory/predictions2.csv"  # เปลี่ยนเป็น path ที่คุณต้องการ
prediction_df.to_csv(file_path_to_save, index=False)
print(f"Predictions saved to {file_path_to_save}")

# Evaluate the model
test_accuracy, test_predictions, test_true, test_features = evaluate_model(knn, test_x_scaled, test_y, dataset_name="Test")
validation_accuracy, validation_predictions, validation_true, validation_features = evaluate_model(knn, validation_x_scaled, validation_y, dataset_name="Validation")

# Print classification reports and confusion matrices
print_evaluation_results(test_true, test_predictions, "Test")
print_evaluation_results(validation_true, validation_predictions, "Validation")

# Example usage of plot_color_distribution_with_prediction:
sample_r, sample_g, sample_b = 38, 54, 87 # ปรับตำแหน่งตัวอย่าง
sample_features = scaler.transform([[sample_r, sample_g, sample_b]])
predicted_class = knn.predict(sample_features)[0]
predicted_color = [k for k, v in color_mapping.items() if v == predicted_class][0]
plot_color_distribution_with_prediction(data, sample_r, sample_g, sample_b, predicted_color)

print(f"มังคุดที่มีค่า R={sample_r}, G={sample_g}, B={sample_b} ถูกจัดให้อยู่ในคลาส: {predicted_class} ({predicted_color})")

# แสดง Classification Report
# รับคลาสที่ไม่ซ้ำกันใน test_y
# แก้ไข: ใช้ unique() เพื่อรับคลาสที่ปรากฏจริงใน test_y
target_names_unique = sorted(test_y.unique())
# แก้ไข: ใช้ target_names_unique เป็น labels และแปลงเป็นชื่อสีสำหรับ target_names
target_names_unique_names = [k for k,v in color_mapping.items() if v in target_names_unique] #แปลงค่าตัวเลขกลับเป็นชื่อสี
print(classification_report(test_y, predictions, labels=target_names_unique, target_names=target_names_unique, zero_division=0))
