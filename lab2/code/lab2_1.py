import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# Вхідний файл, який містить дані
input_file = 'income_data.txt'

# Завантаження даних
X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 25000

# Читання даних
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        label = data[-1]  # Останній елемент - мітка
        if label == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])  # Додати всі, крім мітки
            y.append(0)  # Мітка для класу '<=50K'
            count_class1 += 1
        elif label == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])  # Додати всі, крім мітки
            y.append(1)  # Мітка для класу '>50K'
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

# Перетворення рядкових даних на числові
label_encoder = []
X_encoded = np.empty(X.shape, dtype=object)  # Додано dtype=object для змішаних даних
for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i].astype(int)
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

# Визначення вхідних даних і міток
X = X_encoded[:, :-1].astype(int)
y = np.array(y)

# Створення SVМ-класифікатора
classifier = OneVsOneClassifier(LinearSVC(random_state=0))

# Розподіл даних на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Навчання класифікатора
classifier.fit(X_train, y_train)

# Прогнозування результату для тестових даних
y_test_pred = classifier.predict(X_test)

# Обчислення метрик якості
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

# Виведення результатів
print("Звіт про класифікацію:\n", classification_report(y_test, y_test_pred))
print(f"Акуратність: {accuracy:.2f}")
print(f"Точність: {precision:.2f}")
print(f"Повнота: {recall:.2f}")
print(f"F1-міра: {f1:.2f}")

# Передбачення результату для тестової точки даних
input_data = ['37', 'Private', '215646', 'HS-grad', '9', 'Never-married', 'Handlers-cleaners', 'Not-in-family', 'White', 'Male', '0', '0', '40', 'United-States']

# Кодування тестової точки даних
input_data_encoded = [-1] * len(input_data)
count = 0
for i, item in enumerate(input_data):
    if item.isdigit():
        input_data_encoded[i] = int(input_data[i])
    else:
        input_data_encoded[i] = label_encoder[count].transform([input_data[i]])[0]
        count += 1

input_data_encoded = np.array(input_data_encoded).reshape(1, -1)
input_data_encoded = input_data_encoded[:, :-1] 

# Використання класифікатора для кодованої точки даних та виведення результату
predicted_class = classifier.predict(input_data_encoded)
print("Передбачений клас для тестової точки:", label_encoder[-1].inverse_transform(predicted_class)[0])