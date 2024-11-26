import numpy as np
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

input_file = 'income_data.txt'

X = []
y = []
count_class1 = 0
count_class2 = 0
max_datapoints = 10000

# Читання даних
with open(input_file, 'r') as f:
    for line in f.readlines():
        if count_class1 >= max_datapoints and count_class2 >= max_datapoints:
            break
        if '?' in line:
            continue
        data = line.strip().split(', ')
        label = data[-1]
        if label == '<=50K' and count_class1 < max_datapoints:
            X.append(data[:-1])
            y.append(0)
            count_class1 += 1
        elif label == '>50K' and count_class2 < max_datapoints:
            X.append(data[:-1])
            y.append(1)
            count_class2 += 1

# Перетворення на масив numpy
X = np.array(X)

label_encoder = []
X_encoded = np.empty(X.shape, dtype=object)
for i in range(X.shape[1]):
    if X[0, i].isdigit():
        X_encoded[:, i] = X[:, i].astype(int)
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(X[:, i])

try:
    X_encoded = X_encoded.astype(int)
except ValueError:
    print("Error: Non-numeric values present after encoding.")

X = X_encoded.astype(int)
y = np.array(y)

# Розподіл даних на тренувальні та тестові
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Створення і навчання класифікатора з поліноміальним ядром
classifier = SVC(kernel='poly', degree=3)  # Зменшення ступеня до 3
classifier.fit(X_train, y_train)

# Прогнозування результату для тестових даних
y_test_pred = classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("\n--- Поліноміальне ядро SVM ---")
print("Звіт про класифікацію:\n", classification_report(y_test, y_test_pred))
print(f"Акуратність: {accuracy:.2f}")
print(f"Точність: {precision:.2f}")
print(f"Повнота: {recall:.2f}")
print(f"F1-міра: {f1:.2f}")
