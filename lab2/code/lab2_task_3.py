import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

# КРОК 2: Побудова та оцінка моделей
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Оцінка моделей
results = []
names = []
print("Результати оцінки моделей (точність):")
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")

# Порівняння алгоритмів на графіку
plt.boxplot(results, labels=names)
plt.title('Порівняння алгоритмів')
plt.xlabel('Модель')
plt.ylabel('Точність')
plt.show()

# КРОК 6: Створення прогнозу на тестовій вибірці
model = SVC(gamma='auto')
model.fit(X_train, y_train)  # Навчаємо модель
predictions = model.predict(X_validation)  # Прогноз на тестовій вибірці

# КРОК 7: Оцінка якості моделі
print("\nТочність моделі на тестовій вибірці:", accuracy_score(y_validation, predictions))
print("\nМатриця помилок:")
print(confusion_matrix(y_validation, predictions))
print("\nЗвіт про класифікацію:")
print(classification_report(y_validation, predictions))

# КРОК 8: Передбачення для нових даних
X_new = np.array([[5, 2.9, 1, 0.2]])
print("\nФорма масиву X_new:", X_new.shape)

new_prediction = model.predict(X_new)
print("\nПрогнозований клас для нових даних:", new_prediction[0])
print("Спрогнозований сорт ірису:", iris.target_names[new_prediction[0]])
