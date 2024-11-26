import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

# Завантажуємо дані
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')

# Розділяємо на вхідні дані та мітки
X, y = data[:, :-1], data[:, -1]

# Розбиваємо дані на три класи
class_0 = np.array(X[y == 0])
class_1 = np.array(X[y == 1])
class_2 = np.array(X[y == 2])

# Розбиваємо дані на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=5)

# Сітка значень параметрів для пошуку
parameter_grid = [
    {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
    {'n_estimators': [25, 50, 100, 250], 'max_depth': [4]}
]

# Метричні характеристики, які будемо використовувати для оцінки
metrics = ['precision_weighted', 'recall_weighted']

# Для кожної метрики виконуємо сітковий пошук
for metric in metrics:
    print(f"\n##### Searching optimal parameters for {metric}")
    
    # Створення класифікатора ExtraTreesClassifier
    classifier = GridSearchCV(
        ExtraTreesClassifier(random_state=0),
        parameter_grid,
        cv=5,
        scoring=metric
    )
    
    # Навчаємо класифікатор
    classifier.fit(X_train, y_train)
    
    # Виводимо результати для кожної комбінації параметрів
    print("\nGrid scores for the parameter grid:")
    for params, avg_score in zip(classifier.cv_results_['params'], classifier.cv_results_['mean_test_score']):
        print(f"{params} --> {round(avg_score, 3)}")
    
    # Виводимо найкращі параметри
    print("\nBest parameters:", classifier.best_params_)
    
    # Прогнозування на тестовому наборі
    y_pred = classifier.predict(X_test)
    
    # Звіт про продуктивність
    print("\nPerformance report:\n")
    print(classification_report(y_test, y_pred))
