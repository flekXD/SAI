import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline

# Генерація даних
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)  # перетворюємо X в колону
y = np.sin(X).flatten() + np.random.uniform(-0.5, 0.5, m)  # Генерація шуму

# Розбиття на навчальні та перевірочні дані
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def plot_learning_curve_pipeline(model, X_train, y_train, X_val, y_val, degree=2, title="Learning Curve"):
    train_errors, val_errors = [], []
    for m in range(1, len(X_train) + 1):
        X_train_subset, y_train_subset = X_train[:m], y_train[:m]

        # Створюємо Pipeline з перетворенням даних та лінійною регресією
        pipeline = Pipeline([
            ("poly_features", PolynomialFeatures(degree=degree)),
            ("lin_reg", LinearRegression())
        ])
        
        # Навчаємо модель
        pipeline.fit(X_train_subset, y_train_subset)

        # Оцінка помилок
        train_predictions = pipeline.predict(X_train_subset)
        val_predictions = pipeline.predict(X_val)

        # Обчислення помилок
        train_errors.append(mean_squared_error(y_train_subset, train_predictions))
        val_errors.append(mean_squared_error(y_val, val_predictions))

    # Побудова графіка
    plt.plot(range(1, len(X_train) + 1), train_errors, label="Training error")
    plt.plot(range(1, len(X_train) + 1), val_errors, label="Validation error")
    plt.title(title)
    plt.xlabel("Number of training examples")
    plt.ylabel("Error")
    plt.ylim(-0.5, 0.5)  # Обмеження по осі Y
    plt.legend()
    plt.show()

# Побудова кривих навчання для лінійної регресії
print("Learning Curve for Linear Regression")
plot_learning_curve_pipeline(LinearRegression(), X_train, y_train, X_val, y_val, degree=1, title="Learning Curve - Linear Regression")

# Побудова кривих навчання для поліноміальної регресії 2-го ступеня
print("Learning Curve for Polynomial Regression (degree=2)")
plot_learning_curve_pipeline(LinearRegression(), X_train, y_train, X_val, y_val, degree=2, title="Learning Curve - Polynomial Degree 2")

# Побудова кривих навчання для поліноміальної регресії 10-го ступеня
print("Learning Curve for Polynomial Regression (degree=10)")
plot_learning_curve_pipeline(LinearRegression(), X_train, y_train, X_val, y_val, degree=10, title="Learning Curve - Polynomial Degree 10")
