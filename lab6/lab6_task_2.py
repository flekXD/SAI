import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Завантаження даних
url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/refs/heads/master/data/renfe_small.csv"
df = pd.read_csv(url)

# Видаляємо рядки, де ціна відсутня або не є числом
df = df.dropna(subset=['price'])
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df = df.dropna(subset=['price'])

# Переводимо ціни в категорії (наприклад, дешево і дорого)
df['price_category'] = df['price'].apply(lambda x: 1 if x > 100 else 0)  # Створення стовпця price_category

# Вибір ознак для моделювання
features = ['origin', 'destination', 'train_type', 'train_class']  # Використовуємо ці стовпці для моделювання
df_dummies = pd.get_dummies(df[features], drop_first=True)  # Перетворюємо категоріальні змінні в числові

# Додаємо стовпець price_category назад в df_dummies
df_dummies['price_category'] = df['price_category'].astype(int)

# Розділяємо дані на тренувальні та тестові
X = df_dummies.drop('price_category', axis=1)
y = df_dummies['price_category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Створення і тренування моделі наївного баєсівського класифікатора
model = GaussianNB()
model.fit(X_train, y_train)

# Оцінка моделі
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Фільтрація даних для цін менше 25 євро
low_price_count = df[df['price'] < 25].shape[0]

# Загальна кількість записів
total_count = df.shape[0]

# Ймовірність отримати ціну менше 25 євро
probability = low_price_count / total_count

print(f"Ймовірність отримати ціну менше 25 євро: {probability:.2%}")


print(f"Точність моделі: {accuracy:.2f}")
