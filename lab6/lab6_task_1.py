import numpy as np

# Функція для розрахунку ймовірності для кожної таблиці (Outlook, Humidity, Wind)
def table_operations(smt, table, row_index):
    total_sum = np.sum(table)
    
    # Сума чисел в рядку
    row_sum = np.sum(table[row_index])
    
    # Сума чисел в стовпці для Yes та No
    column_Y = np.sum(table[:, 0])  # Сума стовпця "Yes"
    column_N = np.sum(table[:, 1])  # Сума стовпця "No"
    
    # Обчислення ймовірності для Yes та No
    res_Y = (smt / column_Y) * (column_Y / total_sum) / (row_sum / total_sum)
    res_N = (smt / column_N) * (column_N / total_sum) / (row_sum / total_sum)
    
    return res_Y, res_N

# Функція для розрахунку ймовірностей для матчу
def calculate_probability_for_match(outlook, humidity, wind):
    # Таблиці ймовірностей для Outlook, Humidity, Wind
    outlook_table = np.array([
        [3, 2],  # Sunny (Yes, No)
        [4, 0],  # Overcast (Yes, No)
        [3, 2]   # Rainy (Yes, No)
    ])

    humidity_table = np.array([
        [3, 4],  # High (Yes, No)
        [6, 1]   # Normal (Yes, No)
    ])

    wind_table = np.array([
        [6, 2],  # Strong (Yes, No)
        [3, 3]   # Weak (Yes, No)
    ])

    # Задано умови для Outlook, Humidity, Wind
    if outlook == 'Sunny':
        outlook_row_index = 0
    elif outlook == 'Overcast':
        outlook_row_index = 1
    elif outlook == 'Rainy':
        outlook_row_index = 2
    else:
        raise ValueError("Invalid outlook condition")

    if humidity == 'High':
        humidity_row_index = 0
    elif humidity == 'Normal':
        humidity_row_index = 1
    else:
        raise ValueError("Invalid humidity condition")

    if wind == 'Strong':
        wind_row_index = 0
    elif wind == 'Weak':
        wind_row_index = 1
    else:
        raise ValueError("Invalid wind condition")

    # Розрахунок ймовірностей для Outlook, Humidity, Wind
    outlook_Y, outlook_N = table_operations(3, outlook_table, outlook_row_index)  # Потрібно передати правильне значення smt
    humidity_Y, humidity_N = table_operations(3, humidity_table, humidity_row_index)  # Аналогічно
    wind_Y, wind_N = table_operations(6, wind_table, wind_row_index)  # І для вітру, передаємо правильне число smt

    # Розрахунок P(Yes) та P(No) для всіх умов
    P_yes_result = outlook_Y * humidity_Y * wind_Y * 9/14  # P(Yes) (наприклад, для конкретного загального числа)
    P_no_result = outlook_N * humidity_N * wind_N * 5/14  # P(No)

    P_yes= P_yes_result/(P_yes_result+P_no_result)
    P_no = P_no_result/(P_yes_result+P_no_result)
    return P_yes, P_no

# Задано умови
outlook = 'Overcast'
humidity = 'High'
wind = 'Strong'

# Розрахунок ймовірностей для Yes і No
P_yes, P_no = calculate_probability_for_match(outlook, humidity, wind)


# Виведення результатів
print(f"P(Yes|Outlook={outlook}, Humidity={humidity}, Wind={wind}) = {P_yes:.6f}")
print(f"P(No|Outlook={outlook}, Humidity={humidity}, Wind={wind}) = {P_no:.6f}")
