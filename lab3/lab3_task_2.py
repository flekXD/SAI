import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

temperature = ctrl.Antecedent(np.arange(0, 41, 1), 'temperature')  # Температура (°C)
temp_change = ctrl.Antecedent(np.arange(-5, 6, 1), 'temp_change')  # Швидкість зміни температури (°C/хв)

ac_regulator = ctrl.Consequent(np.arange(-90, 91, 1), 'ac_regulator')  # Кут регулятора кондиціонера

# Функції належності для температури
temperature['very_cold'] = fuzz.trimf(temperature.universe, [0, 0, 10])
temperature['cold'] = fuzz.trimf(temperature.universe, [5, 10, 20])
temperature['normal'] = fuzz.trimf(temperature.universe, [15, 20, 25])
temperature['warm'] = fuzz.trimf(temperature.universe, [20, 30, 35])
temperature['very_warm'] = fuzz.trimf(temperature.universe, [30, 40, 40])

# Функції належності для швидкості зміни температури
temp_change['negative'] = fuzz.trimf(temp_change.universe, [-5, -5, 0])
temp_change['zero'] = fuzz.trimf(temp_change.universe, [-1, 0, 1])
temp_change['positive'] = fuzz.trimf(temp_change.universe, [0, 5, 5])

# Функції належності для регулятора
ac_regulator['large_left'] = fuzz.trimf(ac_regulator.universe, [-90, -90, -45])
ac_regulator['small_left'] = fuzz.trimf(ac_regulator.universe, [-45, -15, 0])
ac_regulator['no_change'] = fuzz.trimf(ac_regulator.universe, [-15, 0, 15])
ac_regulator['small_right'] = fuzz.trimf(ac_regulator.universe, [0, 15, 45])
ac_regulator['large_right'] = fuzz.trimf(ac_regulator.universe, [45, 90, 90])

# Правила нечіткої логіки
rules = [
    ctrl.Rule(temperature['very_warm'] & temp_change['positive'], ac_regulator['large_left']),
    ctrl.Rule(temperature['very_warm'] & temp_change['negative'], ac_regulator['small_left']),
    ctrl.Rule(temperature['very_warm'] & temp_change['zero'], ac_regulator['large_left']),
    ctrl.Rule(temperature['warm'] & temp_change['positive'], ac_regulator['large_left']),
    ctrl.Rule(temperature['warm'] & temp_change['negative'], ac_regulator['no_change']),
    ctrl.Rule(temperature['warm'] & temp_change['zero'], ac_regulator['small_left']),
    ctrl.Rule(temperature['very_cold'] & temp_change['negative'], ac_regulator['large_right']),
    ctrl.Rule(temperature['very_cold'] & temp_change['positive'], ac_regulator['small_right']),
    ctrl.Rule(temperature['very_cold'] & temp_change['zero'], ac_regulator['large_right']),
    ctrl.Rule(temperature['cold'] & temp_change['negative'], ac_regulator['large_right']),
    ctrl.Rule(temperature['cold'] & temp_change['positive'], ac_regulator['no_change']),
    ctrl.Rule(temperature['cold'] & temp_change['zero'], ac_regulator['small_right']),
    ctrl.Rule(temperature['normal'] & temp_change['positive'], ac_regulator['small_left']),
    ctrl.Rule(temperature['normal'] & temp_change['negative'], ac_regulator['small_right']),
    ctrl.Rule(temperature['normal'] & temp_change['zero'], ac_regulator['no_change'])
]

# Система керування
control_system = ctrl.ControlSystem(rules)
ac_simulation = ctrl.ControlSystemSimulation(control_system)

# Візуалізація
def visualize_memberships():
    temperature.view()
    temp_change.view()
    ac_regulator.view()

def simulate_and_visualize(temp, temp_rate_change):
    ac_simulation.input['temperature'] = temp
    ac_simulation.input['temp_change'] = temp_rate_change
    ac_simulation.compute()
    ac_regulator.view(sim=ac_simulation)

# Візуалізація функцій належності
visualize_memberships()

# Тестування моделі з візуалізацією
simulate_and_visualize(28, 2)

plt.show()
