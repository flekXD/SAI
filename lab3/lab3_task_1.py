import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

temperature = ctrl.Antecedent(np.arange(0, 101, 1), 'temperature')  # Температура [0-100 °C]
pressure = ctrl.Antecedent(np.arange(0, 101, 1), 'pressure')        # Напір [0-100 %]

hot_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'hot_valve')     # Кут крана гарячої води
cold_valve = ctrl.Consequent(np.arange(-90, 91, 1), 'cold_valve')   # Кут крана холодної води

# Функції належності для температури
temperature['cold'] = fuzz.trimf(temperature.universe, [0, 0, 50])
temperature['cool'] = fuzz.trimf(temperature.universe, [0, 50, 75])
temperature['warm'] = fuzz.trimf(temperature.universe, [50, 75, 100])
temperature['hot'] = fuzz.trimf(temperature.universe, [75, 100, 100])

# Функції належності для напору
pressure['low'] = fuzz.trimf(pressure.universe, [0, 0, 50])
pressure['medium'] = fuzz.trimf(pressure.universe, [25, 50, 75])
pressure['high'] = fuzz.trimf(pressure.universe, [50, 100, 100])

# Функції належності для кутів
hot_valve['large_left'] = fuzz.trimf(hot_valve.universe, [-90, -90, -45])
hot_valve['medium_left'] = fuzz.trimf(hot_valve.universe, [-60, -30, 0])
hot_valve['small_left'] = fuzz.trimf(hot_valve.universe, [-30, 0, 30])
hot_valve['no_change'] = fuzz.trimf(hot_valve.universe, [-15, 0, 15])
hot_valve['small_right'] = fuzz.trimf(hot_valve.universe, [0, 30, 60])
hot_valve['medium_right'] = fuzz.trimf(hot_valve.universe, [30, 60, 90])
hot_valve['large_right'] = fuzz.trimf(hot_valve.universe, [45, 90, 90])

cold_valve['large_left'] = fuzz.trimf(cold_valve.universe, [-90, -90, -45])
cold_valve['medium_left'] = fuzz.trimf(cold_valve.universe, [-60, -30, 0])
cold_valve['small_left'] = fuzz.trimf(cold_valve.universe, [-30, 0, 30])
cold_valve['no_change'] = fuzz.trimf(cold_valve.universe, [-15, 0, 15])
cold_valve['small_right'] = fuzz.trimf(cold_valve.universe, [0, 30, 60])
cold_valve['medium_right'] = fuzz.trimf(cold_valve.universe, [30, 60, 90])
cold_valve['large_right'] = fuzz.trimf(cold_valve.universe, [45, 90, 90])

# Правила нечіткої логіки
rules = [
    ctrl.Rule(temperature['hot'] & pressure['high'], 
              (hot_valve['medium_left'], cold_valve['medium_right'])),
    ctrl.Rule(temperature['hot'] & pressure['medium'], 
              cold_valve['medium_right']),
    ctrl.Rule(temperature['warm'] & pressure['high'], 
              hot_valve['small_left']),
    ctrl.Rule(temperature['warm'] & pressure['low'], 
              (hot_valve['small_right'], cold_valve['small_right'])),
    ctrl.Rule(temperature['cool'] & pressure['medium'], 
              (hot_valve['no_change'], cold_valve['no_change'])),
    ctrl.Rule(temperature['cool'] & pressure['high'], 
              (hot_valve['medium_right'], cold_valve['medium_left'])),
    ctrl.Rule(temperature['cool'] & pressure['low'], 
              (hot_valve['medium_right'], cold_valve['small_left'])),
    ctrl.Rule(temperature['cold'] & pressure['low'], 
              hot_valve['large_right']),
    ctrl.Rule(temperature['cold'] & pressure['high'], 
              (hot_valve['medium_left'], cold_valve['medium_right'])),
    ctrl.Rule(temperature['warm'] & pressure['high'], 
              (hot_valve['small_left'], cold_valve['small_left'])),
    ctrl.Rule(temperature['warm'] & pressure['low'], 
              (hot_valve['small_right'], cold_valve['small_right']))
]

# Система керування
control_system = ctrl.ControlSystem(rules)
mixer = ctrl.ControlSystemSimulation(control_system)

# Візуалізація
def visualize_memberships():
    temperature.view()
    pressure.view()
    hot_valve.view()
    cold_valve.view()

def simulate_and_visualize(temp, pres):
    mixer.input['temperature'] = temp
    mixer.input['pressure'] = pres
    mixer.compute()
    hot_valve.view(sim=mixer)
    cold_valve.view(sim=mixer)

visualize_memberships()

simulate_and_visualize(70, 80)

plt.show()