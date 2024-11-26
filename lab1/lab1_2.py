def step(x):
    return 1 if x > 0 else 0

def or_neuron(x1, x2):
    return step(x1 + x2 - 0.5)

def and_neuron(x1, x2):
    return step(x1 + x2 - 1.5)

def xor_perceptron(x1, x2):
    y1 = or_neuron(x1, x2)
    y2 = and_neuron(x1, x2)
    
    output = step(y1 - y2 - 0.5)
    return output


print(xor_perceptron(0, 0))
print(xor_perceptron(0, 1))
print(xor_perceptron(1, 0))
print(xor_perceptron(1, 1))
