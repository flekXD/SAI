def xor(x1, x2):
    return (x1 or x2) and not (x1 and x2)


print(xor(0, 0))  
print(xor(0, 1)) 
print(xor(1, 0))
print(xor(1, 1))
