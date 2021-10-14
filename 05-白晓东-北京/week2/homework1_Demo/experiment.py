def calc(*numbers):
    sum = 0
    for n in numbers:
        sum = sum + n * n
    return sum

a = [1, 3, 4]
print(calc(*a))
