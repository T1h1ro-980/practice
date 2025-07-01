import torch

# 2.1 Простые вычисления с градиентами
# Создайте тензоры x, y, z с requires_grad=True
# Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
# Найдите градиенты по всем переменным
# Проверьте результат аналитически

x = torch.rand(2, 5, requires_grad = True)
y = torch.rand(2, 5, requires_grad = True)
z = torch.rand(2, 5, requires_grad = True)
f = x**2 + y**2 + z**2 + 2*x*y*z
f.sum().backward()

print(f"x: \n{x}\ny: \n{y}\nz: \n{z}\n")
print(f"torch df/dx:\n {x.grad}")
print(f"torch df/dy:\n {y.grad}")  
print(f"torch df/dz:\n {z.grad}\n")  

"""
Для проверки возьмем частные производные по каждой переменной (x, y, z)

df/dx = 2x + 2yz
df/dy = 2y + 2xz
df/dz = 2z + 2xy

(значения переменных берем из 0 индекса (1 точка))
df/dx = 2 * 0.3961 + 2 * 0.3992 * 0.2362 = 0,98078208 (0.9807)
df/dy = 2 * 0.3992 + 2 * 0.3961 * 0.2362 = 0,98551764 (0.9856)
df/dz = 2 * 0.2362 + 2 * 0.3961 * 0.3992 = 0,78864624 (0.7887)
Сравнив аналитические результаты с результатами работы torch, можно понять что
все значения соотвуетствуют друг другу, а значит проверка успешная
"""

# 2.2 Градиент функции потерь
# Реализуйте функцию MSE (Mean Squared Error):
# MSE = (1/n) * Σ(y_pred - y_true)^2
# где y_pred = w * x + b (линейная функция)
# Найдите градиенты по w и b

x = torch.tensor([1, 2, 3, 4, 5, 6], dtype = torch.float32, requires_grad=True)
w = torch.tensor(0.5, dtype = torch.float32, requires_grad = True)
b = torch.tensor(1.0, dtype = torch.float32, requires_grad = True)

y_pred = w * x + b
y_true = torch.tensor([1.5, 2, 2.5, 3, 3.5, 4], dtype = torch.float32)

MSE = ((y_pred - y_true) ** 2).mean()

MSE.backward()

print(f"df/dw: {w.grad}") # Градиенты равны 0, т.к. MSE = 0, потому что w и b "идеально подобранны"
print(f"df/db: {b.grad}")

# 2.3 Цепное правило
# Реализуйте составную функцию: f(x) = sin(x^2 + 1)
# Найдите градиент df/dx
# Проверьте результат с помощью torch.autograd.grad

x = torch.rand((1, 15), dtype = torch.float32, requires_grad=True)
f = torch.sin(x**2 + 1)

f.sum().backward(retain_graph=True)
print(f"Градиент df/dx (backward()):\n {x.grad}")

autograd = torch.autograd.grad(outputs = f.sum(), inputs = x)
print(f"Градиент df/dx (autograd):\n {autograd[0]}")

# Вывод:
# x: 
# tensor([[0.4011, 0.2655, 0.3079, 0.3827, 0.7372],
#         [0.0573, 0.0692, 0.6031, 0.2869, 0.7802]], requires_grad=True)
# y: 
# tensor([[0.0011, 0.9110, 0.6575, 0.8671, 0.7798],
#         [0.6472, 0.2326, 0.2197, 0.2097, 0.4885]], requires_grad=True)
# z: 
# tensor([[0.2847, 0.3494, 0.0139, 0.7128, 0.1098],
#         [0.3785, 0.2805, 0.5899, 0.8648, 0.9129]], requires_grad=True)

# torch df/dx:
#  tensor([[0.8029, 1.1676, 0.6342, 2.0015, 1.6456],
#         [0.6045, 0.2688, 1.4654, 0.9365, 2.4524]])
# torch df/dy:
#  tensor([[0.2307, 2.0076, 1.3236, 2.2798, 1.7216],
#         [1.3377, 0.5040, 1.1509, 0.9156, 2.4015]])
# torch df/dz:
#  tensor([[0.5703, 1.1826, 0.4328, 2.0892, 1.3694],
#         [0.8312, 0.5932, 1.4447, 1.8499, 2.5881]])

# df/dw: 0.0
# df/db: 0.0
# Градиент df/dx (backward()):
#  tensor([[ 0.3240,  0.2978,  0.2285, -0.5687,  0.2446, -0.1724,  0.0791,  0.1874,
#           0.1703,  0.2739, -0.3576, -0.6392,  0.3241,  0.1785,  0.2808]])
# Градиент df/dx (autograd):
#  tensor([[ 0.3240,  0.2978,  0.2285, -0.5687,  0.2446, -0.1724,  0.0791,  0.1874,
#           0.1703,  0.2739, -0.3576, -0.6392,  0.3241,  0.1785,  0.2808]])
