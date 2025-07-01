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