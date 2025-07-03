# 3.1 Исследование гиперпараметров
# Проведите эксперименты с различными:
# - Скоростями обучения (learning rate)
# - Размерами батчей
# - Оптимизаторами (SGD, Adam, RMSprop)
# Визуализируйте результаты в виде графиков или таблиц


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset
import matplotlib.pyplot as plt

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def learning(learning_rate, batch_size, optim_type):
# Генерируем данные
    X, y = make_regression_data(n=200)
    
    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    if optim_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optim_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # Обучаем модель
    epochs = 100

    ALPHA = 0.01 # константа отвечающая за степень регуляризации
    reg_type = None # тип регуляризации ("l1", "l2", None)

    STOP_CONST = 0.0001 # константа для критерия остановки
    last_weight = 0 # хранит веса прошлой тиерации обучения

    history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            if reg_type == "l2": reg = (ALPHA * (model.linear.weight ** 2).sum()) # l2 регуляризация
            elif reg_type == "l1": reg = (ALPHA * (abs(model.linear.weight).sum())) # l1 регуляризация
            else: reg = 0
            loss = criterion(y_pred, batch_y) + reg # Добавляем к loss регуляризатор
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() 
        avg_loss = total_loss / (i + 1)
        history.append(avg_loss)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

        # использует евклидову норму для того чтобы сравнить тензор с числом правильно
        weight_diff = torch.norm(model.linear.weight - last_weight, p=2).item()

        # критерий остановки
        if weight_diff < STOP_CONST:
            break

        last_weight = model.linear.weight.clone().detach()
    return history

if __name__ == '__main__':
    results = {}

    params = [
        (0.01, 16, "SGD"),
        (0.001, 16, "Adam"),
        (0.01, 32, "RMSprop"),
        (0.01, 64, "SGD"),
        (0.005, 16, "Adam"),
    ]

    for lr, bs, opt in params:
        label = f"{opt} | lr={lr} | bs={bs}"
        print(f"\n==> Обучение: {label}")
        loss_history = learning(lr, bs, opt)
        results[label] = loss_history

    # Построим графики
    plt.figure(figsize=(12, 6))
    for label, loss_history in results.items():
        plt.plot(loss_history, label=label)
    
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Сравнение метрик обучения")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ==> Обучение: SGD | lr=0.01 | bs=16
# Размер датасета: 200
# Количество батчей: 13
# Epoch 10: loss=0.1446
# Epoch 20: loss=0.1007
# Epoch 30: loss=0.0726
# Epoch 40: loss=0.0538
# Epoch 50: loss=0.0401
# Epoch 60: loss=0.0309
# Epoch 70: loss=0.0236
# Epoch 80: loss=0.0195
# Epoch 90: loss=0.0166
# Epoch 100: loss=0.0146

# ==> Обучение: Adam | lr=0.001 | bs=16
# Размер датасета: 200
# Количество батчей: 13
# Epoch 10: loss=0.5087
# Epoch 20: loss=0.4029
# Epoch 30: loss=0.3267
# Epoch 40: loss=0.2935
# Epoch 50: loss=0.2697
# Epoch 60: loss=0.2524
# Epoch 70: loss=0.2291
# Epoch 80: loss=0.2138
# Epoch 90: loss=0.1955
# Epoch 100: loss=0.1840

# ==> Обучение: RMSprop | lr=0.01 | bs=32
# Размер датасета: 200
# Количество батчей: 7
# Epoch 10: loss=0.2230
# Epoch 20: loss=0.1390
# Epoch 30: loss=0.0685
# Epoch 40: loss=0.0345
# Epoch 50: loss=0.0176
# Epoch 60: loss=0.0110
# Epoch 70: loss=0.0091
# Epoch 80: loss=0.0093
# Epoch 90: loss=0.0095
# Epoch 100: loss=0.0091

# ==> Обучение: SGD | lr=0.01 | bs=64
# Размер датасета: 200
# Количество батчей: 4
# Epoch 10: loss=0.4004
# Epoch 20: loss=0.2044
# Epoch 30: loss=0.1437
# Epoch 40: loss=0.1207
# Epoch 50: loss=0.1491
# Epoch 60: loss=0.1199
# Epoch 70: loss=0.1243
# Epoch 80: loss=0.1002
# Epoch 90: loss=0.0885
# Epoch 100: loss=0.1037

# ==> Обучение: Adam | lr=0.005 | bs=16
# Размер датасета: 200
# Количество батчей: 13
# Epoch 10: loss=0.4183
# Epoch 20: loss=0.2773
# Epoch 30: loss=0.1702
# Epoch 40: loss=0.1041
# Epoch 50: loss=0.0580
# Epoch 60: loss=0.0340
# Epoch 70: loss=0.0210
# Epoch 80: loss=0.0155
# Epoch 90: loss=0.0127
# Epoch 100: loss=0.0116

# Визуализация метрик в README.md


# 3.2 Feature Engineering
# Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset, create_new_dataset
import matplotlib.pyplot as plt
from homework_datasets import CustomDataset

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)


def learning(learning_rate, batch_size, optim_type, dataset_path, target_name, in_features):
    dataset = CustomDataset(dataset_path, target_name)
    X, y = dataset.X, dataset.y
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=in_features)
    criterion = nn.MSELoss()
    if optim_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif optim_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optim_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)

    # Обучаем модель
    epochs = 100

    ALPHA = 0.01 # константа отвечающая за степень регуляризации
    reg_type = None # тип регуляризации ("l1", "l2", None)

    STOP_CONST = 0.0001 # константа для критерия остановки
    last_weight = 0 # хранит веса прошлой тиерации обучения

    history = []

    for epoch in range(1, epochs + 1):
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            if reg_type == "l2": reg = (ALPHA * (model.linear.weight ** 2).sum()) # l2 регуляризация
            elif reg_type == "l1": reg = (ALPHA * (abs(model.linear.weight).sum())) # l1 регуляризация
            else: reg = 0
            loss = criterion(y_pred, batch_y) + reg # Добавляем к loss регуляризатор
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() 
        avg_loss = total_loss / (i + 1)
        history.append(avg_loss)
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

        # использует евклидову норму для того чтобы сравнить тензор с числом правильно
        weight_diff = torch.norm(model.linear.weight - last_weight, p=2).item()

        # критерий остановки
        if weight_diff < STOP_CONST:
            break

        last_weight = model.linear.weight.clone().detach()
    return history

if __name__ == '__main__':
    results = {}
    create_new_dataset("LinRegDataset.csv")

    params = [
        (0.01, 16, "SGD", "LinRegDataset.csv", "Y", 4),
        (0.01, 16, "SGD", "LinRegDatasetEdited.csv", "Y", 9),
    ]

    for lr, bs, opt, dataset_path, target_name, in_features in params:
        label = f"{opt} | lr={lr} | bs={bs} | ds={dataset_path}"
        print(f"\n==> Обучение: {label}")
        loss_history = learning(lr, bs, opt, dataset_path, target_name, in_features)
        results[label] = loss_history

    # Построим графики
    plt.figure(figsize=(12, 6))
    for label, loss_history in results.items():
        plt.plot(loss_history, label=label)
    
    plt.xlabel("Эпоха")
    plt.ylabel("Loss")
    plt.title("Сравнение метрик обучения")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
