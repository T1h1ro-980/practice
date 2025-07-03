# 1.1 Расширение линейной регрессии 
# Модифицируйте существующую линейную регрессию:
# - Добавьте L1 и L2 регуляризацию
# - Добавьте early stopping

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    # Генерируем данные
    X, y = make_regression_data(n=200)
    
    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель
    epochs = 100

    ALPHA = 0.01 # константа отвечающая за степень регуляризации
    reg_type = None # тип регуляризации ("l1", "l2", None)

    STOP_CONST = 0.0001 # константа для критерия остановки
    last_weight = 0 # хранит веса прошлой тиерации обучения

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
        if epoch % 1 == 0:
            log_epoch(epoch, avg_loss)

        # использует евклидову норму для того чтобы сравнить тензор с числом правильно
        weight_diff = torch.norm(model.linear.weight - last_weight, p=2).item()

        # критерий остановки
        if weight_diff < STOP_CONST:
            break

        last_weight = model.linear.weight.clone().detach()
    
    # Сохраняем модель
    torch.save(model.state_dict(), 'linreg_torch.pth')
    
    # Загружаем модель
    new_model = LinearRegression(in_features=1)
    new_model.load_state_dict(torch.load('linreg_torch.pth'))
    new_model.eval() 

# Вывод
# Размер датасета: 200
# Количество батчей: 7
# Epoch 1: loss=0.2972
# Epoch 2: loss=0.2315
# Epoch 3: loss=0.1919
# Epoch 4: loss=0.1678
# Epoch 5: loss=0.1481
# Epoch 6: loss=0.1194
# Epoch 7: loss=0.0997
# Epoch 8: loss=0.0905
# Epoch 9: loss=0.0786
# Epoch 10: loss=0.0657
# Epoch 11: loss=0.0566
# Epoch 12: loss=0.0495
# Epoch 13: loss=0.0438
# Epoch 14: loss=0.0429
# Epoch 15: loss=0.0342
# Epoch 16: loss=0.0285
# Epoch 17: loss=0.0277
# Epoch 18: loss=0.0252
# Epoch 19: loss=0.0213
# Epoch 20: loss=0.0211
# Epoch 21: loss=0.0189
# Epoch 22: loss=0.0168
# Epoch 23: loss=0.0157
# Epoch 24: loss=0.0142
# Epoch 25: loss=0.0138
# Epoch 26: loss=0.0140
# Epoch 27: loss=0.0128
# Epoch 28: loss=0.0118
# Epoch 29: loss=0.0115
# Epoch 30: loss=0.0114
# Epoch 31: loss=0.0110
# Epoch 32: loss=0.0110
# Epoch 33: loss=0.0108
# Epoch 34: loss=0.0109
# Epoch 35: loss=0.0112
# Epoch 36: loss=0.0099
# Epoch 37: loss=0.0124
# Epoch 38: loss=0.0106
# Epoch 39: loss=0.0104
# Epoch 40: loss=0.0107
# Epoch 41: loss=0.0104
# Epoch 42: loss=0.0112
# Epoch 43: loss=0.0109
# Epoch 44: loss=0.0109
# Epoch 45: loss=0.0099
# Epoch 46: loss=0.0093
# Epoch 47: loss=0.0102
# Epoch 48: loss=0.0119
# Epoch 49: loss=0.0107
# Epoch 50: loss=0.0101
# Epoch 51: loss=0.0104
# Epoch 52: loss=0.0107
# Epoch 53: loss=0.0108
# Epoch 54: loss=0.0100
# Epoch 55: loss=0.0104
# Epoch 56: loss=0.0104
# Epoch 57: loss=0.0097
# Epoch 58: loss=0.0102
# Epoch 59: loss=0.0103
# Epoch 60: loss=0.0102
# Epoch 61: loss=0.0103
# Epoch 62: loss=0.0106
# Epoch 63: loss=0.0104
# Epoch 64: loss=0.0099
# Epoch 65: loss=0.0097
# Epoch 66: loss=0.0097
# Epoch 67: loss=0.0098
# Epoch 68: loss=0.0106
# Epoch 69: loss=0.0097
# Epoch 70: loss=0.0102
# Epoch 71: loss=0.0107
# Epoch 72: loss=0.0107
# Epoch 73: loss=0.0097
# Epoch 74: loss=0.0107
# Epoch 75: loss=0.0098
# Epoch 76: loss=0.0107


# 1.2 Расширение логистической регрессии
# Модифицируйте существующую логистическую регрессию:
# - Добавьте поддержку многоклассовой классификации
# - Реализуйте метрики: precision, recall, F1-score, ROC-AUC
# - Добавьте визуализацию confusion matrix


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_classification_data, accuracy, log_epoch, ClassificationDataset, precision, recall, calculate_tp_tn_fp_fn, print_multiclass_confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    # Генерируем данные
    X, y = make_classification_data(source = "load_iris")
    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=4, num_classes = 3) # Меняем на кол-во классов и призаков
    criterion = nn.CrossEntropyLoss() # Меняем 
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        
        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            # Вычисляем accuracy
            y_pred = torch.argmax(logits, dim=1)
            acc = accuracy(y_pred, batch_y)
            total_loss += loss.item()
            total_acc += acc

        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)
        
        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc)
    

    # Сохраняем модель
    torch.save(model.state_dict(), 'logreg_torch.pth')
    
    # Загружаем модель
    new_model = LogisticRegression(in_features=4, num_classes=3)
    new_model.load_state_dict(torch.load('logreg_torch.pth'))
    new_model.eval() 
    plt.show()


# Вывод
# Размер датасета: 150
# Количество батчей: 5
# Epoch 10: loss=0.6299, acc=0.6460
# Epoch 20: loss=0.5459, acc=0.7477
# Epoch 30: loss=0.4171, acc=0.7426
# Epoch 40: loss=0.2583, acc=0.9568
# Epoch 50: loss=0.3560, acc=0.8381
# Epoch 60: loss=0.2211, acc=0.9375
# Epoch 70: loss=0.2322, acc=0.9290
# Epoch 80: loss=0.1926, acc=0.9568
# Epoch 90: loss=0.1939, acc=0.9568
# Epoch 100: loss=0.1850, acc=0.9443
