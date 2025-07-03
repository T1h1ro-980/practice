from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset

# 2.1 Кастомный Dataset класс
# Создайте кастомный класс датасета для работы с CSV файлами:
# - Загрузка данных из файла
# - Предобработка (нормализация, кодирование категорий)
# - Поддержка различных форматов данных (категориальные, числовые, бинарные и т.д.)

class CustomDataset(Dataset):
    def __init__(self, path, target: str):
        df = pd.read_csv(path)
        df_y = df[target]
        df_X = df.drop(columns=[target])
        
        cat_cols = df_X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = df_X.select_dtypes(include=[np.number]).columns.tolist()
        
        
        if cat_cols:
            encoder = OneHotEncoder(sparse_output = False)
            X_cat = encoder.fit_transform(df_X[cat_cols])
        else:
            X_cat = np.empty((len(df), 0))

        if num_cols:
            scaler = MinMaxScaler()
            X_num = scaler.fit_transform(df_X[num_cols])
        else:
            X_num = np.empty((len(df), 0))
        
        X_processed = np.hstack([X_num, X_cat])
        
        self.X = torch.tensor(X_processed, dtype=torch.float32)
        self.y = torch.tensor(df_y.values, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 2.2 Эксперименты с различными датасетами 
# Найдите csv датасеты для регрессии и бинарной классификации и, 
# применяя наработки из предыдущей части задания, обучите линейную 
# и логистическую регрессию

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import log_epoch

class LinearRegression(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.linear(x)

if __name__ == '__main__':
    
    # Создаём датасет и даталоадер
    dataset = CustomDataset("LinRegDataset.csv", "Y")
    X, y = dataset.X, dataset.y
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')
    
    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=4)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель
    epochs = 300

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





#------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_classification_data, accuracy, log_epoch, ClassificationDataset
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
    dataset = CustomDataset("LinRegDataset.csv", "Y")
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

