# 2.1 Сравнение моделей разной ширины
# Создайте модели с различной шириной слоев:
# - Узкие слои: [64, 32, 16]
# - Средние слои: [256, 128, 64]
# - Широкие слои: [1024, 512, 256]
# - Очень широкие слои: [2048, 1024, 512]
# 
# Для каждого варианта:
# - Поддерживайте одинаковую глубину (3 слоя)
# - Сравните точность и время обучения
# - Проанализируйте количество параметров

import torch
from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import plot_training_history, count_parameters
from time import time

configs_sizes = [
    [64, 32, 16],
    [256, 128, 64],
    [1024, 512, 256],
    [2048, 1024, 512]
                ]


for current_sizes in configs_sizes:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers = [
            {"type": "linear", "size": current_sizes[0]},
            {"type": "relu"},
            {"type": "linear", "size": current_sizes[1]},
            {"type": "relu"},
            {"type": "linear", "size": current_sizes[2]},
            {"type": "relu"},
            {"type": "linear", "size": 10}
        ]
    ).to(device)

    print(f"Model parameters: {count_parameters(model)}")

    start_time = time()
    history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
    end_time = time()
    print(f"Время обучения: {end_time-start_time}")

    plot_training_history(history) 

# Выводы в README.md

# 2.2 Оптимизация архитектуры 
# Найдите оптимальную архитектуру:
# - Используйте grid search для поиска лучшей комбинации
# - Попробуйте различные схемы изменения ширины (расширение, сужение, постоянная)
# - Визуализируйте результаты в виде heatmap

import torch
from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import plot_training_history, count_parameters
from time import time

import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Возможные размеры
sizes = [32, 64, 128, 256, 512]

# Схемы изменения ширины:
def expanding_sizes(sizes):
    return [a for a in itertools.product(sizes, repeat=3) if a[0] < a[1] < a[2]]

def contracting_sizes(sizes):
    return [a for a in itertools.product(sizes, repeat=3) if a[0] > a[1] > a[2]]

def constant_sizes(sizes):
    return [(a, a, a) for a in sizes]

expanding = expanding_sizes(sizes)
contracting = contracting_sizes(sizes)
constant = constant_sizes(sizes)

# Все варианты
all_configs = {
    "expanding": expanding,
    "contracting": contracting,
    "constant": constant
}

results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, test_loader = get_mnist_loaders(batch_size=64)

for scheme_name, configs in all_configs.items():
    print(f"Scheme: {scheme_name}")
    for config in configs:
        model = FullyConnectedModel(
            input_size=784,
            num_classes=10,
            layers = [
                {"type": "linear", "size": config[0]},
                {"type": "relu"},
                {"type": "linear", "size": config[1]},
                {"type": "relu"},
                {"type": "linear", "size": config[2]},
                {"type": "relu"},
                {"type": "linear", "size": 10}
            ]
        ).to(device)

        params = count_parameters(model)

        start_time = time()
        history = train_model(model, train_loader, test_loader, epochs=3, device=str(device))  # для ускорения 3 эпохи
        end_time = time()

        test_acc = history['test_accs'][-1]  # последний Test Acc
        train_acc = history['train_accs'][-1]
        train_time = end_time - start_time

        results.append({
            "scheme": scheme_name,
            "layer1": config[0],
            "layer2": config[1],
            "layer3": config[2],
            "params": params,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "time": train_time
        })

# Преобразуем в DataFrame
df = pd.DataFrame(results)

# Пример визуализации для схемы "expanding" — heatmap test accuracy (слой1 на оси X, слой3 на оси Y)
for scheme in all_configs.keys():
    df_sub = df[df['scheme'] == scheme]
    pivot_table = df_sub.pivot_table(values='test_acc', index='layer3', columns='layer1')
    plt.figure(figsize=(8,6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap='viridis')
    plt.title(f'Test Accuracy Heatmap ({scheme})')
    plt.xlabel('Layer 1 size')
    plt.ylabel('Layer 3 size')
    plt.show()
