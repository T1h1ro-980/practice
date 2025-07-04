# 1.1 Сравнение моделей разной глубины
# Создайте и обучите модели с различным количеством слоев:
# - 1 слой (линейный классификатор)
# - 2 слоя (1 скрытый)
# - 3 слоя (2 скрытых)
# - 5 слоев (4 скрытых)
# - 7 слоев (6 скрытых)
# 
# Для каждого варианта:
# - Сравните точность на train и test
# - Визуализируйте кривые обучения
# - Проанализируйте время обучения

import torch
from datasets import get_mnist_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import plot_training_history, count_parameters
from time import time

configs_layers = [
    [
        {"type": "linear", "size": 10}
    ],
    [
        {"type": "linear", "size": 512}, 
        {"type": "relu"}, 
        {"type": "linear", "size": 10}
    ], # 2 слоя
    [
        {"type": "linear", "size": 512},  
        {"type": "relu"},
        {"type": "linear", "size": 256}, 
        {"type": "relu"},
        {"type": "linear", "size": 10}    
    ], # 3 слоя
    [
        {"type": "linear", "size": 512},  
        {"type": "relu"},
        {"type": "linear", "size": 256}, 
        {"type": "relu"},
        {"type": "linear", "size": 128}, 
        {"type": "relu"},
        {"type": "linear", "size": 64}, 
        {"type": "relu"},
        {"type": "linear", "size": 10}     
    ], # 5 слоев
    [
        {"type": "linear", "size": 512},  
        {"type": "relu"},
        {"type": "linear", "size": 256}, 
        {"type": "relu"},
        {"type": "linear", "size": 128}, 
        {"type": "relu"},
        {"type": "linear", "size": 64}, 
        {"type": "relu"},
        {"type": "linear", "size": 32},
        {"type": "relu"},
        {"type": "linear", "size": 16}, 
        {"type": "relu"},
        {"type": "linear", "size": 10}     
    ], # 5 слоев
        ]


for current_layers in configs_layers:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_mnist_loaders(batch_size=64)

    model = FullyConnectedModel(
        input_size=784,
        num_classes=10,
        layers = current_layers
    ).to(device)

    print(f"Model parameters: {count_parameters(model)}")

    start_time = time()
    history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
    end_time = time()
    print(f"Время обучения: {end_time-start_time}")

    plot_training_history(history) 


# Выводы в README.md

# 1.2 Анализ переобучения
# Исследуйте влияние глубины на переобучение:
# - Постройте графики train/test accuracy по эпохам
# - Определите оптимальную глубину для каждого датасета
# - Добавьте Dropout и BatchNorm, сравните результаты
# - Проанализируйте, когда начинается переобучение

import torch
from datasets import get_cifar_loaders
from models import FullyConnectedModel
from trainer import train_model
from utils import plot_training_history, count_parameters
from time import time

configs_layers = [
    [
        {"type": "linear", "size": 10}
    ],
    [
        {"type": "linear", "size": 512}, 
        {"type": "relu"}, 
        {"type": "linear", "size": 10}
    ], # 2 слоя
    [
        {"type": "linear", "size": 512},  
        {"type": "relu"},
        {"type": "linear", "size": 256}, 
        {"type": "relu"},
        {"type": "linear", "size": 10}    
    ], # 3 слоя
    [
        {"type": "linear", "size": 512},  
        {"type": "relu"},
        {"type": "linear", "size": 256}, 
        {"type": "relu"},
        {"type": "linear", "size": 128}, 
        {"type": "relu"},
        {"type": "linear", "size": 64}, 
        {"type": "relu"},
        {"type": "linear", "size": 10}     
    ], # 5 слоев
    [
        {"type": "linear", "size": 512},  
        {"type": "relu"},
        {"type": "linear", "size": 256}, 
        {"type": "relu"},
        {"type": "linear", "size": 128}, 
        {"type": "relu"},
        {"type": "linear", "size": 64}, 
        {"type": "relu"},
        {"type": "linear", "size": 32},
        {"type": "relu"},
        {"type": "linear", "size": 16}, 
        {"type": "relu"},
        {"type": "linear", "size": 10}     
    ], # 7 слоев
    [
        {"type": "linear", "size": 512}, 
        {"type": "batchnorm"},      # BatchNorm после линейного слоя
        {"type": "relu"}, 
        {"type": "dropout", "p": 0.5},  # Dropout с вероятностью 0.5
        {"type": "linear", "size": 10}
    ]
        ]


for current_layers in configs_layers:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = get_cifar_loaders(batch_size=64)

    model = FullyConnectedModel(
        input_size=3*32*32,
        num_classes=10,
        layers = current_layers
    ).to(device)

    print(f"Model parameters: {count_parameters(model)}")

    start_time = time()
    history = train_model(model, train_loader, test_loader, epochs=5, device=str(device))
    end_time = time()
    print(f"Время обучения: {end_time-start_time}")

    plot_training_history(history) 

# Выводы в README.md