import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_training_history(history):
    """Визуализирует историю обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history['train_losses'], label='Train Loss')
    ax1.plot(history['test_losses'], label='Test Loss')
    ax1.set_title('Loss')
    ax1.legend()
    
    ax2.plot(history['train_accs'], label='Train Acc')
    ax2.plot(history['test_accs'], label='Test Acc')
    ax2.set_title('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()


def count_parameters(model):
    """Подсчитывает количество параметров модели"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    """Сохраняет модель"""
    torch.save(model.state_dict(), path)


def load_model(model, path):
    """Загружает модель"""
    model.load_state_dict(torch.load(path))
    return model


def compare_models(fc_history, cnn_history, residual_history):
    """Сравнивает результаты полносвязной и сверточной сетей"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(fc_history['test_accs'], label='FC Network', marker='o')
    ax1.plot(cnn_history['test_accs'], label='CNN', marker='s')
    ax1.plot(residual_history['test_accs'], label='Residual CNN', marker='*')
    ax1.set_title('Test Accuracy Comparison')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(fc_history['test_losses'], label='FC Network', marker='o')
    ax2.plot(cnn_history['test_losses'], label='CNN', marker='s')
    ax2.plot(residual_history['test_losses'], label='Residual CNN', marker='*')
    ax2.set_title('Test Loss Comparison')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show() 

def plot_confusion_matrix(model, data_loader, class_names, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

import matplotlib.pyplot as plt

def plot_gradient_flow(named_parameters):
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, param in named_parameters:
        if param.requires_grad and param.grad is not None:
            layers.append(name)
            ave_grads.append(param.grad.abs().mean().item())
            max_grads.append(param.grad.abs().max().item())
    
    plt.figure(figsize=(10, 5))
    plt.plot(ave_grads, label='mean gradient')
    plt.plot(max_grads, label='max gradient')
    plt.hlines(0, 0, len(ave_grads)-1, linestyle='dashed', color='black')
    plt.xticks(ticks=range(len(layers)), labels=layers, rotation=90)
    plt.xlabel("Layers")
    plt.ylabel("Gradient")
    plt.title("Gradient flow")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import torch

def plot_feature_maps(model, image_tensor, layer_index=0, num_features=6, device='cpu'):
    """
    Визуализирует карты признаков (feature maps) указанного сверточного слоя.

    Параметры:
    - model: torch.nn.Module — обученная модель
    - image_tensor: torch.Tensor — входное изображение размером [1, C, H, W]
    - layer_index: int — индекс слоя в model.layers, с которого взять feature map
    - num_features: int — сколько карт признаков показать
    - device: str — 'cpu' или 'cuda'
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    model = model.to(device)

    # Пропустить изображение через модель до нужного слоя
    x = image_tensor
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i == layer_index:
                feature_maps = x
                break
        else:
            raise ValueError(f"Слой с индексом {layer_index} не найден в model.layers")

    # Визуализация
    num_features = min(num_features, feature_maps.shape[1])
    plt.figure(figsize=(15, 5))
    for i in range(num_features):
        plt.subplot(1, num_features, i + 1)
        plt.imshow(feature_maps[0, i].cpu(), cmap='viridis')
        plt.title(f"Feature {i}")
        plt.axis('off')
    plt.suptitle(f"Feature maps at layer {layer_index}", fontsize=14)
    plt.tight_layout()
    plt.show()
