import torch
import numpy as np
from torch.utils.data import Dataset
import pandas as pd

# Создайте новые признаки для улучшения модели:
# - Полиномиальные признаки
# - Взаимодействия между признаками
# - Статистические признаки (среднее, дисперсия)
# Сравните качество с базовой моделью
import pandas as pd
import numpy as np
from itertools import combinations

def create_new_dataset(old_path):
    df = pd.read_csv(old_path)
    
    num_df = df.select_dtypes(include=[np.number])
    
    new_features = pd.DataFrame(index=df.index)
    
    for col in num_df.columns:
        new_features[f"{col}_squared"] = num_df[col] ** 2
    
    for col1, col2 in combinations(num_df.columns, 2):
        new_features[f"{col1}_x_{col2}"] = num_df[col1] * num_df[col2]
    
    new_features["row_mean"] = num_df.mean(axis=1)
    new_features["row_std"] = num_df.std(axis=1)
    
    df_new = pd.concat([df, new_features], axis=1)
    df_new.to_csv("/home/egikor/ML/Pratic in git/HOMEWORK_2/LinRegDatasetEdited.csv", index=False)
    return df_new

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class ClassificationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_regression_data(n=100, noise=0.1, source='random'):
    if source == 'random':
        X = torch.rand(n, 1)
        w, b = 2.0, -1.0
        y = w * X + b + noise * torch.randn(n, 1)
        return X, y
    elif source == 'diabetes':
        from sklearn.datasets import load_diabetes
        data = load_diabetes()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.float32).unsqueeze(1)
        return X, y
    else:
        raise ValueError('Unknown source')

def make_classification_data(n=100, source='random'):
    if source == 'random':
        X = torch.rand(n, 2)
        w = torch.tensor([2.0, -3.0])
        b = 0.5
        logits = X @ w + b
        y = (logits > 0).float().unsqueeze(1)
        return X, y
    elif source == 'load_iris':
        from sklearn.datasets import load_iris # новый датасет с 4 классами
        data = load_iris()
        X = torch.tensor(data['data'], dtype=torch.float32)
        y = torch.tensor(data['target'], dtype=torch.long)
        return X, y
    else:
        raise ValueError('Unknown source')

def mse(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean().item()

def accuracy(y_pred, y_true):
    return (y_pred == y_true).float().mean().item()

def calculate_tp_tn_fp_fn(y_pred, y_true, num_classes):
    tp = torch.zeros(num_classes)
    tn = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    for cls in range(num_classes):
        tp[cls] = ((y_pred == cls) & (y_true == cls)).sum().item()
        tn[cls] = ((y_pred != cls) & (y_true != cls)).sum().item()
        fp[cls] = ((y_pred == cls) & (y_true != cls)).sum().item()
        fn[cls] = ((y_pred != cls) & (y_true == cls)).sum().item()

    return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}


def precision(y_pred, y_true, num_classes):
    tp = torch.zeros(num_classes)
    fp = torch.zeros(num_classes)
    
    for cls in range(num_classes):
        tp[cls] = ((y_pred == cls) & (y_true == cls)).sum().item()
        fp[cls] = ((y_pred == cls) & (y_true != cls)).sum().item()
    return tp / tp + fp


def recall(y_pred, y_true, num_classes):
    tp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes)
    
    for cls in range(num_classes):
        tp[cls] = ((y_pred == cls) & (y_true == cls)).sum().item()
        fn[cls] = ((y_pred != cls) & (y_true == cls)).sum().item()
    return tp / tp + fn

def print_multiclass_confusion_matrix(tp, fn, tn, fp, num_classes, class_names=None):
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    for i in range(num_classes):
        header = f"\nConfusion Matrix for {class_names[i]} (one-vs-all):\n"
        header += " " * 12 + "Predicted\n"
        header += " " * 11 + "     | Positive | Negative |\n"
        header += "-" * 33 + "\n"

        row_positive = f"Actual Positive | {tp[i]:^8} | {fn[i]:^8} |\n"
        row_negative = f"Actual Negative | {fp[i]:^8} | {tn[i]:^8} |\n"

        print(header + row_positive + row_negative)

# precision, recall, F1-score, ROC-AUC

def log_epoch(epoch, loss, **metrics):
    msg = f"Epoch {epoch}: loss={loss:.4f}"
    for k, v in metrics.items():
        msg += f", {k}={v:.4f}"
    print(msg)