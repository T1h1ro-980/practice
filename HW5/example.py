import torch
from torchvision import transforms
from PIL import Image
from datasets import CustomImageDataset
from utils import show_images, show_single_augmentation, show_multiple_augmentations
from extra_augs import (AddGaussianNoise, RandomErasingCustom, CutOut, 
                       Solarize, Posterize, AutoContrast, ElasticTransform)


# Загрузка датасета без аугментаций
root = '/home/egikor/ML/Pratic in git/lesson5-augmentations/data/train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))

# Берем одно изображение для демонстрации
original_img, label = dataset[0]
class_names = dataset.get_class_names()
print(f"Оригинальное изображение, класс: {class_names[label]}")

# Демонстрация каждой аугментации отдельно
print("\n=== Демонстрация отдельных аугментаций ===")

# 1. Гауссов шум
noise_aug = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.2)
])
noise_img = noise_aug(original_img)
show_single_augmentation(original_img, noise_img, "Гауссов шум")

# 2. Random Erasing
erase_aug = transforms.Compose([
    transforms.ToTensor(),
    RandomErasingCustom(p=1.0)
])
erase_img = erase_aug(original_img)
show_single_augmentation(original_img, erase_img, "Random Erasing")

# 3. CutOut
cutout_aug = transforms.Compose([
    transforms.ToTensor(),
    CutOut(p=1.0, size=(32, 32))
])
cutout_img = cutout_aug(original_img)
show_single_augmentation(original_img, cutout_img, "CutOut")

# 4. Solarize
solarize_aug = transforms.Compose([
    transforms.ToTensor(),
    Solarize(threshold=128)
])
solarize_img = solarize_aug(original_img)
show_single_augmentation(original_img, solarize_img, "Solarize")

# 5. Posterize
posterize_aug = transforms.Compose([
    transforms.ToTensor(),
    Posterize(bits=4)
])
posterize_img = posterize_aug(original_img)
show_single_augmentation(original_img, posterize_img, "Posterize")

# 6. AutoContrast
autocontrast_aug = transforms.Compose([
    transforms.ToTensor(),
    AutoContrast(p=1.0)
])
autocontrast_img = autocontrast_aug(original_img)
show_single_augmentation(original_img, autocontrast_img, "AutoContrast")

# 7. Elastic Transform
elastic_aug = transforms.Compose([
    transforms.ToTensor(),
    ElasticTransform(p=1.0, alpha=1, sigma=50)
])
elastic_img = elastic_aug(original_img)
show_single_augmentation(original_img, elastic_img, "Elastic Transform")

# Демонстрация стандартных аугментаций torchvision
print("\n=== Стандартные аугментации torchvision ===")

standard_augs = [
    ("RandomHorizontalFlip", transforms.RandomHorizontalFlip(p=1.0)),
    ("RandomCrop", transforms.RandomCrop(200, padding=20)),
    ("ColorJitter", transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)),
    ("RandomRotation", transforms.RandomRotation(degrees=30)),
    ("RandomGrayscale", transforms.RandomGrayscale(p=1.0))
]

augmented_imgs = []
titles = []

for name, aug in standard_augs:
    aug_transform = transforms.Compose([
        aug,
        transforms.ToTensor()
    ])
    aug_img = aug_transform(original_img)
    augmented_imgs.append(aug_img)
    titles.append(name)

show_multiple_augmentations(original_img, augmented_imgs, titles)

# Демонстрация комбинированных аугментаций
print("\n=== Комбинированные аугментации ===")

combined_aug = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(200, padding=20),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    AddGaussianNoise(0., 0.1),
    CutOut(p=0.5)
])

combined_imgs = []
for i in range(8):
    combined_img = combined_aug(original_img)
    combined_imgs.append(combined_img)

show_images(combined_imgs, title="Комбинированные аугментации") 