# 1.1 Сравнение на MNIST
Сравните производительность на MNIST:
- Полносвязная сеть (3-4 слоя)
- Простая CNN (2-3 conv слоя)
- CNN с Residual Block

Для каждого варианта:
- Обучите модель с одинаковыми гиперпараметрами
- Сравните точность на train и test множествах
- Измерьте время обучения и инференса
- Визуализируйте кривые обучения
- Проанализируйте количество параметров

Создал три модели (SimpleCNN, CNNWithResidual, FullyConnectedModel) и обучил их (гиперпараметры: input_channels=1, num_classes=10)

Далее сделал визуализацию и вывод нужных метрик
![image](https://github.com/user-attachments/assets/b372827b-c64f-49c6-8f54-f3f89d1b3225)

Как видно из графиков CNN модели показывают больший accuracy чем FC модель

Время обучения больше всего у CNN с Residual Block

А кол-во параметров больше всего у полносвязной сети

# 1.2 Сравнение на CIFAR-10
 Сравните производительность на CIFAR-10:
- Полносвязная сеть (глубокая)
- CNN с Residual блоками
- CNN с регуляризацией и Residual блоками

Для каждого варианта:
- Обучите модель с одинаковыми гиперпараметрами
- Сравните точность и время обучения
- Проанализируйте переобучение
- Визуализируйте confusion matrix
- Исследуйте градиенты (gradient flow)

Сделал то же самое, но с датасетом CIFAR

![image](https://github.com/user-attachments/assets/2ff25876-8e73-459e-a345-dc3c3862f1cb)

И как видно из графиков результаты почти не изменились

### Визуализация FC модели:
- confusion matrix
![image](https://github.com/user-attachments/assets/1df046de-560b-4876-bbf2-5c78478026cf)
- gradient flow
![image](https://github.com/user-attachments/assets/c9b87a8c-b1fb-4896-a1c4-8445bb934068)

### Визуализация CNN модели:
- confusion matrix
![image](https://github.com/user-attachments/assets/715dc695-3bde-4ae8-9a4f-7670145074c1)
- gradient flow
![image](https://github.com/user-attachments/assets/ce206404-877e-4328-bba0-0779b9b3da31)


### Визуализация CNN с регуляризацией и Residual блоками:
- confusion matrix
![image](https://github.com/user-attachments/assets/7f370aaa-8a33-4a59-9544-888fee5c9f40)
- gradient flow
![image](https://github.com/user-attachments/assets/b0c5631b-8b1c-4bfa-948a-1cab657449c0)

# 2.1 Влияние размера ядра свертки
Исследуйте влияние размера ядра свертки:
- 3x3 ядра
- 5x5 ядра
- 7x7 ядра
- Комбинация разных размеров (1x1 + 3x3)

Для каждого варианта:
- Поддерживайте одинаковое количество параметров
- Сравните точность и время обучения
- Проанализируйте рецептивные поля
- Визуализируйте активации первого слоя

Сделал три модели без комбинации и одну с комбинацией
  
a. 3x3
  
- loss и acc
    
![image](https://github.com/user-attachments/assets/6425763e-2cf5-4964-b86a-20ea1b279c21)

- time
    
![image](https://github.com/user-attachments/assets/9312f41d-3a59-44b4-a8e0-88a05618dacf)

b. 5x5
  
- loss и acc
    
![image](https://github.com/user-attachments/assets/485187eb-ccd1-4199-aae8-d2dff7f41cdb)

- time
    
![image](https://github.com/user-attachments/assets/bd4aa55a-928f-49e5-a289-cdeb517ff917)

c. 7x7
  
- loss и acc
![image](https://github.com/user-attachments/assets/c7890b26-d089-4f4a-8b1d-8587c553c26e)
    
- time
    
![image](https://github.com/user-attachments/assets/b74db89c-9f0b-4fee-bd46-f4d22d7f0fff)


d. 1x1 и 3x3
  
- loss и acc
    
![image](https://github.com/user-attachments/assets/bdeda736-6a61-471f-9a5b-355536d9ada1)

- time
    
![image](https://github.com/user-attachments/assets/a0476570-1e7e-42fd-9152-f1eadd6e8701)

Визуализация активации первого слоя

a. 3x3

![image](https://github.com/user-attachments/assets/e80bee53-1be4-4e2d-a829-f23a17bd2118)

b. 5x5

![image](https://github.com/user-attachments/assets/dc3ab3a9-4c56-445d-ab5a-c3e8a7a101a4)

c. 7x7
  
![image](https://github.com/user-attachments/assets/7433357e-379e-43af-a0cf-1fdfe200452a)

d. 1x1 и 3x3

![image](https://github.com/user-attachments/assets/cc9f9a21-c766-414f-8c4d-57d2df128697)

# 2.2 Влияние глубины CNN

Исследуйте влияние глубины CNN:
- Неглубокая CNN (2 conv слоя)
- Средняя CNN (4 conv слоя)
- Глубокая CNN (6+ conv слоев)
- CNN с Residual связями

Для каждого варианта:
- Сравните точность и время обучения
- Проанализируйте vanishing/exploding gradients
- Исследуйте эффективность Residual связей
- Визуализируйте feature maps

Сделал класс который позволяет указывать слои явно (конфиг)

Далее создал модели с указанными параметрами

Графики:
### Неглубокая CNN
![image](https://github.com/user-attachments/assets/9f736d4c-d38b-4189-923b-dfeb77969f96)

### Средняя CNN
![image](https://github.com/user-attachments/assets/8b024c13-3767-462e-a280-1f5b4d699389)

### Глубокая CNN 
![image](https://github.com/user-attachments/assets/c063a8d4-c402-4d1a-bea9-4d9dfff56373)

### CNN с Residual связями
![image](https://github.com/user-attachments/assets/218778f7-0409-4a1b-996b-d979aee5f5cd)

Как видно из графиков, чем больше слоев, тем сильнее градиент затухает. Но CNN с Residual связями решает эту проблему
