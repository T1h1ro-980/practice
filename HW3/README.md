Выводы к 1.1
- 1 слой
![image](https://github.com/user-attachments/assets/d5611de7-09ea-40b3-a38b-9e53982d803b)

  Train Loss: 0.2705, Train Acc: 0.9245
  
  Test Loss: 0.2788, Test Acc: 0.9223

  Время обучения: 59.321749448776245
  
- 2 слоя
![image](https://github.com/user-attachments/assets/87c612ac-2c22-4e61-95cc-60b225181f1e)
  Train Loss: 0.0353, Train Acc: 0.9884
  
  Test Loss: 0.0790, Test Acc: 0.9775

  Время обучения: 65.88357543945312

- 3 слоя
![image](https://github.com/user-attachments/assets/ccb2161a-7813-4f04-9b80-4668dc22b516)
    Train Loss: 0.0387, Train Acc: 0.9878
  
    Test Loss: 0.0704, Test Acc: 0.9795

    Время обучения: 67.09625720977783

- 5 слоев
![image](https://github.com/user-attachments/assets/02222ed1-432b-4001-a0ce-427cbd8fee25)
  Train Loss: 0.0484, Train Acc: 0.9851
  
  Test Loss: 0.0721, Test Acc: 0.9789
  
  Время обучения: 70.27900958061218

- 7 слоев
![image](https://github.com/user-attachments/assets/21c35dc7-5a7b-47b0-9eb4-9b6fa78c6e7e)
  Train Loss: 0.0557, Train Acc: 0.9834
  
  Test Loss: 0.0925, Test Acc: 0.9775

  Время обучения: 73.23612761497498

Результаты показали что при увеличении модели до 3 слоев, качество модели увеличивается, но затем качество ухудшается а иногда приводит и к переобучению (небольшому). Лучшее качество модели при 3 слоях

Также при увеличении количества слоев увеличивается и время обучения модели

Выводы для 1.2 

Графики accuracy 
- 1 слой
  
  ![image](https://github.com/user-attachments/assets/ce4b158a-7d43-4f37-89d7-76eb739e3075)

- 2 слоя
  
  ![image](https://github.com/user-attachments/assets/7a675e25-840a-4fb5-b13f-002dea747526)

- 3 слоя
  
  ![image](https://github.com/user-attachments/assets/a9d42a80-e83b-47f6-8211-60647c04c211)

- 5 слоев
  
  ![image](https://github.com/user-attachments/assets/f3668c22-7161-4e8e-8d61-5ce69c22136c)

- 7 слоев
  
  ![image](https://github.com/user-attachments/assets/82d78691-11b5-4371-9b81-5496ad727142)

Опитмальная длинна слоев для датасета MNIST - 3, для CIFAR - 5

График при новых слоях:

![image](https://github.com/user-attachments/assets/cd019ee7-c5ee-4c86-aec2-984d074ef29a)

Сравнение результатов:

- 1 слой
  
  Train Loss: 1.7056, Train Acc: 0.4158
  
  Test Loss: 1.7580, Test Acc: 0.3949

- 2 слоя
  
  Train Loss: 1.2194, Train Acc: 0.5755
  
  Test Loss: 1.3877, Test Acc: 0.5216

- 3 слоя
  
  Train Loss: 1.1789, Train Acc: 0.5836
  
  Test Loss: 1.3683, Test Acc: 0.5227

- 5 слоев
  
  Train Loss: 1.1981, Train Acc: 0.5754
  
  Test Loss: 1.3426, Test Acc: 0.5279

- 7 слоев
  
  Train Loss: 1.2250, Train Acc: 0.5708
  
  Test Loss: 1.3769, Test Acc: 0.5159

- С Dropout и BatchNorm
  
  Train Loss: 1.4852, Train Acc: 0.4794
  
  Test Loss: 1.4131, Test Acc: 0.5053

Как видно из выводов точность немного ухудшилась по сравнению с предыдущими моделями

Переобучение происходит при 7 слоях, т.к. качество модели начинает ухудшаться по сравнению с прошлыми результатами



Выводы к 2.1 
- [64, 32, 16]
  
  Model parameters: 53128
  
  Train Loss: 0.0860, Train Acc: 0.9732
  
  Test Loss: 0.1101, Test Acc: 0.9672
  
  Время обучения: 72.41468286514282

- [256, 128, 64]
  
  Model parameters: 242872
  
  Train Loss: 0.0457, Train Acc: 0.9854
  
  Test Loss: 0.0726, Test Acc: 0.9773
  
  Время обучения: 73.02111387252808

- [1024, 512, 256]
  
  Model parameters: 1462648
  
  Train Loss: 0.0439, Train Acc: 0.9866
  
  Test Loss: 0.0833, Test Acc: 0.9794
  
  Время обучения: 74.77809453010559

- [2048, 1024, 512]
  
  Model parameters: 4235896
  
  Train Loss: 0.0465, Train Acc: 0.9867
  
  Test Loss: 0.0924, Test Acc: 0.9719
  
  Время обучения: 79.48529267311096

Как видно из выводов, при увеличении широты слоев, сильно увеличивается число параметров, качество модели меняется не сильно как и время обучения
