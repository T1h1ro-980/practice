# Задание 1: Стандартные аугментации torchvision

В этом задании я взял картинки из 5 разных классов и применил к ним аугментации \
сначала по одной на каждую картинку потом несколько на одну

Примененные аугментации:
RandomHorizontalFlip ->
ColorJitter ->
RandomRotation ->
RandomGrayscale

### Итог:
Результат применения отдельных аугментации 
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/1%20task/result_aug_0.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/1%20task/result_aug_1.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/1%20task/result_aug_2.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/1%20task/result_aug_3.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/1%20task/result_aug_4.png)

Результат применения комбинированных аугментации 
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/1%20task/result_aug_all.png)

# Задание 2: Кастомные аугментации

В этом задании я сделал три кастомных аугментации как было сказано в примере (лучайное размытие, случайная перспектива, случайная яркость) \
И прменил каждую аугментацию к изображению каждого класса
(классы лежат в custom_augs.py)

### Итог:
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/2%20task/result_custom_aug_0.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/2%20task/result_custom_aug_1.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/2%20task/result_custom_aug_2.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/2%20task/result_custom_aug_3.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/2%20task/result_custom_aug_4.png)

# Задание 3: Анализ датасета
В этом задании я сначала сделал новый класс который не делает resize изображений, прошелся по всему датасету (train и test) \
и посчитал максимум, минимум и среднее по размеру (вычисления были по ширине и высоте отдельно)

По итогу получились следующие результаты: 

### Вывод результатов:
Класс Гароу : Длинна 130 \
Класс Генос : Длинна 130 \
Класс Сайтама : Длинна 130 \
Класс Соник : Длинна 130 \
Класс Татсумаки : Длинна 130 \
Класс Фубуки : Длинна 130 

Максимальный размер: 736 x 1308 \
Минимальный размер: 210 x 220 \
Средний размер: 545 x 629 

И визуализировал распределения по этим величинам
### Визуализация:
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/3%20task/distribution_classes.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/3%20task/distribution_size.png)


# Задание 4: Pipeline аугментаций
В этом задании я реализовал класс по интерфейсу данном в условии
Далее сделал три "конфига" пайплайнов аугментации и премнил их на изображения
### Итог:
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/4%20task/result_light_aug.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/4%20task/result_medium_aug.png)
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/4%20task/result_heavy_aug.png)
В последнем получилась опечатка в названии конфига (должно быть heavy)

# Задание 5: Эксперимент с размерами

В этом задании я использовал датасеты с разным количеством размеров изображении и посчитал нужные величины

### Вывод посчитанных величин:
Время аугментации размера (64, 64) : 0.0540 \
Время загрузки изображений размера (64, 64) : 0.6869 \
Потребление памяти: 0.0207 МБ


Время аугментации размера (128, 128) : 0.0772  \
Время загрузки изображений размера (128, 128) : 0.7954 \
Потребление памяти: 0.0184 МБ


Время аугментации размера (224, 224) : 0.1156  \
Время загрузки изображений размера (224, 224) : 0.9609 \
Потребление памяти: 0.0156 МБ


Время аугментации размера (512, 512) : 0.2785  \
Время загрузки изображений размера (512, 512) : 1.2115 \
Потребление памяти: 0.0151 МБ

А далее построил графики зависимостей посчитанных величин от размера изображения
### Визуализация:
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/5%20task/size_vs_any.png)

Почему то количество используемой памяти уменьшалась с увеличением размера изображений, скорее всего из-за того что память была посчитана неправильно либо на подсчет виляли множество внешних факторов

# Задание 6: Дообучение предобученных моделей

В этом задании я дообучил модель resnet18 и сделал визуализацию loss и accuracy

### Визуализация:
![Image alt](https://github.com/T1h1ro-980/practice/blob/main/HW5/results/6%20task/loss_acc.png)
