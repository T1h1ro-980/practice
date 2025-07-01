import torch
import time

# 3.1 Подготовка данных
# Создайте большие матрицы размеров:
# - 64 x 1024 x 1024
# - 128 x 512 x 512
# - 256 x 256 x 256
# Заполните их случайными числами

matrix_1 = torch.rand(64, 1024, 1024, dtype = torch.float32)
matrix_2 = torch.rand(128, 512, 512, dtype = torch.float32)
matrix_3 = torch.rand(256, 256, 256, dtype = torch.float32)

# 3.2 Функция измерения времени
# Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на GPU
# Используйте time.time() для измерения на CPU

def check_time_for_GPU(func, **params):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    func(**params)

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event.elapsed_time(end_event)
    return elapsed_time_ms / 1000 # переводим в секунды 

def check_time_for_CPU(func, **params):
    start_time = time.time()
    func(**params)
    end_time = time.time()
    return abs(start_time - end_time)

device = torch.device("cuda")

#3.3 Сравнение операций
# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов

# Для каждой операции:
# 1. Измерьте время на CPU
# 2. Измерьте время на GPU (если доступен)
# 3. Вычислите ускорение (speedup)
# 4. Выведите результаты в табличном виде

results = []

# Матричное умножение (torch.matmul)
tensor1_cpu = torch.rand((64, 1024, 1024))
tensor2_cpu = torch.rand((64, 1024, 1024))
CPU_time = check_time_for_CPU(func = torch.matmul, input=tensor1_cpu, other=tensor2_cpu)

tensor1_gpu = tensor1_cpu.to(device)
tensor2_gpu = tensor2_cpu.to(device)
GPU_time = check_time_for_GPU(func = torch.matmul, input=tensor1_gpu, other=tensor2_gpu)

results.append(("Матричное умножение", CPU_time, GPU_time))

# Поэлементное сложение
def addition(tensor1, tensor2):
    return tensor1 + tensor2
tensor1_cpu = torch.rand((64, 1024, 1024))
tensor2_cpu = torch.rand((64, 1024, 1024))
CPU_time = check_time_for_CPU(func = addition, tensor1 = tensor1_cpu, tensor2 = tensor2_cpu)

tensor1_gpu = tensor1_cpu.to(device)
tensor2_gpu = tensor2_cpu.to(device)
GPU_time = check_time_for_GPU(func = addition, tensor1 = tensor1_gpu, tensor2 = tensor2_gpu)

results.append(("Поэлементное сложение", CPU_time, GPU_time))

# Поэлементное умножение
def multiplication(tensor1, tensor2):
    return tensor1 * tensor2
tensor1_cpu = torch.rand((64, 1024, 1024))
tensor2_cpu = torch.rand((64, 1024, 1024))
CPU_time = check_time_for_CPU(func = multiplication, tensor1 = tensor1_cpu, tensor2 = tensor2_cpu)

tensor1_gpu = tensor1_cpu.to(device)
tensor2_gpu = tensor2_cpu.to(device)
GPU_time = check_time_for_GPU(func = multiplication, tensor1 = tensor1_gpu, tensor2 = tensor2_gpu)

results.append(("Поэлементное умножение", CPU_time, GPU_time))

# Транспонирование
def transposition(tensor1):
    return tensor1.T
tensor1_cpu = torch.rand((64, 1024, 1024))
CPU_time = check_time_for_CPU(func = transposition, tensor1 = tensor1_cpu)

tensor1_gpu = tensor1_cpu.to(device)
GPU_time = check_time_for_GPU(func = transposition, tensor1 = tensor1_gpu)

results.append(("Транспонирование", CPU_time, GPU_time))
# Вычисление суммы всех элементов
def sum(tensor1):
    return tensor1.sum()
tensor1_cpu = torch.rand((64, 1024, 1024))
CPU_time = check_time_for_CPU(func = sum, tensor1 = tensor1_cpu)

tensor1_gpu = tensor1_cpu.to(device)
GPU_time = check_time_for_GPU(func = sum, tensor1 = tensor1_gpu)

results.append(("Сумма всех элементов", CPU_time, GPU_time))

# Таблица
print(f"{'Операция':<25} | {'CPU (мс)':>10} | {'GPU (мс)':>10} | {'Ускорение':>10}")
print("-" * 25 + "-+-" + "-" * 12 + "-+-" + "-" * 12 + "-+-" + "-" * 12)

for name, cpu, gpu in results:
    speedup = cpu / gpu if gpu > 0 else float("inf")
    print(f"{name:<25} | {cpu:10.4f} | {gpu:10.4f} | {speedup:10.4f}x")

# 3.4 Анализ результатов
# в README.md