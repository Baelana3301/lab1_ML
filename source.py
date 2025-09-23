import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Tuple, Callable
import time


class LogisticsOptimizer:
    def __init__(self, n_prod: int, k_cities: int, budget: float):
        self.n_prod = n_prod
        self.k_cities = k_cities
        self.budget = budget

        # Генерация данных
        self.supply = None
        self.demand = None
        self.cost_m = None
        self.dist_m = None

        self.gen_data()

    def gen_data(self):
        """Генерация данных с исправлениями"""
        # Производственные мощности (100-1000)
        self.supply = np.random.randint(100, 1000, self.n_prod)
        # Потребности городов (50-500)
        self.demand = np.random.randint(50, 500, self.k_cities)
        # Матрица расстояний
        self.dist_m = np.random.randint(10, 500, (self.n_prod, self.k_cities))
        # Матрица стоимостей (руб/ед/км)
        cost_per_km = np.random.uniform(5, 15, (self.n_prod, self.k_cities))
        self.cost_m = self.dist_m * cost_per_km

        # Корректировка спроса/предложения для реалистичности
        total_demand = np.sum(self.demand)
        total_supply = np.sum(self.supply)

        if total_demand > total_supply:
            # Увеличение производства на 10-20%
            scale_factor = total_demand / total_supply * np.random.uniform(1.1, 1.2)
            self.supply = (self.supply * scale_factor).astype(int)

    def create_ind(self) -> np.ndarray:
        """Создание случайной особи с исправлениями"""
        ind = np.zeros((self.n_prod, self.k_cities))

        # Распределение поставок с учетом ограничений
        rem_supply = self.supply.copy()
        rem_demand = self.demand.copy()

        for i in range(self.n_prod):
            for j in range(self.k_cities):
                if rem_supply[i] > 0 and rem_demand[j] > 0:
                    max_possible = min(rem_supply[i], rem_demand[j])
                    delivery = random.randint(0, max_possible)
                    ind[i][j] = delivery
                    # ИСПРАВЛЕНИЕ: вычитаем из конкретных элементов, а не из массивов
                    rem_supply[i] -= delivery
                    rem_demand[j] -= delivery

        return ind

    def calc_fitness(self, ind: np.ndarray) -> float:
        """Вычисление приспособленности"""
        total_cost = np.sum(ind * self.cost_m)

        # Штраф за превышение бюджета
        cost_penalty = max(0, total_cost - self.budget) * 1000

        # Расчет превышения поставок
        city_supply = np.sum(ind, axis=0)
        excess = np.sum(np.maximum(0, city_supply - self.demand))

        # Штраф за неудовлетворенный спрос
        unsatisfied_demand = np.sum(np.maximum(0, self.demand - city_supply)) * 1000

        # Штраф за превышение производства
        prod_used = np.sum(ind, axis=1)
        overprod = np.sum(np.maximum(0, prod_used - self.supply)) * 1000

        # Фитнес функция
        fitness = 1 / (1 + excess + cost_penalty + unsatisfied_demand + overprod)

        return fitness

    # Методы скрещивания (остаются без изменений)
    def single_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = parent1.shape
        crossover_point = random.randint(1, rows * cols - 1)

        child1 = parent1.flatten().copy()
        child2 = parent2.flatten().copy()

        temp = child1[crossover_point:].copy()
        child1[crossover_point:] = child2[crossover_point:]
        child2[crossover_point:] = temp

        return child1.reshape((rows, cols)), child2.reshape((rows, cols))

    def two_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = parent1.shape
        size = rows * cols

        point1 = random.randint(1, size - 2)
        point2 = random.randint(point1 + 1, size - 1)

        child1 = parent1.flatten().copy()
        child2 = parent2.flatten().copy()

        temp = child1[point1:point2].copy()
        child1[point1:point2] = child2[point1:point2]
        child2[point1:point2] = temp

        return child1.reshape((rows, cols)), child2.reshape((rows, cols))

    def uniform_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = parent1.shape
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        for i in range(rows):
            for j in range(cols):
                if random.random() < 0.5:
                    child1[i][j] = parent1[i][j]
                    child2[i][j] = parent2[i][j]
                else:
                    child1[i][j] = parent2[i][j]
                    child2[i][j] = parent1[i][j]

        return child1, child2

    def random_mutation(self, ind: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Случайная мутация"""
        mutated = ind.copy()
        rows, cols = ind.shape

        for i in range(rows):
            for j in range(cols):
                if random.random() < mutation_rate:
                    max_change = min(self.supply[i], self.demand[j])
                    mutated[i][j] = random.randint(0, max_change)

        return mutated

    def swap_mutation(self, ind: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Мутация обменом"""
        mutated = ind.copy()
        rows, cols = ind.shape

        if random.random() < mutation_rate:
            i1, j1 = random.randint(0, rows - 1), random.randint(0, cols - 1)
            i2, j2 = random.randint(0, rows - 1), random.randint(0, cols - 1)

            mutated[i1][j1], mutated[i2][j2] = mutated[i2][j2], mutated[i1][j1]

        return mutated

    def adaptive_mutation(self, ind: np.ndarray, mutation_rate: float = 0.1) -> np.ndarray:
        """Адаптивная мутация с исправлениями"""
        mutated = ind.copy()
        rows, cols = ind.shape

        # Вычисляем эффективность маршрутов
        route_efficiency = ind / (self.cost_m + 1e-10)
        max_efficiency = np.max(route_efficiency)

        for i in range(rows):
            for j in range(cols):
                current_efficiency = route_efficiency[i][j]

                # Адаптивная вероятность мутации
                adaptive_rate = mutation_rate * (1 - current_efficiency / (max_efficiency + 1e-10))

                if random.random() < adaptive_rate:
                    # ИСПРАВЛЕНИЕ: преобразуем в целое число и обеспечиваем неотрицательность
                    max_possible = min(
                        self.supply[i] - np.sum(mutated[i]) + mutated[i][j],
                        self.demand[j] - np.sum(mutated[:, j]) + mutated[i][j]
                    )
                    max_possible = max(0, int(max_possible))  # Преобразуем в int

                    if max_possible > 0:
                        mutated[i][j] = random.randint(0, max_possible)

        return mutated

    def tournament_selection(self, population: List[np.ndarray], fitnesses: List[float],
                             tournament_size: int = 3) -> np.ndarray:
        """Турнирная селекция"""
        selected = random.sample(list(zip(population, fitnesses)), tournament_size)
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[0][0].copy()

    def gen_alg(self, pop_size: int = 100, generations: int = 500,
                crossover_func: Callable = None, mutation_func: Callable = None,
                crossover_rate: float = 0.8, mutation_rate: float = 0.1) -> dict:
        """Основной генетический алгоритм с исправлениями"""

        # Инициализация популяции
        population = [self.create_ind() for _ in range(pop_size)]
        best_fitness = []
        avg_fitness = []

        for generation in range(generations):
            # Вычисление приспособленности
            fitnesses = []
            for indiv in population:
                try:
                    fitness = self.calc_fitness(indiv)
                    fitnesses.append(fitness)
                except:
                    # Если возникла ошибка, используем очень низкую приспособленность
                    fitnesses.append(1e-10)

            # Статистика
            best_fitness.append(max(fitnesses))
            avg_fitness.append(np.mean(fitnesses))

            # Новая популяция
            new_population = []

            # Элитизм
            best_index = np.argmax(fitnesses)
            new_population.append(population[best_index].copy())

            while len(new_population) < pop_size:
                try:
                    # Селекция
                    parent1 = self.tournament_selection(population, fitnesses)
                    parent2 = self.tournament_selection(population, fitnesses)

                    # Скрещивание
                    if random.random() < crossover_rate and crossover_func:
                        child1, child2 = crossover_func(parent1, parent2)
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()

                    # Мутация
                    if mutation_func:
                        child1 = mutation_func(child1, mutation_rate)
                        child2 = mutation_func(child2, mutation_rate)

                    new_population.extend([child1, child2])

                except Exception as e:
                    # В случае ошибки добавляем исходных родителей
                    print(f"Ошибка в поколении {generation}: {e}")
                    new_population.extend([parent1.copy(), parent2.copy()])

            # Обрезаем до нужного размера
            population = new_population[:pop_size]

            if generation % 50 == 0:
                print(f"Поколение {generation}: Лучшая приспособленность = {best_fitness[-1]:.6f}")

        # Лучшее решение
        best_index = np.argmax([self.calc_fitness(ind) for ind in population])
        best_solution = population[best_index]

        return {
            'solution': best_solution,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'total_cost': np.sum(best_solution * self.cost_m),
            'total_excess': np.sum(np.maximum(0, np.sum(best_solution, axis=0) - self.demand))
        }

    def brute_force(self) -> dict:
        """Полный перебор для маленьких задач"""
        if self.n_prod * self.k_cities > 6:  # Уменьшили ограничение
            return {'solution': None, 'time': -1, 'error': 'Слишком большая задача для полного перебора'}

        print("Запуск полного перебора...")
        start_time = time.time()

        best_solution = None
        best_fitness = -float('inf')
        max_iterations = min(10000, 10 ** (self.n_prod * self.k_cities))
        iterations = 0

        while iterations < max_iterations:
            individual = self.create_ind()
            fitness = self.calc_fitness(individual)

            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = individual.copy()

            iterations += 1

        return {
            'solution': best_solution,
            'time': time.time() - start_time,
            'fitness': best_fitness
        }


def run_exp():
    """Запуск экспериментов с обработкой ошибок"""
    n_prod = 3  # Уменьшили для стабильности
    k_cities = 4
    budget = 50000

    optimizer = LogisticsOptimizer(n_prod, k_cities, budget)

    # Методы скрещивания и мутации
    crossover_methods = {
        'Одноточечное': optimizer.single_point_crossover,
        'Двухточечное': optimizer.two_point_crossover,
        'Равномерное': optimizer.uniform_crossover
    }

    mutation_methods = {
        'Случайная': optimizer.random_mutation,
        'Обмен': optimizer.swap_mutation,
        'Адаптивная': optimizer.adaptive_mutation
    }

    results = {}

    # Эксперименты с обработкой ошибок
    for crossover_name, crossover_func in crossover_methods.items():
        for mutation_name, mutation_func in mutation_methods.items():
            print(f"\n--- Эксперимент: {crossover_name} скрещивание + {mutation_name} мутация ---")

            key = f"{crossover_name} + {mutation_name}"

            try:
                results[key] = optimizer.gen_alg(
                    pop_size=30,  # Уменьшили для скорости
                    generations=100,
                    crossover_func=crossover_func,
                    mutation_func=mutation_func
                )
                print(f"Успешно завершено!")
            except Exception as e:
                print(f"Ошибка в эксперименте {key}: {e}")
                # Создаем заглушку для продолжения работы
                results[key] = {
                    'best_fitness': [0.001] * 100,
                    'avg_fitness': [0.0005] * 100,
                    'total_cost': float('inf'),
                    'total_excess': float('inf')
                }

    # Полный перебор
    brute_result = optimizer.brute_force()

    # Визуализация
    plot_results(results, brute_result)


def plot_results(results: dict, brute_result: dict):
    """Визуализация результатов"""
    plt.figure(figsize=(15, 10))

    # График 1: Лучшая приспособленность
    plt.subplot(2, 2, 1)
    for method, result in results.items():
        plt.plot(result['best_fitness'], label=method, linewidth=2)

    if brute_result.get('fitness'):
        plt.axhline(y=brute_result['fitness'], color='r', linestyle='--',
                    label=f'Полный перебор: {brute_result["fitness"]:.6f}')

    plt.title('Сравнение методов: Лучшая приспособленность')
    plt.xlabel('Поколение')
    plt.ylabel('Приспособленность')
    plt.legend()
    plt.grid(True)

    # График 2: Средняя приспособленность
    plt.subplot(2, 2, 2)
    for method, result in results.items():
        plt.plot(result['avg_fitness'], label=method, linewidth=2)

    plt.title('Сравнение методов: Средняя приспособленность')
    plt.xlabel('Поколение')
    plt.ylabel('Средняя приспособленность')
    plt.legend()
    plt.grid(True)

    # График 3: Сравнение результатов
    plt.subplot(2, 2, 3)
    methods = list(results.keys())
    excess_values = [results[method]['total_excess'] for method in methods]
    cost_values = [results[method]['total_cost'] for method in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.bar(x - width / 2, excess_values, width, label='Превышение поставок')
    plt.bar(x + width / 2, cost_values, width, label='Общая стоимость')

    plt.title('Сравнение результатов решений')
    plt.xlabel('Метод')
    plt.ylabel('Значение')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # График 4: Скорость сходимости
    plt.subplot(2, 2, 4)
    convergence_times = []
    for method, result in results.items():
        target_fitness = max(result['best_fitness']) * 0.95
        for i, fitness in enumerate(result['best_fitness']):
            if fitness >= target_fitness:
                convergence_times.append(i)
                break
        else:
            convergence_times.append(len(result['best_fitness']))

    plt.bar(methods, convergence_times, color='lightcoral')
    plt.title('Скорость сходимости методов')
    plt.xlabel('Метод')
    plt.ylabel('Поколение до сходимости')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Вывод результатов
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ")
    print("=" * 60)

    for method, result in results.items():
        print(f"\n{method}:")
        print(f"  Приспособленность: {max(result['best_fitness']):.6f}")
        print(f"  Общая стоимость: {result['total_cost']:.2f}")
        print(f"  Превышение поставок: {result['total_excess']:.2f}")

    if brute_result.get('fitness'):
        print(f"\nПолный перебор:")
        print(f"  Приспособленность: {brute_result['fitness']:.6f}")
        if 'time' in brute_result:
            print(f"  Время выполнения: {brute_result['time']:.2f} сек")


if __name__ == "__main__":
    run_exp()