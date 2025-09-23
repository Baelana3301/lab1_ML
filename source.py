import numpy as np
import matplotlib as plt
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

    #Функция генерации данных
    def gen_data(self):
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

    # Создание случайной особи
    def create_ind(self) -> np.ndarray:
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
                    rem_supply -= delivery
                    rem_demand -= delivery

        return ind

    # Вычисление приспособленности (fitness)
    def calc_fitness(self, ind: np.ndarray):
        total_cost = np.sum(ind * self.cost_m)
        # Штраф за превышение транспортных затрат
        cost_penalty = max(0, total_cost - self.budget) * 1000
        # Расчет превышения поставок
        city_supply = np.sum(ind, axis=0)
        excess = np.sum(np.maximum(0, city_supply - self.demand))
        # Штраф за неудовлетворенный спрос
        unsatisfied_demand = np.sum(np.maximum(0, self.demand - city_supply)) * 1000
        # Штраф за превышение производства
        prod_used = np.sum(ind, axis=1)
        overprod = np.sum(np.maximum(0, prod_used - self.supply)) * 1000
        # Фитнес функция, минимизирование превышения и штрафов
        fitness = 1 / (1 + excess + cost_penalty + unsatisfied_demand + overprod)

        return fitness

    # Одноточечное скрещивание
    def single_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = parent1.shape
        crossover_point = random.randint(1, rows * cols - 1)

        child1 = parent1.flatten().copy()
        child2 = parent2.flatten().copy()

        # Обмен генами после точки скрещивания
        temp = child1[crossover_point:].copy()
        child1[crossover_point:] = child2[crossover_point:].copy()
        child2[crossover_point:] = temp

        return child1.reshape((rows, cols)), child2.reshape((rows, cols))

    # Двухточечное скрещивание
    def two_point_crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rows, cols = parent1.shape
        size = rows * cols

        point1 = random.randint(1, size - 2)
        point2 = random.randint(point1 + 1, size - 1)

        child1 = parent1.flatten().copy()
        child2 = parent2.flatten().copy()

        # Обмен генами между двумя точками
        temp = child1[point1:point2].copy()
        child1[point1:point2] = child2[point1:point2]
        child2[point1:point2] = temp

        return child1.reshape((rows, cols)), child2.reshape((rows, cols))

    # Равномерное скрещивание
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



def run():
    n_prod = 4
    k_cities = 5
    budget = 50000

if __name__ == "__main__":
    run()