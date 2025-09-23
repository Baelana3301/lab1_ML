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

def run():
    n_prod = 4
    k_cities = 5
    budget = 50000

if __name__ == "__main__":
    run()