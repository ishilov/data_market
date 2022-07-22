import numpy as np
from scipy import stats
import matplotlib as plt

class Seller:
    def __init__(self, probabilities: np.array, values: np.array, wager: float) -> None:

        self.forecast = [probabilities, values]
        self.wager = wager
        self.distribution = self._distribution

    def _distribution(self):
        dist = stats.rv_histogram([self.probabilities, self.values])

        return dist

    def _plot_forecast(self):
        support = np.linspace(*self.distribution.support(), 100)

        plt.plot(support, self.distribution.pdf(support))

class MarketOperator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_forecasts(*sellers):
        fig, ax = plt.subplots()

        for seller in sellers:
            seller._plot_forecast()

        fig;

    @staticmethod
    def aggregation(*sellers, type = 'QA') -> np.array:
        wager_sum = sum([seller.wager for seller in sellers])

        for seller in sellers:
            seller.partial_wager = seller.wager / wager_sum

        if type == 'LOP':
            aggregated_forecast = lambda x: sum([seller.partial_wager * seller.distribution.pdf(x) for seller in sellers])

        if type == 'QA':

    @staticmethod
    def scoring(seller) -> float:
        pass

    @staticmethod
    def payoff(buyer, *sellers) -> np.array:
        pass

class Buyer:
    def __init__(self, base_forecast, utility) -> None:
        self.base_forecast = base_forecast
        self.utility = utility

    