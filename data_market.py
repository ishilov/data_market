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
    def _plot_aggregation(*sellers, agg_forecast):
        supp = np.sort(np.concatenate([np.linspace(*seller.distribution.support(), 1000) for seller in sellers]))

        plt.plot(supp, agg_forecast(supp))

    @staticmethod
    def aggregation(*sellers, type = 'QA', plot = False) -> np.array:
        wager_sum = sum([seller.wager for seller in sellers])

        for seller in sellers:
            seller.partial_wager = seller.wager / wager_sum

        if type == 'LOP':
            aggregated_forecast = lambda x: sum([seller.partial_wager * seller.distribution.pdf(x) for seller in sellers])
    
        if type == 'QA':
            aggregated_ppf = lambda x: sum([seller.partial_wager * seller.distribution.ppf(x) for seller in sellers])

            class aggregated_rv(stats.rv_continuous):
                def _ppf(self, x):
                    return aggregated_ppf(x)

            rv = aggregated_rv()

            agg_data = rv.rvs(size=1000000)

            probas_res, values_res, _ = plt.hist(agg_data, bins = 200)

            result_rv = stats.rv_histogram([probas_res, values_res])

            aggregated_forecast = result_rv.pdf

        if plot:
            MarketOperator._plot_aggregation(sellers, aggregated_forecast)

        return aggregated_forecast


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

    