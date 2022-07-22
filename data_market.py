import numpy as np
from scipy import stats
import matplotlib.pyplot as plt 

class Seller:
    def __init__(self, probabilities: np.array, values: np.array, wager: float) -> None:

        self.forecast = [probabilities, values]
        self.wager = wager
        self.forecast_rv = self._distribution()

    def _distribution(self):
        dist = stats.rv_histogram([self.forecast[0], self.forecast[1]])

        return dist

    def _plot_forecast(self):
        support = np.linspace(*self.forecast_rv.support(), 1000)

        plt.plot(support, self.forecast_rv.pdf(support))

class MarketOperator:
    def __init__(self) -> None:
        pass

    @staticmethod
    def plot_forecasts(sellers: list, scale = True):
        fig, ax = plt.subplots()
        
        if scale:
            MarketOperator.scaler(sellers)

        for seller in sellers:
            seller._plot_forecast()

        fig;

    @staticmethod
    def scaler(sellers):
        #Rewrite this whole functionality

        supp = np.sort(np.concatenate([np.linspace(*seller.forecast_rv.support(), 1000) for seller in sellers]))

        for seller in sellers:
            scaled_values = (seller.forecast[1] - supp[0]) / (supp[-1] - supp[0])
            
            seller.non_scaled_forecast_rv = seller.forecast_rv
            seller.forecast_rv = stats.rv_histogram([seller.forecast[0], scaled_values])


    @staticmethod
    def _plot_aggregation(sellers: list, agg_forecast):
        supp = np.sort(np.concatenate([np.linspace(*seller.forecast_rv.support(), 1000) for seller in sellers]))

        if agg_forecast.support() == (0,1):
            supp = np.linspace(0,1, 1000)

        plt.plot(supp, agg_forecast.pdf(supp));

    @staticmethod
    def aggregation(sellers: list, type = 'QA', plot = False, scale = True):
        '''Returns a random variable, which is an instance of of stats.rv_continuous class'''
        if scale:
            MarketOperator.scaler(sellers)

        wager_sum = sum([seller.wager for seller in sellers])

        for seller in sellers:
            seller.partial_wager = seller.wager / wager_sum

        if type == 'LOP':
            aggregated_forecast_pdf = lambda x: sum([seller.partial_wager * seller.forecast_rv.pdf(x) for seller in sellers])
            supp = np.sort(np.concatenate([np.linspace(*seller.forecast_rv.support(), 1000) for seller in sellers]))

            class aggregated_rv(stats.rv_continuous):
                def _pdf(self, x):
                    return aggregated_forecast_pdf(x)

                def _get_support(self):
                    return supp[0], supp[-1]

            aggregated_forecast = aggregated_rv()
    
        if type == 'QA':
            aggregated_ppf = lambda x: sum([seller.partial_wager * seller.forecast_rv.ppf(x) for seller in sellers])

            class aggregated_rv(stats.rv_continuous):
                def _ppf(self, x):
                    return aggregated_ppf(x)

            rv = aggregated_rv()
            agg_data = rv.rvs(size=1000000)
            probas_res, values_res = np.histogram(agg_data, bins = 200, density=True)

            #rewrite this scaling later
            if scale:
                values_res = (values_res - np.min(values_res)) / (np.max(values_res) - np.min(values_res))

            result_rv = stats.rv_histogram([probas_res, values_res])
            aggregated_forecast = result_rv

        if plot:
            MarketOperator._plot_aggregation(sellers, aggregated_forecast)

        return aggregated_forecast

    @staticmethod
    def _task_indicator(task, support: np.array) -> np.array:
        return support >= task

    @staticmethod
    def scoring(forecast_rv, task, type = 'CRPS') -> float:
        if type == 'CRPS':
            support = np.linspace(*forecast_rv.support(), 1000)
            task_cdf = MarketOperator._task_indicator(task, support)
            integrand = (forecast_rv.cdf(support) - task_cdf) ** 2

            return np.trapz(integrand, x = support)


    @staticmethod
    def payoff(buyer, *sellers) -> np.array:
        pass

class Buyer:
    def __init__(self, base_forecast, utility) -> None:
        self.base_forecast = base_forecast
        self.utility = utility

    