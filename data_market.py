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

class Buyer:
    def __init__(self, probabilities: np.array, values: np.array, utility) -> None:
        self.forecast = [probabilities, values]
        self.base_forecast = stats.rv_histogram([self.forecast[0], self.forecast[1]])
        self.utility = utility

class MarketOperator:

    @staticmethod
    def _scale_distribution(values, probabilities, support):
        scaled_values = (values - support[0]) / (support[-1] - support[0])
        scaled_distribution = stats.rv_histogram([probabilities, scaled_values])

        return scaled_distribution

    @staticmethod
    def _task_indicator(task, support: np.array) -> np.array:
        return support >= task

class Market:
    '''Each instance of Market class is defined by the list of sellers, buyer and the task. 
    '''
    def __init__(self, sellers: list[Seller], buyer: Buyer, task) -> None:
        self.sellers = sellers
        self.score_dict = {f'Seller #{id}' : 0 for id, seller in enumerate(sellers)}

        total_support = [np.linspace(*seller.forecast_rv.support(), 1000) for seller in self.sellers]
        total_support.append(np.linspace(*buyer.base_forecast.support(), 1000))
        self.total_supp = np.sort(np.concatenate(total_support))

        self.scaled_dict = self._scale_forecasts()
        self.task = task
        self.buyer = buyer

    def _scale_forecasts(self):
        scaled_dict = {f'Seller #{id}' : MarketOperator._scale_distribution(values = seller.forecast[1], 
                                                                            probabilities = seller.forecast[0], 
                                                                            support= self.total_supp) for id, seller in enumerate(self.sellers)}

        return scaled_dict

    def plot_forecasts(self):
        fig, ax = plt.subplots()

        for seller in self.sellers:
            seller._plot_forecast()

        fig;

    def plot_scaled_forecasts(self):
        fig, ax = plt.subplots()

        for distribution in self.scaled_dict.values():
            support = np.linspace(*distribution.support(), 1000)

            plt.plot(support, distribution.pdf(support))

        fig;

    def _plot_aggregation(self, agg_forecast):

        supp = np.linspace(*agg_forecast.support(), 1000)

        plt.plot(supp, agg_forecast.pdf(supp));

    def scaled_aggregation(self, type = 'QA', plot = False):
        wager_sum = sum([seller.wager for seller in self.sellers]) 
        partial_wagers = [seller.wager / wager_sum for seller in self.sellers]

        scaled_total_support = [np.linspace(*self.scaled_dict[f'Seller #{id}'].support(), 1000) for id, seller in enumerate(self.sellers)]
        scaled_total_support.append(np.linspace(*MarketOperator._scale_distribution(values= self.buyer.forecast[1],
                                                                    probabilities= self.buyer.forecast[0],
                                                                    support= self.total_supp).support(), 1000))

        scaled_total_support = np.sort(np.concatenate(scaled_total_support))

        if type == 'LOP':
            aggregated_forecast_pdf = lambda x: sum([partial_wagers[id] * self.scaled_dict[f'Seller #{id}'].pdf(x) for id, seller in enumerate(self.sellers)])                                                      

            class aggregated_rv(stats.rv_continuous):
                def _pdf(self, x):
                    return aggregated_forecast_pdf(x)

                def _cdf_single(self, x):
                    support = np.linspace(0, x, 1000)
                    return np.trapz(self._pdf(support), x = support)

                def _get_support(self):
                    return scaled_total_support[0], scaled_total_support[-1]

            scaled_aggregated_forecast = aggregated_rv()

        if type == 'QA':
            aggregated_ppf = lambda x: sum([partial_wagers[id] * seller.forecast_rv.ppf(x) for id, seller in enumerate(self.sellers)])

            class aggregated_rv(stats.rv_continuous):
                def _ppf(self, x):
                    return aggregated_ppf(x)

            rv = aggregated_rv()
            agg_data = rv.rvs(size=1000000)
            probas_res, values_res = np.histogram(agg_data, bins = 200, density=True)

            #values_res = (values_res - np.min(values_res)) / (np.max(values_res) - np.min(values_res))
            values_res = (values_res - self.total_supp[0]) / (self.total_supp[-1] - self.total_supp[0])

            result_rv = stats.rv_histogram([probas_res, values_res])
            scaled_aggregated_forecast = result_rv

        if plot:
            self._plot_aggregation(scaled_aggregated_forecast)

        return scaled_aggregated_forecast


    def aggregation(self, type = 'QA', plot = False):
        '''Returns a random variable, which is an instance of of stats.rv_continuous class'''

        wager_sum = sum([seller.wager for seller in self.sellers]) 
        partial_wagers = [seller.wager / wager_sum for seller in self.sellers]

        if type == 'LOP':
            aggregated_forecast_pdf = lambda x: sum([partial_wagers[id] * seller.forecast_rv.pdf(x) for id, seller in enumerate(self.sellers)])
            supp = self.total_supp

            class aggregated_rv(stats.rv_continuous):
                def _pdf(self, x):
                    return aggregated_forecast_pdf(x)

                def _cdf_single(self, x):
                    support = np.linspace(0, x, 1000)
                    return np.trapz(self._pdf(support), x = support)

                def _get_support(self):
                    return supp[0], supp[-1]

            aggregated_forecast = aggregated_rv()
    
        if type == 'QA':
            aggregated_ppf = lambda x: sum([partial_wagers[id] * seller.forecast_rv.ppf(x) for id, seller in enumerate(self.sellers)])

            class aggregated_rv(stats.rv_continuous):
                def _ppf(self, x):
                    return aggregated_ppf(x)

            rv = aggregated_rv()
            agg_data = rv.rvs(size=1000000)
            probas_res, values_res = np.histogram(agg_data, bins = 200, density=True)


            result_rv = stats.rv_histogram([probas_res, values_res])
            aggregated_forecast = result_rv

        if plot:
            self._plot_aggregation(aggregated_forecast)

        return aggregated_forecast

    def _scale_task(self):
        self.scaled_task = (self.task - self.total_supp[0]) / (self.total_supp[-1] - self.total_supp[0])

    def make_scaling(self):
        self._scale_forecasts()
        self._scale_task()

    def _scoring(self, forecast_rv, task, type = 'CRPS') -> float:
        if type == 'CRPS':
            support = np.linspace(-500, 500, 10000)
            task_cdf = MarketOperator._task_indicator(task, support)
            integrand = (forecast_rv.cdf(support) - task_cdf) ** 2

            return 1 - np.trapz(integrand, x = support)

    def make_scoring(self):
        for id, seller in enumerate(self.sellers):
            self.score_dict[f'Seller #{id}'] = self._scoring(self.scaled_dict[f'Seller #{id}'], self.scaled_task)
            

    def _skill_component(self):
        list_skill_payoff = []
        weigted_total_scoring = sum([seller.wager * self.score_dict[f'Seller #{id}'] for id, seller in enumerate(self.sellers)])
        wager_sum = sum([seller.wager for seller in self.sellers])

        for id, seller in enumerate(self.sellers):
            personal_score = self.score_dict[f'Seller #{id}']
            payoff = seller.wager * (1 + personal_score - weigted_total_scoring / wager_sum)
            list_skill_payoff.append(payoff)

        return list_skill_payoff

    def _utililty_component(self):
        list_utility_payoff = []
        buyers_score = self._scoring(MarketOperator._scale_distribution(values = self.buyer.forecast[1], 
                                                                                probabilities= self.buyer.forecast[0],
                                                                                support= self.total_supp), self.scaled_task)

        self.buyers_score = buyers_score

        if self.buyer.utility > 0:
            weigted_total_scoring = sum([seller.wager * self.score_dict[f'Seller #{id}'] for id, seller in enumerate(self.sellers)
                                                                 if self.score_dict[f'Seller #{id}'] > buyers_score])
            for id, seller in enumerate(self.sellers):
                personal_score = self.score_dict[f'Seller #{id}']
                payoff = self.buyer.utility * (personal_score * seller.wager) / weigted_total_scoring if personal_score > buyers_score else 0
                list_utility_payoff.append(payoff)

        return list_utility_payoff

    def calculate_payoffs(self) -> np.array:
        self.make_scoring()
        list_skill_payoff = np.array(self._skill_component())
        print(list_skill_payoff)
        list_utility_payoff = np.array(self._utililty_component())
        print(list_utility_payoff)
        list_wagers = np.array([seller.wager for seller in self.sellers])

        return list_skill_payoff + list_utility_payoff - list_wagers
