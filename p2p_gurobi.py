import gurobipy as gp
import numpy as np

class FirstStageModel:

    @staticmethod
    def da_purchase(agent, model):

        model.addVar(lb = 0,
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'Agent {agent.id} day-ahead purchase')

        model.update()

    @staticmethod
    def da_sale(agent, model):

        model.addVar(lb = 0,
                    ub = 10, #float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'Agent {agent.id} day-ahead sale')

        model.update()

    @staticmethod
    def rt_purchase(agent, model):

        for proba, proba_val in enumerate(agent.probabilities):
            if proba_val > 0:
                model.addVar(lb = 0,
                            ub = float('inf'),
                            vtype = gp.GRB.CONTINUOUS,
                            name = f'Agent {agent.id} proba {proba} real-time purchase')

        model.update()

    @staticmethod
    def rt_sale(agent, model):

        for proba, proba_val in enumerate(agent.probabilities):
            if proba_val > 0:
                model.addVar(lb = 0,
                            ub = 10, #float('inf'),
                            vtype = gp.GRB.CONTINUOUS,
                            name = f'Agent {agent.id} proba {proba} real-time sale')

            model.update()

    @staticmethod
    def energy_trading_var(agent, agents, model):

        for agent_2 in agents:
            if agent.connections[agent_2.id]:
                    model.addVar(lb = - agent.kappa[agent_2.id],
                                ub = agent.kappa[agent_2.id],
                                vtype = gp.GRB.CONTINUOUS,
                                name = f'q_{agent.id}_{agent_2.id}')

        model.update()

    @staticmethod
    def bilateral_trading_constraint(agent, agents, model):
        for agent_2 in agents:
            if agent.connections[agent_2.id]:
                model.addConstr(model.getVarByName(f'q_{agent.id}_{agent_2.id}')  
                                + model.getVarByName(f'q_{agent_2.id}_{agent.id}') == 0, 
                                name = f'Bilateral trading for pair ({agent.id}, {agent_2.id})')

        model.update()

    @staticmethod
    def trading_sum_calc(agent, agents, model, weights = False):
        lExpr = gp.LinExpr()

        if weights:
            for agent_2 in agents:
                if agent.connections[agent_2.id]:
                    lExpr.add(model.getVarByName(f'q_{agent.id}_{agent_2.id}') * agent.trading_cost[agent_2.id])

        else:
            for agent_2 in agents:
                if agent.connections[agent_2.id]:
                    lExpr.add(model.getVarByName(f'q_{agent.id}_{agent_2.id}'))

        return lExpr

    @staticmethod
    def net_trading(agent,model):
        model.addVar(lb = - float('inf'),
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'Agent {agent.id} net trading')

        model.update()
        
    @staticmethod
    def net_trading_constraint(agent, agents, model):
        model.addConstr(model.getVarByName(f'Agent {agent.id} net trading') ==  FirstStageModel.trading_sum_calc(agent, agents, model, weights=False), name = 'Net trading constraint')

    @staticmethod
    def balance_constraint(agent, agents, model):

        for proba, proba_val in enumerate(agent.probabilities):
            if proba_val > 0:
                model.addConstr(agent.demand
                                - agent.generation[proba]
                                - model.getVarByName(f'Agent {agent.id} day-ahead purchase')
                                + model.getVarByName(f'Agent {agent.id} day-ahead sale')
                                - model.getVarByName(f'Agent {agent.id} proba {proba} real-time purchase')
                                + model.getVarByName(f'Agent {agent.id} proba {proba} real-time sale')
                                - model.getVarByName(f'Agent {agent.id} net trading') == 0,
                                name= f'SD balance for agent {agent.id} proba {proba}')

        model.update()

    @staticmethod
    def set_objective(agent, price_da_buy, price_da_sell, price_rt_buy, price_rt_sell, model):
        lExpr = gp.LinExpr()

        for proba, proba_val in enumerate(agent.probabilities):
            if proba_val > 0:
                lExpr.add(proba_val * model.getVarByName(f'Agent {agent.id} proba {proba} real-time purchase') * price_rt_buy
                        - proba_val * model.getVarByName(f'Agent {agent.id} proba {proba} real-time sale') * price_rt_sell)

        lExpr.add(model.getVarByName(f'Agent {agent.id} day-ahead purchase') * price_da_buy
                - model.getVarByName(f'Agent {agent.id} day-ahead sale') * price_da_sell)

        lExpr.add(model.getVarByName(f'Agent {agent.id} net trading') * (price_da_buy - 0.1))

        return lExpr


class SecondStageModelDeterministic:
    @staticmethod
    def da_purchase(agent, model):

        model.addVar(lb = 0,
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'Agent {agent.id} day-ahead purchase')

        model.update()

    @staticmethod
    def da_sale(agent, model):

        model.addVar(lb = 0,
                    ub = 10, #float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'Agent {agent.id} day-ahead sale')

        model.update()

    def da_purchase(agent, model):

        model.addVar(lb = 0,
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'Agent {agent.id} real-time purchase')

        model.update()

    @staticmethod
    def da_sale(agent, model):

        model.addVar(lb = 0,
                    ub = 10, #float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = f'Agent {agent.id} real-time sale')

        model.update()
    
    @staticmethod
    def energy_trading_var(agent, agents, model):

        for agent_2 in agents:
            if agent.connections[agent_2.id]:
                    model.addVar(lb = - agent.kappa[agent_2.id],
                                ub = agent.kappa[agent_2.id],
                                vtype = gp.GRB.CONTINUOUS,
                                name = f'q_{agent.id}_{agent_2.id}')

        model.update()

    @staticmethod
    def bilateral_trading_constraint(agent, agents, model):
        for agent_2 in agents:
            if agent.connections[agent_2.id]:
                model.addConstr(model.getVarByName(f'q_{agent.id}_{agent_2.id}')  
                                + model.getVarByName(f'q_{agent_2.id}_{agent.id}') == 0, 
                                name = f'Bilateral trading for pair ({agent.id}, {agent_2.id})')

        model.update()


class FirstStageMarket:
    def __init__(self, agents, model,
                price_da_buy, price_da_sell, price_rt_buy, price_rt_sell) -> None:

        self.agents = agents
        self.model = model
        self.price_da_buy = price_da_buy
        self.price_da_sell = price_da_sell
        self.price_rt_buy = price_rt_buy
        self.price_rt_sell = price_rt_sell

    def build_model(self):
        obj = gp.LinExpr()

        for agent in self.agents:
            FirstStageModel.da_purchase(agent, self.model)
            FirstStageModel.da_sale(agent, self.model)
            FirstStageModel.rt_purchase(agent, self.model)
            FirstStageModel.rt_sale(agent, self.model)
            FirstStageModel.energy_trading_var(agent, self.agents, self.model)
            FirstStageModel.net_trading(agent, self.model)

        for agent in self.agents:
            FirstStageModel.net_trading_constraint(agent, self.agents, self.model)
            FirstStageModel.balance_constraint(agent, self.agents, self.model)
            FirstStageModel.bilateral_trading_constraint(agent, self.agents, self.model)

        for agent in self.agents:
            agent.objective = FirstStageModel.set_objective(agent, self.price_da_buy, self.price_da_sell, self.price_rt_buy, self.price_rt_sell, self.model)
            obj.add(agent.objective)

        self.model.setObjective(obj, gp.GRB.MINIMIZE)

class Agents:
    def __init__(self, id, probabilities, generation_values, demand, connections, kappa) -> None:
        self.id = id
        self.probabilities = probabilities
        self.generation = generation_values
        self.demand = demand
        self.connections = connections
        self.kappa = np.ma.MaskedArray(kappa, 
                                    mask = np.logical_not(self.connections), 
                                    fill_value = 0).filled() 

        
