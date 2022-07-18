import gurobipy as gp

class Centralized:

    @staticmethod
    def lt_var(model, probabilities):

        lt_purchase = model.addVar(lb = 0,
                        ub = float('inf'),
                        vtype = gp.GRB.CONTINUOUS,
                        name = 'long term purchase')

        model.update()

        return lt_purchase

    @staticmethod
    def im_var(model, probabilities):

        im_purchase = model.addVars(len(probabilities),
                        lb = 0,
                        ub = float('inf'),
                        vtype = gp.GRB.CONTINUOUS,
                        name = 'intermediate market purchase')

        model.update()

        return im_purchase

    @staticmethod
    def rt_var(model, probabilities):

        rt_purchase = model.addVars(len(probabilities),
                        lb = 0,
                        ub = float('inf'),
                        vtype = gp.GRB.CONTINUOUS,
                        name = 'real time purchase')

        model.update()

        return rt_purchase

    @staticmethod
    def two_stage_objective(model, lt_purchase, rt_purchase, probabilities, p_lt : float, p_rt : float):
        #rewrite it with scalar product and quicksum

        objective = gp.LinExpr()
        objective.add(lt_purchase * p_lt)

        for ind, proba in enumerate(probabilities):
            objective.add(proba * (rt_purchase[ind] * p_rt))

        return objective

    @staticmethod
    def balance_constraint(model, lt_purchase, rt_purchase, generation, demand):
        
        for ind, gen_value in enumerate(generation):
            model.addConstr(lt_purchase + rt_purchase[ind] + gen_value >= demand)

        model.update()

class GurobiSolution:
    #rewrite in generic form
    def __init__(self, model, probabilities, generation) -> None:
        self.model = model
        self.probabilities = probabilities
        self.generation = generation

    def build_centralized_2stage_model(self, demand, p_lt, p_rt):
        lt_purchase = Centralized.lt_var(self.model, self.probabilities)
        rt_purchase = Centralized.rt_var(self.model, self.probabilities)

        Centralized.balance_constraint(self.model, lt_purchase, rt_purchase, self.generation, demand)

        objective = Centralized.two_stage_objective(self.model, lt_purchase, rt_purchase, self.probabilities, p_lt, p_rt)

        self.model.setObjective(objective, gp.GRB.MINIMIZE)

        self.model.update()
