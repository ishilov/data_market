import gurobipy as gp
import numpy as np

class TSOModel:
    
    @staticmethod
    def da_purchase(model):

        model.addVar(lb = 0.01,
                    #ub=0,
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = 'Price for day-ahead purchase')

        model.update()

    @staticmethod
    def da_sell(model):

        model.addVar(lb = 0.01,
                    #ub=0,
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = 'Price for day-ahead sell')

        model.update()

    @staticmethod
    def rt_purchase(model):

        model.addVar(lb = 0.01,
                    #ub=0,
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = 'Price for real-time purchase')

        model.update()

    @staticmethod
    def rt_sell(model):

        model.addVar(lb = 0.01,
                    #ub=0,
                    ub = float('inf'),
                    vtype = gp.GRB.CONTINUOUS,
                    name = 'Price for real-time sell')

        model.update()

    @staticmethod
    def constraints_c(model, c_da_b, c_da_s, c_rt_b, c_rt_s):
        model.addConstr(model.getVarByName('Price for day-ahead purchase') >= c_da_b)
        model.addConstr(model.getVarByName('Price for day-ahead sell') <= c_da_s)
        model.addConstr(model.getVarByName('Price for real-time purchase') >= c_rt_b)
        model.addConstr(model.getVarByName('Price for real-time sell') <= c_rt_s)

        model.update()

    @staticmethod
    def constraints_b(model):
        model.addConstr(model.getVarByName('Price for real-time purchase') - model.getVarByName('Price for day-ahead purchase') >= 0.01)
        model.addConstr(model.getVarByName('Price for day-ahead purchase') - model.getVarByName('Price for day-ahead sell') >= 0.01)
        model.addConstr(model.getVarByName('Price for day-ahead sell') - model.getVarByName('Price for real-time sell') >= 0.01)

        model.update()

    @staticmethod
    def set_objective(model, G_da, G_rt, gamma, c_da_b, c_da_s, c_rt_b, c_rt_s):
        lExpr = gp.LinExpr()

        for G_da_k in G_da:
            if G_da_k >= 0:
                lExpr.add(G_da_k * (model.getVarByName('Price for day-ahead purchase') + c_da_b))
            else:
                lExpr.add(- G_da_k * (model.getVarByName('Price for day-ahead sell') + c_da_s))

        for G_rt_k in G_rt:
            if G_rt_k >= 0:
                lExpr.add(G_rt_k * (model.getVarByName('Price for real-time purchase') + c_rt_b))
            else:
                lExpr.add(- G_rt_k * (model.getVarByName('Price for real-time sell') + c_rt_s))

        for k in range(len(G_da)):
            model.addVar(lb = 0,
                #ub=0,
                ub = float('inf'),
                vtype = gp.GRB.CONTINUOUS,
                name = f'Absolute value for community {k}')
        
            model.update()
            
            model.addConstr(model.getVarByName(f'Absolute value for community {k}') >= (G_da[k] + G_rt[k]))
            model.addConstr(model.getVarByName(f'Absolute value for community {k}') >= -(G_da[k] + G_rt[k]))

            lExpr.add(gamma * model.getVarByName(f'Absolute value for community {k}'))

        return lExpr

class TSOOptimization:
    def __init__(self, model, G_da, G_rt, gamma, c_da_b, c_da_s, c_rt_b, c_rt_s) -> None:
        self.model = model
        self.g_da = G_da
        self.g_rt = G_rt
        self.gamma = gamma
        self.c_da_b = c_da_b
        self.c_da_s = c_da_s
        self.c_rt_b = c_rt_b
        self.c_rt_s = c_rt_s

    def build_model(self):
        TSOModel.da_purchase(self.model)
        TSOModel.da_sell(self.model)
        TSOModel.rt_purchase(self.model)
        TSOModel.rt_sell(self.model)

        TSOModel.constraints_c(self.model, self.c_da_b, self.c_da_s, self.c_rt_b, self.c_rt_s)
        TSOModel.constraints_b(self.model)

        obj = TSOModel.set_objective(self.model, self.g_da, self.g_rt, self.gamma, self.c_da_b, self.c_da_s, self.c_rt_b, self.c_rt_s)

        self.model.setObjective(obj, gp.GRB.MINIMIZE)

        self.model.update()



        
