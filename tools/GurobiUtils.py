from gurobipy import Model, GRB

listOfVars = []

def define_1d_int_var_array(gurobi_model: Model, n : int, label: str):
    return gurobi_model.addVars(n, vtype=GRB.INTEGER, name=label)

def define_2d_bool_var_array_array(gurobi_model, size_x, size_y, name):
    return gurobi_model.addVars(size_x, size_y, vtype=GRB.BINARY, name=name)

def define_1d_bool_var_array(gurobiModel: Model, n : int, label: str):
    return gurobiModel.addVars(n, vtype=GRB.BINARY, name=label)