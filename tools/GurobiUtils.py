from gurobipy import GRB, Model, tupledict

class Variables:
    def __init__(self, model: Model):

        layout: DataInstance = model._layout
        n = layout.n  # EXPL: Number of elements

        self.l = model.addVars(n, vtype=GRB.INTEGER, name='L')
        self.r = model.addVars(n, vtype=GRB.INTEGER, name='R')
        self.t = model.addVars(n, vtype=GRB.INTEGER, name='T')
        self.b = model.addVars(n, vtype=GRB.INTEGER, name='B')
        self.h = model.addVars(n, vtype=GRB.INTEGER, name='H')
        self.w = model.addVars(n, vtype=GRB.INTEGER, name='W')

        self.above = model.addVars(n, n, vtype=GRB.BINARY, name='ABOVE')  # EXPL: one elem is above the other
        self.on_left = model.addVars(n, n, vtype=GRB.BINARY,
                                     name='LEFT')  # EXPL: one elem is to the left of the other

        self.lag = model.addVars(n, vtype=GRB.BINARY,
                                 name='LAG')  # EXPL: left alignment group enabled?
        self.rag = model.addVars(n, vtype=GRB.BINARY,
                                 name='RAG')  # EXPL: right aligment group enabled?
        self.tag = model.addVars(n, vtype=GRB.BINARY,
                                 name='TAG')  # EXPL: top alignment group enabled?
        self.bag = model.addVars(n, vtype=GRB.BINARY,
                                 name='BAG')  # EXPL: bottom alignment group enabled?

        self.v_lag = model.addVars(n, vtype=GRB.INTEGER, name='vLAG')  # EXPL: ???
        self.v_rag = model.addVars(n, vtype=GRB.INTEGER, name='vRAG')
        self.v_tag = model.addVars(n, vtype=GRB.INTEGER, name='vTAG')
        self.v_bag = model.addVars(n, vtype=GRB.INTEGER, name='vBAG')

        self.at_lag = model.addVars(n, n, vtype=GRB.BINARY, name='zLAG')
        self.at_rag = model.addVars(n, n, vtype=GRB.BINARY, name='zRAG')
        self.at_tag = model.addVars(n, n, vtype=GRB.BINARY, name='zTAG')
        self.at_bag = model.addVars(n, n, vtype=GRB.BINARY, name='zBAG')