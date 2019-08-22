from gurobipy import *

class Variables:
    def __init__(self, model):

        layout: DataInstance = model._layout
        n = layout.n  # EXPL: Number of elements

        self.L = model.addVars(n, vtype=GRB.INTEGER, name='L')
        self.R = model.addVars(n, vtype=GRB.INTEGER, name='R')
        self.T = model.addVars(n, vtype=GRB.INTEGER, name='T')
        self.B = model.addVars(n, vtype=GRB.INTEGER, name='B')
        self.H = model.addVars(n, vtype=GRB.INTEGER, name='H')
        self.W = model.addVars(n, vtype=GRB.INTEGER, name='W')

        self.ABOVE = model.addVars(n, n, vtype=GRB.BINARY, name='ABOVE')  # EXPL: one elem is above the other
        self.LEFT = model.addVars(n, n, vtype=GRB.BINARY,
                                  name='LEFT')  # EXPL: one elem is to the left of the other

        self.LAG = model.addVars(n, vtype=GRB.BINARY,
                                 name='LAG')  # EXPL: left alignment group enabled?
        self.RAG = model.addVars(n, vtype=GRB.BINARY,
                                 name='RAG')  # EXPL: right aligment group enabled?
        self.TAG = model.addVars(n, vtype=GRB.BINARY,
                                 name='TAG')  # EXPL: top alignment group enabled?
        self.BAG = model.addVars(n, vtype=GRB.BINARY,
                                 name='BAG')  # EXPL: bottom alignment group enabled?

        self.vLAG = model.addVars(n, vtype=GRB.INTEGER, name='vLAG')  # EXPL: ???
        self.vRAG = model.addVars(n, vtype=GRB.INTEGER, name='vRAG')
        self.vTAG = model.addVars(n, vtype=GRB.INTEGER, name='vTAG')
        self.vBAG = model.addVars(n, vtype=GRB.INTEGER, name='vBAG')

        self.elemAtLAG = model.addVars(n, n, vtype=GRB.BINARY, name='zLAG')
        self.elemAtRAG = model.addVars(n, n, vtype=GRB.BINARY, name='zRAG')
        self.elemAtTAG = model.addVars(n, n, vtype=GRB.BINARY, name='zTAG')
        self.elemAtBAG = model.addVars(n, n, vtype=GRB.BINARY, name='zBAG')