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

        # Multiples of grid to allow for constraining the layout to a grid
        self.lg = model.addVars(n, vtype=GRB.INTEGER, name='LG')
        self.tg = model.addVars(n, vtype=GRB.INTEGER, name='TG')
        self.wg = model.addVars(n, vtype=GRB.INTEGER, name='WG')
        self.hg = model.addVars(n, vtype=GRB.INTEGER, name='HG')


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

        # Use the current layout as the starting solution
        '''
        for i, element in enumerate(layout.elements):
            if element.x > 0:
                self.l[i].start = element.x
            if element.y > 0:
                self.t[i].start = element.y
            if element.width > 0:
                self.w[i].start = element.width
            if element.height > 0:
                self.h[i].start = element.height
        '''