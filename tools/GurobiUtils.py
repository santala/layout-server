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
        '''
        self.lg = model.addVars(n, vtype=GRB.INTEGER, name='LG')
        self.tg = model.addVars(n, vtype=GRB.INTEGER, name='TG')
        self.wg = model.addVars(n, vtype=GRB.INTEGER, name='WG')
        self.hg = model.addVars(n, vtype=GRB.INTEGER, name='HG')
        '''

        # TODO: proper negative bounds
        self.resize_width = model.addVars(n, lb=-1000, vtype=GRB.INTEGER, name='resizeW')
        self.resize_width_abs = model.addVars(n, vtype=GRB.INTEGER, name='resizeWAbs')
        self.resize_height = model.addVars(n, lb=-1000, vtype=GRB.INTEGER, name='resizeH')
        self.resize_height_abs = model.addVars(n, vtype=GRB.INTEGER, name='resizeHAbs')

        self.move_x = model.addVars(n, lb=-1000, vtype=GRB.INTEGER, name='moveX')
        self.move_x_abs = model.addVars(n, vtype=GRB.INTEGER, name='moveXAbs')
        self.move_y = model.addVars(n, lb=-1000, vtype=GRB.INTEGER, name='moveY')
        self.move_y_abs = model.addVars(n, vtype=GRB.INTEGER, name='moveYAbs')

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

        self.v_lag = model.addVars(n, vtype=GRB.INTEGER, name='vLAG')  # EXPL: Pixel values for alignment groups?
        self.v_rag = model.addVars(n, vtype=GRB.INTEGER, name='vRAG')
        self.v_tag = model.addVars(n, vtype=GRB.INTEGER, name='vTAG')
        self.v_bag = model.addVars(n, vtype=GRB.INTEGER, name='vBAG')

        self.at_lag = model.addVars(n, n, vtype=GRB.BINARY, name='zLAG') # EXPL: Assignment matrix of elements to alignment groups
        self.at_rag = model.addVars(n, n, vtype=GRB.BINARY, name='zRAG')
        self.at_tag = model.addVars(n, n, vtype=GRB.BINARY, name='zTAG')
        self.at_bag = model.addVars(n, n, vtype=GRB.BINARY, name='zBAG')

        # Use the current layout as the starting solution

        for i, element in enumerate(layout.elements):
            if element.x > 0:
                self.l[i].Start = element.x / model._grid_size + 1
            if element.y > 0:
                self.t[i].Start = element.y / model._grid_size + 1
            if element.width > 0:
                self.w[i].Start = element.width / model._grid_size + 1
            if element.height > 0:
                self.h[i].Start = element.height / model._grid_size + 1
                ''''''
