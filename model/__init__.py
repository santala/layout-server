class Element:
    def __init__(self):
        self.width  = None
        self.height = None
        self.X = None
        self.Y = None
        self.id = None
        self.area = None
        self.PenaltyIfSkipped = None
    

class Layout:
    def __init__(self):
        self.canvasWidth = None
        self.canvasHeight = None
        self.elements = [] 
        self.id = None
        self.N = None
        self.Xsum = 0
        self.Ysum = 0
        self.Wsum = 0
        self.Hsum = 0
        self.AreaSum = 0