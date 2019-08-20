class Element:
    def __init__(self):
        self.width = None
        self.height = None
        self.x = None
        self.y = None
        self.id = None
        self.area = None
        self.penalty_if_skipped = None


class Layout:
    def __init__(self, layout_dict=None):
        self.canvas_width = None
        self.canvas_height = None
        self.elements = []
        self.id = None
        self.n = None
        self.x_sum = 0
        self.y_sum = 0
        self.w_sum = 0
        self.h_sum = 0
        self.area_sum = 0

        if layout_dict is not None:
            self.canvas_width = layout_dict.get('canvasWidth')
            self.canvas_height = layout_dict.get('canvasHeight')
            self.id = layout_dict.get('id')

            for element_dict in layout_dict.get('elements'):
                element = Element()
                element.id = element_dict.get('id')
                element.x = element_dict.get('x')
                element.y = element_dict.get('y')
                element.width = element_dict.get('width')
                element.height = element_dict.get('height')
                self.elements.append(element)

            self.n = len(self.elements)