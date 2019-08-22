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
    def __init__(self, props=None):

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

        if props is not None:
            props = props.get("layouts")[0]  # TODO: edit the format

            self.canvas_width = props.get('canvasWidth')
            self.canvas_height = props.get('canvasHeight')
            self.id = str(props.get('id'))

            for element_dict in props.get('elements'):
                element = Element()
                element.id = element_dict.get('id')
                element.x = element_dict.get('x')
                element.y = element_dict.get('y')
                element.width = element_dict.get('width')
                element.height = element_dict.get('height')
                self.elements.append(element)

            self.n = len(self.elements)