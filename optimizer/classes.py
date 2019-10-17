from operator import itemgetter

class Layout:
    def __init__(self, props: dict):
        # TODO: format validation


        self.id = str(props.get('id'))
        self.canvas_width = props.get('canvasWidth', 0)
        self.canvas_height = props.get('canvasHeight', 0)
        self.canvas_aspect_ratio = self.canvas_width / self.canvas_height
        self.canvas_area = self.canvas_width * self.canvas_height

        self.solution_count = props.get('NumOfSolutions', None)

        self.elements = [
            Element(element_props, self) for element_props in props.get('elements', [])
        ]

        for element in self.elements:
            parents = [other for other in self.elements if element is not other and element.is_contained_within(other)]
            print('parents',element.id, [p.id for p in parents])
            if len(parents) == 1:
                element.parent = parents[0]
            elif len(parents) > 1:
                # If there are multiple containing elements, pick the smallest as the parent
                element.parent = min(parents, itemgetter('area'))

        self.n = len(self.elements)

        # The following are for the layout difference algorithm
        # TODO: consider making this a separate method

        self.x_sum = sum([abs(element.x0) for element in self.elements])
        self.y_sum = sum([abs(element.y0) for element in self.elements])
        self.w_sum = sum([abs(element.width) for element in self.elements])
        self.h_sum = sum([abs(element.height) for element in self.elements])
        self.area_sum = sum([element.area for element in self.elements])



class Element:
    def __init__(self, props: dict, layout: Layout):

        self.layout = layout

        self.id = str(props.get('id'))
        self.elementType = props.get('type', None)
        self.componentName = props.get('componentName', '?')

        self.x0 = int(props.get('x', 0))
        self.y0 = int(props.get('y', 0))
        self.width = props.get('width', 1)   # TODO: choose default number values and/or validate input
        self.height = props.get('height', 1)
        self.x1 = self.x0 + self.width
        self.y1 = self.y0 + self.height
        self.area = self.width * self.height


        self.constrainLeft = bool(props.get('constrainLeft', False))
        self.constrainRight = bool(props.get('constrainRight', False))
        self.constrainTop = bool(props.get('constrainTop', False))
        self.constrainBottom = bool(props.get('constrainBottom', False))
        self.constrainWidth = bool(props.get('constrainWidth', False))
        self.constrainHeight = bool(props.get('constrainHeight', False))

        self.isLocked = bool(props.get('isLocked', False))

        self.parent = None



    def overlap_width(self, other):
        return (self.width + other.width) - (max(self.x0 + self.width, other.x0 + other.width) - min(self.x0, other.x0))

    def overlap_height(self, other):
        return (self.height + other.height) - (max(self.y0 + self.height, other.y0 + other.height) - min(self.y0, other.y0))

    def overlap_area(self, other):
        return self.overlap_width(other) * self.overlap_height(other)

    def do_overlap(self, other):
        return self.overlap_width(other) > 0 and self.overlap_height(other) > 0

    def is_contained_within(self, other):
        return self.overlap_area(other) == self.area

    def is_above(self, other):
        return self.y1 <= other.y0

    def is_on_left(self, other):
        return self.x1 <= other.x0