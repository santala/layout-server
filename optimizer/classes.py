from enum import Enum
from operator import itemgetter

class Edge(Enum):
    NONE = None
    TOP = 'top'
    RIGHT = 'right'
    BOTTOM = 'bottom'
    LEFT = 'left'

class Layout:
    def __init__(self, props: dict):
        # TODO: format validation


        self.id = str(props.get('id'))
        self.canvas_width = props.get('canvasWidth', 0)
        self.canvas_height = props.get('canvasHeight', 0)
        self.canvas_aspect_ratio = self.canvas_width / self.canvas_height
        self.canvas_area = self.canvas_width * self.canvas_height

        # If this element contains children, align them into a grid
        # By default, top level elements are aligned in a grid
        self.enable_grid = bool(props.get('enableGrid', True))

        self.element_list = [
            Element(element_props, self)
            for element_props in props.get('elements', [])
        ]

        self.element_dict = {
            element.id: element
            for element in self.element_list
        }


        for element in self.element_list:
            parents = [other for other in self.element_list if element is not other and element.is_contained_within(other)]
            if len(parents) == 1:
                element.parent_id = parents[0].id
            elif len(parents) > 1:
                # If there are multiple containing elements, pick the smallest as the parent
                element.parent_id = min(parents, itemgetter('area')).id

        self.n = len(self.element_list)


        # The following are for the layout difference algorithm
        # TODO: consider making this a separate method

        self.x_sum = sum([abs(element.x0) for element in self.element_list])
        self.y_sum = sum([abs(element.y0) for element in self.element_list])
        self.w_sum = sum([abs(element.width) for element in self.element_list])
        self.h_sum = sum([abs(element.height) for element in self.element_list])
        self.area_sum = sum([element.area for element in self.element_list])





class Element:
    def __init__(self, props: dict, layout: Layout):

        self.layout = layout

        self.id = str(props.get('id'))
        self.element_type = props.get('type', None)
        self.component_name = props.get('componentName', '?')

        self.x0 = int(props.get('x', 0))
        self.y0 = int(props.get('y', 0))
        self.width = props.get('width', 1)   # TODO: choose default number values and/or validate input
        self.height = props.get('height', 1)
        self.x1 = self.x0 + self.width
        self.y1 = self.y0 + self.height
        self.area = self.width * self.height

        # If this element contains children, align them into a grid
        # By default, elements contained within other elements are not aligned in a grid
        self.enable_grid = bool(props.get('enableGrid', False))

        self.constrain_left = bool(props.get('constrainLeft', False))
        self.constrain_right = bool(props.get('constrainRight', False))
        self.constrain_top = bool(props.get('constrainTop', False))
        self.constrain_bottom = bool(props.get('constrainBottom', False))
        self.constrain_width = bool(props.get('constrainWidth', False))
        self.constrain_height = bool(props.get('constrainHeight', False))

        self.isLocked = bool(props.get('isLocked', False))

        self.parent_id = None

        self.snap_to_edge = Edge(props.get('snapToEdge', None))
        self.snap_priority = int(props.get('snapPriority', 1))
        self.snap_margin = int(props.get('snapMargin', 0))

        # TODO: move these to client side
        if self.snap_to_edge is None and False:
            if 'ABB Stripe' in self.component_name:
                self.snap_to_edge = Edge.TOP
                self.snap_priority = 1
            elif 'Collapsible' in self.component_name:
                self.snap_to_edge = Edge.LEFT
                self.snap_priority = 2



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