from enum import Enum
from math import sqrt
from operator import attrgetter


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
        self.canvas_width = float(props.get('canvasWidth', 0))
        self.canvas_height = float(props.get('canvasHeight', 0))
        self.canvas_aspect_ratio = self.canvas_width / self.canvas_height
        self.canvas_area = self.canvas_width * self.canvas_height

        self.x0 = 0
        self.y0 = 0
        self.x1 = self.canvas_width
        self.y1 = self.canvas_height

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


        # TODO: prevent elements from being each others parents
        # EXPL: this will happen if elements have identical size and position

        for element in self.element_list:
            parents = [other for other in self.element_list if element is not other and element.is_contained_within(other)]
            if len(parents) == 1:
                element.parent = parents[0]
                #element.parent_id = parents[0].id
            elif len(parents) > 1:
                # If there are multiple containing elements, pick the smallest as the parent
                element.parent = min(parents, key=attrgetter('area'))
                #element.parent_id = min(parents, key=attrgetter('area')).id
            else:
                element.parent = self # Define the layout as the parent
                #element.parent_id = self.id

        self.n = len(self.element_list)

        self.depth = max([e.get_ancestor_count() for e in self.element_list])


        # The following are for the layout difference algorithm
        # TODO: consider making this a separate method

        self.x_sum = sum([abs(element.x0) for element in self.element_list])
        self.y_sum = sum([abs(element.y0) for element in self.element_list])
        self.w_sum = sum([abs(element.width) for element in self.element_list])
        self.h_sum = sum([abs(element.height) for element in self.element_list])
        self.area_sum = sum([element.area for element in self.element_list])


    def get_ancestor_count(self):
        return 0


class Element:
    def __init__(self, props: dict, layout: Layout):

        self.layout = layout

        self.id = str(props.get('id'))
        self.element_type = props.get('type', None)
        self.component_name = props.get('componentName', '?')

        self.x0 = round(float(props.get('x', 0)))
        self.y0 = round(float(props.get('y', 0)))
        self.width = round(float(props.get('width', 1)))   # TODO: choose default number values and/or validate input
        self.height = round(float(props.get('height', 1)))
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

        self.parent = None
        #self.parent_id = None

        self.snap_to_edge = Edge(props.get('snapToEdge', None))
        self.snap_priority = int(props.get('snapPriority', 1))
        self.snap_margin = int(props.get('snapMargin', 0))

        # TODO: move these to client side
        if self.snap_to_edge is Edge.NONE:
            if 'ABB Stripe' in self.component_name:
                self.snap_to_edge = Edge.TOP
                self.snap_priority = 1
            elif 'Collapsible' in self.component_name or 'Left pane' in self.component_name:
                self.snap_to_edge = Edge.LEFT
                self.snap_priority = 3
            elif 'Menu Bar' in self.component_name:
                self.snap_to_edge = Edge.TOP
                self.snap_priority = 2

        # TODO: these should be on client side as well
        # TODO: base unit width should come from client side as well
        self.fixed_width = None
        self.fixed_height = None

        if self.element_type == 'component':
            print(self.component_name)
            if 'Mobile Menu Bar' in self.component_name:
                self.fixed_height = 6
            elif 'Menu Bar' in self.component_name:
                print('!!!')
                self.fixed_height = 5
            elif 'ABB Stripe' in self.component_name:
                self.fixed_height = 4
            elif 'Select-32' in self.component_name:
                self.fixed_height = 9
            elif 'Slider' in self.component_name:
                self.fixed_height = 8
            elif 'H1' in self.component_name:
                self.fixed_height = 6
            elif 'Left pane / Compact mode' in self.component_name:
                self.fixed_width = 12

    def get_parent_id(self):
        if self.parent is None:
            return None
        else:
            return self.parent.id

    def get_ancestor_count(self):
        if self.parent is None:
            return 0
        else:
            return 1 + self.parent.get_ancestor_count()

    def overlap_width(self, other):
        return (self.width + other.width) - (max(self.x1, other.x1) - min(self.x0, other.x0))

    def overlap_height(self, other):
        return (self.height + other.height) - (max(self.y1, other.y1) - min(self.y0, other.y0))

    def overlap_area(self, other):
        return self.overlap_width(other) * self.overlap_height(other)

    def does_overlap(self, other):
        return self.overlap_width(other) > 0 and self.overlap_height(other) > 0

    def is_contained_within(self, other):
        return self.does_overlap(other) and self.overlap_area(other) == self.area

    def is_above(self, other):
        return self.y1 <= other.y0

    def is_on_left(self, other):
        return self.x1 <= other.x0

    def distance_to(self, other):
        if self.does_overlap(other):
            return 0
        else:
            h_dist = min(abs(self.x0 - other.x1), abs(self.x1 - other.x0))
            v_dist = min(abs(self.y0 - other.y1), abs(self.y1 - other.y0))

            if self.overlap_width(other) <= 0:
                return h_dist
            elif self.overlap_height(other) <= 0:
                return v_dist
            else:
                return sqrt(h_dist**2, v_dist**2)


    def is_adjacent(self, other):
        # Element is considered adjacent, if it can be on the same ‘row’ or ‘column’ and is ‘close enough’

        # Elements won’t be considered adjacent if they belong to different groups
        if self.get_parent_id() != other.get_parent_id():
            print('Different groups')
            return False

        if self.overlap_width(other) <= 0 and self.overlap_height(other):
            # The other element is not in the same row or column
            print('Not in same row or col')
            return False

        distance_to_other = self.distance_to(other)

        if self.overlap_width(other) > 0 and distance_to_other > max(self.height, other.height):
            print('Too far away (v)')
            return False
        elif self.overlap_height(other) > 0 and distance_to_other > max(self.width, other.width):
            print('Too far away (h)')
            return False

        closer_elements_in_same_group = [
            e for e in self.layout.element_list
            if e is not self and e.get_parent_id() == self.get_parent_id() and self.distance_to(e) < distance_to_other
        ]

        for e in closer_elements_in_same_group:
            if self.is_above(other) == self.is_above(e) or self.is_on_left(other) == self.is_on_left(e):
                # There is an element between self and other
                print('There is an element between self and other')
                print(self.id, other.id, e.id)
                return False

        return True