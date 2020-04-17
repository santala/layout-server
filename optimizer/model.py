from functools import lru_cache
from typing import Dict
from itertools import permutations, product
from gurobi import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError


class LayoutProps:
    def __init__(self, props: dict):
        self.id = str(props.get('id'), '')
        self.width = float(props.get('canvasWidth', 0))
        self.height = float(props.get('canvasHeight', 0))


class ElementProps:
    def __init__(self, props: dict):
        self.id = str(props.get('id'))
        self.element_type = props.get('type', None)
        self.component_name = props.get('componentName', '?')
        self.x0 = float(props.get('x', 0))
        self.y0 = float(props.get('y', 0))
        self.w = float(props.get('width', 0))
        self.h = float(props.get('height', 0))
        self.x1 = self.x0 + self.w
        self.y1 = self.y0 + self.h


class Layout:
    def __init__(self, m: Model, props: LayoutProps, fixed_width: bool = True, fixed_height: bool = True):
        self.m = m
        self.initial = props

        self.w = m.addVar(lb=1, vtype=GRB.INTEGER)
        self.h = m.addVar(lb=1, vtype=GRB.INTEGER)


class Element:
    def __init__(self, m: Model, props: ElementProps):
        self.m = m
        self.initial = props

        # Variables

        self.x0 = m.addVar(vtype=GRB.INTEGER)
        self.y0 = m.addVar(vtype=GRB.INTEGER)
        self.x1 = m.addVar(vtype=GRB.INTEGER)
        self.y1 = m.addVar(vtype=GRB.INTEGER)

        self.w = m.addVar(vtype=GRB.INTEGER)
        m.addConstr(self.x1 - self.x0 == self.w)
        self.h = m.addVar(vtype=GRB.INTEGER)
        m.addConstr(self.y1 - self.y0 == self.h)

    @lru_cache
    def is_left_of(self, other):
        return self.initial.x1 <= other.props.x0

    @lru_cache
    def is_above(self, other):
        return self.initial.y1 <= other.props.y0

    @lru_cache
    def is_content(self):
        return self.parent() is not None

    @lru_cache
    def parent(self):
        potential_parents = [other for other in self.m._elements if other is not self and self.is_within(other)]
        if len(potential_parents) == 0:
            return None
        else:
            # If there are multiple containing elements, pick the one with smallest area
            return min(potential_parents, key=lambda e: e.initial.w * e.initial.h)

    @lru_cache
    def is_within(self, other):
        return self.initial.x0 > other.initial.x0 and self.initial.y0 > other.initial.y0 \
               and self.initial.x1 < other.initial.x1 and self.y1 < other.initial.y1


def solve(layout_props: dict, time_out: int = 30):

    m = Model("DesignSystem")

    m.Params.MIPFocus = 1
    m.Params.TimeLimit = time_out

    m.Params.OutputFlag = 1

    layout = Layout(m, layout_props)
    m._layout = layout
    elements = [Element(m, layout, element_props) for element_props in layout_props.get('elements', [])]
    m._elements = elements

    for element, other in permutations(elements, 2):

        if element.is_content() and element.parent() == other.parent():
            # Maintain relationships
            if element.is_left_of(other):
                m.addConstr(element.x1 <= other.x0)
            if element.is_above(other):
                m.addConstr(element.y1 <= other.y0)

