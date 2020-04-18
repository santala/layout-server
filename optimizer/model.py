import sys
import math
from functools import lru_cache
from typing import Dict, List
from itertools import permutations, product
from gurobi import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError, Var


class LayoutProps:
    def __init__(self, props: dict):
        self.id = str(props.get('id'))
        self.w = float(props.get('canvasWidth', 0))
        self.h = float(props.get('canvasHeight', 0))


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

        self.w = m.addVar(lb=1, vtype=GRB.CONTINUOUS)
        self.h = m.addVar(lb=1, vtype=GRB.CONTINUOUS)


class Element:
    def __init__(self, m: Model, props: ElementProps):
        self.m = m
        self.initial = props

        # Variables

        self.x0 = m.addVar(vtype=GRB.CONTINUOUS)
        self.y0 = m.addVar(vtype=GRB.CONTINUOUS)
        self.x1 = m.addVar(vtype=GRB.CONTINUOUS)
        self.y1 = m.addVar(vtype=GRB.CONTINUOUS)

        self.w = m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(self.x1 - self.x0 == self.w)
        self.h = m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(self.y1 - self.y0 == self.h)

    @lru_cache(maxsize=None)
    def is_left_of(self, other):
        return self.initial.x1 <= other.initial.x0

    @lru_cache(maxsize=None)
    def is_above(self, other):
        return self.initial.y1 <= other.initial.y0

    @lru_cache(maxsize=None)
    def is_content(self):
        return self.parent() is not None

    @lru_cache(maxsize=None)
    def parent(self):
        potential_parents = [other for other in self.m._elements if other is not self and self.is_within(other)]
        if len(potential_parents) == 0:
            return None
        else:
            # If there are multiple containing elements, pick the one with smallest area
            return min(potential_parents, key=lambda e: e.initial.w * e.initial.h)

    @lru_cache(maxsize=None)
    def is_within(self, other):
        return self.initial.x0 > other.initial.x0 and self.initial.y0 > other.initial.y0 \
               and self.initial.x1 < other.initial.x1 and self.initial.y1 < other.initial.y1


def solve(layout_dict: dict, time_out: int = 30):

    m = Model("DesignSystem")

    m.Params.OutputFlag = 1

    m.Params.TimeLimit = time_out
    m.Params.MIPFocus = 1
    m.Params.ModelSense = GRB.MINIMIZE
    m.Params.Presolve = -1  # -1=auto, 0=off, 1=conservative, 2=aggressive

    layout = Layout(m, LayoutProps(layout_dict))
    m._layout = layout
    elements = [Element(m, ElementProps(element_dict)) for element_dict in layout_dict.get('elements', [])]
    m._elements = elements

    # Fix layout size
    m.addConstr(layout.w == layout.initial.w)
    m.addConstr(layout.h == layout.initial.h)

    # Match the internal layout of identical groups
    #

    for element, other in permutations(elements, 2):
        if element.is_content() and element.parent() == other.parent():
            # Maintain relationships
            if element.is_left_of(other):
                m.addConstr(element.x1 <= other.x0)
            if element.is_above(other):
                m.addConstr(element.y1 <= other.y0)

    maintain_relationships(m, elements)
    apply_horizontal_grid(m, elements, 0, layout.w, 24, 16, 8)
    apply_vertical_baseline(m, elements, 8)

    size_loss = get_size_loss(m, elements)

    # Minimize downscaling of elements
    m.setObjective(size_loss)

    try:
        m.optimize()

        if m.Status in [GRB.Status.OPTIMAL, GRB.Status.INTERRUPTED, GRB.Status.TIME_LIMIT]:

            layouts = []

            for s in range(m.SolCount):
                m.Params.SolutionNumber = s

                element_props = []

                for element in elements:
                    element_props.append({
                        'id': element.initial.id,
                        'x': element.x0.X,
                        'y': element.y0.X,
                        'width': element.w.X,
                        'height': element.h.X,
                    })

                layouts.append({
                    'solutionNumber': s,
                    'canvasWidth': layout.w.X,
                    'canvasHeight': layout.h.X,
                    'elements': element_props
                })

            try:
                print('Size loss', size_loss.getValue())
            except:
                e = sys.exc_info()[0]
                print(e)

            return {
                'status': 0,
                'layouts': layouts
            }
        else:
            if m.Status == GRB.Status.INFEASIBLE:
                m.computeIIS()
                #m.write("output/SimoPracticeModel.ilp")
            print('Non-optimal status:', m.Status)

            return {'status': 1}

    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ": " + str(e))
        raise e


def maintain_relationships(m: Model, elements: List[Element]):
    for element, other in permutations(elements, 2):
        if element.is_left_of(other):
            m.addConstr(element.x1 <= other.x0)
        if element.is_above(other):
            m.addConstr(element.y1 <= other.y0)


def get_size_loss(m: Model, elements: List[Element]):
    max_initial_width = max([element.initial.w for element in elements])
    max_initial_height = max([element.initial.h for element in elements])

    size_loss = LinExpr(0)

    for element in elements:
        # Smaller elements should be scaled down less
        element_width_loss = m.addVar(lb=0)
        m.addConstr(element_width_loss >= element.initial.w - element.w)
        size_loss.add(element_width_loss * max_initial_width / element.initial.w)

        element_height_loss = m.addVar(lb=0)
        m.addConstr(element_height_loss >= element.initial.h - element.h)
        size_loss.add(element_height_loss * max_initial_height / element.initial.h)

    return size_loss


def apply_horizontal_grid(m: Model, elements: List[Element], offset: Var, grid_width: Var, col_count: int, margin: float, gutter: float):

    gutter_count = col_count - 1
    col_width = m.addVar(vtype=GRB.CONTINUOUS)
    m.addConstr(grid_width == 2 * margin + col_count * col_width + gutter_count * gutter)

    # Ensure that the whole grid is used
    starting_in_first_col = LinExpr(0)
    ending_in_last_col = LinExpr(0)

    for i, element in enumerate(elements):

        col_start_flags = m.addVars(range(col_count), vtype=GRB.BINARY)
        m.addConstr(col_start_flags.sum() == 1)
        col_end_flags = m.addVars(range(col_count), vtype=GRB.BINARY)
        m.addConstr(col_end_flags.sum() == 1)

        starting_in_first_col.add(col_start_flags[0])
        ending_in_last_col.add(col_end_flags[col_count - 1])

        for col_index in range(col_count):
            col_x0 = LinExpr(offset + margin + col_index * (col_width + gutter))
            m.addConstr(col_start_flags[col_index] * element.x0 == col_start_flags[col_index] * col_x0)
            col_x1 = LinExpr(offset + margin + (col_index + 1) * (col_width + gutter) - gutter)
            m.addConstr(col_end_flags[col_index] * element.x1 == col_end_flags[col_index] * col_x1)

    m.addConstr(starting_in_first_col >= 1)
    m.addConstr(ending_in_last_col >= 1)


def apply_vertical_baseline(m: Model, elements: List[Element], baseline_height: int):

    for element in elements:
        start_line = m.addVar(vtype=GRB.INTEGER)
        end_line = m.addVar(vtype=GRB.INTEGER)
        m.addConstr(baseline_height * start_line == element.y0)
        m.addConstr(baseline_height * end_line == element.y1)