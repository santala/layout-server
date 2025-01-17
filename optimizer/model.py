import sys
import math
import time
from collections import namedtuple
from enum import Enum
from functools import lru_cache
from typing import List, Optional, Union, Tuple
from itertools import combinations, permutations
from gurobi import GRB, LinExpr, Model, max_, min_, GurobiError, Var


Margin = namedtuple('Margin', 'top right bottom left')
Padding = namedtuple('Margin', 'top right bottom left')
BBox = namedtuple('BBox', 'x0 y0 x1 y1')


class Alignment(Enum):
    CHROME_TOP = 'ChromeTop'
    CHROME_RIGHT = 'ChromeRight'
    CHROME_BOTTOM = 'ChromeBottom'
    CHROME_LEFT = 'ChromeLeft'


class Cardinal(Enum):
    N = 'N'
    E = 'E'
    S = 'S'
    W = 'W'


class Intercardinal(Enum):
    NE = 'NE'
    SE = 'SE'
    SW = 'SW'
    NW = 'NW'


class LayoutProps:
    def __init__(self, props: dict):
        self.id = str(props.get('id'))
        self.w = float(props.get('canvasWidth', 0))
        self.h = float(props.get('canvasHeight', 0))
        self.max_width = self.w
        self.max_height = self.h


class ElementProps:
    def __init__(self, props: dict):
        self.id = str(props.get('id'))
        self.name = str(props.get('name'))
        self.element_type = props.get('type', None)
        self.component_name = props.get('componentName', '?')
        self.x0 = float(props.get('x', 0))
        self.y0 = float(props.get('y', 0))
        self.w = float(props.get('width', 0))
        self.h = float(props.get('height', 0))
        self.x1 = self.x0 + self.w
        self.y1 = self.y0 + self.h
        self.fixed_width = bool(props.get('fixedWidth', False))
        self.fixed_height = bool(props.get('fixedHeight', False))
        self.min_width = float(props.get('minWidth', 0))
        self.min_height = float(props.get('minHeight', 0))

        self.fixed_width = 'Left pane / Compact' in self.component_name
        self.fixed_height = any(name in self.component_name for name in [
            'Stripe', 'Main Menu', 'Select', 'button'
        ])

        if 'Stripe' in self.component_name:
            self.h = 32
        if 'Select' in self.component_name:
            self.h = 80
        if 'button--32' in self.component_name:
            self.h = 32
        if 'Main Menu' in self.component_name:
            self.h = 48
        if 'Left pane / Compact' in self.component_name:
            self.w = 96
        if 'Left pane / Full' in self.component_name:
            self.min_width = 96


class Layout:
    def __init__(self, m: Model, props: LayoutProps):
        self.m = m
        self.initial: LayoutProps = props

        self.w = m.addVar(lb=1, vtype=GRB.CONTINUOUS)
        self.h = m.addVar(lb=1, vtype=GRB.CONTINUOUS)


class Element:
    def __init__(self, m: Model, props: ElementProps):
        self.m = m
        self.initial: ElementProps = props

        self.x0 = m.addVar(vtype=GRB.CONTINUOUS)
        self.y0 = m.addVar(vtype=GRB.CONTINUOUS)
        self.x1 = m.addVar(vtype=GRB.CONTINUOUS)
        self.y1 = m.addVar(vtype=GRB.CONTINUOUS)

        self.w = m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(self.x1 - self.x0 == self.w)
        self.h = m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(self.y1 - self.y0 == self.h)

    @lru_cache(maxsize=None)
    def is_chrome(self) -> bool:
        for name in ["Menu Bar", "Stripe", "Left pane", "Footer"]:
            if name in self.initial.component_name:
                return True
        return False

    @lru_cache(maxsize=None)
    def get_alignment(self) -> Optional[Alignment]:
        for name, alignment in {
            'Stripe': Alignment.CHROME_TOP,
            'Menu Bar': Alignment.CHROME_TOP,
            'Left pane': Alignment.CHROME_LEFT,
            'Footer': Alignment.CHROME_BOTTOM,
        }.items():
            if name in self.initial.component_name:
                return alignment
        return None

    @lru_cache(maxsize=None)
    def get_priority(self) -> int:
        for name, priority in {
            'Stripe': 1,
            'Menu Bar': 2,
            'Left pane': 3,
            'Footer': 4,
        }.items():
            if name in self.initial.component_name:
                return priority
        return 999

    @lru_cache(maxsize=None)
    def get_padding(self) -> Padding:
        if "Card" in self.initial.component_name:
            return Padding(80, 32, 32, 32)
        else:
            return Padding(32, 32, 32, 32)

    @lru_cache(maxsize=None)
    def is_left_of(self, other: 'Element') -> bool:
        return self.initial.x1 <= other.initial.x0

    @lru_cache(maxsize=None)
    def is_above(self, other: 'Element') -> bool:
        return self.initial.y1 <= other.initial.y0

    @lru_cache(maxsize=None)
    def overlaps_horizontally_with(self, other: 'Element') -> bool:
        return not self.is_left_of(other) and not other.is_left_of(self)

    @lru_cache(maxsize=None)
    def overlaps_vertically_with(self, other: 'Element') -> bool:
        return not self.is_above(other) and not other.is_above(self)

    @lru_cache(maxsize=None)
    def overlaps_with(self, other: 'Element') -> bool:
        return self.overlaps_horizontally_with(other) and self.overlaps_vertically_with(other)

    @lru_cache(maxsize=None)
    def distance_to(self, other: 'Element') -> float:
        # Returns negative distance if the elements overlap each other

        horizontal_dist = max(self.initial.x0, other.initial.x0) - min(self.initial.x1, other.initial.x1)
        vertical_dist = max(self.initial.y0, other.initial.y0) - min(self.initial.y1, other.initial.y1)

        if horizontal_dist < 0 and vertical_dist < 0:
            # Overlap, return the value closer to zero
            return max(horizontal_dist, vertical_dist)
        elif horizontal_dist < 0:
            return vertical_dist
        elif vertical_dist < 0:
            return horizontal_dist
        else:
            return math.sqrt(horizontal_dist**2 + vertical_dist**2)

    @lru_cache(maxsize=None)
    def direction_to(self, other: 'Element') -> Union[Cardinal, Intercardinal, None]:
        if other.overlaps_with(self):
            return None
        elif other.overlaps_vertically_with(self):
            return Cardinal.W if other.is_left_of(self) else Cardinal.E
        elif other.overlaps_horizontally_with(other):
            return Cardinal.N if other.is_above(self) else Cardinal.S
        elif other.is_above(self):
            return Intercardinal.NW if other.is_left_of(self) else Intercardinal.NE
        else:
            return Intercardinal.SW if other.is_left_of(self) else Intercardinal.SE

    @lru_cache(maxsize=None)
    def is_neighbor_of(self, other: 'Element') -> bool:
        dir_to_other = self.direction_to(other)
        if dir_to_other is None:
            return False
        dist_to_other = self.distance_to(other)

        for third in [e for e in self.m._elements if e not in [self, other]]:
            dir_to_third = self.direction_to(third)
            if dir_to_third != dir_to_other:
                continue
            dist_to_third = self.distance_to(third)
            if 0 <= dist_to_third < dist_to_other:
                return False

        return True

    @lru_cache(maxsize=None)
    def parent(self) -> Optional['Element']:
        potential_parents = [other for other in self.m._elements if other is not self and self.is_within(other)]
        if len(potential_parents) == 0:
            return None
        else:
            # If there are multiple containing elements, pick the one with smallest area
            return min(potential_parents, key=lambda e: e.initial.w * e.initial.h)

    @lru_cache(maxsize=None)
    def children(self) -> List['Element']:
        return [other for other in self.m._elements if other.parent() == self]

    @lru_cache(maxsize=None)
    def is_within(self, other: 'Element'):
        return self.initial.x0 >= other.initial.x0 and self.initial.y0 >= other.initial.y0 \
               and self.initial.x1 <= other.initial.x1 and self.initial.y1 <= other.initial.y1 \
               and self.initial.w * self.initial.h < other.initial.w * other.initial.h

    @lru_cache(maxsize=None)
    def coordinates_match_with(self, other: 'Element') -> bool:
        parent = self.parent() if self.parent() is not None else self.m._layout
        other_parent = other.parent() if other.parent() is not None else self.m._layout
        return (self.initial.x0 - parent.initial.x0) == (other.initial.x0 - other_parent.initial.x0) \
               and (self.initial.y0 - parent.initial.y0) == (other.initial.y0 - other_parent.initial.y0) \
               and (self.initial.x1 - parent.initial.x0) == (other.initial.x1 - other_parent.initial.x0) \
               and (self.initial.y1 - parent.initial.y0) == (other.initial.y1 - other_parent.initial.y0)

    @lru_cache(maxsize=None)
    def match_children(self, other: 'Element') -> List[Tuple['Element', 'Element']]:
        children = self.children()
        other_children = other.children()
        if len(children) == 0 or len(other_children) == 0 or len(children) != len(other_children):
            return []
        else:
            matched_children = []
            matched_other_children = []
            for element in children:
                for other in other_children:
                    if element.coordinates_match_with(other) and other not in matched_other_children:
                        matched_children.append(element)
                        matched_other_children.append(other)
                        break
            if len(matched_children) == len(children):
                return list(zip(matched_children, matched_other_children))
            else:
                return []


def solve(layout_dict: dict, time_out: int = 30, **kwargs) -> dict:

    start = time.time()

    m = Model("DesignSystem")

    m.Params.OutputFlag = 0

    m.Params.TimeLimit = time_out
    m.Params.MIPFocus = 1
    m.Params.ModelSense = GRB.MINIMIZE
    m.Params.Presolve = -1  # -1=auto, 0=off, 1=conservative, 2=aggressive
    m.Params.PoolSearchMode = 0
    m.Params.PoolSolutions = 1

    layout = Layout(m, LayoutProps(layout_dict))
    m._layout = layout
    elements = [Element(m, ElementProps(element_dict)) for element_dict in layout_dict.get('elements', [])]
    m._elements = elements

    # Fix layout size
    m.addConstr(layout.w == layout.initial.w)
    m.addConstr(layout.h == layout.initial.h)

    # Layout Parameters
    column_count = kwargs.get('columns', 24)
    grid_margin = kwargs.get('margin', 16)
    grid_gutter = kwargs.get('gutter', 8)
    baseline_height = kwargs.get('baseline', 8)

    apply_component_constraints(m, elements)

    chrome_elements = [element for element in elements if element.is_chrome()]
    horizontal_chrome_elements = [
        element for element in chrome_elements
        if element.get_alignment() in [Alignment.CHROME_BOTTOM, Alignment.CHROME_BOTTOM]
    ]

    apply_vertical_baseline(m, horizontal_chrome_elements, baseline_height)

    if len(chrome_elements) > 0:
        content_area = align_chrome(m, chrome_elements)
    else:
        content_area = BBox(0, 0, layout.w, layout.h)

    content_area = apply_padding(content_area, Padding(grid_margin, grid_margin, grid_margin, grid_margin))

    top_level_elements = [element for element in elements if element.parent() is None and not element.is_chrome()]

    keep_within(m, content_area, top_level_elements)

    keep_relations(m, top_level_elements)
    keep_alignment(m, top_level_elements)

    apply_horizontal_grid(m, top_level_elements, content_area.x0, content_area.x1, column_count, grid_gutter)
    make_edges_even(m, top_level_elements, content_area)
    apply_vertical_baseline(m, top_level_elements, baseline_height)

    vertical_min_dist(m, top_level_elements, grid_gutter)

    link_group_layouts(m, top_level_elements)

    equal_dim_constrs = keep_equal_dim(m, top_level_elements)
    equal_dist_constrs = keep_equal_dist(m, top_level_elements)
    rel_size_constrs = keep_rel_size(m, top_level_elements)

    alignment_obj = LinExpr()
    upscaling_obj = LinExpr()

    for top_level_element in top_level_elements:

        children = top_level_element.children()

        keep_within(m, top_level_element, children)

        keep_relations(m, children)
        keep_alignment(m, children)
        keep_equal_dim(m, children)
        keep_equal_dist(m, children)

        apply_vertical_baseline(m, children, baseline_height)

        alignment_obj.add(get_alignment_expr(m, children), 10)
        alignment_obj.add(get_fill_container_expr(m, top_level_element, children), 10)
        alignment_obj.add(get_snapping_expr(m, children, close=8, far=32))
        upscaling_obj.add(get_upscaling_expr(m, children))

    downscaling_obj = get_downscaling_expr(m, elements)

    m.setObjectiveN(alignment_obj, 0, priority=10, weight=10, abstol=0, reltol=0)
    m.setObjectiveN(downscaling_obj, 1, priority=5, weight=1, abstol=0, reltol=0)
    m.setObjectiveN(upscaling_obj, 2, priority=1, weight=1, abstol=0, reltol=0)

    try:
        m.optimize()

        if m.Status == GRB.Status.INFEASIBLE:
            m.remove(rel_size_constrs)
            m.optimize()

        if m.Status == GRB.Status.INFEASIBLE:
            m.remove(equal_dim_constrs)
            m.optimize()

        if m.Status == GRB.Status.INFEASIBLE:
            m.remove(equal_dist_constrs)
            m.optimize()

        if m.Status in [GRB.Status.OPTIMAL, GRB.Status.INTERRUPTED, GRB.Status.TIME_LIMIT]:

            layouts = []

            for s in range(m.SolCount):
                m.Params.SolutionNumber = s

                element_props = []

                for top_level_element in elements:
                    element_props.append({
                        'id': top_level_element.initial.id,
                        'x': top_level_element.x0.X,
                        'y': top_level_element.y0.X,
                        'width': top_level_element.w.X,
                        'height': top_level_element.h.X,
                    })

                layouts.append({
                    'id': layout.initial.id,
                    'solutionNumber': s,
                    'canvasWidth': layout.w.X,
                    'canvasHeight': layout.h.X,
                    'elements': element_props
                })

            try:
                print('Size loss', downscaling_obj.getValue())
                print('Alignment', alignment_obj.getValue())
            except:
                e = sys.exc_info()[0]
                print(e)

            runtime = time.time() - start
            print('Runtime', runtime, 'seconds')

            return {
                'status': 0,
                'layouts': layouts
            }
        else:
            return {'status': 1}

    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ": " + str(e))
        raise e


def keep_within(m: Model, container: Union[Element, BBox], elements: List[Element]):
    if isinstance(container, Element):
        padding = container.get_padding()
    else:
        padding = Padding(0, 0, 0, 0)
    bbox = apply_padding(container, padding)
    for element in elements:
        m.addConstr(element.x0 >= bbox.x0)
        m.addConstr(element.y0 >= bbox.y0)
        m.addConstr(element.x1 <= bbox.x1)
        m.addConstr(element.y1 <= bbox.y1)


def get_fill_container_expr(m: Model, container: Union[Element, BBox], elements: List[Element]):
    fill_container_expr = LinExpr(0)

    if len(elements) == 0:
        return fill_container_expr

    if isinstance(container, Element):
        padding = container.get_padding()
    else:
        padding = Padding(0, 0, 0, 0)
    bbox = apply_padding(container, padding)

    min_x0 = m.addVar(vtype=GRB.CONTINUOUS)
    m.addConstr(min_x0 == min_([e.x0 for e in elements]))
    m.addConstr(min_x0 == bbox.x0)

    min_y0 = m.addVar(vtype=GRB.CONTINUOUS)
    m.addConstr(min_y0 == min_([e.y0 for e in elements]))
    m.addConstr(min_y0 == bbox.y0)

    max_x1 = m.addVar(vtype=GRB.CONTINUOUS)
    m.addConstr(max_x1 == max_([e.x1 for e in elements]))
    m.addConstr(max_x1 <= bbox.x1)
    fill_container_expr.add(bbox.x1 - max_x1)

    max_y1 = m.addVar(vtype=GRB.CONTINUOUS)
    m.addConstr(max_y1 == max_([e.y1 for e in elements]))
    m.addConstr(max_y1 <= bbox.y1)
    fill_container_expr.add(bbox.y1 - max_y1)

    return fill_container_expr


def keep_relations(m: Model, elements: List[Element]):
    for element, other in permutations(elements, 2):
        if element.is_left_of(other):
            m.addConstr(element.x1 <= other.x0)
        if element.is_above(other):
            m.addConstr(element.y1 <= other.y0)


def keep_rel_size(m: Model, elements: List[Element], factor: float = 1.2):
    constraints = []
    for element, other in permutations(elements, 2):
        if element.initial.w > other.initial.w * factor:
            constraints.append(m.addConstr(element.w >= other.w))
        if element.initial.h > other.initial.h * factor:
            constraints.append(m.addConstr(element.h >= other.h))

    return constraints


def keep_alignment(m: Model, elements: List[Element]):
    for element, other in permutations(elements, 2):
        if element.initial.x0 == other.initial.x0:
            m.addConstr(element.x0 == other.x0)
        if element.initial.y0 == other.initial.y0:
            m.addConstr(element.y0 == other.y0)
        if element.initial.x1 == other.initial.x1:
            m.addConstr(element.x1 == other.x1)
        if element.initial.y1 == other.initial.y1:
            m.addConstr(element.y1 == other.y1)


def keep_equal_dim(m: Model, elements: List[Element]):
    constraints = []
    for element, other in permutations(elements, 2):
        if element.is_neighbor_of(other):
            if element.initial.w == other.initial.w == 0:
                constraints.append(m.addConstr(element.w == other.w))
            if element.initial.h == other.initial.h:
                constraints.append(m.addConstr(element.h == other.h))
    return constraints


def keep_equal_dist(m: Model, elements: List[Element]):
    constraints = []
    for element in elements:
        for other, third in combinations([e for e in elements if e is not element], 2):
            if element.is_neighbor_of(other) and element.is_neighbor_of(third):
                dist_to_other = element.distance_to(other)
                dist_to_third = element.distance_to(third)
                if dist_to_other == dist_to_third:
                    dist_to_other_var = get_distance_var(m, element, other)
                    constraints.append(m.addConstr(dist_to_other_var == get_distance_var(m, element, third)))
    return constraints


def vertical_min_dist(m: Model, elements: List[Element], min_distance: Var):
    for element, other in permutations(elements, 2):
        if element.is_above(other):
            m.addConstr(element.y1 + min_distance <= other.y0)


def get_snapping_expr(m: Model, elements: List[Element], close: float, far: float):
    snapping = LinExpr(0)
    for element, other in permutations(elements, 2):
        if element.is_neighbor_of(other):
            initial_distance = element.distance_to(other)
            distance_var = get_distance_var(m, element, other)
            m.addConstr(distance_var >= close)
            snapping_error = m.addVar(lb=0, vtype=GRB.CONTINUOUS)
            if initial_distance <= (close + far) / 2:
                m.addConstr(snapping_error >= distance_var - close)
            elif initial_distance <= far:
                m.addConstr(snapping_error >= distance_var - far)
                m.addConstr(snapping_error >= far - distance_var)
            else:
                m.addConstr(distance_var >= far)
            snapping.add(snapping_error)

    return snapping


def get_distance_var(m: Model, element: Element, other: Element):
    distance_var = m.addVar(vtype=GRB.CONTINUOUS)
    direction = element.direction_to(other)

    if direction == Cardinal.W:
        m.addConstr(distance_var == element.x0 - other.x1)
    elif direction == Cardinal.E:
        m.addConstr(distance_var == other.x0 - element.x1)
    elif direction == Cardinal.N:
        m.addConstr(distance_var == element.y0 - other.y1)
    elif direction == Cardinal.S:
        m.addConstr(distance_var == other.y0 - element.y1)
    else:
        pass # Distance is undefined for intercardinal directions

    return distance_var


def get_downscaling_expr(m: Model, elements: List[Element]):
    downscaling = LinExpr(0)
    if len(elements) == 0:
        return downscaling

    max_initial_width = max([element.initial.w for element in elements])
    max_initial_height = max([element.initial.h for element in elements])

    for element in elements:
        # Smaller elements should be scaled down less
        element_width_loss = m.addVar(lb=0)
        m.addConstr(element_width_loss >= element.initial.w - element.w)
        downscaling.add(element_width_loss * max_initial_width / element.initial.w)

        element_height_loss = m.addVar(lb=0)
        m.addConstr(element_height_loss >= element.initial.h - element.h)
        downscaling.add(element_height_loss * max_initial_height / element.initial.h)

    return downscaling


def get_upscaling_expr(m: Model, elements: List[Element]):
    upscaling = LinExpr(0)
    horizontal_threshold = 1.
    vertical_threshold = .5

    if len(elements) == 0:
        return upscaling

    max_initial_width = max([element.initial.w for element in elements])
    max_initial_height = max([element.initial.h for element in elements])

    for element in elements:
        # Smaller elements should be scaled up less
        excessive_element_width_increase = m.addVar(lb=0)
        m.addConstr(excessive_element_width_increase >= element.w - element.initial.w * (1 + horizontal_threshold))
        upscaling.add(excessive_element_width_increase * max_initial_width / element.initial.w)

        excessive_element_height_increase = m.addVar(lb=0)
        m.addConstr(excessive_element_height_increase >= element.h - element.initial.h * (1 + vertical_threshold))
        upscaling.add(excessive_element_height_increase * max_initial_height / element.initial.h)

    return upscaling


def apply_horizontal_grid(m: Model, elements: List[Element], grid_x0: Var, grid_x1: Var, pref_col_count: int, gutter: float):
    if len(elements) == 0:
        return
    col_count = max(pref_col_count, get_min_col_count(elements))

    gutter_count = col_count - 1
    col_width = m.addVar(vtype=GRB.CONTINUOUS)
    m.addConstr(grid_x1 == grid_x0 + col_count * col_width + gutter_count * gutter)

    # Ensure that the whole grid is used
    starting_in_first_col = LinExpr(0)
    ending_in_last_col = LinExpr(0)
    layout = m._layout

    for i, element in enumerate(elements):

        col_start_flags = m.addVars(range(col_count), vtype=GRB.BINARY)
        m.addConstr(col_start_flags.sum() == 1)
        col_end_flags = m.addVars(range(col_count), vtype=GRB.BINARY)
        m.addConstr(col_end_flags.sum() == 1)

        starting_in_first_col.add(col_start_flags[0])
        ending_in_last_col.add(col_end_flags[col_count - 1])

        for col_index in range(col_count):
            col_x0 = LinExpr(grid_x0 + col_index * (col_width + gutter))
            m.addConstr(element.x0 >= col_x0 - layout.initial.w * (1 - col_start_flags[col_index]))
            m.addConstr(element.x0 <= col_x0 + layout.initial.w * (1 - col_start_flags[col_index]))

            col_x1 = LinExpr(grid_x0 + (col_index + 1) * (col_width + gutter) - gutter)
            m.addConstr(element.x1 >= col_x1 - layout.initial.w * (1 - col_end_flags[col_index]))
            m.addConstr(element.x1 <= col_x1 + layout.initial.w * (1 - col_end_flags[col_index]))

    m.addConstr(starting_in_first_col >= 1)
    m.addConstr(ending_in_last_col >= 1)


def make_edges_even(m: Model, elements: List[Element], bbox: BBox):
    for element in elements:
        others_on_left = [other for other in elements if other.is_left_of(element)]
        if len(others_on_left) == 0:
            m.addConstr(element.x0 == bbox.x0)

        others_above = [other for other in elements if other.is_above(element)]
        if len(others_above) == 0:
            m.addConstr(element.y0 == bbox.y0)

        others_on_right = [other for other in elements if element.is_left_of(other)]
        if len(others_on_right) == 0:
            m.addConstr(element.x1 == bbox.x1)

        others_below = [other for other in elements if element.is_above(other)]
        if len(others_below) == 0:
            m.addConstr(element.y1 == bbox.y1)


def apply_vertical_baseline(m: Model, elements: List[Element], baseline_height: int):
    for element in elements:
        start_line = m.addVar(vtype=GRB.INTEGER)
        end_line = m.addVar(vtype=GRB.INTEGER)
        m.addConstr(baseline_height * start_line == element.y0)
        if not element.initial.fixed_height:
            m.addConstr(baseline_height * end_line == element.y1)


def apply_padding(bbox: BBox, padding: Padding):
    return BBox(
        bbox.x0 + padding.left,
        bbox.y0 + padding.top,
        bbox.x1 - padding.right,
        bbox.y1 - padding.bottom,
    )


def align_chrome(m: Model, chrome_elements: List[Element]) -> BBox:
    layout = m._layout
    sorted_chrome_elements = sorted(chrome_elements, key=lambda e: e.get_priority())

    last_above = None
    last_on_left = None
    last_below = None
    last_on_right = None

    for element in sorted_chrome_elements:
        if element.get_alignment() is not Alignment.CHROME_RIGHT:
            if last_on_left is None:
                m.addConstr(element.x0 == 0)
            else:
                m.addConstr(element.x0 == last_on_left.x1)

        if element.get_alignment() is not Alignment.CHROME_BOTTOM:
            if last_above is None:
                m.addConstr(element.y0 == 0)
            else:
                m.addConstr(element.y0 == last_above.y1)

        if element.get_alignment() is not Alignment.CHROME_LEFT:
            if last_on_left is None:
                m.addConstr(element.x1 == layout.w)
            else:
                m.addConstr(element.x1 == last_on_left.x0)

        if element.get_alignment() is not Alignment.CHROME_TOP:
            if last_below is None:
                m.addConstr(element.y1 == layout.h)
            else:
                m.addConstr(element.y1 == last_above.y0)

        if element.get_alignment() is Alignment.CHROME_LEFT:
            last_on_left = element
        if element.get_alignment() is Alignment.CHROME_BOTTOM:
            last_below = element
        if element.get_alignment() is Alignment.CHROME_RIGHT:
            last_on_right = element
        if element.get_alignment() is Alignment.CHROME_TOP:
            last_above = element

    content_x0 = m.addVar(vtype=GRB.INTEGER)
    content_y0 = m.addVar(vtype=GRB.INTEGER)
    content_x1 = m.addVar(vtype=GRB.INTEGER)
    content_y1 = m.addVar(vtype=GRB.INTEGER)

    if last_on_left is None:
        m.addConstr(content_x0 == 0)
    else:
        m.addConstr(content_x0 == last_on_left.x1)

    if last_above is None:
        m.addConstr(content_y0 == 0)
    else:
        m.addConstr(content_y0 == last_above.y1)

    if last_on_right is None:
        m.addConstr(content_x1 == layout.w)
    else:
        m.addConstr(content_x1 == last_on_right.x0)

    if last_below is None:
        m.addConstr(content_y1 == layout.h)
    else:
        m.addConstr(content_y1 == last_below.y0)

    return BBox(content_x0, content_y0, content_x1, content_y1)


def apply_component_constraints(m: Model, elements: List[Element]):
    for element in elements:
        if element.initial.fixed_width:
            m.addConstr(element.w == element.initial.w)
        else:
            m.addConstr(element.w >= element.initial.min_width)
        if element.initial.fixed_height:
            m.addConstr(element.h == element.initial.h)
        else:
            m.addConstr(element.h >= element.initial.min_height)


def get_alignment_expr(m: Model, elements: List[Element]):
    layout: Layout = m._layout
    alignment = LinExpr(0)

    var_count = 0

    for element, other in combinations(elements, 2):
        if element.overlaps_horizontally_with(other) and not element.overlaps_with(other):
            elements_align_start = m.addVar(vtype=GRB.BINARY)
            m.addConstr(element.x0 >= other.x0 - layout.initial.max_width * (1 - elements_align_start))
            m.addConstr(element.x0 <= other.x0 + layout.initial.max_width * (1 - elements_align_start))
            elements_align_end = m.addVar(vtype=GRB.BINARY)
            m.addConstr(element.x1 >= other.x1 - layout.initial.max_width * (1 - elements_align_end))
            m.addConstr(element.x1 <= other.x1 + layout.initial.max_width * (1 - elements_align_end))
            alignment.add(2 - elements_align_start - elements_align_end)
            var_count += 2
        elif element.overlaps_vertically_with(other) and not element.overlaps_with(other):
            elements_align_start = m.addVar(vtype=GRB.BINARY)
            m.addConstr(element.y0 >= other.y0 - layout.initial.max_height * (1 - elements_align_start))
            m.addConstr(element.y0 <= other.y0 + layout.initial.max_height * (1 - elements_align_start))
            elements_align_end = m.addVar(vtype=GRB.BINARY)
            m.addConstr(element.y1 >= other.y1 - layout.initial.max_height * (1 - elements_align_end))
            m.addConstr(element.y1 <= other.y1 + layout.initial.max_height * (1 - elements_align_end))
            alignment.add(2 - elements_align_start - elements_align_end)
            var_count += 2

    return alignment


def link_group_layouts(m: Model, parents: List[Element]):
    for parent, other_parent in combinations(parents, 2):
        if parent.initial.w == other_parent.initial.w and parent.initial.h == other_parent.initial.h:
            matched_children = parent.match_children(other_parent)
            if len(matched_children) > 0:
                m.addConstr(parent.w == other_parent.w)
                m.addConstr(parent.h == other_parent.h)
                for child, other_child in matched_children:
                    m.addConstr(child.x0 - parent.x0 == other_child.x0 - other_parent.x0)
                    m.addConstr(child.y0 - parent.y0 == other_child.y0 - other_parent.y0)
                    m.addConstr(child.x1 - parent.x1 == other_child.x1 - other_parent.x1)
                    m.addConstr(child.y1 - parent.y1 == other_child.y1 - other_parent.y1)


def get_min_col_count(elements: List[Element]) -> int:
    start = [e.initial.x0 for e in elements]
    end = [e.initial.x1 for e in elements]
    return get_min_grid_width(start, end)


def get_min_row_count(elements: List[Element]) -> int:
    start = [e.initial.y0 for e in elements]
    end = [e.initial.y1 for e in elements]
    return get_min_grid_width(start, end)


def get_min_grid_width(start: List[float], end: List[float]) -> int:

    m = Model('GridWidth')
    m.Params.OutputFlag = 0
    m.Params.TimeLimit = 5

    start_vars = m.addVars(len(start), lb=0, ub=len(start), vtype=GRB.INTEGER)
    end_vars = m.addVars(len(start), lb=1, ub=len(start) + 1, vtype=GRB.INTEGER)

    grid_width = m.addVar(lb=0, vtype=GRB.INTEGER)

    for i in range(len(start)):
        m.addConstr(start_vars[i] + 1 <= end_vars[i])
        m.addConstr(grid_width >= end_vars[i])
        for j in range(len(start)):
            if end[i] <= start[j]:
                m.addConstr(end_vars[i] <= start_vars[j])

    m.setObjective(grid_width, GRB.MINIMIZE)

    try:
        m.optimize()
    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ": " + str(e))
        raise e

    return int(grid_width.X)