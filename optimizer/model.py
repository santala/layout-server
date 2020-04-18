import sys
import math
from collections import namedtuple
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Union, Tuple
from itertools import combinations, permutations, product
from gurobi import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError, Var


Margin = namedtuple('Margin', 'top right bottom left')
Padding = namedtuple('Margin', 'top right bottom left')
BoundingBox = namedtuple('BBox', 'x0 y0 x1 y1')


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
        self.fixed_height = 'Stripe' in self.component_name or 'Main Menu' in self.component_name
        print(self.component_name)


class Layout:
    def __init__(self, m: Model, props: LayoutProps, fixed_width: bool = True, fixed_height: bool = True):
        self.m = m
        self.initial: LayoutProps = props

        self.w = m.addVar(lb=1, vtype=GRB.CONTINUOUS)
        self.h = m.addVar(lb=1, vtype=GRB.CONTINUOUS)


class Element:
    def __init__(self, m: Model, props: ElementProps):
        self.m = m
        self.initial: ElementProps = props

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
    def is_chrome(self) -> bool:
        for name in ["Left pane", "Menu Bar", "Stripe"]:
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
    def get_margin(self) -> Margin:
        # TODO: make configurable somehow
        if self.is_chrome():
            return Margin(0, 0, 0, 0)
        else:
            return Margin(8, 8, 8, 8)

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
    def neighbors(self):
        return [e for e in self.m._elements if e is not self and e.is_neighbor_of(self)]

    @lru_cache(maxsize=None)
    def is_content(self) -> bool:
        return self.parent() is not None

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
        return self.initial.x0 > other.initial.x0 and self.initial.y0 > other.initial.y0 \
               and self.initial.x1 < other.initial.x1 and self.initial.y1 < other.initial.y1


def solve(layout_dict: dict, time_out: int = 30):

    m = Model("DesignSystem")

    m.Params.OutputFlag = 0

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
    if False:
        m.addConstr(layout.h == layout.initial.h)

    # Layout Parameters
    column_count = 24
    grid_margin = 16
    grid_gutter = 8
    baseline_height = 8
    tolerance = 8

    # TODO: Match distances to neighbors
    # TODO: Grid gaps
    # TODO: Match the internal layout of identical groups
    # TODO: Consider supporting reflow and resolution changes
    # TODO: Consider supporting locking aspect ratio

    apply_component_specific_constraints(m, elements)

    chrome_elements = [element for element in elements if element.is_chrome()]
    horizontal_chrome_elements = [
        element for element in chrome_elements
        if element.get_alignment() in [Alignment.CHROME_BOTTOM, Alignment.CHROME_BOTTOM]
    ]
    top_level_elements = [element for element in elements if element.parent() is None and not element.is_chrome()]
    content_elements = [element for element in elements if element.parent() is not None and not element.is_chrome()]

    if len(chrome_elements) > 0:
        content_area = align_chrome(m, chrome_elements)
    else:
        content_area = BoundingBox(0, 0, layout.w, layout.h)

    apply_vertical_baseline(m, horizontal_chrome_elements, baseline_height)

    maintain_relationships(m, top_level_elements)
    maintain_alignment(m, top_level_elements)
    maintain_matching_neighbor_dimensions(m, top_level_elements, tolerance)
    #maintain_matching_dimensions(m, top_level_elements, tolerance)
    maintain_matching_neighbor_distances(m, top_level_elements, tolerance)
    #keep_neighbors_together(m, top_level_elements, grid_gutter)
    #snap_distances(m, top_level_elements, grid_gutter, 4 * grid_gutter)
    apply_horizontal_grid(m, top_level_elements, content_area.x0, content_area.x1, column_count, grid_margin, grid_gutter)
    make_edges_even(m, top_level_elements, apply_padding(content_area, Padding(grid_margin, grid_margin, grid_margin, grid_margin)))
    apply_vertical_baseline(m, top_level_elements, baseline_height)
    contain_within(m, apply_padding(content_area, Padding(grid_margin, grid_margin, grid_margin, grid_margin)), top_level_elements)
    #bind_to_edges_of(m, apply_padding(content_area, Padding(grid_margin, grid_margin, grid_margin, grid_margin)), top_level_elements)

    for top_level_element in top_level_elements:

        children = top_level_element.children()
        contain_within(m, top_level_element, children)
        bind_to_edges_of(m, top_level_element, children)
        maintain_relationships(m, children)
        maintain_alignment(m, children)
        maintain_matching_neighbor_dimensions(m, children, tolerance)
        #maintain_matching_dimensions(m, children, tolerance)
        maintain_matching_neighbor_distances(m, children, tolerance)
        #keep_neighbors_together(m, children, grid_gutter)
        #snap_distances(m, children)

    apply_vertical_baseline(m, content_elements, baseline_height)

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

                for top_level_element in elements:
                    element_props.append({
                        'id': top_level_element.initial.id,
                        'x': top_level_element.x0.X,
                        'y': top_level_element.y0.X,
                        'width': top_level_element.w.X,
                        'height': top_level_element.h.X,
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


def contain_within(m: Model, container: Union[Element, BoundingBox], elements: List[Element]):
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


def bind_to_edges_of(m: Model, container: Union[Element, BoundingBox], elements: List[Element]):
    if len(elements) == 0:
        return

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
    m.addConstr(max_x1 == bbox.x1)

    max_y1 = m.addVar(vtype=GRB.CONTINUOUS)
    m.addConstr(max_y1 == max_([e.y1 for e in elements]))
    m.addConstr(max_y1 == bbox.y1)


def maintain_relationships(m: Model, elements: List[Element]):
    for element, other in permutations(elements, 2):
        if element.is_left_of(other):
            m.addConstr(element.x1 + element.get_margin().right <= other.x0)
        if element.is_above(other):
            m.addConstr(element.y1 + element.get_margin().bottom <= other.y0)


def maintain_alignment(m: Model, elements: List[Element], tolerance: float = 8):
    for element, other in permutations(elements, 2):
        if abs(element.initial.x0 - other.initial.x0) <= tolerance:
            m.addConstr(element.x0 == other.x0)
        if abs(element.initial.y0 - other.initial.y0) <= tolerance:
            m.addConstr(element.y0 == other.y0)
        if abs(element.initial.x1 - other.initial.x1) <= tolerance:
            m.addConstr(element.x1 == other.x1)
        if abs(element.initial.y1 - other.initial.y1) <= tolerance:
            m.addConstr(element.y1 == other.y1)


def maintain_matching_dimensions(m: Model, elements: List[Element], tolerance: float = 8):
    for element, other in permutations(elements, 2):
        if abs(element.initial.w - other.initial.w) <= tolerance:
            m.addConstr(element.w == other.w)
        if abs(element.initial.h - other.initial.h) <= tolerance:
            m.addConstr(element.h == other.h)


def maintain_matching_neighbor_dimensions(m: Model, elements: List[Element], tolerance: float = 8):
    for element, other in permutations(elements, 2):
        if element.is_neighbor_of(other):
            if abs(element.initial.w - other.initial.w) <= tolerance:
                m.addConstr(element.w == other.w)
            if abs(element.initial.h - other.initial.h) <= tolerance:
                m.addConstr(element.h == other.h)


def maintain_matching_neighbor_distances(m: Model, elements: List[Element], tolerance: float = 8):
    for element in elements:
        for other, third in combinations([e for e in elements if e is not element], 2):
            if element.is_neighbor_of(other) and element.is_neighbor_of(third):
                dist_to_other = element.distance_to(other)
                dist_to_third = element.distance_to(third)
                if abs(dist_to_other - dist_to_third) <= tolerance:
                    m.addConstr(get_distance_var(m, element, other) == get_distance_var(m, element, third))


def keep_neighbors_together(m: Model, elements: List[Element], distance):
    for element, other in permutations(elements, 2):
        if element.is_neighbor_of(other):
            m.addConstr(get_distance_var(m, element, other) == distance)


def snap_distances(m: Model, elements: List[Element], close: float = 8, far: float = 32): # TODO: rename as something better
    for element, other in permutations(elements, 2):
        if element.is_neighbor_of(other):
            distance = element.distance_to(other)
            distance_var = get_distance_var(m, element, other)
            if 0 <= distance <= (close + far) / 2:
                m.addConstr(distance_var == close)
            elif (close + far) / 2 < distance <= far + (close + far) / 2:
                m.addConstr(distance_var == far)


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


def apply_horizontal_grid(m: Model, elements: List[Element], grid_x0: Var, grid_x1: Var, col_count: int, margin: float, gutter: float):
    gutter_count = col_count - 1
    col_width = m.addVar(vtype=GRB.CONTINUOUS)
    m.addConstr(grid_x1 == grid_x0 + 2 * margin + col_count * col_width + gutter_count * gutter)

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
            col_x0 = LinExpr(grid_x0 + margin + col_index * (col_width + gutter))
            m.addConstr(col_start_flags[col_index] * element.x0 == col_start_flags[col_index] * col_x0)
            col_x1 = LinExpr(grid_x0 + margin + (col_index + 1) * (col_width + gutter) - gutter)
            m.addConstr(col_end_flags[col_index] * element.x1 == col_end_flags[col_index] * col_x1)

        # There are no other elements before this one, this element should start in the first column
        others_before = [other for other in elements if other.is_left_of(element)]
        if len(others_before) == 0:
            m.addConstr(col_start_flags[0] == 1)


    m.addConstr(starting_in_first_col >= 1)
    m.addConstr(ending_in_last_col >= 1)


def make_edges_even(m: Model, elements: List[Element], bbox: BoundingBox):
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
        m.addConstr(baseline_height * end_line == element.y1)


def apply_padding(bbox: BoundingBox, padding: Padding):
    return BoundingBox(
        bbox.x0 + padding.left,
        bbox.y0 + padding.top,
        bbox.x1 - padding.right,
        bbox.y1 - padding.bottom,
    )


def align_chrome(m: Model, chrome_elements: List[Element]) -> BoundingBox:
    layout = m._layout
    sorted_chrome_elements = sorted(chrome_elements, key=lambda e: e.get_priority())

    last_above = None
    last_on_left = None
    last_below = None
    last_on_right = None

    for element in sorted_chrome_elements:
        if element.get_alignment() is not Alignment.CHROME_RIGHT:
            if last_on_left is None:
                m.addConstr(element.x0 == element.get_margin().left)
            else:
                m.addConstr(element.x0 == last_on_left.x1 + element.get_margin().left)

        if element.get_alignment() is not Alignment.CHROME_BOTTOM:
            if last_above is None:
                m.addConstr(element.y0 == element.get_margin().top)
            else:
                m.addConstr(element.y0 == last_above.y1 + element.get_margin().top)

        if element.get_alignment() is not Alignment.CHROME_LEFT:
            if last_on_left is None:
                m.addConstr(element.x1 == layout.w - element.get_margin().left)
            else:
                m.addConstr(element.x1 == last_on_left.x0 - element.get_margin().left)

        if element.get_alignment() is not Alignment.CHROME_TOP:
            if last_below is None:
                m.addConstr(element.y1 == layout.h - element.get_margin().bottom)
            else:
                m.addConstr(element.y1 == last_above.y0 - element.get_margin().bottom)

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
        m.addConstr(content_x0 == last_on_left.x1 + last_on_left.get_margin().right)

    if last_above is None:
        m.addConstr(content_y0 == 0)
    else:
        m.addConstr(content_y0 == last_above.y1 + last_above.get_margin().bottom)

    if last_on_right is None:
        m.addConstr(content_x1 == layout.w)
    else:
        m.addConstr(content_x1 == last_on_right.x0 - last_on_left.get_margin().left)

    if last_below is None:
        m.addConstr(content_y1 == layout.h)
    else:
        m.addConstr(content_y1 == last_below.y0 - last_above.get_margin().top)

    return BoundingBox(content_x0, content_y0, content_x1, content_y1)


def apply_component_specific_constraints(m: Model, elements: List[Element]):
    for element in elements:
        if element.initial.fixed_width:
            m.addConstr(element.w == element.initial.w)
        if element.initial.fixed_height:
            m.addConstr(element.h == element.initial.h)
