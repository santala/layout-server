from collections import namedtuple
from enum import Enum
from itertools import product, permutations
from math import ceil, floor, sqrt
from typing import List

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError

from .classes import Layout, Element

def equal_width_columns(m: Model, elements: List[Element], available_width, available_height, elem_width, elem_height):

    elem_count = len(elements)
    elem_ids = [e.id for e in elements]

    # TODO: compute a proper maximum column/row count
    max_col_count = elem_count
    max_row_count = elem_count

    # MINIMIZE LAYOUT COLUMNS
    # TODO: what about preferred number of columns?


    # Element coordinates in base units
    x0, y0, x1, y1 = add_coord_vars(m, elem_ids, available_width, available_height)

    width, height = [
        add_diff_vars(m, elem_ids, var1, var2)
        for var1, var2 in [(x1, x0), (y1, y0)]
    ]

    m.addConstrs((
        elem_width[i] == width[i]
        for i in elem_ids
    ))
    m.addConstrs((
        elem_height[i] == height[i]
        for i in elem_ids
    ))

    x0_diff, y0_diff, x1_diff, y1_diff = [
        add_pairwise_diff_vars(m, elem_ids, vars)
        for vars in [x0, y0, x1, y1]
    ]

    x0_less_than, y0_less_than, x1_less_than, y1_less_than = [
        add_less_than_vars(m, elem_ids, vars)
        for vars in [x0_diff, y0_diff, x1_diff, y1_diff]
    ]

    x0x1_diff, x1x0_diff, y0y1_diff, y1y0_diff  = [
        add_pairwise_cross_diff_vars(m, elem_ids, var1, var2)
        for var1, var2 in [(x0, x1), (x1, x0), (y0, y1), (y1, y0)]
    ]

    x0_less_than_x1, x1_less_than_x0, y0_less_than_y1, y1_less_than_y0 = [
        add_less_than_vars(m, elem_ids, vars)
        for vars in [x0x1_diff, x1x0_diff, y0y1_diff, y1y0_diff]
    ]

    # Element coordinates in rows and columns
    c0, r0, c1, r1 = add_coord_vars(m, elem_ids, max_col_count, max_row_count)

    col_span, row_span = [
        add_diff_vars(m, elem_ids, var1, var2)
        for var1, var2 in [(c1, c0), (r1, r0)]
    ]

    c0_diff, r0_diff, c1_diff, r1_diff = [add_pairwise_diff_vars(m, elem_ids, vars) for vars in [c0, r0, c1, r1]]

    c0_less_than, r0_less_than, c1_less_than, r1_less_than = [add_less_than_vars(m, elem_ids, vars) for vars in [c0_diff, r0_diff, c1_diff, r1_diff]]


    c0c1_diff, c1c0_diff, r0r1_diff, r1r0_diff = [
        add_pairwise_cross_diff_vars(m, elem_ids, var1, var2)
        for var1, var2 in [(c0, c1), (c1, c0), (r0, r1), (r1, r0)]
    ]

    c0_less_than_c1, c1_less_than_c0, r0_less_than_r1, r1_less_than_r0 = [
        add_less_than_vars(m, elem_ids, vars)
        for vars in [c0c1_diff, c1c0_diff, r0r1_diff, r1r0_diff]
    ]


    # Link order of coordinates, e.g.
    # * if x0[A] < x0[B], then c0[A] < c0[B]
    # * if x0[A] = x0[B], then c0[A] = c0[B]
    # * if x0[A] > x0[B], then c0[A] > c0[B]
    m.addConstrs((x0_less_than[i1, i2] == c0_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((x1_less_than[i1, i2] == c1_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((y0_less_than[i1, i2] == r0_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((y1_less_than[i1, i2] == r1_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))

    m.addConstrs((x0_less_than_x1[i1, i2] == c0_less_than_c1[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((x1_less_than_x0[i1, i2] == c1_less_than_c0[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((y0_less_than_y1[i1, i2] == r0_less_than_r1[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((y1_less_than_y0[i1, i2] == r1_less_than_r0[i1, i2] for i1, i2 in permutations(elem_ids, 2)))

    # COLUMN/ROW SIZE & COUNT

    col_width = m.addVar(lb=1, vtype=GRB.INTEGER) # in base units
    col_count = m.addVar(lb=1, vtype=GRB.INTEGER)
    m.addConstr(col_count == max_(c1))

    row_height = m.addVar(lb=1, vtype=GRB.INTEGER) # in base units
    row_count= m.addVar(lb=1, vtype=GRB.INTEGER)
    m.addConstr(row_count == max_(r1))


    # Column width must be at max the smallest difference between two column lines
    m.addConstrs((
        (x0_less_than[i1, i2] == 1) >> (x0_diff[i2, i1] >= col_width)
        for i1, i2 in permutations(elem_ids, 2)
    ))
    m.addConstrs((
        (x1_less_than[i1, i2] == 1) >> (x1_diff[i2, i1] >= col_width)
        for i1, i2 in permutations(elem_ids, 2)
    ))

    # Row height must be at max the smallest difference between two row lines
    m.addConstrs((
        (y0_less_than[i1, i2] == 1) >> (y0_diff[i2, i1] >= row_height)
        for i1, i2 in permutations(elem_ids, 2)
    ))
    m.addConstrs((
        (y1_less_than[i1, i2] == 1) >> (y1_diff[i2, i1] >= row_height)
        for i1, i2 in permutations(elem_ids, 2)
    ))

    c0_one_less_than = add_one_less_than_vars(m, elem_ids, c0_less_than, c0_diff)
    c1_one_less_than = add_one_less_than_vars(m, elem_ids, c1_less_than, c1_diff)

    # Consecutive column lines must be column width apart
    # e.g. if element A starts in column 2, and element B starts in column 3, x0[B] - x0[A] == col_width
    m.addConstrs((
        (c0_one_less_than[i1, i2] == 1) >> (x0_diff[i2, i1] == col_width)
        for i1, i2 in permutations(elem_ids, 2)
    ))
    m.addConstrs((
        (c1_one_less_than[i1, i2] == 1) >> (x1_diff[i2, i1] == col_width)
        for i1, i2 in permutations(elem_ids, 2)
    ))

    # Prevent overlap
    above, on_left = get_directional_relationships(m, elem_ids, x0, x1, y0, y1)

    in_same_col, on_same_row = add_overlap_vars(m, elem_ids, above, on_left)

    prevent_overlap(m, elem_ids, in_same_col, on_same_row)

    # Minimize gaps in the grid

    c0_equals_c1 = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY)
    m.addConstrs((
        (c0_equals_c1[i1, i2] == 1) >> (c0[i1] == c1[i2])
        for i1, i2 in permutations(elem_ids, 2)
    ))

    no_gap_on_left = m.addVars(elem_ids, vtype=GRB.BINARY)
    m.addConstrs((
        no_gap_on_left[i] >= 1 - c0[i] # If element is in the first column (c0[i]==0), there is no gap on the left
        for i in elem_ids
    ))
    neighbor_exists_on_left = m.addVars(elem_ids, vtype=GRB.BINARY)
    m.addConstrs((
        neighbor_exists_on_left[i1] == and_(on_same_row[i1, i2], c0_equals_c1[i1, i2])
        for i1, i2 in permutations(elem_ids, 2)
    ))
    m.addConstrs((
        no_gap_on_left[i] >= neighbor_exists_on_left[i]
        for i in elem_ids
    ))


    if True:
        gap_count = elem_count - no_gap_on_left.sum()
    else:
        cell_count = add_area_vars(m, elem_ids, col_span, row_span, max_col_count, max_row_count)
        total_cell_count = cell_count.sum()

        grid_area = add_area_var(m, col_count, row_count, max_col_count, max_row_count)

        gap_count = grid_area - total_cell_count


    number_of_groups_expr = LinExpr()
    number_of_groups_expr.add(col_count)
    number_of_groups_expr.add(row_count)
    m.addConstr(number_of_groups_expr >= compute_minimum_grid(elem_count), name='PreventOvertOptimization')

    # MINIMIZE DIFFERENCE BETWEEN LEFT AND RIGHT MARGINS
    left_margin = m.addVar(lb=0, vtype=GRB.INTEGER, name='LeftMargin')
    m.addConstr(left_margin == min_(x0), name='LinkLeftMargin')

    max_x1 = m.addVar(lb=0, vtype=GRB.INTEGER, name='MaxX1')
    m.addConstr(max_x1 == max_(x1), name='LinkMaxX1')

    right_margin = m.addVar(lb=0, vtype=GRB.INTEGER, name='MaxX1')
    m.addConstr(right_margin == available_width - max_x1, name='LinkRightMargin')

    margin_diff_loose_abs = m.addVar(lb=0, vtype=GRB.INTEGER, name='MarginDiffLooseAbs')
    m.addConstr(margin_diff_loose_abs >= left_margin - right_margin, name='LinkMarginDiffLooseAbs1')
    m.addConstr(margin_diff_loose_abs >= right_margin - left_margin, name='LinkMarginDiffLooseAbs2')

    margin_diff_abs_expr = LinExpr()
    margin_diff_abs_expr.add(margin_diff_loose_abs)



    height_error = LinExpr(0) # TODO

    def get_rel_xywh(element_id):
        # Returns the element position (relative to the grid top left corner)

        # Attribute Xn refers to the variable value in the solution selected using SolutionNumber parameter.
        # When SolutionNumber equals 0 (default), Xn refers to the variable value in the best solution.
        # https://www.gurobi.com/documentation/8.1/refman/xn.html#attr:Xn
        x = x0[element_id].Xn
        y = y0[element_id].Xn
        w = width[element_id].Xn
        h = height[element_id].Xn

        return x, y, w, h

    return get_rel_xywh, margin_diff_abs_expr, height_error, gap_count, above, on_left


def add_coord_vars(m: Model, elem_ids, available_width, available_height):
    x0 = m.addVars(elem_ids, lb=0, vtype=GRB.INTEGER)
    y0 = m.addVars(elem_ids, lb=0, vtype=GRB.INTEGER)
    x1 = m.addVars(elem_ids, lb=1, vtype=GRB.INTEGER)
    y1 = m.addVars(elem_ids, lb=1, vtype=GRB.INTEGER)

    m.addConstrs((x0[i] <= x1[i] - 1 for i in elem_ids)) # x0 < x1 sanity
    m.addConstrs((y0[i] <= y1[i] - 1 for i in elem_ids)) # y0 < y1 sanity
    m.addConstrs((x1[i] <= available_width for i in elem_ids)) # contain to available width
    m.addConstrs((y1[i] <= available_height for i in elem_ids)) # contain to available height

    return x0, y0, x1, y1

def add_diff_vars(m: Model, ids: List, var1, var2):
    diff = m.addVars(ids, lb=-GRB.INFINITY, vtype=GRB.INTEGER)
    m.addConstrs((
        diff[i] == var1[i] - var2[i]
        for i in ids
    ))
    return diff

def add_pairwise_diff_vars(m: Model, ids: List, var: tupledict):
    diff = m.addVars(permutations(ids, 2), lb=-GRB.INFINITY, vtype=GRB.INTEGER)
    m.addConstrs((
        diff[i1, i2] == var[i1] - var[i2]
        for i1, i2 in permutations(ids, 2)
    ))
    return diff

def add_pairwise_cross_diff_vars(m: Model, ids: List, var1: tupledict, var2: tupledict):
    diff = m.addVars(permutations(ids, 2), lb=-GRB.INFINITY, vtype=GRB.INTEGER)
    m.addConstrs((
        diff[i1, i2] == var1[i1] - var2[i2]
        for i1, i2 in permutations(ids, 2)
    ))
    return diff

def add_less_than_vars(m: Model, ids: List, diff: tupledict):
    less_than = m.addVars(permutations(ids, 2), vtype=GRB.BINARY)
    m.addConstrs((
        (less_than[i1, i2] == 1) >> (diff[i1, i2] <= -1)
        for i1, i2 in permutations(ids, 2)
    ))
    m.addConstrs((
        (less_than[i1, i2] == 0) >> (diff[i1, i2] >= 0)
        for i1, i2 in permutations(ids, 2)
    ))
    return less_than

def add_one_less_than_vars(m: Model, ids: List, less_than: tupledict, diff: tupledict):
    # Variable whose value is true if the difference between two elements equals 1
    one_less_than = m.addVars(permutations(ids, 2), vtype=GRB.BINARY)
    # The first two constraints make sure that if one_less_than == false, then diff != 1
    m.addConstrs((
        one_less_than[i1, i2] <= less_than[i1, i2]
        # i.e. if less_than == false, then one_less_than == false
        for i1, i2 in permutations(ids, 2)
    ))
    m.addConstrs((
        (less_than[i1, i2] == 1) >> (diff[i2, i1] >= 2 - one_less_than[i1, i2])
        # i.e. if less_than == true && one_less_than == false, then diff >= 2
        for i1, i2 in permutations(ids, 2)
    ))
    # The last constraint makes sure that if one_less_than == true, then diff == 1
    m.addConstrs((
        (one_less_than[i1, i2] == 1) >> (diff[i2, i1] == 1)
        # TODO: test this alternative: one_less_than[i1, i2] == one_less_than[i1, i2] * diff[i2, i1]
        for i1, i2 in permutations(ids, 2)
    ))

    return one_less_than




def get_directional_relationships(m: Model, elem_ids: List[str], x0: tupledict, x1: tupledict, y0: tupledict, y1: tupledict):

    # TODO: use consistent x1 and y1 coordinates, i.e. choose whether they are inclusive or exclusive

    above = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY)
    '''
    m.addConstrs((
        # TODO compare performance
        above[e1, e2] * (row_start[e2] - row_end[e1] - 1) + (1 - above[e1, e2]) * (row_end[e1] - row_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove1')
    '''
    m.addConstrs((
        # TODO compare performance
        (above[e1, e2] == 1) >> (y1[e1] <= y0[e2])
        # above[e1, e2] * (row_start[e2] - row_end[e1] - 1) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove1')
    m.addConstrs((
        # TODO compare performance
        (above[e1, e2] == 0) >> (y1[e1] + 1 >= y0[e2])
        # (1 - above[e1, e2]) * (row_end[e1] - row_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove2')

    m.addConstrs((
        above[e1, e2] + above[e2, e1] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='AboveSanity')  # TODO: check if sanity checks are necessary

    on_left = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY)
    m.addConstrs((
        # TODO compare performance
        (on_left[e1, e2] == 1) >> (x1[e1] <= x0[e2])
        # on_left[e1, e2] * (col_start[e2] - col_end[e1] - 1) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOnLeft1')
    m.addConstrs((
        # TODO compare performance
        (on_left[e1, e2] == 0) >> (x1[e1] + 1 >= x0[e2])
        # (1 - on_left[e1, e2]) * (col_end[e1] - col_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOnLeft2')
    m.addConstrs((
        on_left[e1, e2] + on_left[e2, e1] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='OnLeftSanity')

    return above, on_left

def add_overlap_vars(m: Model, elem_ids: List[str], above: tupledict, on_left: tupledict):
    horizontal_overlap = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY)
    m.addConstrs((
        horizontal_overlap[e1, e2] == 1 - (on_left[e1, e2] + on_left[e2, e1])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkHorizontalOverlap')

    vertical_overlap = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY)
    m.addConstrs((
        vertical_overlap[e1, e2] == 1 - (above[e1, e2] + above[e2, e1])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkVerticalOverlap')

    return horizontal_overlap, vertical_overlap

def prevent_overlap(m: Model, ids: List, horizontal_overlap: tupledict, vertical_overlap: tupledict):
    m.addConstrs((
        horizontal_overlap[e1, e2] + vertical_overlap[e1, e2] <= 1
        for e1, e2 in permutations(ids, 2)
    ))

def add_area_var(m: Model, width, height, max_width, max_height):
    widths = range(1, max_width + 1)
    heights = range(1, max_height + 1)

    chosen_width = m.addVars(widths, vtype=GRB.BINARY)
    m.addConstr(chosen_width.sum() == 1)  # One option must always be selected
    m.addConstrs((
        # TODO compare performance:
        (chosen_width[w] == 1) >> (width == w)
        # chosen_width[w] * w == chosen_width[w] * width
        # chosen_width[w] * (width - w) == 0
        for w in widths
    ))

    chosen_height = m.addVars(heights, vtype=GRB.BINARY)
    m.addConstr(chosen_height.sum() == 1) # One option must always be selected
    m.addConstrs((
        # TODO compare performance:
        (chosen_height[h] == 1) >> (height == h)
        # chosen_height[h] * w == chosen_height[h] * height
        # chosen_height[h] * (height - h) == 0
        for h in heights
    ))

    chosen_area = m.addVars(product(widths, heights), vtype=GRB.BINARY)
    m.addConstr(chosen_area.sum() == 1) # One option must always be selected
    m.addConstrs((
        chosen_area[w, h] == and_(chosen_width[w], chosen_height[h])
        for w, h in product(widths, heights)
    ))

    # The area of the elements in terms of cells, i.e. col_span * row_span
    # cell_coverage: row_span_equals[e,n] >> cell_coverage == n * row_span
    area = m.addVar(vtype=GRB.INTEGER, lb=1)
    m.addConstrs((
        # Using indicator constraint to avoid quadratic constraints
        (chosen_area[w, h] == 1) >> (area == w * h)
        for w, h in product(widths, heights)
    ))

    return area

def add_area_vars(m: Model, ids, width, height, max_width, max_height):
    widths = range(1, max_width + 1)
    heights = range(1, max_height + 1)

    chosen_width = m.addVars(product(ids, widths), vtype=GRB.BINARY, name='SelectedColumnCount')
    m.addConstrs((
        chosen_width.sum(i) == 1 # One option must always be selected
        for i in ids
    ))
    m.addConstrs((
        # TODO compare performance:
        (chosen_width[i, w] == 1) >> (width[i] == w)
        # chosen_width[w] * w == chosen_width[w] * width
        # chosen_width[i, w] * (width[i] - w) == 0
        for i, w in product(ids, widths)
    ))

    chosen_height = m.addVars(product(ids, heights), vtype=GRB.BINARY, name='SelectedRowCount')
    m.addConstrs((
        chosen_height.sum(i) == 1  # One option must always be selected
        for i in ids
    ))
    m.addConstrs((
        # TODO compare performance:
        (chosen_height[i, h] == 1) >> (height[i] == h)
        # chosen_height[h] * w == chosen_height[h] * height
        # chosen_height[i, h] * (height[i] - h) == 0
        for i, h in product(ids, heights)
    ))

    chosen_area = m.addVars(product(ids, widths, heights), vtype=GRB.BINARY)
    m.addConstrs((
        chosen_area.sum(i) == 1  # One option must always be selected
        for i in ids
    ))
    m.addConstrs((
        chosen_area[i, w, h] == and_(chosen_width[i, w], chosen_height[i, h])
        for i, w, h in product(ids, widths, heights)
    ))

    # The area of the elements in terms of cells, i.e. col_span * row_span
    # cell_coverage: row_span_equals[e,n] >> cell_coverage == n * row_span
    area = m.addVars(ids, vtype=GRB.INTEGER, lb=1)
    m.addConstrs((
        # Using indicator constraint to avoid quadratic constraints
        (chosen_area[i, w, h] == 1) >> (area[i] == w * h)
        for i, w, h in product(ids, widths, heights)
    ))

    return area



def compute_minimum_grid(n: int) -> int:
    min_grid_width = int(sqrt(n))
    elements_in_min_grid = min_grid_width**2
    extra_elements = n - elements_in_min_grid
    if extra_elements == 0:
        result = 4 * min_grid_width
    else:
        extra_columns = int(extra_elements / min_grid_width)
        remainder = (extra_elements - (extra_columns * min_grid_width))
        if remainder == 0:
            result = (4 * min_grid_width) + (2 * extra_columns)
        else:
            result = (4 * min_grid_width) + (2 * extra_columns) + 2
    return result