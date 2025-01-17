from collections import namedtuple
from enum import Enum
from itertools import product, permutations
from math import ceil, floor, sqrt
from typing import List

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, or_, QuadExpr, GurobiError

from .classes import Layout, Element

from optimizer import util





def equal_width_columns(m: Model, elements: List[Element], available_width, available_height, width, height, gutter_width, offset_x, offset_y):

    # TODO: desktop web layouts take too long for presolve (check the number of top level elements)
    # TODO: try to simplify this grid alignment, but add an objective to standardize grid width
    # TODO: e.g. for above: add binary variable to check if width is correct multiple of col width (minimize wrong values)
    # TODO: (maybe) add variables so that if element is wider than another, its colspan must also be wider

    # TODO: (maybe) add weight for size loss based on element original size (smaller elements should be scaled down less)

    # TODO: make a simpler version of this for inner alignment
    # TODO: (maybe add rule that if two elements are next to each other, and their top and bottom edges align, link those coordinates together)
    # TODO: (maybe) add an objective to maintain horizontal and vertical overlap (e.g. if at least half of the other element share the same vertical/horizontal space)



    elem_count = len(elements)
    elem_ids = [e.id for e in elements]

    # TODO: compute a proper maximum column/row count
    max_col_count = 12
    max_row_count = 100

    # MINIMIZE LAYOUT COLUMNS
    # TODO: what about preferred number of columns?


    # Element coordinates in base units
    x0, y0, x1, y1 = util.add_coord_vars(m, elem_ids, available_width, available_height)

    m.addConstrs((
        x0[e] + width[e] == x1[e]
        for e in elem_ids
    ))
    m.addConstrs((
        y0[e] + height[e] == y1[e]
        for e in elem_ids
    ))

    m.addConstrs((
        width[i] >= 1
        for i in elem_ids
    ))
    m.addConstrs((
        height[i] >= 1
        for i in elem_ids
    ))

    w_diff, h_diff = [
        util.add_pairwise_diff(m, elem_ids, var)
        for var in [width, height]
    ]

    x0_diff, y0_diff, x1_diff, y1_diff = [
        util.add_pairwise_diff(m, elem_ids, var)
        for var in [x0, y0, x1, y1]
    ]

    x0_less_than, y0_less_than, x1_less_than, y1_less_than = [
        util.add_less_than_vars(m, elem_ids, vars)
        for vars in [x0_diff, y0_diff, x1_diff, y1_diff]
    ]

    x0x1_diff, y0y1_diff  = [
        util.add_pairwise_diff(m, elem_ids, var1, var2)
        for var1, var2 in [(x0, x1), (y0, y1)]
    ]

    # Element coordinates in rows and columns
    c0, r0, c1, r1 = util.add_coord_vars(m, elem_ids, max_col_count, max_row_count)

    c0_min = m.addVar(vtype=GRB.INTEGER)
    m.addConstr(c0_min == min_(c0))
    m.addConstr(c0_min == 0) # At least one element must be in the first column

    col_span, row_span = [
        add_diff_vars(m, elem_ids, var1, var2)
        for var1, var2 in [(c1, c0), (r1, r0)]
    ]

    cs_diff, rs_diff = [
        util.add_pairwise_diff(m, elem_ids, var)
        for var in [col_span, row_span]
    ]

    c0_diff, r0_diff, c1_diff, r1_diff = [util.add_pairwise_diff(m, elem_ids, var) for var in [c0, r0, c1, r1]]

    c0_less_than, r0_less_than, c1_less_than, r1_less_than = [util.add_less_than_vars(m, elem_ids, vars) for vars in [c0_diff, r0_diff, c1_diff, r1_diff]]




    w_less_than, h_less_than, cs_less_than, rs_less_than = [
        util.add_less_than_vars(m, elem_ids, var)
        for var in [w_diff, h_diff, cs_diff, rs_diff]
    ]





    # Link order of variables, e.g.
    # * if x0[A] < x0[B], then c0[A] < c0[B]
    # * if x0[A] = x0[B], then c0[A] = c0[B]
    # * if x0[A] > x0[B], then c0[A] > c0[B]
    m.addConstrs((x0_less_than[i1, i2] == c0_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((x1_less_than[i1, i2] == c1_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((y0_less_than[i1, i2] == r0_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((y1_less_than[i1, i2] == r1_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))

    m.addConstrs((w_less_than[i1, i2] == cs_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((h_less_than[i1, i2] == rs_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))

    # COLUMN/ROW SIZE & COUNT

    grid_width = m.addVar(lb=1, vtype=GRB.INTEGER)
    actual_width = m.addVar(lb=1, vtype=GRB.INTEGER)
    m.addConstr(actual_width == grid_width - gutter_width)
    width_error = m.addVar(lb=1, vtype=GRB.INTEGER)
    m.addConstr(width_error == available_width - actual_width)

    col_width = m.addVar(lb=1, vtype=GRB.INTEGER) # in base units
    m.addConstr(col_width >= gutter_width + 1)

    col_count = m.addVar(lb=1, vtype=GRB.INTEGER)
    m.addConstr(col_count == max_(c1))

    col_counts = range(1, max_col_count + 1)
    col_count_selected = m.addVars(col_counts, vtype=GRB.BINARY)
    m.addConstr(col_count_selected.sum() == 1)
    m.addConstrs((
        (col_count_selected[c] == 1) >> (col_count == c)
        for c in col_counts
    ))
    m.addConstrs((
        (col_count_selected[c] == 1) >> (grid_width == c * col_width)
        for c in col_counts
    ))
    m.addConstr(actual_width <= available_width)
    m.addConstr(actual_width + col_width - gutter_width >= available_width + 1) # This should stretch the column width to match the layout width

    row_height = m.addVar(lb=1, vtype=GRB.INTEGER) # in base units
    m.addConstr(row_height >= gutter_width + 1)
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

    # If two elements are different widths, the difference must be at least one col_width
    m.addConstrs((
        (w_less_than[i1, i2] == 1) >> (w_diff[i2, i1] >= col_width)
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

    # If two elements are different heights, the difference must be at least one row_height
    m.addConstrs((
        (h_less_than[i1, i2] == 1) >> (h_diff[i2, i1] >= row_height)
        for i1, i2 in permutations(elem_ids, 2)
    ))



    # Consecutive column/row lines must be column width/row height apart
    # e.g. if element A starts in column 2, and element B starts in column 3, x0[B] - x0[A] == col_width
    c0_one_less_than = add_one_less_than_vars(m, elem_ids, c0_less_than, c0_diff)
    c1_one_less_than = add_one_less_than_vars(m, elem_ids, c1_less_than, c1_diff)

    m.addConstrs((
        (c0_one_less_than[i1, i2] == 1) >> (x0_diff[i2, i1] == col_width)
        for i1, i2 in permutations(elem_ids, 2)
    ))
    m.addConstrs((
        (c1_one_less_than[i1, i2] == 1) >> (x1_diff[i2, i1] == col_width)
        for i1, i2 in permutations(elem_ids, 2)
    ))

    r0_one_less_than = add_one_less_than_vars(m, elem_ids, r0_less_than, r0_diff)
    r1_one_less_than = add_one_less_than_vars(m, elem_ids, r1_less_than, r1_diff)

    m.addConstrs((
        (r0_one_less_than[i1, i2] == 1) >> (y0_diff[i2, i1] == row_height)
        for i1, i2 in permutations(elem_ids, 2)
    ))
    m.addConstrs((
        (r1_one_less_than[i1, i2] == 1) >> (y1_diff[i2, i1] == row_height)
        for i1, i2 in permutations(elem_ids, 2)
    ))


    # Prevent overlap
    util.prevent_overlap(m, elem_ids, x0x1_diff, y0y1_diff, min_distance=gutter_width)

    util.preserve_relationships(m, elements, x0, x1, y0, y1)

    # Prevent empty rows and columns

    # Horizontal
    c0_equals_c1 = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY)
    m.addConstrs((
        (c0_equals_c1[i1, i2] == 1) >> (c0[i1] == c1[i2])
        for i1, i2 in permutations(elem_ids, 2)
    ))

    m.addConstrs((
        # Set gutter TODO: check if this is necessary
        (c0_equals_c1[i1, i2] == 1) >> (x0[i1] - x1[i2] == gutter_width)
        for i1, i2 in permutations(elem_ids, 2)
    ))

    in_first_col = m.addVars(elem_ids, vtype=GRB.BINARY)
    m.addConstrs((
        (in_first_col[i] == 1) >> (c0[i] == 0)
        for i in elem_ids
    ))
    m.addConstrs((
        (in_first_col[i] == 0) >> (c0[i] >= 1)
        for i in elem_ids
    ))


    something_on_left = m.addVars(elem_ids, vtype=GRB.BINARY)
    m.addConstrs((
        #something_on_left[i] <= c0_equals_c1.sum(i)
        # TODO compare performance
        something_on_left[i] == max_(c0_equals_c1.select(i, '*'))
        for i in elem_ids
    ))

    no_gap_on_left = m.addVars(elem_ids, vtype=GRB.BINARY)
    m.addConstrs((
        no_gap_on_left[i] == or_(something_on_left[i], in_first_col[i])
        for i in elem_ids
    ))



    # Vertical
    r0_equals_r1 = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY)
    m.addConstrs((
        (r0_equals_r1[i1, i2] == 1) >> (r0[i1] == r1[i2])
        for i1, i2 in permutations(elem_ids, 2)
    ))

    m.addConstrs((
        # Set gutter
        (r0_equals_r1[i1, i2] == 1) >> (y0[i1] - y1[i2] == gutter_width)
        for i1, i2 in permutations(elem_ids, 2)
    ))

    on_first_row = m.addVars(elem_ids, vtype=GRB.BINARY)
    m.addConstrs((
        (on_first_row[i] == 1) >> (r0[i] == 0)
        for i in elem_ids
    ))
    m.addConstrs((
        (on_first_row[i] == 0) >> (r0[i] >= 1)
        for i in elem_ids
    ))

    something_above = m.addVars(elem_ids, vtype=GRB.BINARY)
    m.addConstrs((
        #something_above[i] <= r0_equals_r1.sum(i)
        # TODO compare performance
        something_above[i] == max_(r0_equals_r1.select(i, '*'))
        for i in elem_ids
    ))

    no_gap_above = m.addVars(elem_ids, vtype=GRB.BINARY)
    m.addConstrs((
        no_gap_above[i] == or_(something_above[i], on_first_row[i])
        for i in elem_ids
    ))

    m.addConstr(no_gap_on_left.sum() == elem_count)
    m.addConstr(no_gap_above.sum() == elem_count)


    # TODO: add expression (needs to be linear) for the column width error, i.e. ((col_span[i] * col_width - gutter_width) - width[i])


    if False:
        cell_count = add_area_vars(m, elem_ids, col_span, row_span, max_col_count, max_row_count)
        total_cell_count = cell_count.sum()

        grid_area = add_area_var(m, col_count, row_count, max_col_count, max_row_count)

        gap_count = grid_area - total_cell_count
    else:
        gap_count = LinExpr(0)


    actual_height = m.addVar(vtype=GRB.INTEGER)
    m.addConstr(actual_height == max_(y1))
    height_error = available_height - actual_height

    def get_rel_xywh(element_id):
        # Returns the element position (relative to the grid top left corner)

        # Attribute Xn refers to the variable value in the solution selected using SolutionNumber parameter.
        # When SolutionNumber equals 0 (default), Xn refers to the variable value in the best solution.
        # https://www.gurobi.com/documentation/8.1/refman/xn.html#attr:Xn
        x = offset_x.getValue() + x0[element_id].Xn
        y = offset_y.getValue() + y0[element_id].Xn
        w = width[element_id].Xn
        h = height[element_id].Xn

        if True:
            print(
                'Cols', col_count.Xn,
                'ColW', col_width.Xn,
                'ElemW', w, 'ElemH', h,
                'ColSpan', col_span[element_id].Xn,
                'Error', ((col_span[element_id].Xn * col_width.Xn - gutter_width.Xn ) - w),
                'Id', element_id
            )

        return x, y, w, h

    return get_rel_xywh, width_error, height_error, gap_count




def add_diff_vars(m: Model, ids: List, var1, var2):
    diff = m.addVars(ids, lb=-GRB.INFINITY, vtype=GRB.INTEGER)
    m.addConstrs((
        diff[i] == var1[i] - var2[i]
        for i in ids
    ))
    return diff



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