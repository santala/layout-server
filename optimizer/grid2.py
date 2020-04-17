from itertools import product, permutations
from math import ceil, floor, sqrt
from typing import List

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, or_, QuadExpr, GurobiError, Var

from .classes import Layout, Element

from optimizer import util


def equal_width_columns(m: Model, elements: List[Element], available_width, available_height, w, h, gutter_width, offset_x, offset_y):

    elem_ids = [e.id for e in elements]

    # TODO: compute a proper maximum column/row count
    max_col_count = 12
    max_row_count = 100

    # VARIABLES

    # Element coordinates in base units
    x0, y0, x1, y1 = util.add_coord_vars(m, elem_ids, available_width, available_height)
    # Element coordinates in rows and columns
    c0, r0, c1, r1 = util.add_coord_vars(m, elem_ids, max_col_count, max_row_count)

    w_diff, h_diff = [util.add_pairwise_diff(m, elem_ids, var) for var in [w, h]]
    x0_diff, y0_diff, x1_diff, y1_diff = [util.add_pairwise_diff(m, elem_ids, var) for var in [x0, y0, x1, y1]]

    x0_less_than, y0_less_than, x1_less_than, y1_less_than = [util.add_less_than_vars(m, elem_ids, vars) for vars in [x0_diff, y0_diff, x1_diff, y1_diff]]

    x0x1_diff, y0y1_diff = [util.add_pairwise_diff(m, elem_ids, var1, var2) for var1, var2 in [(x0, x1), (y0, y1)]]

    col_span, row_span = [add_diff_vars(m, elem_ids, var1, var2) for var1, var2 in [(c1, c0), (r1, r0)]]

    cs_diff, rs_diff = [util.add_pairwise_diff(m, elem_ids, var) for var in [col_span, row_span]]

    c0_diff, r0_diff, c1_diff, r1_diff = [util.add_pairwise_diff(m, elem_ids, var) for var in [c0, r0, c1, r1]]

    c0_less_than, r0_less_than, c1_less_than, r1_less_than = [util.add_less_than_vars(m, elem_ids, vars) for vars in [c0_diff, r0_diff, c1_diff, r1_diff]]

    w_less_than, h_less_than, cs_less_than, rs_less_than = [util.add_less_than_vars(m, elem_ids, var) for var in [w_diff, h_diff, cs_diff, rs_diff]]

    # AVAILABLE WIDTH vs ACTUAL WIDTH

    grid_width = m.addVar(lb=1, vtype=GRB.INTEGER)
    actual_width = m.addVar(lb=1, vtype=GRB.INTEGER)
    m.addConstr(actual_width == grid_width - gutter_width)
    width_error = m.addVar(lb=1, vtype=GRB.INTEGER)
    m.addConstr(width_error == available_width - actual_width)
    m.addConstr(actual_width <= available_width)

    # COLUMN WIDTH
    col_width = m.addVar(lb=1, vtype=GRB.INTEGER)  # in base units
    # must be wider than gutter
    m.addConstr(col_width >= gutter_width + 1)

    # The width error must be less than one column width
    m.addConstr(width_error + 1 <= col_width - gutter_width)  # This should stretch the column width to match the layout width

    # COLUMN COUNT
    col_count = m.addVar(lb=1, ub=max_col_count, vtype=GRB.INTEGER)
    # is equal to the rightmost end column
    m.addConstr(col_count == max_(c1))

    # helpers for column count
    col_count_options = range(1, max_col_count + 1)
    col_count_selected = m.addVars(col_count_options, vtype=GRB.BINARY)
    m.addConstr(col_count_selected.sum() == 1)
    for c in col_count_options:
        m.addConstr((col_count_selected[c] == 1) >> (col_count == c))
        m.addConstr((col_count_selected[c] == 1) >> (grid_width == c * col_width))

    # GRID ROWS

    row_height = m.addVar(lb=1, vtype=GRB.INTEGER)  # in base units
    m.addConstr(row_height >= gutter_width + 1)
    row_count = m.addVar(lb=1, ub=max_row_count, vtype=GRB.INTEGER)
    m.addConstr(row_count == max_(r1))

    # Consecutive column/row lines must be column width/row height apart
    # e.g. if element A starts in column 2, and element B starts in column 3, x0[B] - x0[A] == col_width
    c0_one_less_than = add_one_less_than_vars(m, elem_ids, c0_less_than, c0_diff)
    c1_one_less_than = add_one_less_than_vars(m, elem_ids, c1_less_than, c1_diff)

    r0_one_less_than = add_one_less_than_vars(m, elem_ids, r0_less_than, r0_diff)
    r1_one_less_than = add_one_less_than_vars(m, elem_ids, r1_less_than, r1_diff)

    for i in elem_ids:

        m.addConstr(x0[i] + w[i] == x1[i])
        m.addConstr(y0[i] + h[i] == y1[i])
        m.addConstr(w[i] >= 1)
        m.addConstr(h[i] >= 1)

    for i, j in permutations(elem_ids, 2):

        # Link order of variables, e.g.
        # * if x0[A] < x0[B], then c0[A] < c0[B]
        # * if x0[A] = x0[B], then c0[A] = c0[B]
        # * if x0[A] > x0[B], then c0[A] > c0[B]
        m.addConstr(x0_less_than[i, j] == c0_less_than[i, j])
        m.addConstr(x1_less_than[i, j] == c1_less_than[i, j])
        m.addConstr(y0_less_than[i, j] == r0_less_than[i, j])
        m.addConstr(y1_less_than[i, j] == r1_less_than[i, j])
        m.addConstr(w_less_than[i, j] == cs_less_than[i, j])
        m.addConstr(h_less_than[i, j] == rs_less_than[i, j])

        # If element x-coordinates differ, they must be at least one column width apart
        m.addConstr((x0_less_than[i, j] == 1) >> (x0[j] - x0[i] >= col_width))
        m.addConstr((x1_less_than[i, j] == 1) >> (x1[j] - x1[i] >= col_width))

        # If element widths differ, they must differ at least one column width
        m.addConstr((w_less_than[i, j] == 1) >> (w[j] - w[i] >= col_width))

        # If element y-coordinates differ, they must be at least one row height apart
        m.addConstr((y0_less_than[i, j] == 1) >> (y0[j] - y0[i] >= row_height))
        m.addConstr((y1_less_than[i, j] == 1) >> (y1[j] - y1[i] >= row_height))

        # If element heights differ, they must differ at least one row height
        m.addConstr((h_less_than[i, j] == 1) >> (h[j] - h[i] >= row_height))

        # If the difference is one row/column, the matching coordinates must be
        # exactly one row height/column width apart
        m.addConstr((c0_one_less_than[i, j] == 1) >> (x0[j] - x0[i] == col_width))
        m.addConstr((c1_one_less_than[i, j] == 1) >> (x1[j] - x1[i] == col_width))
        m.addConstr((r0_one_less_than[i, j] == 1) >> (y0[j] - y0[i] == row_height))
        m.addConstr((r1_one_less_than[i, j] == 1) >> (y1[j] - y1[i] == row_height))

    # GRID CONSTRAINTS
    # no gaps in columns
    reduce_gaps_in_grid(m, elem_ids, c0, c1, x0, x1, gutter_width)
    # no gaps in rows
    reduce_gaps_in_grid(m, elem_ids, r0, r1, y0, y1, gutter_width)

    # Prevent overlap
    util.prevent_overlap(m, elem_ids, x0x1_diff, y0y1_diff, min_distance=gutter_width)

    util.preserve_relationships(m, elements, x0, x1, y0, y1)

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
        width = w[element_id].Xn
        height = h[element_id].Xn

        if True:
            print(
                'Cols', col_count.Xn,
                'ColW', col_width.Xn,
                'ElemW', width, 'ElemH', height,
                'ColSpan', col_span[element_id].Xn,
                'Error', ((col_span[element_id].Xn * col_width.Xn - gutter_width.Xn ) - width),
                'Id', element_id
            )

        return x, y, width, height

    return get_rel_xywh, width_error, height_error, gap_count


def reduce_gaps_in_grid(m: Model, ids: List, grid_start: tupledict, grid_end: tupledict, coord_start: tupledict, coord_end: tupledict, gutter_width: Var):
    id_pairs = list(permutations(ids, 2))

    starts_from_zero = m.addVars(ids, vtype=GRB.BINARY)
    no_gap_on_left = m.addVars(ids, vtype=GRB.BINARY)
    grid_start_end_diff = m.addVars(id_pairs, vtype=GRB.INTEGER, lb=-GRB.INFINITY)
    coord_start_end_diff = m.addVars(id_pairs, vtype=GRB.INTEGER, lb=-GRB.INFINITY)
    starts_right_after = m.addVars(id_pairs, vtype=GRB.BINARY)

    for i in ids:
        # Element must either start either in the 1st column,
        # or right after another element ends
        m.addConstr(starts_from_zero[i] + no_gap_on_left[i] >= 1)
        # If the element starts from zero, the start coord must equal 0
        m.addConstr((starts_from_zero[i] == 1) >> (grid_start[i] == 0))
        # If this element does not start right after any other, there is a gap
        m.addConstr(no_gap_on_left[i] <= starts_right_after.sum(i, '*'))

    for i, j in id_pairs:
        # Difference between the start row/column (inclusive) of this element
        # and the end row/column (exclusive) of another element
        m.addConstr(grid_start_end_diff[i, j] == grid_start[i] - grid_end[j])
        # Difference in coordinate units
        m.addConstr(coord_start_end_diff[i, j] == coord_start[i] - coord_end[j])
        # If the element starts right after another, its start row/column
        # must equal the end row/column of the other
        m.addConstr((starts_right_after[i, j] == 1) >> (grid_start_end_diff[i, j] == 0))
        # and the coordinate unit difference must match the gutter
        m.addConstr((starts_right_after[i, j] == 1) >> (coord_start_end_diff[i, j] == gutter_width))




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
        one_less_than[i, j] <= less_than[i, j]
        # i.e. if less_than == false, then one_less_than == false
        for i, j in permutations(ids, 2)
    ))
    m.addConstrs((
        (less_than[i, j] == 1) >> (diff[j, i] >= 2 - one_less_than[i, j])
        # i.e. if less_than == true && one_less_than == false, then diff >= 2
        for i, j in permutations(ids, 2)
    ))
    # The last constraint makes sure that if one_less_than == true, then diff == 1
    m.addConstrs((
        (one_less_than[i, j] == 1) >> (diff[j, i] == 1)
        # TODO: test this alternative: one_less_than[i, j] == one_less_than[i, j] * diff[j, i]
        for i, j in permutations(ids, 2)
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


def less_than(m: Model, v1: Var, v2: Var):
    lt = m.addVar(vtype=GRB.BINARY)
    m.addConstr((lt == 1) >> (v1 <= v2 - 1))
    m.addConstr((lt == 0) >> (v1 >= v2))
    return lt
