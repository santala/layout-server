from collections import namedtuple
from enum import Enum
from itertools import product, permutations
from math import ceil, floor, sqrt
from typing import List

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, or_, QuadExpr, GurobiError

from .classes import Layout, Element

from optimizer import util

def improve_alignment(m: Model, elements: List[Element], available_width, available_height, elem_width, elem_height):

    elem_ids = [element.id for element in elements]
    elem_count = len(elem_ids)

    elem_x0 = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementX0')
    elem_y0 = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementY0')
    elem_x1 = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementX1')
    elem_y1 = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementY1')

    m.addConstrs((
        elem_x0[e] + elem_width[e] == elem_x1[e]
        for e in elem_ids
    ), name='LinkX1ToWidth')
    m.addConstrs((
        elem_y0[e] + elem_height[e] == elem_y1[e]
        for e in elem_ids
    ), name='LinkY1ToHeight')

    # Constrain element to the available area
    m.addConstrs((
        elem_x1[e] <= available_width
        for e in elem_ids
    ), name='ContainElementToAvailableWidth')
    m.addConstrs((
        elem_y1[e] <= available_height
        for e in elem_ids
    ), name='ContainElementToAvailableHeight')



    x0_diff, y0_diff, x1_diff, y1_diff = [
        m.addVars(permutations(elem_ids, 2), lb=-GRB.INFINITY, vtype=GRB.INTEGER, name=name)
        for name in ['X0Diff', 'Y0Diff', 'X1Diff', 'Y1Diff']
    ]
    for diff, var in zip([x0_diff, y0_diff, x1_diff, y1_diff], [elem_x0, elem_y0, elem_x1, elem_y1]):
        m.addConstrs((
            diff[i1, i2] == var[i1] - var[i2]
            for i1, i2 in permutations(elem_ids, 2)
        ))

    x0_less_than = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='X0LessThan')
    m.addConstrs((
        (x0_less_than[i1, i2] == 1) >> (x0_diff[i1, i2] <= -1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkX0LessThan1')
    m.addConstrs((
        (x0_less_than[i1, i2] == 0) >> (x0_diff[i1, i2] >= 0)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkX0LessThan2')

    y0_less_than = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='Y0LessThan')
    m.addConstrs((
        (y0_less_than[i1, i2] == 1) >> (y0_diff[i1, i2] <= -1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY0LessThan1')
    m.addConstrs((
        (y0_less_than[i1, i2] == 0) >> (y0_diff[i1, i2] >= 0)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY0LessThan2')

    x1_less_than = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='X1LessThan')
    m.addConstrs((
        (x1_less_than[i1, i2] == 1) >> (x1_diff[i1, i2] <= -1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkX1LessThan1')
    m.addConstrs((
        (x1_less_than[i1, i2] == 0) >> (x1_diff[i1, i2] >= 0)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkX1LessThan2')

    y1_less_than = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='Y1LessThan')
    m.addConstrs((
        (y1_less_than[i1, i2] == 1) >> (y1_diff[i1, i2] <= -1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY1LessThan1')
    m.addConstrs((
        (y1_less_than[i1, i2] == 0) >> (y1_diff[i1, i2] >= 0)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY1LessThan2')

    # ALT NUMBER OF GROUPS
    x0_group = m.addVars(elem_ids, lb=1, ub=elem_count, vtype=GRB.INTEGER, name='X0Group')
    y0_group = m.addVars(elem_ids, lb=1, ub=elem_count, vtype=GRB.INTEGER, name='Y0Group')
    x1_group = m.addVars(elem_ids, lb=1, ub=elem_count, vtype=GRB.INTEGER, name='X1Group')
    y1_group = m.addVars(elem_ids, lb=1, ub=elem_count, vtype=GRB.INTEGER, name='Y1Group')
    m.addConstrs((
        (x0_less_than[i1, i2] == 1) >> (x0_group[i1] <= x0_group[i2] - 1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkX0Group1')
    m.addConstrs((
        (x0_less_than[i1, i2] == 0) >> (x0_group[i1] >= x0_group[i2])
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkX0Group2')
    m.addConstrs((
        (y0_less_than[i1, i2] == 1) >> (y0_group[i1] <= y0_group[i2] - 1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY0Group1')
    m.addConstrs((
        (y0_less_than[i1, i2] == 0) >> (y0_group[i1] >= y0_group[i2])
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY0Group2')
    m.addConstrs((
        (x1_less_than[i1, i2] == 1) >> (x1_group[i1] <= x1_group[i2] - 1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkX1Group1')
    m.addConstrs((
        (x1_less_than[i1, i2] == 0) >> (x1_group[i1] >= x1_group[i2])
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkX1Group2')
    m.addConstrs((
        (y1_less_than[i1, i2] == 1) >> (y1_group[i1] <= y1_group[i2] - 1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY1Group1')
    m.addConstrs((
        (y1_less_than[i1, i2] == 0) >> (y1_group[i1] >= y1_group[i2])
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY1Group2')

    x0_group_count = m.addVar(lb=1, ub=elem_count, vtype=GRB.INTEGER, name='X0GroupCount')
    y0_group_count = m.addVar(lb=1, ub=elem_count, vtype=GRB.INTEGER, name='Y0GroupCount')
    x1_group_count = m.addVar(lb=1, ub=elem_count, vtype=GRB.INTEGER, name='X1GroupCount')
    y1_group_count = m.addVar(lb=1, ub=elem_count, vtype=GRB.INTEGER, name='Y1GroupCount')
    m.addConstr(x0_group_count == max_(x0_group))
    m.addConstr(y0_group_count == max_(y0_group))
    m.addConstr(x1_group_count == max_(x1_group))
    m.addConstr(y1_group_count == max_(y1_group))

    total_group_count = x0_group_count + y0_group_count + x1_group_count + y1_group_count
    m.addConstr(total_group_count >= compute_minimum_grid(elem_count)) # Prevent over-optimization

    util.preserve_relationships(m, elements, elem_x0, elem_x1, elem_y0, elem_y1)

    x0x1_diff, y0y1_diff = [
        util.add_pairwise_diff(m, elem_ids, var1, var2)
        for var1, var2 in [(elem_x0, elem_x1), (elem_y0, elem_y1)]
    ]

    util.prevent_overlap(m, elem_ids, x0x1_diff, y0y1_diff, min_distance=1) # TODO: configurable minimum width

    def get_rel_xywh(element_id):
        # Returns the element position (relative to the parent top left corner)

        # Attribute Xn refers to the variable value in the solution selected using SolutionNumber parameter.
        # When SolutionNumber equals 0 (default), Xn refers to the variable value in the best solution.
        # https://www.gurobi.com/documentation/8.1/refman/xn.html#attr:Xn
        x = elem_x0[element_id].Xn
        y = elem_y0[element_id].Xn
        w = elem_width[element_id].Xn
        h = elem_height[element_id].Xn

        return x, y, w, h

    return get_rel_xywh, total_group_count

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


def get_directional_relationships(m: Model, elem_ids: List[str], x0: tupledict, x1: tupledict, y0: tupledict, y1: tupledict):

    # TODO: use consistent x1 and y1 coordinates, i.e. choose whether they are inclusive or exclusive

    above = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='Above')
    '''
    m.addConstrs((
        # TODO compare performance
        above[e1, e2] * (row_start[e2] - row_end[e1] - 1) + (1 - above[e1, e2]) * (row_end[e1] - row_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove1')
    '''
    m.addConstrs((
        # TODO compare performance
        (above[e1, e2] == 1) >> (y1[e1] + 1 <= y0[e2])
        # above[e1, e2] * (row_start[e2] - row_end[e1] - 1) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove1')
    m.addConstrs((
        # TODO compare performance
        (above[e1, e2] == 0) >> (y1[e1] >= y0[e2])
        # (1 - above[e1, e2]) * (row_end[e1] - row_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove2')

    m.addConstrs((
        above[e1, e2] + above[e2, e1] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='AboveSanity')  # TODO: check if sanity checks are necessary

    on_left = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='OnLeft')
    m.addConstrs((
        # TODO compare performance
        (on_left[e1, e2] == 1) >> (x1[e1] + 1 <= x0[e2])
        # on_left[e1, e2] * (col_start[e2] - col_end[e1] - 1) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOnLeft1')
    m.addConstrs((
        # TODO compare performance
        (on_left[e1, e2] == 0) >> (x1[e1] >= x0[e2])
        # (1 - on_left[e1, e2]) * (col_end[e1] - col_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOnLeft2')
    m.addConstrs((
        on_left[e1, e2] + on_left[e2, e1] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='OnLeftSanity')

    return DirectionalRelationships(above=above, on_left=on_left)