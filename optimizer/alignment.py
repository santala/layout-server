from collections import namedtuple
from enum import Enum
from itertools import product, permutations
from math import ceil, floor, sqrt
from typing import List

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, or_, QuadExpr, GurobiError

from .classes import Layout, Element

from optimizer import util

def improve_alignment(m: Model, elements: List[Element], available_width, available_height, width, height):

    elem_ids = [element.id for element in elements]
    elem_count = len(elem_ids)

    x0, y0, x1, y1 = util.add_coord_vars(m, elem_ids, available_width, available_height)

    # TODO: preference that all of the available space is used, i.e. elements should span

    # TODO: preference that distance to adjacent elements would be constant
    # TODO: e.g. if an element has two adjacent elements that are the same distance away, prefer to keep those distances equal

    # TODO: (maybe) if two elements are ≤ 16px (2*‘related distance’) from each other, prefer them to be 8px distance,
    # TODO: i.e. treat them as ‘related’

    m.addConstrs((
        x0[e] + width[e] == x1[e]
        for e in elem_ids
    ))
    m.addConstrs((
        y0[e] + height[e] == y1[e]
        for e in elem_ids
    ))

    x0_diff, y0_diff, x1_diff, y1_diff = [
        util.add_pairwise_diff(m, elem_ids, var)
        for var in [x0, y0, x1, y1]
    ]

    x0_less_than, y0_less_than, x1_less_than, y1_less_than = [
        util.add_less_than_vars(m, elem_ids, vars)
        for vars in [x0_diff, y0_diff, x1_diff, y1_diff]
    ]

    # Element coordinates in rows and columns

    c0, r0, c1, r1 = util.add_coord_vars(m, elem_ids, elem_count, elem_count)

    c0_diff, r0_diff, c1_diff, r1_diff = [util.add_pairwise_diff(m, elem_ids, var) for var in [c0, r0, c1, r1]]

    c0_less_than, r0_less_than, c1_less_than, r1_less_than = [util.add_less_than_vars(m, elem_ids, vars) for vars in
                                                              [c0_diff, r0_diff, c1_diff, r1_diff]]

    # Link order of variables, e.g.
    # * if x0[A] < x0[B], then c0[A] < c0[B]
    # * if x0[A] = x0[B], then c0[A] = c0[B]
    # * if x0[A] > x0[B], then c0[A] > c0[B]
    
    m.addConstrs((x0_less_than[i1, i2] == c0_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((x1_less_than[i1, i2] == c1_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((y0_less_than[i1, i2] == r0_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))
    m.addConstrs((y1_less_than[i1, i2] == r1_less_than[i1, i2] for i1, i2 in permutations(elem_ids, 2)))


    max_c0 = m.addVar(lb=0, ub=elem_count, vtype=GRB.INTEGER, name='X0GroupCount')
    max_r0 = m.addVar(lb=0, ub=elem_count, vtype=GRB.INTEGER, name='Y0GroupCount')
    max_c1 = m.addVar(lb=0, ub=elem_count, vtype=GRB.INTEGER, name='X1GroupCount')
    max_r1 = m.addVar(lb=0, ub=elem_count, vtype=GRB.INTEGER, name='Y1GroupCount')
    m.addConstr(max_c0 == max_(c0))
    m.addConstr(max_r0 == max_(r0))
    m.addConstr(max_c1 == max_(c1))
    m.addConstr(max_r1 == max_(r1))

    
    total_group_count = 4 + max_c0 + max_r0 + max_c1 + max_r1
    #m.addConstr(total_group_count >= compute_minimum_grid(elem_count)) # Prevent over-optimization
    # TODO: minimum grid computation should probably take into account preservation of relationships,
    # TODO: but this can be a bit complex to compute. Seems to work fine without the above constraint though.

    util.preserve_relationships(m, elements, x0, x1, y0, y1)

    x0x1_diff, y0y1_diff = [
        util.add_pairwise_diff(m, elem_ids, var1, var2)
        for var1, var2 in [(x0, x1), (y0, y1)]
    ]

    util.prevent_overlap(m, elem_ids, x0x1_diff, y0y1_diff, min_distance=1) # TODO: configurable minimum width

    # Preserve alignment

    loss_of_alignment = LinExpr(0)

    for element, other in permutations(elements, 2):
        if element.is_adjacent(other):

            if element.x0 == other.x0:
                x0_equals = util.add_equals_var(m, x0[element.id], x0[other.id])
                loss_of_alignment.add(1 - x0_equals)
            if element.x1 == other.x1:
                x1_equals = util.add_equals_var(m, x1[element.id], x1[other.id])
                loss_of_alignment.add(1 - x1_equals)
            if element.y0 == other.y0:
                y0_equals = util.add_equals_var(m, y0[element.id], y0[other.id])
                loss_of_alignment.add(1 - y0_equals)
            if element.y1 == other.y1:
                y1_equals = util.add_equals_var(m, y1[element.id], y1[other.id])
                loss_of_alignment.add(1 - y1_equals)
            if element.width == other.width:
                w_equals = util.add_equals_var(m, width[element.id], width[other.id])
                loss_of_alignment.add(1 - w_equals)
            if element.height == other.height:
                h_equals = util.add_equals_var(m, height[element.id], height[other.id])
                loss_of_alignment.add(1 - h_equals)

    def get_rel_xywh(element_id):
        # Returns the element position (relative to the parent top left corner)

        # Attribute Xn refers to the variable value in the solution selected using SolutionNumber parameter.
        # When SolutionNumber equals 0 (default), Xn refers to the variable value in the best solution.
        # https://www.gurobi.com/documentation/8.1/refman/xn.html#attr:Xn
        x = x0[element_id].Xn
        y = y0[element_id].Xn
        w = width[element_id].Xn
        h = height[element_id].Xn

        print('Loss of alignment', loss_of_alignment.getValue())

        return x, y, w, h

    quality = LinExpr()
    quality.add(total_group_count, 1)
    quality.add(loss_of_alignment, 10)

    return get_rel_xywh, quality

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

