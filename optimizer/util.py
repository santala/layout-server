from collections import namedtuple
from enum import Enum
from itertools import product, permutations
from math import ceil, floor, sqrt
from typing import List

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, or_, QuadExpr, GurobiError

from .classes import Layout, Element

def preserve_relationships(m: Model, elements: List[Element], x0, x1, y0, y1):
    for element, other in permutations(elements, 2):
        if element.y1 <= other.y0: # element is above the other
            m.addConstr(y1[element.id] <= y0[other.id])
        if element.x1 <= other.x0: # element is on the left side of the other
            m.addConstr(x1[element.id] <= x0[other.id])
        # TODO: test these out
        if element.x1 >= other.x0:
            pass#m.addConstr(x1[element.id] >= x0[other.id])

def prevent_overlap(m: Model, elem_ids: List[str], x0x1diff: tupledict, y0y1diff: tupledict, min_distance: int=0):

    distance = m.addVars(permutations(elem_ids, 2), lb=-GRB.INFINITY, vtype=GRB.INTEGER)

    m.addConstrs((
        distance[i1, i2] == max_(x0x1diff[i1, i2], x0x1diff[i2, i1], y0y1diff[i1, i2], y0y1diff[i2, i1])
        for i1, i2 in permutations(elem_ids, 2)
    ))
    m.addConstrs((
        distance[i1, i2] >= min_distance
        for i1, i2 in permutations(elem_ids, 2)
    ))


def add_pairwise_diff(m: Model, ids: List, var1: tupledict, var2: tupledict=None):
    if var2 is None:
        var2 = var1
    diff = m.addVars(permutations(ids, 2), lb=-GRB.INFINITY, vtype=GRB.INTEGER)
    m.addConstrs((
        diff[i1, i2] == var1[i1] - var2[i2]
        for i1, i2 in permutations(ids, 2)
    ))
    return diff
