from typing import Callable
from difflib import SequenceMatcher
from itertools import product

from gurobipy import GRB, LinExpr, Model, tupledict

from .classes import Layout, Element


def solve(layout1: Layout, layout2: Layout):

    m = Model('GLayoutDifference')

    # Variables

    element_mapping = m.addVars(layout1.n, layout2.n, vtype=GRB.BINARY, name='ElementMapping')
    layout1_unmapped = m.addVars(layout1.n, vtype=GRB.BINARY, name='UnmappedInLayout1')
    layout2_unmapped = m.addVars(layout2.n, vtype=GRB.BINARY, name='UnmappedInLayout2')

    # Constraints

    # For each element in either layout, check that it is unassigned, or assigned to only one element in the other layout
    for i1 in range(layout1.n):
        m.addConstr(layout1_unmapped[i1] + element_mapping.sum(i1, '*') == 1, name='Element' + str(i1) + 'InLayout1Is(Un)MappedOnce')

    for i2 in range(layout2.n):
        m.addConstr(layout2_unmapped[i2] + element_mapping.sum('*', i2) == 1, name='Element' + str(i2) + 'InLayout2Is(Un)MappedOnce')

    # Map as many elements from the first layout as possible
    m.addConstr(layout1_unmapped.sum() == max(layout1.n - layout2.n, 0), name='MapMaxElementsFromLayout1')

    # Objectives

    # objective_euclidean_move_resize

    euclidean_move_expr = element_mapping.prod(get_prod_coeff(euclidean_move_between, layout1, layout2))
    euclidean_resize_expr = element_mapping.prod(get_prod_coeff(euclidean_resize_between, layout1, layout2))

    # component match

    element_similarity_expr = element_mapping.prod(get_prod_coeff(element_similarity, layout1, layout2))

    # number of elements TODO: add this
    # size of canvas TODO: add this
    # objective_elements_lost TODO: probably unnecessary (if this is used, there should be a way to define the importance of each element)

    element_ignored_expr = layout1_unmapped.prod({(i): e.area / e.layout.area_sum for i, e in enumerate(layout1.elements)})


    full_obj_expr = LinExpr()
    full_obj_expr.add(euclidean_move_expr)
    full_obj_expr.add(euclidean_resize_expr)
    full_obj_expr.add(element_ignored_expr)
    full_obj_expr.add(element_similarity_expr, 100)

    m.setObjective(full_obj_expr, GRB.MINIMIZE)

    m.Params.OutputFlag = 0
    m.optimize()


    element_mapping_dict = []

    for e1 in range(layout1.n):
        for e2 in range(layout2.n):
            if element_mapping[e1, e2].X == 1:  # ‘X’ is the value of the variable in the current solution
                element_mapping_dict.append((layout1.elements[e1].id, layout2.elements[e2].id))

    if m.Status == GRB.Status.OPTIMAL:
        # TODO: consider adding metric for difference in screen size
        return {
            'status': 0,
            'euclideanDifference': round(full_obj_expr.getValue() * 10000 - element_ignored_expr.getValue() * 10000),
            'elementsGainedPenalty': 0,
            'elementsLostPenalty': round(element_ignored_expr.getValue() * 10000),
            'elementMapping': element_mapping_dict
        }
    else:
        print('Non-optimal status:', m.Status)
        return {'status': 1}

def get_prod_coeff(coeff_func: Callable[[Element, Element], float], layout1: Layout, layout2: Layout) -> dict:
    # Returns a dict that can be used as an argument for tupledict.prod() method
    # https://www.gurobi.com/documentation/8.1/refman/py_tupledict_prod.html
    return {
        (i1, i2): coeff_func(e1, e2)
        for (i1, e1), (i2, e2) in product(enumerate(layout1.elements), enumerate(layout2.elements))
    }

def euclidean_move_between(e1: Element, e2: Element):
    delta_x = abs(e1.x - e2.x)
    delta_y = abs(e1.y - e2.y)
    return ((delta_x / (e1.layout.x_sum + e2.layout.x_sum)) + (delta_y / (e1.layout.y_sum + e2.layout.y_sum))) \
        * ((e1.area + e2.area) / (e1.layout.area_sum + e2.layout.area_sum))


def euclidean_resize_between(e1, e2):
    delta_w = abs(e1.width - e2.width)
    delta_h = abs(e1.height - e2.height)
    return ((delta_w / (e1.layout.w_sum + e2.layout.w_sum)) + (delta_h / (e1.layout.h_sum + e2.layout.h_sum))) \
        * ((e1.area + e2.area) / (e1.layout.area_sum + e2.layout.area_sum))

def element_similarity(e1: Element, e2: Element) -> float:
    # Returns a similarity measure between 0 (same element type or same component type) and 1 (different element types)
    if e1.elementType != e2.elementType:
        return 1
    elif e1.elementType == 'component': # Same element type, which is component
        # Use the component name similarity as a metric of component similarity
        return 1 - max(
            SequenceMatcher(None, e1.componentName, e2.componentName).ratio(),
            SequenceMatcher(None, e2.componentName, e1.componentName).ratio()
        )
    else: # Same element type, but not components (e.g. text)
        return 0

def canvas_resize(layout1: Layout, layout2: Layout):
    pass