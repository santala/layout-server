from typing import Callable
from difflib import SequenceMatcher
from itertools import product

from gurobipy import GRB, LinExpr, Model, tupledict

from .classes import Layout, Element


def solve(layout1: Layout, layout2: Layout):
    '''
    This function finds a mapping between the elements of the two layouts, and computes the distance between
    the two layouts based on that mapping.

    :param layout1: A sketch
    :param layout2: A template
    :return: Measures of the distance between the two layouts
    '''

    m = Model('GLayoutDifference')

    # Variables

    element_mapping: tupledict = m.addVars(layout1.n, layout2.n, vtype=GRB.BINARY, name='ElementMapping')
    layout1_unmapped: tupledict = m.addVars(layout1.n, vtype=GRB.BINARY, name='UnmappedInLayout1')
    layout2_unmapped: tupledict = m.addVars(layout2.n, vtype=GRB.BINARY, name='UnmappedInLayout2')

    # CONSTRAINTS

    # For each element in either layout, check that it is unassigned, or assigned to only one element in the other layout
    for i1 in range(layout1.n):
        m.addConstr(layout1_unmapped[i1] + element_mapping.sum(i1, '*') == 1, name='Element' + str(i1) + 'InLayout1Is(Un)MappedOnce')

    for i2 in range(layout2.n):
        m.addConstr(layout2_unmapped[i2] + element_mapping.sum('*', i2) == 1, name='Element' + str(i2) + 'InLayout2Is(Un)MappedOnce')

    # Map as many elements from the first layout as possible.
    m.addConstr(layout1_unmapped.sum() == max(layout1.n - layout2.n, 0), name='MapMaxElementsFromLayout1')

    # OBJECTIVES

    # Minimize the relative distance of paired elements
    euclidean_distance_expr = element_mapping.prod(get_prod_coeff(euclidean_distance, layout1, layout2))

    # Minimize the relative size difference of paired elements
    euclidean_size_diff_expr = element_mapping.prod(get_prod_coeff(euclidean_size_diff, layout1, layout2))

    # Maximize the similarity of paired elements
    element_similarity_expr = element_mapping.prod(get_prod_coeff(element_similarity, layout1, layout2))

    # In cases, where all of the elements from the first layout can’t be mapped (i.e. the second layout has fewer
    # elements), prioritize mapping of larger elements
    element_ignored_expr = layout1_unmapped.prod({(i): e.area / e.layout.area_sum for i, e in enumerate(layout1.elements)})

    # TODO: consider taking weights as input
    obj_expr = LinExpr()
    obj_expr.add(euclidean_distance_expr)
    obj_expr.add(euclidean_size_diff_expr)
    obj_expr.add(element_ignored_expr)
    obj_expr.add(element_similarity_expr, 100)

    m.setObjective(obj_expr, GRB.MINIMIZE)

    m.Params.OutputFlag = 0
    m.optimize()

    # MEASURES INDEPENDENT OF MAPPING

    # Difference in the number of elements in the layouts
    element_count_diff = abs(layout2.n - layout1.n) / layout1.n

    # Canvas area difference
    canvas_area_diff = abs(layout2.canvas_area - layout1.canvas_area) / layout1.canvas_area

    # Canvas aspect ratio difference
    canvas_aspect_ratio_diff = abs(layout2.canvas_aspect_ratio - layout1.canvas_aspect_ratio) / layout1.canvas_aspect_ratio

    # number of elements TODO: add this
    # size of canvas TODO: add this

    if m.Status == GRB.Status.OPTIMAL:
        # TODO: consider adding metric for difference in screen size
        return {
            'success': True,
            'gurobiStatus': m.Status,
            'mappingDistance': round(obj_expr.getValue() * 10000),
            'measures': {
                'euclideanDistance': round(euclidean_distance_expr.getValue() * 10000),
                'euclideanSizeDiff': round(euclidean_size_diff_expr.getValue() * 10000),
                'elementDissimilarity': round(element_similarity_expr.getValue() * 10000),
                'ignoredElements': round(element_ignored_expr.getValue() * 10000),
                'elementCountDiff': round(element_count_diff * 10000),
                'canvasAreaDiff': round(canvas_area_diff * 10000),
                'canvasAspectRatioDiff': round(canvas_aspect_ratio_diff * 10000),
            },
            'elementMapping': [
                (e1.id, e2.id)
                for (i1, e1), (i2, e2) in product(enumerate(layout1.elements), enumerate(layout2.elements))
                if element_mapping[i1, i2].X == 1
            ]
        }
    else:
        print('Non-optimal status:', m.Status)
        return {
            'success': False,
            'gurobiStatus': m.Status
        }

def get_prod_coeff(coeff_func: Callable[[Element, Element], float], layout1: Layout, layout2: Layout) -> dict:
    # Returns a dict that can be used as an argument for tupledict.prod() method
    # https://www.gurobi.com/documentation/8.1/refman/py_tupledict_prod.html
    return {
        (i1, i2): coeff_func(e1, e2)
        for (i1, e1), (i2, e2) in product(enumerate(layout1.elements), enumerate(layout2.elements))
    }

def euclidean_distance(e1: Element, e2: Element):
    delta_x = abs(e1.x - e2.x)
    delta_y = abs(e1.y - e2.y)
    return ((delta_x / (e1.layout.x_sum + e2.layout.x_sum)) + (delta_y / (e1.layout.y_sum + e2.layout.y_sum))) \
        * ((e1.area + e2.area) / (e1.layout.area_sum + e2.layout.area_sum))


def euclidean_size_diff(e1, e2):
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
