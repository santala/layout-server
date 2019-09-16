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

    for i2 in range(layout1.n):
        m.addConstr(layout2_unmapped[i2] + element_mapping.sum('*', i2) == 1, name='Element' + str(i2) + 'InLayout2Is(Un)MappedOnce')

    # Map as many elements from the first layout as possible
    m.addConstr(layout1_unmapped.sum() == max(layout1.n - layout2.n, 0), name='MapMaxElementsFromLayout1')

    # Objectives

    # objective_euclidean_move_resize

    # https://www.gurobi.com/documentation/8.1/refman/py_tupledict_prod.html
    objective_euclidean_move = element_mapping.prod(euclidean_move_coeff(layout1, layout2))

    for i1 in range(layout1.n):
        for i2 in range(layout2.n):
            # EXPL: TODO: confirm this
            # EXPL: penalty is the ‘EuclideanMoveResize’ distance between two elements from different layouts
            # EXPL: this code adds a term that equals the penalty if the the elements are paired up,
            # EXPL: but is zero if they are not assigned
            weights = penalty_assignment[i1][i2]
            variable = element_assignment[i1, i2]
            objective_euclidean_move_resize.addTerms(weights, variable)

    # component match TODO: separate this from the above
    # number of elements TODO: add this
    # size of canvas TODO: add this
    # objective_elements_lost TODO: probably unnecessary (if this is used, there should be a way to define the importance of each element)
    # objective_elements_gained TODO: probably unnecessary


def euclidean_move_coeff(layout1: Layout, layout2: Layout) -> dict:
    # Return a dictionary of the relative euclidean distance between each possible pair of elements
    return {
        (i1, i2): euclidean_move_between(layout1, e1, layout2, e2)
        for (i1, e1), (i2, e2) in product(enumerate(layout1.elements), enumerate(layout2.elements))
    }

def euclidean_move_between(layout1: Layout, element1: Element, layout2: Layout, element2: Element):
    delta_x = abs(element1.x - element2.x)
    delta_y = abs(element1.y - element2.y)
    return ((delta_x / (layout1.x_sum + layout2.x_sum)) + (delta_y / (layout1.y_sum + layout2.y_sum))) \
    * ((element1.area + element2.area) / (layout1.area_sum + layout2.area_sum))

def euclidean_resize(layout1: Layout, layout2: Layout):
    pass

def component_similarity(layout1: Layout, layout2: Layout):
    pass

def canvas_resize(layout1: Layout, layout2: Layout):
    pass