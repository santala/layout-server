from gurobipy import GRB, Model, tupledict
from gurobipy.gurobipy import LinExpr

from . import Layout


def solve(layout1: Layout, layout2: Layout, penalty_assignment: list) -> dict:

    # TODO: check whether re-using the same model is feasible
    model = Model('GLayoutCompare')

    # EXPL: Z = element mapping (a boolean matrix)
    # EXPL: UF = extra elements in the first layout (a list of booleans)
    # EXPL: US = extra elements in the second layout (a list of booleans)
    # TODO: maybe use the term ‘orphan’ for unassigned elements?
    element_assignment, unassigned1, unassigned2 = define_variables(model, layout1, layout2)
    # EXPL: Compute the penalty of the mapping plus penalties incurred by the unassigned elements
    objective_euclidean_move_resize, objective_elements_lost, objective_elements_gained, objective_full \
        = define_objectives(model, layout1, layout2, element_assignment, unassigned1, unassigned2, penalty_assignment)
    define_constraints(model, layout1, layout2, element_assignment, unassigned1, unassigned2)
    set_control_parameters(model)
    model.optimize()

    element_mapping = []

    for e1 in range(layout1.n):
        for e2 in range(layout2.n):
            if element_assignment[e1, e2].getAttr('X') == 1: # ‘X’ is the value of the variable in the current solution
                element_mapping.append((layout1.elements[e1].id, layout2.elements[e2].id))

    if model.Status == GRB.Status.OPTIMAL:
        return {
            'status': 1,
            'euclideanDifference': round(objective_euclidean_move_resize.getValue() * 10000),
            'elementsGainedPenalty': round(objective_elements_gained.getValue() * 10000),
            'elementsLostPenalty': round(objective_elements_lost.getValue() * 10000),
            'elementMapping': element_mapping
        }
    else:
        return { 'status': 0 }


def set_control_parameters(model: Model):
    model.Params.OutputFlag = 0


def define_constraints(model: Model, layout1: Layout, layout2: Layout, element_mapping, unassigned1, unassigned2):
    #Forward -- First to second
    for i1 in range(layout1.n):
        element1_assignment = LinExpr()
        # EXPL: for each element, check that it is unassigned…
        element1_assignment.addTerms([1], [unassigned1[i1]])
        for i2 in range(layout2.n):
            # EXPL: …or assigned to only one element in the other layout
            element1_assignment.addTerms([1], [element_mapping[i1, i2]])
        model.addConstr(element1_assignment == 1, name="AssignFirstForElement(" + str(i1) + ")")

    #Reverse -- Second to first
    for i2 in range(layout2.n):
        element2_assignment = LinExpr()
        element2_assignment.addTerms([1], [unassigned2[i2]])
        for i1 in range(layout1.n):
            element2_assignment.addTerms(1, element_mapping[i1, i2])
        model.addConstr(element2_assignment == 1, name="AssignSecondForElement(" + str(i2) + ")")

def define_objectives(gurobi_model: Model, layout1: Layout, layout2: Layout,
                      element_assignment, unassigned1, unassigned2, penalty_assignment) -> (LinExpr, LinExpr, LinExpr, LinExpr):
    objective_euclidean_move_resize = LinExpr()
    objective_elements_lost = LinExpr()
    objective_elements_gained = LinExpr()
    objective_full = LinExpr()

    # Element Assignment
    # EXPL: loop through possible element pairs
    for countInFirst in range(layout1.n):
        for countInSecond in range(layout2.n):
            # EXPL: TODO: confirm this
            # EXPL: penalty is the ‘EuclideanMoveResize’ distance between two elements from different layouts
            # EXPL: this code adds a term that equals the penalty if the the elements are paired up,
            # EXPL: but is zero if they are not assigned
            weights = penalty_assignment[countInFirst][countInSecond]
            variable = element_assignment[countInFirst, countInSecond]
            objective_euclidean_move_resize.addTerms(weights, variable)

            # EXPL: TODO: check how penaltySkipped works
    #UnAssigned from first
    for countInFirst in range(layout1.n):
        objective_elements_lost.addTerms(layout1.elements[countInFirst].PenaltyIfSkipped, unassigned1[countInFirst])

    for countInSecond in range(layout2.n):
        objective_elements_gained.addTerms(layout2.elements[countInSecond].PenaltyIfSkipped, unassigned2[countInSecond])

    # TODO: EXPL: are ‘lost’ and ‘gained’ good terms to use?
    # EXPL: ‘Lost’ here refers to elements from the first layout that are don’t correspond to any element in the second
    # EXPL: layout. ‘Gained’ refers to elements in the seconds layout that don’t have a mapping.
    # EXPL: So, if the two layouts are combined, ‘lost’ elements are those that we need, but don’t have an clear place,
    # EXPL: and ‘gained’ elements are those that are ‘extra’.
    objective_full.add(objective_euclidean_move_resize, 1)
    objective_full.add(objective_elements_lost, 1)
    objective_full.add(objective_elements_gained, 1)
    # EXPL: minimize penalty
    gurobi_model.setObjective(objective_full, GRB.MINIMIZE)

    return objective_euclidean_move_resize, objective_elements_lost, objective_elements_gained, objective_full


def define_variables(gurobi_model: Model, first_layout: Layout, second_layout: Layout)\
        -> (tupledict, tupledict, tupledict):
    element_mapping = gurobi_model.addVars(first_layout.n, second_layout.n, vtype=GRB.BINARY, name='ZAssignment')
    unassigned_in_first = gurobi_model.addVars(first_layout.n, vtype=GRB.BINARY, name='UnassignedInFirstLayout')
    unassigned_in_second = gurobi_model.addVars(second_layout.n, vtype=GRB.BINARY, name='UnassignedInSecondLayout')
    return element_mapping, unassigned_in_first, unassigned_in_second
