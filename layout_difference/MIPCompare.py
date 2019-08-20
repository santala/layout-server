from gurobipy import GRB, Model
from gurobipy.gurobipy import LinExpr

from tools.GurobiUtils import define_1d_bool_var_array, define_2d_bool_var_array_array
from . import Layout


def solve(first_layout: Layout, second_layout: Layout, PenaltyAssignment) -> dict:

    # TODO: check whether re-using the same model is feasible
    gurobi_model = Model('GLayoutCompare')

    # EXPL: Z = element mapping (a boolean matrix)
    # EXPL: UF = extra elements in the first layout (a list of booleans)
    # EXPL: US = extra elements in the second layout (a list of booleans)
    Z, UF, US = define_variables(gurobi_model, first_layout, second_layout)
    # EXPL: Compute the penalty of the mapping plus penalties incurred by the unassigned elements
    objective_euclidean_move_resize, objective_elements_lost, objective_elements_gained, objective_full \
        = define_objectives(gurobi_model, first_layout, second_layout, Z, UF, US, PenaltyAssignment)
    define_constraints(gurobi_model, first_layout, second_layout, Z, UF, US)
    set_control_parameters(gurobi_model)
    gurobi_model.optimize()

    element_mapping = []

    for e1 in range(first_layout.n):
        for e2 in range(second_layout.n):
            if Z[e1, e2].getAttr('X') == 1: # ‘X’ is the value of the variable in the current solution
                element_mapping.append((first_layout.elements[e1].id, second_layout.elements[e2].id))


    if gurobi_model.Status == GRB.Status.OPTIMAL:
        return {
            'status': 1,
            'euclideanDifference': round(objective_euclidean_move_resize.getValue() * 10000),
            'elementsGained': round(objective_elements_gained.getValue() * 10000),
            'elementsLost': round(objective_elements_lost.getValue() * 10000),
            'elementMapping': element_mapping
        }
    else:
        return { 'status': 0 }


def set_control_parameters(gurobi_model):
    gurobi_model.Params.OutputFlag = 0


def define_constraints(gurobi_model: Model, firstLayout:Layout, secondLayout:Layout, Z, UF, US):
    #Forward -- First to second
    for countInFirst in range(firstLayout.n):
        assignmentOfThisElement = LinExpr()
        # EXPL: for each element, check that it is unassigned…
        assignmentOfThisElement.addTerms([1],[UF[countInFirst]])
        for countInSecond in range(secondLayout.n):
            # EXPL: …or assigned to only one element in the other layout
            assignmentOfThisElement.addTerms([1],[Z[countInFirst,countInSecond]])
        gurobi_model.addConstr(assignmentOfThisElement == 1, "AssignFirstForElement(" + str(countInFirst) + ")")

    #Reverse -- Second to first
    for countInSecond in range(secondLayout.n):
        assignmentOfThisElement = LinExpr()
        assignmentOfThisElement.addTerms([1],[US[countInSecond]])
        for countInFirst in range(firstLayout.n):
            assignmentOfThisElement.addTerms(1,Z[countInFirst,countInSecond])
        gurobi_model.addConstr(assignmentOfThisElement == 1, "AssignSecondForElement(" + str(countInSecond) + ")")


def define_objectives(gurobi_model: Model, first_layout: Layout, second_layout: Layout, Z, UF, US, PenaltyAssignment):
    objective_euclidean_move_resize = LinExpr()
    objective_elements_lost = LinExpr()
    objective_elements_gained = LinExpr()
    objective_full = LinExpr()

    # Element Assignment
    # EXPL: loop through possible element pairs
    for countInFirst in range(first_layout.n):
        for countInSecond in range(second_layout.n):
            # EXPL: TODO: confirm this
            # EXPL: penalty is the ‘EuclideanMoveResize’ distance between two elements from different layouts
            # EXPL: this code adds a term that equals the penalty if the the elements are paired up,
            # EXPL: but is zero if they are not assigned
            weightage = PenaltyAssignment[countInFirst][countInSecond]
            variable = Z[countInFirst,countInSecond]
            objective_euclidean_move_resize.addTerms(weightage, variable)

            # EXPL: TODO: check how penaltySkipped works
    #UnAssigned from first
    for countInFirst in range(first_layout.n):
        objective_elements_lost.addTerms(first_layout.elements[countInFirst].PenaltyIfSkipped, UF[countInFirst])

    for countInSecond in range(second_layout.n):
        objective_elements_gained.addTerms(second_layout.elements[countInSecond].PenaltyIfSkipped, US[countInSecond])

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


def define_variables(gurobi_model: Model, firstLayout:Layout, secondLayout:Layout):
    Z = define_2d_bool_var_array_array(gurobi_model, firstLayout.n, secondLayout.n, "ZAssignment")
    UF = define_1d_bool_var_array(gurobi_model, firstLayout.n, "UnassignedInFirstLayout")
    US = define_1d_bool_var_array(gurobi_model, secondLayout.n, "UnassignedInSecondLayout")
    return Z, UF, US
