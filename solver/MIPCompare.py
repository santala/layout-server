from gurobipy import GRB, Model
from gurobipy.gurobipy import LinExpr

from pprint import pprint

from tools.GurobiUtils import define1DBoolVarArray, define2DBoolVarArrayArray
from model import Layout
#from solver.PrepareParameters import PenaltyAssignment


gurobi_model = Model('GLayoutCompare')
objective_euclidean_move_resize = LinExpr()
objective_elements_lost = LinExpr()
objective_elements_gained = LinExpr()
objective_full = LinExpr()


def solve(first_layout: Layout, second_layout: Layout, PenaltyAssignment) -> dict:
    global gurobi_model, objective_euclidean_move_resize, objective_elements_lost, objective_elements_gained, objective_full

    gurobi_model = Model('GLayoutCompare')
    objective_euclidean_move_resize = LinExpr()
    objective_elements_lost = LinExpr()
    objective_elements_gained = LinExpr()
    objective_full = LinExpr()

    # EXPL: Z = element mapping (a boolean matrix)
    # EXPL: UF = extra elements in the first layout (a list of booleans)
    # EXPL: US = extra elements in the second layout (a list of booleans)
    Z, UF, US = define_variables(first_layout, second_layout)
    # EXPL: Compute the penalty of the mapping plus penalties incurred by the unassigned elements
    define_objectives(first_layout, second_layout, Z, UF, US, PenaltyAssignment)
    define_constraints(first_layout, second_layout, Z, UF, US)
    set_control_parameters()
    gurobi_model.optimize()

    element_mapping = []

    for e1 in range(first_layout.N):
        for e2 in range(second_layout.N):
            print(e1, e2, Z[e1, e2].getAttr('X'))
            if Z[e1, e2].getAttr('X') == 1:
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


def set_control_parameters():
    gurobi_model.Params.OutputFlag = 0

def define_constraints(firstLayout:Layout, secondLayout:Layout, Z, UF, US):
    #Forward -- First to second
    for countInFirst in range(firstLayout.N):
        assignmentOfThisElement = LinExpr()
        # EXPL: for each element, check that it is unassigned…
        assignmentOfThisElement.addTerms([1],[UF[countInFirst]])
        for countInSecond in range(secondLayout.N):
            # EXPL: …or assigned to only one element in the other layout
            assignmentOfThisElement.addTerms([1],[Z[countInFirst,countInSecond]])
        gurobi_model.addConstr(assignmentOfThisElement == 1, "AssignFirstForElement(" + str(countInFirst) + ")")

    #Reverse -- Second to first
    for countInSecond in range(secondLayout.N):
        assignmentOfThisElement = LinExpr()
        assignmentOfThisElement.addTerms([1],[US[countInSecond]])
        for countInFirst in range(firstLayout.N):
            assignmentOfThisElement.addTerms(1,Z[countInFirst,countInSecond])
        gurobi_model.addConstr(assignmentOfThisElement == 1, "AssignSecondForElement(" + str(countInSecond) + ")")

def define_objectives(first_layout: Layout, second_layout: Layout, Z, UF, US, PenaltyAssignment):
    # Element Assignment
    # EXPL: loop through possible element pairs
    for countInFirst in range(first_layout.N):
        for countInSecond in range(second_layout.N):
            # EXPL: TODO: confirm this
            # EXPL: penalty is the ‘EuclideanMoveResize’ distance between two elements from different layouts
            # EXPL: this code adds a term that equals the penalty if the the elements are paired up,
            # EXPL: but is zero if they are not assigned
            weightage = PenaltyAssignment[countInFirst][countInSecond]
            variable = Z[countInFirst,countInSecond]
            objective_euclidean_move_resize.addTerms(weightage, variable)

            # EXPL: TODO: check how penaltySkipped works
    #UnAssigned from first
    for countInFirst in range(first_layout.N):
        objective_elements_lost.addTerms(first_layout.elements[countInFirst].PenaltyIfSkipped, UF[countInFirst])

    for countInSecond in range(second_layout.N):
        objective_elements_gained.addTerms(second_layout.elements[countInSecond].PenaltyIfSkipped, US[countInSecond])

    # TODO: EXPL: what does it mean for element to be ‘lost’ or ‘gained’?
    objective_full.add(objective_euclidean_move_resize, 1)
    objective_full.add(objective_elements_lost, 1)
    objective_full.add(objective_elements_gained, 1)
    # EXPL: minimize penalty
    gurobi_model.setObjective(objective_full, GRB.MINIMIZE)

def define_variables(firstLayout:Layout, secondLayout:Layout):
    Z = define2DBoolVarArrayArray(gurobi_model, firstLayout.N, secondLayout.N, "ZAssignment")
    UF = define1DBoolVarArray(gurobi_model, firstLayout.N, "UnassignedInFirstLayout")
    US = define1DBoolVarArray(gurobi_model, secondLayout.N, "UnassignedInSecondLayout")
    return Z, UF, US
