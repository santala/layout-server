from typing import List, Tuple

from itertools import combinations, permutations

from gurobipy import *
from tools.GurobiUtils import *
from tools.JSONLoader import Layout, Element
from tools.Constants import *
import math

from collections import namedtuple

Solution = namedtuple('Solution', 'x, y, w, h, objective_value')


def apply_template(layout: Layout, template: Layout, element_mapping: List[Tuple[str, str]]) -> dict:
    try:
        model = Model("GLayout")
        model._layout = layout
        model._template = template

        var = Variables(model)

        model._var = var




    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ": " + str(e))
        return {'status': 1}

    except AttributeError as e:
        print('AttributeError:' + str(e))
        return {'status': 1}

def define_layout_combination_objective(model: Model, layout1: Layout, layout2: Layout, element_mapping: List[Tuple[str, str]], var: Variables):


    pass

def solve(layout: Layout) -> dict:

    try:
        model = Model("GLayout")
        model._layout = layout

        var = Variables(model)

        model._var = var

        set_variable_bounds(layout, var)

        objective_grid_count, objective_lt = define_objectives(model, layout, var)

        set_constraints(model, layout, var)

        model._solutions = []
        model._solution_number = 1

        model.write("output/NirajPracticeModel.lp")

        set_control_params(model)


        model.optimize(tap_solutions)

        #TODO (from Niraj) check if solution was found. If yes, set the better objective bounds on future solutions

        #gurobi.computeIIS()
        #gurobi.write("IIS.ilp")

        #gurobi.optimize(tapSolutions)
        #reportResult(BAG, H, L, LAG, N, OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT, RAG, T, TAG, W, data, gurobi,vBAG, vLAG,vRAG, vTAG)

        # TODO: EXPL: analyze brute force purpose
        #repeatBruteForceExecutionForMoreResults(BAG, H, L, LAG, LEFT, ABOVE, N, OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT, RAG, T, TAG,W, data, gurobi, vBAG, vLAG, vRAG, vTAG)

    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ": " + str(e))
        return {'status': 1}

    except AttributeError as e:
        print('AttributeError:' + str(e))
        return {'status': 1}

    if model.Status == GRB.Status.OPTIMAL or model.Status == GRB.Status.INTERRUPTED:

        elements = [
            {
                'id': element.id,
                'x': var.l[i].getAttr('X'),  # ‘X’ is the value of the variable in the current solution
                'y': var.t[i].getAttr('X'),
                'width': var.w[i].getAttr('X'),
                'height': var.h[i].getAttr('X'),
            } for i, element in enumerate(layout.elements)
        ]

        return {
            'status': 0,
            'layout': {
                'canvasWidth': layout.canvas_width,
                'canvasHeight': layout.canvas_height,
                'elements': elements
            }
        }
    else:
        if model.Status == GRB.Status.INFEASIBLE:
            model.computeIIS()
            model.write("output/NirajPracticeModel.ilp")
        print('Non-optimal status:', model.Status)
        return {'status': 1}


def repeatBruteForceExecutionForMoreResults(model: Model, layout: Layout, var: Variables):
    for topElem in range(layout.n):
        for bottomElem in range(layout.n):
            if (topElem != bottomElem):

                temporaryConstraint = model.addConstr(var.on_left[topElem, bottomElem] == 1)
                model.optimize(tap_solutions)
                model.remove(temporaryConstraint)

                temporaryConstraint = model.addConstr(var.above[topElem, bottomElem] == 1)
                model.optimize(tap_solutions)
                model.remove(temporaryConstraint)


def set_control_params(model: Model):
    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 1
    #gurobi.Params.MIPGap = 0.01
    #gurobi.Params.TimeLimit = 75
    model.Params.MIPGapAbs = 0.97
    model.Params.LogFile = "output/GurobiLog.txt"
    model.Params.OutputFlag = 0


def set_constraints(model: Model, layout: Layout, var: Variables):
    # TODO: Why are these _constraints_? I.e. are these for e.g. locking elements to their place?

    # EXPL: constraints in short
    # * If element X, Y, or aspectRatio is defined, lock them.
    # * If element has preference towards some edge, prevent other elements from being closer to that edge.
    # * Sanity constraint that coordinates match with width and height
    # * For every element pair, one element has to be either to the left of or above the other.
    # * For every element that is to the left of or above another, there has to be min. of padding between the edges
    # * Every element edge has to align with only one alignment group
    # * If an alignment group is not enabled, no edge should be associated with it.
    # * If an edge is associated with an alignment group, the coordinates have to match.

    # Known Position constraints X Y
    for i, element in enumerate(layout.elements):
        '''
        print(element.isLocked)
        if not element.isLocked:
            print('Element', element.id, 'is not locked.')
            continue
        '''

        if element.x is not None and element.x >= 0 and element.constraintLeft:
            model.addConstr(var.l[i] == element.x, "PrespecifiedXOfElement("+str(i)+")")

        if element.y is not None and element.y >= 0 and element.constraintTop:
            model.addConstr(var.t[i] == element.y, "PrespecifiedYOfElement("+str(i)+")")

        if element.constraintRight:
            model.addConstr(var.l[i] + var.w[i] == element.x + element.width, "PrespecifiedROfElement("+str(i)+")")

        if element.constraintBottom:
            model.addConstr(var.t[i] + var.h[i] == element.y + element.height, "PrespecifiedBOfElement("+str(i)+")")

        if element.constraintWidth:
            model.addConstr(var.w[i] == element.width, "PrespecifiedWOfElement("+str(i)+")")

        if element.constraintWidth:
            model.addConstr(var.h[i] == element.height, "PrespecifiedHOfElement("+str(i)+")")
        
        if element.aspectRatio is not None and element.aspectRatio > 0.001:
            # EXPL: Does this lock element aspect ratio?
            model.addConstr(var.w[i] == element.aspectRatio * var.h[i],
                             "PrespecifiedAspectRatioOfElement("+str(i)+")")
    # Known Position constraints TOP BOTTOM LEFT RIGHT
    coeffsForAbsolutePositionExpression = []
    varsForAbsolutePositionExpression = []


    for (i, element), (j, other) in permutations(enumerate(layout.elements), 2):
        # j = the index of the other element
        # EXPL: loop through element pairs, handling every pair twice (both orderings)

        # EXPL: handle preferences for positioning
        if element.verticalPreference.lower() == "top":
            varsForAbsolutePositionExpression.append(var.above[j, i])
            coeffsForAbsolutePositionExpression.append(1.0)
        if element.verticalPreference.lower() == "bottom":
            varsForAbsolutePositionExpression.append(var.above[i, j])
            coeffsForAbsolutePositionExpression.append(1.0)
        if element.horizontalPreference.lower() == "left":
            varsForAbsolutePositionExpression.append(var.on_left[j, i])
            coeffsForAbsolutePositionExpression.append(1.0)
        if element.horizontalPreference.lower() == "right":
            varsForAbsolutePositionExpression.append(var.on_left[i, j])
            coeffsForAbsolutePositionExpression.append(1.0)

    expression = LinExpr(coeffsForAbsolutePositionExpression, varsForAbsolutePositionExpression)
    # EXPL: This constraint prevents any design where an element is closer to an edge than another element that has a
    # EXPL: preference for that edge.
    model.addConstr(expression == 0, "DisableNon-PermittedBasedOnPrespecified")
    # Height/Width/L/R/T/B Summation Sanity
    for i in range(layout.n):
        # EXPL: a sanity constraint that coordinates match width and height
        model.addConstr(var.w[i] + var.l[i] == var.r[i], "R-L=W(" + str(i) + ")")
        model.addConstr(var.h[i] + var.t[i] == var.b[i], "B-T=H(" + str(i) + ")")
    # MinMax limits of Left-Above interactions
    for i, j in combinations(range(layout.n), 2):
        # EXPL: Loop through element pairs (each pair is handled only once)
        # EXPL: apparently a no overlap constraint: i.e. one element has to be at least either on the left side
        # EXPL: or above the other. Conversely, if neither element is above or to the left of the other, they
        # EXPL: overlap.
        model.addConstr(
            var.above[i, j] + var.above[j, i] + var.on_left[i, j] + var.on_left[
                j, i] >= 1,
            "NoOverlap(" + str(i) + str(j) + ")")
        # EXPL: The following three constraints prevent locating the element on multiple sides of the other.
        # EXPL: I.e. only one element can be ‘the left one’ or ‘the top one’.
        model.addConstr(
            var.above[i, j] + var.above[j, i] + var.on_left[i, j] + var.on_left[
                j, i] <= 2,
            "UpperLimOfQuadrants(" + str(i) + str(j) + ")")
        model.addConstr(var.above[i, j] + var.above[j, i] <= 1,
                         "Anti-symmetryABOVE(" + str(i) + str(j) + ")")
        model.addConstr(var.on_left[i, j] + var.on_left[j, i] <= 1,
                         "Anti-symmetryLEFT(" + str(i) + str(j) + ")")
    # Interconnect L-R-LEFT and T-B-ABOVE
    for (i, element), (j, other) in permutations(enumerate(layout.elements), 2):
        # EXPL: Loop through element pairs (every pair is handled twice)
        # TODO: Check how HPAD_SPECIFICATION and VPAD_SPECIFICATION are defined
        # EXPL: If element is to the right of the other, check that element right edge is at least HPAD distance
        # EXPL: from the other left edge.
        # EXPL: The canvas width term is just a way to make sure the constraint is true if element is not to the
        # EXPL: left of the other.
        model.addConstr(
            var.r[i] + HPAD_SPECIFICATION <= var.l[j] + (1 - var.on_left[i, j]) * layout.canvas_width
            , (str(i) + "(ToLeftOf)" + str(j)))
        # EXPL: Same rule as above but vertically:
        model.addConstr(
            var.b[i] + VPAD_SPECIFICATION <= var.t[j] + (1 - var.above[i, j]) * layout.canvas_height
            , (str(i) + "(Above)" + str(j)))
        # EXPL: If element is not to the left of the other, check that element right edge + HPAD exceed the left
        # EXPL: of the other.
        # EXPL: The canvas width term is just a way to make sure the constraint is true if element is to the
        # EXPL: left of the other.
        model.addConstr(
            (var.l[j] - var.r[i] - HPAD_SPECIFICATION) <= layout.canvas_width * var.on_left[i, j]
            , (str(i) + "(ConverseOfToLeftOf)" + str(j)))
        # EXPL: Same as above but vertical
        model.addConstr(
            (var.t[j] - var.b[i] - VPAD_SPECIFICATION) <= layout.canvas_height * var.above[i, j]
            , (str(i) + "(ConverseOfAboveOf)" + str(j)))
    # One Alignment-group for every edge of every element
    # EXPL: The canvas is divided horizontally and vertically into ‘alignment groups’, or grid lines.
    # EXPL: Each element has to have an alignment group (or a grid line) for each of its edges.
    for i in range(layout.n):
        coeffsForLAG = []
        coeffsForRAG = []
        coeffsForTAG = []
        coeffsForBAG = []
        varsForLAG = []
        varsForRAG = []
        varsForTAG = []
        varsForBAG = []
        for alignmentGroupIndex in range(layout.n):
            # TODO: does N equal data.N ? If not, check how data.N is computed.
            # EXPL: data.N corresponds to the number of alignment groups
            # EXPL: elemAt*AG is a boolean matrix of whether an edge of an element aligns with a given ‘alignment group’
            varsForLAG.append(var.at_lag[i, alignmentGroupIndex])
            coeffsForLAG.append(1)
            varsForRAG.append(var.at_rag[i, alignmentGroupIndex])
            coeffsForRAG.append(1)
            varsForTAG.append(var.at_tag[i, alignmentGroupIndex])
            coeffsForTAG.append(1)
            varsForBAG.append(var.at_bag[i, alignmentGroupIndex])
            coeffsForBAG.append(1)

        # EXPL: following constraints make sure that an element edge aligns with only one alignment group
        model.addConstr(LinExpr(coeffsForLAG, varsForLAG) == 1, "OneLAGForElement[" + str(i) + "]")
        model.addConstr(LinExpr(coeffsForTAG, varsForTAG) == 1, "OneTAGForElement[" + str(i) + "]")
        model.addConstr(LinExpr(coeffsForBAG, varsForBAG) == 1, "OneBAGForElement[" + str(i) + "]")
        model.addConstr(LinExpr(coeffsForRAG, varsForRAG) == 1, "OneRAGForElement[" + str(i) + "]")
    # Assign alignment groups to elements only if groups are enabled
    for alignmentGroupIndex in range(layout.n):
        for i in range(layout.n):
            # TODO: EXPL: apparently, LAG is a boolean matrix telling, whether an ‘alignment group’ (= grid line) is
            # TODO: EXPL: enabled. If it is not, any element edge should not align with it either.
            model.addConstr(var.at_lag[i, alignmentGroupIndex] <= var.lag[alignmentGroupIndex])
            model.addConstr(var.at_rag[i, alignmentGroupIndex] <= var.rag[alignmentGroupIndex])
            model.addConstr(var.at_tag[i, alignmentGroupIndex] <= var.tag[alignmentGroupIndex])
            model.addConstr(var.at_bag[i, alignmentGroupIndex] <= var.bag[alignmentGroupIndex])
    # Correlate alignment groups value with element edge if assigned
    for alignmentGroupIndex in range(layout.n):
        for i in range(layout.n):
            # EXPL: If element is assigned to alignment group, check that * edge is less than or equal to v*AG
            model.addConstr(var.l[i] <= var.v_lag[alignmentGroupIndex] + layout.canvas_width * (
                        1 - var.at_lag[i, alignmentGroupIndex]),
                             "MinsideConnectL[" + str(i) + "]ToLAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.r[i] <= var.v_rag[alignmentGroupIndex] + layout.canvas_width * (
                        1 - var.at_rag[i, alignmentGroupIndex]),
                             "MinsideConnectR[" + str(i) + "]ToRAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.t[i] <= var.v_tag[alignmentGroupIndex] + layout.canvas_height * (
                        1 - var.at_tag[i, alignmentGroupIndex]),
                             "MinsideConnectT[" + str(i) + "]ToTAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.b[i] <= var.v_bag[alignmentGroupIndex] + layout.canvas_height * (
                        1 - var.at_bag[i, alignmentGroupIndex]),
                             "MinsideConnectB[" + str(i) + "]ToBAG[" + str(alignmentGroupIndex) + "]")

            # EXPL: If element is assigned to alignment group, check that * edge is greater than or equal to v*AG
            model.addConstr(var.l[i] >= var.v_lag[alignmentGroupIndex] - layout.canvas_width * (
                        1 - var.at_lag[i, alignmentGroupIndex]),
                             "MaxsideConnectL[" + str(i) + "]ToLAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.r[i] >= var.v_rag[alignmentGroupIndex] - layout.canvas_width * (
                        1 - var.at_rag[i, alignmentGroupIndex]),
                             "MaxsideConnectR[" + str(i) + "]ToRAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.t[i] >= var.v_tag[alignmentGroupIndex] - layout.canvas_height * (
                        1 - var.at_tag[i, alignmentGroupIndex]),
                             "MaxsideConnectT[" + str(i) + "]ToTAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.b[i] >= var.v_bag[alignmentGroupIndex] - layout.canvas_height * (
                        1 - var.at_bag[i, alignmentGroupIndex]),
                             "MaxsideConnectB[" + str(i) + "]ToBAG[" + str(alignmentGroupIndex) + "]")


def define_objectives(model: Model, layout: Layout, var: Variables) -> (LinExpr, LinExpr):
    # EXPL: Constraints
    # * every element right and bottom edge can be at max (maxX, maxY)
    # * grid count >= minimum possible grid count (grid count = LAG*2 + TAG*2 + BAG*1 + RAG*1)
    # EXPL: Objectives
    # * Minimize (grid_count + lt * .001)
    # * LT = T + L + 2B + 2R - W - H

    # EXPL: Model.addVar(…): Add a decision variable to a model.
    x_max = model.addVar(vtype=GRB.INTEGER, name="maxX")
    y_max = model.addVar(vtype=GRB.INTEGER, name="maxY")
    for i in range(layout.n):
        # EXPL: addConstr ( lhs, sense, rhs=None, name="" ): Add a constraint to a model.
        # lhs: Left-hand side for new linear constraint. Can be a constant, a Var, a LinExpr, or a TempConstr.
        # sense: Sense for new linear constraint (GRB.LESS_EQUAL, GRB.EQUAL, or GRB.GREATER_EQUAL).
        # rhs: Right-hand side for new linear constraint. Can be a constant, a Var, or a LinExpr.
        # TODO: CONFIRM THIS: the >= operator is probably overloaded to produce the above arguments
        # EXPL: to think the constraint below another way,
        # EXPL: every element right and bottom coordinates can be at max (maxX, maxY)
        model.addConstr(x_max >= var.r[i])
        model.addConstr(y_max >= var.b[i])

    objective_grid_count = LinExpr(0.0) # EXPL: Initialize a linear expression with a constant
    for i in range(layout.n):
        # EXPL: LinExpr.addTerms ( coeffs, vars ):
        objective_grid_count.addTerms([2.0, 2.0], [var.lag[i], var.tag[i]])
        objective_grid_count.addTerms([1.0, 1.0], [var.bag[i], var.rag[i]])
    objective_lt = LinExpr(0)
    for i in range(layout.n):
        objective_lt.addTerms([1, 1, 2, 2, -1, -1],
                              [var.t[i], var.l[i], var.b[i], var.r[i], var.w[i], var.h[i]])
    full_objective = LinExpr(0)
    # EXPL: LinExpr.add( expr, mult=1.0 ): Add one linear expression into another.
    full_objective.add(objective_grid_count, 1)
    full_objective.add(objective_lt, 0.001)
    #Objective.add(maxX, 10)
    #Objective.add(maxY, 10)

    # EXPL: Maximum number of grid lines is at minimum something
    model.addConstr(objective_grid_count >= (compute_minimum_grid(layout.n)))
    # EXPL: Minimize grid line count
    model.setObjective(full_objective, GRB.MINIMIZE)
    return objective_grid_count, objective_lt


def set_variable_bounds(layout: Layout, var: Variables):

    for i, element in enumerate(layout.elements):
        var.l[i].LB = 0                                                 # EXPL: Lower bound for left edge
        var.l[i].UB = layout.canvas_width - element.minWidth  # EXPL: Upper bound for left edge

        var.r[i].LB = element.minWidth
        var.r[i].UB = layout.canvas_width

        var.t[i].LB = 0
        var.t[i].UB = layout.canvas_height - element.minHeight

        var.b[i].LB = element.minHeight
        var.b[i].UB = layout.canvas_height

        var.w[i].LB = element.minWidth
        var.w[i].UB = element.maxWidth

        var.h[i].LB = element.minHeight
        var.h[i].UB = element.maxHeight

        var.v_lag[i].LB = 0
        var.v_lag[i].UB = layout.canvas_width - 1

        var.v_rag[i].LB = 1
        var.v_rag[i].UB = layout.canvas_width

        var.v_tag[i].LB = 0
        var.v_tag[i].UB = layout.canvas_height - 1

        var.v_bag[i].LB = 1
        var.v_bag[i].UB = layout.canvas_height


def compute_minimum_grid(n: int) -> int:
    min_grid_width = int(math.sqrt(n))
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


def tap_solutions(model: Model, where):
    if where == GRB.Callback.MIPSOL:
        layout: Layout = model._layout
        var: Variables = model._var

        obj_value = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        lower_bound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        percent_gap = (obj_value - lower_bound) / lower_bound

        t = model.cbGet(GRB.Callback.RUNTIME)
        if percent_gap > 0.2:
            if t < 5 or t < layout.n:
                # TODO: ask Niraj why?
                print("Neglected poor solution")
                return
        print("Entering solution because t=",t," and gap%=",percent_gap)

        obj_value = math.floor(obj_value * 10000) / 10000.0
        print("Tapped into Solution No",
              model._solution_number,
              "of objective value ",
              obj_value,
              "with lower bound at",
              lower_bound)
        x_list = [model.cbGetSolution(var.l[i]) for i in range(layout.n)]
        y_list = [model.cbGetSolution(var.t[i]) for i in range(layout.n)]
        w_list = [model.cbGetSolution(var.w[i]) for i in range(layout.n)]
        h_list = [model.cbGetSolution(var.h[i]) for i in range(layout.n)]

        solution = Solution(obj_value, x_list, y_list, w_list, h_list)
        if solution in model._solutions:
            print("** Neglecting a repeat solution **")
            return
        else:
            model._solutions.append(solution)
            model._solution_number += 1

