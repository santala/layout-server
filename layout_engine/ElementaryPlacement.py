from typing import List, Tuple

from gurobipy import *
from tools.GurobiUtils import *
from tools.JSONLoader import Layout
from tools.JSonExportUtility import *
from tools.PlotUtility import *
from tools.Constants import *
from . import SolutionManager
import math, time
import tools.GurobiUtils

def solve(layout: Layout) -> dict:

    try:
        model = Model("GLayout")
        model._layout = layout

        var = Variables(model)

        model._var = var

        set_variable_bounds(layout, var)

        objective_grid_count, objective_lt = define_objectives(model, layout, var)

        set_constraints(model, layout, var)

        model._solution_number = 1

        model.write("output/NirajPracticeModel.lp")

        set_control_params(model)
        model._hash_to_solution = dict()

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
        return {'status': 0}

    except AttributeError  as e:
        print('AttributeError:' + str(e))
        return {'status': 0}

    if model.Status == GRB.Status.OPTIMAL:

        elements = []

        for e in range(layout.n):

            elements.append({
                'id': layout.elements[e].id,
                'x': var.L[e].getAttr('X'), # ‘X’ is the value of the variable in the current solution
                'y': var.T[e].getAttr('X'),
                'width': var.W[e].getAttr('X'),
                'height': var.H[e].getAttr('X'),
            })

        return {
            'status': 1,
            'layout': {
                'canvasWidth': layout.canvasWidth,
                'canvasHeight': layout.canvasHeight,
                'elements': elements
            }
        }
    else:
        return {'status': 0}


def repeatBruteForceExecutionForMoreResults(model: Model, layout: Layout, var: Variables):
    for topElem in range(layout.n):
        for bottomElem in range(layout.n):
            if (topElem != bottomElem):

                temporaryConstraint = model.addConstr(var.LEFT[topElem, bottomElem] == 1)
                model.optimize(tap_solutions)
                model.remove(temporaryConstraint)

                temporaryConstraint = model.addConstr(var.ABOVE[topElem, bottomElem] == 1)
                model.optimize(tap_solutions)
                model.remove(temporaryConstraint)

def reportResult(BAG, H, L, LAG, N, OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT, RAG, T, TAG, W, data, gurobi, vBAG, vLAG, vRAG,vTAG):
    print("Value of grid measure is: ", OBJECTIVE_GRIDCOUNT.getValue())
    print("Value of LT objective is: ", OBJECTIVE_LT.getValue())
    for solNo in range(gurobi.Params.PoolSolutions):
        Hval, Lval, Tval, Wval = extract_variable_values(N, H, L, T, W, gurobi, solNo)

        # Output
        SaveToJSon(N, data.canvasWidth, data.canvasHeight, Lval, Tval, Wval, Hval, 100+solNo, data, gurobi.getObjective().getValue())

        printResultToConsole(N, BAG, LAG, RAG, TAG, vBAG, vLAG, vRAG, vTAG)

        DrawPlotOnPage(N, data.canvasWidth, data.canvasHeight, Lval, Tval, Wval, Hval, 100+solNo)


def set_control_params(gurobi):
    gurobi.Params.PoolSearchMode = 2
    gurobi.Params.PoolSolutions = 1
    #gurobi.Params.MIPGap = 0.01
    #gurobi.Params.TimeLimit = 75
    gurobi.Params.MIPGapAbs = 0.97
    gurobi.Params.LogFile = "output/GurobiLog.txt"
    gurobi.Params.OutputFlag = 0


#def set_constraints(ABOVE, B, BAG, H, L, LAG, LEFT, N, R, RAG, T, TAG, W, data, elemAtBAG, elemAtLAG, elemAtRAG, elemAtTAG, gurobi, vBAG, vLAG, vRAG, vTAG):
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
    for element in range(layout.n):
        if (layout.elements[element].X is not None and layout.elements[element].X >= 0):
            # EXPL: Does this lock element X coordinate? Answer: most probably
            model.addConstr(var.L[element] == layout.elements[element].X, "PrespecifiedXOfElement(", element, ")")
        if (layout.elements[element].Y is not None and layout.elements[element].Y >= 0):
            # EXPL: Does this lock element Y coordinate?
            model.addConstr(var.T[element] == layout.elements[element].Y, "PrespecifiedYOfElement(", element, ")")
        if (layout.elements[element].aspectRatio is not None and layout.elements[element].aspectRatio > 0.001):
            # EXPL: Does this lock element aspect ratio?
            model.addConstr(var.W[element] == layout.elements[element].aspectRatio * var.H[element],
                             "PrespecifiedAspectRatioOfElement(", element, ")")
    # Known Position constraints TOP BOTTOM LEFT RIGHT
    coeffsForAbsolutePositionExpression = []
    varsForAbsolutePositionExpression = []
    for element in range(layout.n):
        for other in range(layout.n):
            # EXPL: loop through element pairs, handling every pair twice
            if (element != other):
                # EXPL: handle preferences for positioning
                if (layout.elements[element].verticalPreference.lower() == "top"):
                    varsForAbsolutePositionExpression.append(var.ABOVE[other, element])
                    coeffsForAbsolutePositionExpression.append(1.0)
                if (layout.elements[element].verticalPreference.lower() == "bottom"):
                    varsForAbsolutePositionExpression.append(var.ABOVE[element, other])
                    coeffsForAbsolutePositionExpression.append(1.0)
                if (layout.elements[element].horizontalPreference.lower() == "left"):
                    varsForAbsolutePositionExpression.append(var.LEFT[other, element])
                    coeffsForAbsolutePositionExpression.append(1.0)
                if (layout.elements[element].horizontalPreference.lower() == "right"):
                    varsForAbsolutePositionExpression.append(var.LEFT[element, other])
                    coeffsForAbsolutePositionExpression.append(1.0)

    expression = LinExpr(coeffsForAbsolutePositionExpression, varsForAbsolutePositionExpression)
    # EXPL: This constraint prevents any design where an element is closer to an edge than another element that has a
    # EXPL: preference for that edge.
    model.addConstr(expression == 0, "Disable non-permitted based on prespecified")
    # Height/Width/L/R/T/B Summation Sanity
    for element in range(layout.n):
        # EXPL: a sanity constraint that coordinates match width and height
        model.addConstr(var.W[element] + var.L[element] == var.R[element], "R-L=W(" + str(element) + ")")
        model.addConstr(var.H[element] + var.T[element] == var.B[element], "B-T=H(" + str(element) + ")")
    # MinMax limits of Left-Above interactions
    for element in range(layout.n):
        for otherElement in range(layout.n):
            # EXPL: Loop through element pairs (each pair is handled only once)
            if (element > otherElement):
                # EXPL: apparently a no overlap constraint: i.e. one element has to be at least either on the left side
                # EXPL: or above the other. Conversely, if neither element is above or to the left of the other, they
                # EXPL: overlap.
                model.addConstr(
                    var.ABOVE[element, otherElement] + var.ABOVE[otherElement, element] + var.LEFT[element, otherElement] + var.LEFT[
                        otherElement, element] >= 1,
                    "NoOverlap(" + str(element) + str(otherElement) + ")")
                # EXPL: The following three constraints prevent locating the element on multiple sides of the other.
                # EXPL: I.e. only one element can be ‘the left one’ or ‘the top one’.
                model.addConstr(
                    var.ABOVE[element, otherElement] + var.ABOVE[otherElement, element] + var.LEFT[element, otherElement] + var.LEFT[
                        otherElement, element] <= 2,
                    "UpperLimOfQuadrants(" + str(element) + str(otherElement) + ")")
                model.addConstr(var.ABOVE[element, otherElement] + var.ABOVE[otherElement, element] <= 1,
                                 "Anti-symmetryABOVE(" + str(element) + str(otherElement) + ")")
                model.addConstr(var.LEFT[element, otherElement] + var.LEFT[otherElement, element] <= 1,
                                 "Anti-symmetryLEFT(" + str(element) + str(otherElement) + ")")
    # Interconnect L-R-LEFT and T-B-ABOVE
    for element in range(layout.n):
        for otherElement in range(layout.n):
            # EXPL: Loop through element pairs (every pair is handled twice)
            if (element != otherElement):
                # TODO: Check how HPAD_SPECIFICATION and VPAD_SPECIFICATION are defined
                # EXPL: If element is to the right of the other, check that element right edge is at least HPAD distance
                # EXPL: from the other left edge.
                # EXPL: The canvas width term is just a way to make sure the constraint is true if element is not to the
                # EXPL: left of the other.
                model.addConstr(
                    var.R[element] + HPAD_SPECIFICATION <= var.L[otherElement] + (1 - var.LEFT[element, otherElement]) * layout.canvasWidth
                    , (str(element) + "(ToLeftOf)" + str(otherElement)))
                # EXPL: Same rule as above but vertically:
                model.addConstr(
                    var.B[element] + VPAD_SPECIFICATION <= var.T[otherElement] + (1 - var.ABOVE[element, otherElement]) * layout.canvasHeight
                    , (str(element) + "(Above)" + str(otherElement)))
                # EXPL: If element is not to the left of the other, check that element right edge + HPAD exceed the left
                # EXPL: of the other.
                # EXPL: The canvas width term is just a way to make sure the constraint is true if element is to the
                # EXPL: left of the other.
                model.addConstr(
                    (var.L[otherElement] - var.R[element] - HPAD_SPECIFICATION) <= layout.canvasWidth * var.LEFT[element, otherElement]
                    , (str(element) + "(ConverseOfToLeftOf)" + str(otherElement)))
                # EXPL: Same as above but vertical
                model.addConstr(
                    (var.T[otherElement] - var.B[element] - VPAD_SPECIFICATION) <= layout.canvasHeight * var.ABOVE[element, otherElement]
                    , (str(element) + "(ConverseOfAboveOf)" + str(otherElement)))
    # One Alignment-group for every edge of every element
    # EXPL: The canvas is divided horizontally and vertically into ‘alignment groups’, or grid lines.
    # EXPL: Each element has to have an alignment group (or a grid line) for each of its edges.
    for element in range(layout.n):
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
            varsForLAG.append(var.elemAtLAG[element, alignmentGroupIndex])
            coeffsForLAG.append(1)
            varsForRAG.append(var.elemAtRAG[element, alignmentGroupIndex])
            coeffsForRAG.append(1)
            varsForTAG.append(var.elemAtTAG[element, alignmentGroupIndex])
            coeffsForTAG.append(1)
            varsForBAG.append(var.elemAtBAG[element, alignmentGroupIndex])
            coeffsForBAG.append(1)

        # EXPL: following constraints make sure that an element edge aligns with only one alignment group
        model.addConstr(LinExpr(coeffsForLAG, varsForLAG) == 1, "OneLAGForElement[" + str(element) + "]")
        model.addConstr(LinExpr(coeffsForTAG, varsForTAG) == 1, "OneTAGForElement[" + str(element) + "]")
        model.addConstr(LinExpr(coeffsForBAG, varsForBAG) == 1, "OneBAGForElement[" + str(element) + "]")
        model.addConstr(LinExpr(coeffsForRAG, varsForRAG) == 1, "OneRAGForElement[" + str(element) + "]")
    # Assign alignment groups to elements only if groups are enabled
    for alignmentGroupIndex in range(layout.n):
        for element in range(layout.n):
            # TODO: EXPL: apparently, LAG is a boolean matrix telling, whether an ‘alignment group’ (= grid line) is
            # TODO: EXPL: enabled. If it is not, any element edge should not align with it either.
            model.addConstr(var.elemAtLAG[element, alignmentGroupIndex] <= var.LAG[alignmentGroupIndex])
            model.addConstr(var.elemAtRAG[element, alignmentGroupIndex] <= var.RAG[alignmentGroupIndex])
            model.addConstr(var.elemAtTAG[element, alignmentGroupIndex] <= var.TAG[alignmentGroupIndex])
            model.addConstr(var.elemAtBAG[element, alignmentGroupIndex] <= var.BAG[alignmentGroupIndex])
    # Correlate alignment groups value with element edge if assigned
    for alignmentGroupIndex in range(layout.n):
        for element in range(layout.n):
            # EXPL: If element is assigned to alignment group, check that * edge is less than or equal to v*AG
            model.addConstr(var.L[element] <= var.vLAG[alignmentGroupIndex] + layout.canvasWidth * (
                        1 - var.elemAtLAG[element, alignmentGroupIndex]),
                             "MinsideConnectL[" + str(element) + "]ToLAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.R[element] <= var.vRAG[alignmentGroupIndex] + layout.canvasWidth * (
                        1 - var.elemAtRAG[element, alignmentGroupIndex]),
                             "MinsideConnectR[" + str(element) + "]ToRAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.T[element] <= var.vTAG[alignmentGroupIndex] + layout.canvasHeight * (
                        1 - var.elemAtTAG[element, alignmentGroupIndex]),
                             "MinsideConnectT[" + str(element) + "]ToTAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.B[element] <= var.vBAG[alignmentGroupIndex] + layout.canvasHeight * (
                        1 - var.elemAtBAG[element, alignmentGroupIndex]),
                             "MinsideConnectB[" + str(element) + "]ToBAG[" + str(alignmentGroupIndex) + "]")

            # EXPL: If element is assigned to alignment group, check that * edge is greater than or equal to v*AG
            model.addConstr(var.L[element] >= var.vLAG[alignmentGroupIndex] - layout.canvasWidth * (
                        1 - var.elemAtLAG[element, alignmentGroupIndex]),
                             "MaxsideConnectL[" + str(element) + "]ToLAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.R[element] >= var.vRAG[alignmentGroupIndex] - layout.canvasWidth * (
                        1 - var.elemAtRAG[element, alignmentGroupIndex]),
                             "MaxsideConnectR[" + str(element) + "]ToRAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.T[element] >= var.vTAG[alignmentGroupIndex] - layout.canvasHeight * (
                        1 - var.elemAtTAG[element, alignmentGroupIndex]),
                             "MaxsideConnectT[" + str(element) + "]ToTAG[" + str(alignmentGroupIndex) + "]")
            model.addConstr(var.B[element] >= var.vBAG[alignmentGroupIndex] - layout.canvasHeight * (
                        1 - var.elemAtBAG[element, alignmentGroupIndex]),
                             "MaxsideConnectB[" + str(element) + "]ToBAG[" + str(alignmentGroupIndex) + "]")


#def define_objectives(N, W, H, B, BAG, L, LAG, R, RAG, T, TAG, data, gurobi):
def define_objectives(model: Model, layout: Layout, var: Variables):
    # EXPL: Constraints
    # * every element right and bottom edge can be at max (maxX, maxY)
    # * grid count >= minimum possible grid count (grid count = LAG*2 + TAG*2 + BAG*1 + RAG*1)
    # EXPL: Objectives
    # * Minimize (grid_count + lt * .001)
    # * LT = T + L + 2B + 2R - W - H

    # EXPL: Model.addVar(…): Add a decision variable to a model.
    maxX = model.addVar(vtype=GRB.INTEGER, name="maxX")
    maxY = model.addVar(vtype=GRB.INTEGER, name="maxY")
    for i in range(layout.n):
        # EXPL: addConstr ( lhs, sense, rhs=None, name="" ): Add a constraint to a model.
        # lhs: Left-hand side for new linear constraint. Can be a constant, a Var, a LinExpr, or a TempConstr.
        # sense: Sense for new linear constraint (GRB.LESS_EQUAL, GRB.EQUAL, or GRB.GREATER_EQUAL).
        # rhs: Right-hand side for new linear constraint. Can be a constant, a Var, or a LinExpr.
        # TODO: CONFIRM THIS: the >= operator is probably overloaded to produce the above arguments
        # EXPL: to think the constraint below another way,
        # EXPL: every element right and bottom coordinates can be at max (maxX, maxY)
        model.addConstr(maxX >= var.R[i])
        model.addConstr(maxY >= var.B[i])

    objective_grid_count = LinExpr(0.0) # EXPL: Initialize a linear expression with a constant
    for i in range(layout.n):
        # EXPL: LinExpr.addTerms ( coeffs, vars ):
        objective_grid_count.addTerms([2.0, 2.0], [var.LAG[i], var.TAG[i]])
        objective_grid_count.addTerms([1.0, 1.0], [var.BAG[i], var.RAG[i]])
    objective_lt = LinExpr(0)
    for i in range(layout.n):
        objective_lt.addTerms([1, 1, 2, 2, -1, -1],
                              [var.T[i], var.L[i], var.B[i], var.R[i], var.W[i], var.H[i]])
    Objective = LinExpr(0)
    # EXPL: LinExpr.add( expr, mult=1.0 ): Add one linear expression into another.
    Objective.add(objective_grid_count, 1)
    Objective.add(objective_lt, 0.001)
    #Objective.add(maxX, 10)
    #Objective.add(maxY, 10)

    # EXPL: Maximum number of grid lines is at minimum something
    model.addConstr(objective_grid_count >= (compute_lower_bound(layout.n)))
    # EXPL: Minimize grid line count
    model.setObjective(Objective, GRB.MINIMIZE)
    return objective_grid_count, objective_lt


def set_variable_bounds(layout: Layout, var: Variables):

    for i in range(layout.n):
        var.L[i].LB = 0                                                 # EXPL: Lower bound for left edge
        var.L[i].UB = layout.canvasWidth - layout.elements[i].minWidth  # EXPL: Upper bound for left edge

        var.R[i].LB = layout.elements[i].minWidth
        var.R[i].UB = layout.canvasWidth

        var.T[i].LB = 0
        var.T[i].UB = layout.canvasHeight - layout.elements[i].minHeight

        var.B[i].LB = layout.elements[i].minHeight
        var.B[i].UB = layout.canvasHeight

        var.W[i].LB = layout.elements[i].minWidth
        var.W[i].UB = layout.elements[i].maxWidth

        var.H[i].LB = layout.elements[i].minHeight
        var.H[i].UB = layout.elements[i].maxHeight

        var.vLAG[i].LB = 0
        var.vLAG[i].UB = layout.canvasWidth - 1

        var.vRAG[i].LB = 1
        var.vRAG[i].UB = layout.canvasWidth

        var.vTAG[i].LB = 0
        var.vTAG[i].UB = layout.canvasHeight - 1

        var.vBAG[i].LB = 1
        var.vBAG[i].UB = layout.canvasHeight


def extract_variable_values(N, H, L, T, W, model: Model, solNo):
    model.Params.SolutionNumber = solNo
    Lval = []
    Tval = []
    Wval = []
    Hval = []
    for element in range(N):
        Lval.append(L[element].xn)
        Tval.append(T[element].xn)
        Wval.append(W[element].xn)
        Hval.append(H[element].xn)
    return Hval, Lval, Tval, Wval


def printResultToConsole(N, BAG, LAG, RAG, TAG, vBAG, vLAG, vRAG, vTAG):
    leftCount = 0
    rightCount = 0
    topCount = 0
    bottomCount = 0
    for index in range(N):
        result = "Index:" + str(index) + ": "
        if (LAG[index].xn > 0.99):
            leftCount = leftCount + 1
            result = result + "Left = " + str(round(vLAG[index].xn)) + ","
        else:
            result = result + "Left = <>,"
        if (TAG[index].xn > 0.99):
            topCount = topCount + 1
            result = result + "Top = " + str(round(vTAG[index].xn)) + ","
        else:
            result = result + "Top = <>,"
        if (RAG[index].xn > 0.99):
            rightCount = rightCount + 1
            result = result + "Right = " + str(round(vRAG[index].xn)) + ","
        else:
            result = result + "Right = <>,"
        if (BAG[index].xn > 0.99):
            bottomCount = bottomCount + 1
            result = result + "Bottom = " + str(round(vBAG[index].xn)) + ","
        else:
            result = result + "Bottom = <>,"
        print(result)

def compute_lower_bound(N : int) -> float:
    floorRootN = math.floor(math.sqrt(N))
    countOfElementsInGrid = floorRootN*floorRootN
    countOfNonGridElements = N - countOfElementsInGrid
    if(countOfNonGridElements == 0):
        result =  4*floorRootN
    else:
        countOfAdditionalFilledColumns = math.floor(countOfNonGridElements/floorRootN)
        remainder = (countOfNonGridElements - (countOfAdditionalFilledColumns*floorRootN))
        if(remainder == 0):
            result = (4*floorRootN) + (2*countOfAdditionalFilledColumns)
        else:
            result = (4 * floorRootN) + (2 * countOfAdditionalFilledColumns) + 2
    print("Min Objective value is "+str(result))
    return result


def tap_solutions(model, where):
    if where == GRB.Callback.MIPSOL:
        layout: Layout = model._layout

        objeValue = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        lowerBound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        percentGap = (objeValue - lowerBound)/lowerBound

        t = model.cbGet(GRB.Callback.RUNTIME)
        if(percentGap > 0.2):
            if t < 5 or t < layout.n:
                print("Neglected poor solution")
                return
        print("Entering solution because t=",t," and gap%=",percentGap)

        objeValue = math.floor(objeValue*10000)/10000.0
        print("Tapped into Solution No",model._solution_number," of objective value ",objeValue," with lower bound at ",lowerBound)
        Hval, Lval, Tval, Wval = extract_var_values_from_partial_solution(model)
        print('???', Hval, Lval, Tval, Wval)
        SolutionManager.build_new_solution(model, objeValue, Lval, Tval, Wval, Hval, model._hash_to_solution)
        #tools.GurobiUtils.solNo = tools.GurobiUtils.solNo + 1
        model._solution_number += 1


def extract_var_values_from_partial_solution(model: Model) -> Tuple[List[int], List[int], List[int], List[int]]:
    layout: Layout = model._layout
    var: Variables = model._var
    Lval = []
    Tval = []
    Wval = []
    Hval = []
    for element in range(layout.n):
        Lval.append(model.cbGetSolution(var.L[element]))
        Tval.append(model.cbGetSolution(var.T[element]))
        Wval.append(model.cbGetSolution(var.W[element]))
        Hval.append(model.cbGetSolution(var.H[element]))
    return Hval, Lval, Tval, Wval