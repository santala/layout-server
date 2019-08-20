from gurobipy import *
from tools.GurobiUtils import *
from tools.JSONLoader import DataInstance
from tools.JSonExportUtility import *
from tools.PlotUtility import *
from tools.Constants import *
from . import SolutionManager
import math
import tools.GurobiUtils

def solve(data: DataInstance):
    print(data)
    try:
        # EXPL: Initiliaze variables
        ABOVE, B, BAG, H, L, LAG, LEFT, N, R, RAG, T, TAG, W, elemAtBAG, elemAtLAG, elemAtRAG, elemAtTAG, gurobi, vBAG, vLAG, vRAG, vTAG  =  defineVars(data)

        # EXPL: Set lower and upper bounds
        setVarNames(B, H, L, R, T, W, data, vBAG, vLAG, vRAG, vTAG)

        # EXPL: Define Objective (the objective is to minimize OBJECTIVE_GRIDCOUNT + 0.001*OBJECTIVE_LT)
        OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT = defineObjectives(N, W, H, B, BAG, L, LAG, R, RAG, T, TAG, data, gurobi)

        setConstraints(ABOVE, B, BAG, H, L, LAG, LEFT, N, R, RAG, T, TAG, W, data, elemAtBAG, elemAtLAG, elemAtRAG,elemAtTAG, gurobi, vBAG, vLAG, vRAG, vTAG)

        globalizeVariablesForOpenAccess(H, L, T, W, data)

        gurobi.write("output/NirajPracticeModel.lp")

        setControlParams(gurobi)

        gurobi._hashToSolution = dict()

        gurobi.optimize(tapSolutions)

        #TODO check if solution was found. If yes, set the better objective bounds on future solutions

        #gurobi.computeIIS()
        #gurobi.write("IIS.ilp")

        #gurobi.optimize(tapSolutions)
        #reportResult(BAG, H, L, LAG, N, OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT, RAG, T, TAG, W, data, gurobi,vBAG, vLAG,vRAG, vTAG)

        repeatBruteForceExecutionForMoreResults(BAG, H, L, LAG, LEFT, ABOVE, N, OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT, RAG, T, TAG,W, data, gurobi, vBAG, vLAG, vRAG, vTAG)

    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ": " + str(e))

    except AttributeError  as e:
        print('AttributeError:' + str(e))


def globalizeVariablesForOpenAccess(H, L, T, W, data):
    tools.GurobiUtils.data = data
    tools.GurobiUtils.solNo = 1
    tools.GurobiUtils.L = L
    tools.GurobiUtils.T = T
    tools.GurobiUtils.W = W
    tools.GurobiUtils.H = H


def repeatBruteForceExecutionForMoreResults(BAG, H, L, LAG, LEFT, ABOVE, N, OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT, RAG, T, TAG, W,data, gurobi, vBAG, vLAG, vRAG, vTAG):
    for topElem in range(N):
        for bottomElem in range(N):
            if (topElem != bottomElem):

                temporaryConstraint = gurobi.addConstr(LEFT[topElem, bottomElem] == 1)
                gurobi.optimize(tapSolutions)
                gurobi.remove(temporaryConstraint)

                temporaryConstraint = gurobi.addConstr(ABOVE[topElem, bottomElem] == 1)
                gurobi.optimize(tapSolutions)
                gurobi.remove(temporaryConstraint)

def reportResult(BAG, H, L, LAG, N, OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT, RAG, T, TAG, W, data, gurobi, vBAG, vLAG, vRAG,vTAG):
    print("Value of grid measure is: ", OBJECTIVE_GRIDCOUNT.getValue())
    print("Value of LT objective is: ", OBJECTIVE_LT.getValue())
    for solNo in range(gurobi.Params.PoolSolutions):
        Hval, Lval, Tval, Wval = extractVariableValues(N, H, L, T, W, gurobi, solNo)

        # Output
        SaveToJSon(N, data.canvasWidth, data.canvasHeight, Lval, Tval, Wval, Hval, 100+solNo, data)

        printResultToConsole(N, BAG, LAG, RAG, TAG, vBAG, vLAG, vRAG, vTAG)

        DrawPlotOnPage(N, data.canvasWidth, data.canvasHeight, Lval, Tval, Wval, Hval, 100+solNo)


def setControlParams(gurobi):
    gurobi.Params.PoolSearchMode = 2
    gurobi.Params.PoolSolutions = 1
    #gurobi.Params.MIPGap = 0.01
    #gurobi.Params.TimeLimit = 75
    gurobi.Params.MIPGapAbs = 0.97
    gurobi.Params.LogFile = "output/GurobiLog.txt"
    gurobi.Params.OutputFlag = 0


def setConstraints(ABOVE, B, BAG, H, L, LAG, LEFT, N, R, RAG, T, TAG, W, data, elemAtBAG, elemAtLAG, elemAtRAG,elemAtTAG, gurobi, vBAG, vLAG, vRAG, vTAG):
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
    for element in range(data.N):
        if (data.elements[element].X is not None and data.elements[element].X >= 0):
            # EXPL: Does this lock element X coordinate? Answer: most probably
            gurobi.addConstr(L[element] == data.elements[element].X, "PrespecifiedXOfElement(", element, ")")
        if (data.elements[element].Y is not None and data.elements[element].Y >= 0):
            # EXPL: Does this lock element Y coordinate?
            gurobi.addConstr(T[element] == data.elements[element].Y, "PrespecifiedYOfElement(", element, ")")
        if (data.elements[element].aspectRatio is not None and data.elements[element].aspectRatio > 0.001):
            # EXPL: Does this lock element aspect ratio?
            gurobi.addConstr(W[element] == data.elements[element].aspectRatio * H[element],
                             "PrespecifiedAspectRatioOfElement(", element, ")")
    # Known Position constraints TOP BOTTOM LEFT RIGHT
    coeffsForAbsolutePositionExpression = []
    varsForAbsolutePositionExpression = []
    for element in range(data.N):
        for other in range(data.N):
            # EXPL: loop through element pairs, handling every pair twice
            if (element != other):
                # EXPL: handle preferences for positioning
                if (data.elements[element].verticalPreference.lower() == "top"):
                    varsForAbsolutePositionExpression.append(ABOVE[other, element])
                    coeffsForAbsolutePositionExpression.append(1.0)
                if (data.elements[element].verticalPreference.lower() == "bottom"):
                    varsForAbsolutePositionExpression.append(ABOVE[element, other])
                    coeffsForAbsolutePositionExpression.append(1.0)
                if (data.elements[element].horizontalPreference.lower() == "left"):
                    varsForAbsolutePositionExpression.append(LEFT[other, element])
                    coeffsForAbsolutePositionExpression.append(1.0)
                if (data.elements[element].horizontalPreference.lower() == "right"):
                    varsForAbsolutePositionExpression.append(LEFT[element, other])
                    coeffsForAbsolutePositionExpression.append(1.0)

    expression = LinExpr(coeffsForAbsolutePositionExpression, varsForAbsolutePositionExpression)
    # EXPL: This constraint prevents any design where an element is closer to an edge than another element that has a
    # EXPL: preference for that edge.
    gurobi.addConstr(expression == 0, "Disable non-permitted based on prespecified")
    # Height/Width/L/R/T/B Summation Sanity
    for element in range(N):
        # EXPL: a sanity constraint that coordinates match width and height
        gurobi.addConstr(W[element] + L[element] == R[element], "R-L=W(" + str(element) + ")")
        gurobi.addConstr(H[element] + T[element] == B[element], "B-T=H(" + str(element) + ")")
    # MinMax limits of Left-Above interactions
    for element in range(N):
        for otherElement in range(N):
            # EXPL: Loop through element pairs (each pair is handled only once)
            if (element > otherElement):
                # EXPL: apparently a no overlap constraint: i.e. one element has to be at least either on the left side
                # EXPL: or above the other. Conversely, if neither element is above or to the left of the other, they
                # EXPL: overlap.
                gurobi.addConstr(
                    ABOVE[element, otherElement] + ABOVE[otherElement, element] + LEFT[element, otherElement] + LEFT[
                        otherElement, element] >= 1,
                    "NoOverlap(" + str(element) + str(otherElement) + ")")
                # EXPL: The following three constraints prevent locating the element on multiple sides of the other.
                # EXPL: I.e. only one element can be ‘the left one’ or ‘the top one’.
                gurobi.addConstr(
                    ABOVE[element, otherElement] + ABOVE[otherElement, element] + LEFT[element, otherElement] + LEFT[
                        otherElement, element] <= 2,
                    "UpperLimOfQuadrants(" + str(element) + str(otherElement) + ")")
                gurobi.addConstr(ABOVE[element, otherElement] + ABOVE[otherElement, element] <= 1,
                                 "Anti-symmetryABOVE(" + str(element) + str(otherElement) + ")")
                gurobi.addConstr(LEFT[element, otherElement] + LEFT[otherElement, element] <= 1,
                                 "Anti-symmetryLEFT(" + str(element) + str(otherElement) + ")")
    # Interconnect L-R-LEFT and T-B-ABOVE
    for element in range(N):
        for otherElement in range(N):
            # EXPL: Loop through element pairs (every pair is handled twice)
            if (element != otherElement):
                # TODO: Check how HPAD_SPECIFICATION and VPAD_SPECIFICATION are defined
                # EXPL: If element is to the right of the other, check that element right edge is at least HPAD distance
                # EXPL: from the other left edge.
                # EXPL: The canvas width term is just a way to make sure the constraint is true if element is not to the
                # EXPL: left of the other.
                gurobi.addConstr(
                    R[element] + HPAD_SPECIFICATION <= L[otherElement] + (1 - LEFT[element, otherElement]) * data.canvasWidth
                    , (str(element) + "(ToLeftOf)" + str(otherElement)))
                # EXPL: Same rule as above but vertically:
                gurobi.addConstr(
                    B[element] + VPAD_SPECIFICATION <= T[otherElement] + (1 - ABOVE[element, otherElement]) * data.canvasHeight
                    , (str(element) + "(Above)" + str(otherElement)))
                # EXPL: If element is not to the left of the other, check that element right edge + HPAD exceed the left
                # EXPL: of the other.
                # EXPL: The canvas width term is just a way to make sure the constraint is true if element is to the
                # EXPL: left of the other.
                gurobi.addConstr(
                    (L[otherElement] - R[element] - HPAD_SPECIFICATION) <= data.canvasWidth * LEFT[element, otherElement]
                    , (str(element) + "(ConverseOfToLeftOf)" + str(otherElement)))
                # EXPL: Same as above but vertical
                gurobi.addConstr(
                    (T[otherElement] - B[element] - VPAD_SPECIFICATION) <= data.canvasHeight * ABOVE[element, otherElement]
                    , (str(element) + "(ConverseOfAboveOf)" + str(otherElement)))
    # One Alignment-group for every edge of every element
    # EXPL: The canvas is divided horizontally and vertically into ‘alignment groups’, or grid lines.
    # EXPL: Each element has to have an alignment group (or a grid line) for each of its edges.
    for element in range(N):
        coeffsForLAG = []
        coeffsForRAG = []
        coeffsForTAG = []
        coeffsForBAG = []
        varsForLAG = []
        varsForRAG = []
        varsForTAG = []
        varsForBAG = []
        for alignmentGroupIndex in range(data.N):
            # TODO: does N equal data.N ? If not, check how data.N is computed.
            # EXPL: data.N corresponds to the number of alignment groups
            # EXPL: elemAt*AG is a boolean matrix of whether an edge of an element aligns with a given ‘alignment group’
            varsForLAG.append(elemAtLAG[element, alignmentGroupIndex])
            coeffsForLAG.append(1)
            varsForRAG.append(elemAtRAG[element, alignmentGroupIndex])
            coeffsForRAG.append(1)
            varsForTAG.append(elemAtTAG[element, alignmentGroupIndex])
            coeffsForTAG.append(1)
            varsForBAG.append(elemAtBAG[element, alignmentGroupIndex])
            coeffsForBAG.append(1)

        # EXPL: following constraints make sure that an element edge aligns with only one alignment group
        gurobi.addConstr(LinExpr(coeffsForLAG, varsForLAG) == 1, "OneLAGForElement[" + str(element) + "]")
        gurobi.addConstr(LinExpr(coeffsForTAG, varsForTAG) == 1, "OneTAGForElement[" + str(element) + "]")
        gurobi.addConstr(LinExpr(coeffsForBAG, varsForBAG) == 1, "OneBAGForElement[" + str(element) + "]")
        gurobi.addConstr(LinExpr(coeffsForRAG, varsForRAG) == 1, "OneRAGForElement[" + str(element) + "]")
    # Assign alignment groups to elements only if groups are enabled
    for alignmentGroupIndex in range(data.N):
        for element in range(N):
            # TODO: EXPL: apparently, LAG is a boolean matrix telling, whether an ‘alignment group’ (= grid line) is
            # TODO: EXPL: enabled. If it is not, any element edge should not align with it either.
            gurobi.addConstr(elemAtLAG[element, alignmentGroupIndex] <= LAG[alignmentGroupIndex])
            gurobi.addConstr(elemAtRAG[element, alignmentGroupIndex] <= RAG[alignmentGroupIndex])
            gurobi.addConstr(elemAtTAG[element, alignmentGroupIndex] <= TAG[alignmentGroupIndex])
            gurobi.addConstr(elemAtBAG[element, alignmentGroupIndex] <= BAG[alignmentGroupIndex])
    # Correlate alignment groups value with element edge if assigned
    for alignmentGroupIndex in range(data.N):
        for element in range(N):
            # EXPL: If element is assigned to alignment group, check that * edge is less than or equal to v*AG
            gurobi.addConstr(L[element] <= vLAG[alignmentGroupIndex] + data.canvasWidth * (
                        1 - elemAtLAG[element, alignmentGroupIndex]),
                             "MinsideConnectL[" + str(element) + "]ToLAG[" + str(alignmentGroupIndex) + "]")
            gurobi.addConstr(R[element] <= vRAG[alignmentGroupIndex] + data.canvasWidth * (
                        1 - elemAtRAG[element, alignmentGroupIndex]),
                             "MinsideConnectR[" + str(element) + "]ToRAG[" + str(alignmentGroupIndex) + "]")
            gurobi.addConstr(T[element] <= vTAG[alignmentGroupIndex] + data.canvasHeight * (
                        1 - elemAtTAG[element, alignmentGroupIndex]),
                             "MinsideConnectT[" + str(element) + "]ToTAG[" + str(alignmentGroupIndex) + "]")
            gurobi.addConstr(B[element] <= vBAG[alignmentGroupIndex] + data.canvasHeight * (
                        1 - elemAtBAG[element, alignmentGroupIndex]),
                             "MinsideConnectB[" + str(element) + "]ToBAG[" + str(alignmentGroupIndex) + "]")

            # EXPL: If element is assigned to alignment group, check that * edge is greater than or equal to v*AG
            gurobi.addConstr(L[element] >= vLAG[alignmentGroupIndex] - data.canvasWidth * (
                        1 - elemAtLAG[element, alignmentGroupIndex]),
                             "MaxsideConnectL[" + str(element) + "]ToLAG[" + str(alignmentGroupIndex) + "]")
            gurobi.addConstr(R[element] >= vRAG[alignmentGroupIndex] - data.canvasWidth * (
                        1 - elemAtRAG[element, alignmentGroupIndex]),
                             "MaxsideConnectR[" + str(element) + "]ToRAG[" + str(alignmentGroupIndex) + "]")
            gurobi.addConstr(T[element] >= vTAG[alignmentGroupIndex] - data.canvasHeight * (
                        1 - elemAtTAG[element, alignmentGroupIndex]),
                             "MaxsideConnectT[" + str(element) + "]ToTAG[" + str(alignmentGroupIndex) + "]")
            gurobi.addConstr(B[element] >= vBAG[alignmentGroupIndex] - data.canvasHeight * (
                        1 - elemAtBAG[element, alignmentGroupIndex]),
                             "MaxsideConnectB[" + str(element) + "]ToBAG[" + str(alignmentGroupIndex) + "]")


def defineObjectives(N, W, H, B, BAG, L, LAG, R, RAG, T, TAG, data, gurobi):
    # EXPL: Constraints
    # * every element right and bottom edge can be at max (maxX, maxY)
    # * grid count >= minimum possible grid count (grid count = LAG*2 + TAG*2 + BAG*1 + RAG*1)
    # EXPL: Objectives
    # * Minimize (grid_count + lt * .001)
    # * LT = T + L + 2B + 2R - W - H

    # EXPL: Model.addVar(…): Add a decision variable to a model.
    maxX = gurobi.addVar(vtype=GRB.INTEGER, name="maxX")
    maxY = gurobi.addVar(vtype=GRB.INTEGER, name="maxY")
    for element in range(data.N):
        # EXPL: addConstr ( lhs, sense, rhs=None, name="" ): Add a constraint to a model.
        # lhs: Left-hand side for new linear constraint. Can be a constant, a Var, a LinExpr, or a TempConstr.
        # sense: Sense for new linear constraint (GRB.LESS_EQUAL, GRB.EQUAL, or GRB.GREATER_EQUAL).
        # rhs: Right-hand side for new linear constraint. Can be a constant, a Var, or a LinExpr.
        # TODO: CONFIRM THIS: the >= operator is probably overloaded to produce the above arguments
        # EXPL: to think the constraint below another way,
        # EXPL: every element right and bottom coordinates can be at max (maxX, maxY)
        gurobi.addConstr(maxX >= R[element])
        gurobi.addConstr(maxY >= B[element])

    OBJECTIVE_GRIDCOUNT = LinExpr(0.0) # EXPL: Initialize a linear expression with a constant
    for element in range(data.N):
        # EXPL: LinExpr.addTerms ( coeffs, vars ):
        OBJECTIVE_GRIDCOUNT.addTerms([2.0, 2.0], [LAG[element], TAG[element]])
        OBJECTIVE_GRIDCOUNT.addTerms([1.0, 1.0], [BAG[element], RAG[element]])
    OBJECTIVE_LT = LinExpr(0)
    for element in range(data.N):

        OBJECTIVE_LT.addTerms([1, 1, 2, 2, -1, -1],
                              [T[element], L[element], B[element], R[element], W[element], H[element]])
    Objective = LinExpr(0)
    # EXPL: LinExpr.add( expr, mult=1.0 ): Add one linear expression into another.
    Objective.add(OBJECTIVE_GRIDCOUNT, 1)
    Objective.add(OBJECTIVE_LT, 0.001)
    #Objective.add(maxX, 10)
    #Objective.add(maxY, 10)

    # EXPL: Maximum number of grid lines is at minimum something
    gurobi.addConstr(OBJECTIVE_GRIDCOUNT >= (calculateLowerBound(N)))
    # EXPL: Minimize grid line count
    gurobi.setObjective(Objective, GRB.MINIMIZE)
    return OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT


def setVarNames(B, H, L, R, T, W, data, vBAG, vLAG, vRAG, vTAG):
    for element in range(data.N):
        L[element].LB = 0                                                   # EXPL: Lower bound for left edge
        L[element].UB = data.canvasWidth - data.elements[element].minWidth  # EXPL: Upper bound for left edge

        R[element].LB = data.elements[element].minWidth
        R[element].UB = data.canvasWidth

        T[element].LB = 0
        T[element].UB = data.canvasHeight - data.elements[element].minHeight

        B[element].LB = data.elements[element].minHeight
        B[element].UB = data.canvasHeight

        W[element].LB = data.elements[element].minWidth
        W[element].UB = data.elements[element].maxWidth

        H[element].LB = data.elements[element].minHeight
        H[element].UB = data.elements[element].maxHeight

        vLAG[element].LB = 0
        vLAG[element].UB = data.canvasWidth - 1

        vRAG[element].LB = 1
        vRAG[element].UB = data.canvasWidth

        vTAG[element].LB = 0
        vTAG[element].UB = data.canvasHeight - 1

        vBAG[element].LB = 1
        vBAG[element].UB = data.canvasHeight


def extractVariableValues(N, H, L, T, W, gurobi, solNo):
    gurobi.Params.SolutionNumber = solNo
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


def defineVars(data):
    gurobi = Model("GLayout")
    L = define_1d_int_var_array(gurobi, data.N, "L")                            # Left?
    R = define_1d_int_var_array(gurobi, data.N, "R")                            # Right?
    T = define_1d_int_var_array(gurobi, data.N, "T")                            # Top?
    B = define_1d_int_var_array(gurobi, data.N, "B")                            # Bottom?
    H = define_1d_int_var_array(gurobi, data.N, "H")                            # Height?
    W = define_1d_int_var_array(gurobi, data.N, "W")                            # Width?
    ABOVE = define_2d_bool_var_array_array(gurobi, data.N, data.N, "ABOVE")
    LEFT = define_2d_bool_var_array_array(gurobi, data.N, data.N, "LEFT")
    N = data.N                                                              # EXPL: Number of elements
    LAG = define_1d_bool_var_array(gurobi, data.N, "LAG")                       # EXPL: left alignment group enabled?
    RAG = define_1d_bool_var_array(gurobi, data.N, "RAG")                       # EXPL: right aligment group enabled?
    TAG = define_1d_bool_var_array(gurobi, data.N, "TAG")                       # EXPL: top alignment group enabled?
    BAG = define_1d_bool_var_array(gurobi, data.N, "BAG")                       # EXPL: bottom alignment group enabled?
    vLAG = define_1d_int_var_array(gurobi, data.N, "vLAG")                      # EXPL: ? maybe the value of the grid line?
    vRAG = define_1d_int_var_array(gurobi, data.N, "vRAG")
    vTAG = define_1d_int_var_array(gurobi, data.N, "vTAG")
    vBAG = define_1d_int_var_array(gurobi, data.N, "vBAG")
    elemAtLAG = define_2d_bool_var_array_array(gurobi, data.N, data.N, "zLAG")   # ???
    elemAtRAG = define_2d_bool_var_array_array(gurobi, data.N, data.N, "zRAG")
    elemAtTAG = define_2d_bool_var_array_array(gurobi, data.N, data.N, "zTAG")
    elemAtBAG = define_2d_bool_var_array_array(gurobi, data.N, data.N, "zBAG")
    return ABOVE, B, BAG, H, L, LAG, LEFT, N, R, RAG, T, TAG, W, elemAtBAG, elemAtLAG, elemAtRAG, elemAtTAG, gurobi, vBAG, vLAG, vRAG, vTAG


def calculateLowerBound(N : int) -> int:
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


def tapSolutions(model, where):
    if where == GRB.Callback.MIPSOL:
        objeValue = model.cbGet(GRB.Callback.MIPSOL_OBJ)
        lowerBound = model.cbGet(GRB.Callback.MIPSOL_OBJBND)
        percentGap = (objeValue - lowerBound)/lowerBound
        printThis = 0
        t = model.cbGet(GRB.Callback.RUNTIME)
        if(percentGap > 0.2):
            if(t < 5 or t < tools.GurobiUtils.data.N):
                print("Neglected poor solution")
                return
        print("Entering solution because t=",t," and gap%=",percentGap)
        percentGap = math.floor(percentGap*100)
        objeValue = math.floor(objeValue*10000)/10000.0
        print("Tapped into Solution No",tools.GurobiUtils.solNo," of objective value ",objeValue," with lower bound at ",lowerBound)
        Hval, Lval, Tval, Wval = extractVariableValuesFromPartialSolution(model)
        SolutionManager.buildNewSolution(objeValue,  Lval,Tval, Wval, Hval, model._hashToSolution)
        tools.GurobiUtils.solNo = tools.GurobiUtils.solNo + 1


def extractVariableValuesFromPartialSolution(gurobi):
    Lval = []
    Tval = []
    Wval = []
    Hval = []
    for element in range(tools.GurobiUtils.data.N):
        Lval.append(gurobi.cbGetSolution(tools.GurobiUtils.L[element]))
        Tval.append(gurobi.cbGetSolution(tools.GurobiUtils.T[element]))
        Wval.append(gurobi.cbGetSolution(tools.GurobiUtils.W[element]))
        Hval.append(gurobi.cbGetSolution(tools.GurobiUtils.H[element]))
    return Hval, Lval, Tval, Wval