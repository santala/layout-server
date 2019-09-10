from typing import List, Tuple

from itertools import combinations, permutations

import time
from gurobipy import *
from tools.GurobiUtils import *
from tools.JSONLoader import Layout, Element
from tools.Constants import *
import math

from collections import namedtuple

Solution = namedtuple('Solution', 'x, y, w, h, objective_value')


def set_combination_constraints_and_objectives(model: Model):

    layout: Layout = model._layout
    template: Layout = model._template
    var: Variables = model._var

    # Count grid lines
    lag = set()
    tag = set()
    rag = set()
    bag = set()

    for element in template.elements:
        lag.add(element.x)
        tag.add(element.y)
        rag.add(element.x+element.width)
        bag.add(element.y+element.height)

    n_lag = len(lag)
    n_tag = len(tag)
    n_rag = len(rag)
    n_bag = len(bag)



    obj_n_lag = LinExpr(0.0)
    obj_n_tag = LinExpr(0.0)
    obj_n_rag = LinExpr(0.0)
    obj_n_bag = LinExpr(0.0)
    for i in range(layout.n):
        obj_n_lag.addTerms([1], [var.lag[i]])
        obj_n_tag.addTerms([1], [var.tag[i]])
        obj_n_rag.addTerms([1], [var.rag[i]])
        obj_n_bag.addTerms([1], [var.bag[i]])


    # Hard constraints for grid lines:
    limit = 3
    model.addConstr(obj_n_lag == [n_lag - limit, n_lag + limit], 'N_LAG')
    model.addConstr(obj_n_tag == [n_tag - limit, n_tag + limit], 'N_TAG')
    model.addConstr(obj_n_rag == [n_rag - limit, n_rag + limit], 'N_RAG')
    model.addConstr(obj_n_bag == [n_bag - limit, n_bag + limit], 'N_BAG')

    # Optimize for same number of grid lines

    # TODO check proper way to set lower bound
    d_n_lag = model.addVar(lb=-1000, vtype=GRB.INTEGER, name='DistanceFromTargetNLAG')
    d_n_tag = model.addVar(lb=-1000, vtype=GRB.INTEGER, name='DistanceFromTargetNTAG')
    d_n_rag = model.addVar(lb=-1000, vtype=GRB.INTEGER, name='DistanceFromTargetNRAG')
    d_n_bag = model.addVar(lb=-1000, vtype=GRB.INTEGER, name='DistanceFromTargetNBAG')
    dabs_n_lag = model.addVar(vtype=GRB.INTEGER, name='AbsDistanceFromTargetNLAG')
    dabs_n_tag = model.addVar(vtype=GRB.INTEGER, name='AbsDistanceFromTargetNTAG')
    dabs_n_rag = model.addVar(vtype=GRB.INTEGER, name='AbsDistanceFromTargetNRAG')
    dabs_n_bag = model.addVar(vtype=GRB.INTEGER, name='AbsDistanceFromTargetNBAG')

    model.addConstr(dabs_n_lag == abs_(d_n_lag), name="absconstr1")
    model.addConstr(dabs_n_tag == abs_(d_n_tag), name="absconstr2")
    model.addConstr(dabs_n_rag == abs_(d_n_rag), name="absconstr3")
    model.addConstr(dabs_n_bag == abs_(d_n_bag), name="absconstr4")

    model.addConstr(d_n_lag == obj_n_lag, name="nconstr1")
    model.addConstr(d_n_tag == obj_n_tag, name="nconstr2")
    model.addConstr(d_n_rag == obj_n_rag, name="nconstr3")
    model.addConstr(d_n_bag == obj_n_bag, name="nconstr4")

    full_objective = dabs_n_lag + dabs_n_tag + dabs_n_rag + dabs_n_bag

    obj_resize = get_resize_expr(model)
    obj_move = get_move_expr(model)

    full_objective.add(obj_move, 100)
    full_objective.add(obj_resize, 100)

    model.setObjective(full_objective, GRB.MINIMIZE)


def solve(layout: Layout, template: Layout=None, results=None) -> dict:



    #try:
    model = Model("GLayout")
    model._layout = layout
    model._template = template
    model._results = results
    model._grid_size = 8

    var = Variables(model)
    model._var = var

    set_variable_bounds(layout, var, model._grid_size)


    set_constraints(model, layout, var)

    if template is not None:
        print('Applying template')
        set_combination_constraints_and_objectives(model)
    else:
        define_objectives(model, layout, var)

    model._solutions = []
    model._solution_number = 1
    model._start_time = time.time()

    model.write("output/NirajPracticeModel.lp")

    set_control_params(model)


    model.optimize(tap_solutions)

    #TODO (from Niraj) check if solution was found. If yes, set the better objective bounds on future solutions



    # TODO: EXPL: analyze brute force purpose
    #repeatBruteForceExecutionForMoreResults(BAG, H, L, LAG, LEFT, ABOVE, N, OBJECTIVE_GRIDCOUNT, OBJECTIVE_LT, RAG, T, TAG,W, data, gurobi, vBAG, vLAG, vRAG, vTAG)
    '''
    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ": " + str(e))
        return {'status': 1}
    '''
    '''
    except AttributeError as e:
        print('AttributeError:' + str(e))
        return {'status': 1}
    '''

    if model.Status in [GRB.Status.OPTIMAL, GRB.Status.INTERRUPTED, GRB.Status.TIME_LIMIT]:

        elements = [
            {
                'id': element.id,
                'x': int(var.l[i].X) * model._grid_size,  # ‘X’ is the value of the variable in the current solution
                'y': int(var.t[i].X) * model._grid_size,
                'width': int(var.w[i].X) * model._grid_size,
                'height': int(var.h[i].X) * model._grid_size,
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
    # https://www.gurobi.com/documentation/8.1/refman/mip_models.html

    model.Params.MIPFocus = 1
    model.Params.TimeLimit = 10

    model.Params.PoolSearchMode = 2
    model.Params.PoolSolutions = 1
    #model.Params.MIPGap = 0.01

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

        model.addConstr(var.resize_width[i] == var.w[i] - math.ceil(element.width / model._grid_size), name="resizeW")
        model.addConstr(var.resize_width_abs[i] == abs_(var.resize_width[i]), name="resizeWAbs")
        model.addConstr(var.resize_height[i] == var.h[i] - math.ceil(element.height / model._grid_size), name="resizeH")
        model.addConstr(var.resize_height_abs[i] == abs_(var.resize_height[i]), name="resizeHAbs")


        if model._results is not None and model._template is not None and element.id in [assignment[0] for assignment in model._results['elementMapping']]:
            # Compute the distance to the position of the corresponding element in the template
            template_elem_id = next((assignment[1] for assignment in model._results['elementMapping'] if assignment[0] == element.id), None)

            print('id', template_elem_id)
            print(model._template.elements)
            template_elem = next((e for e in model._template.elements if e.id == template_elem_id), None)
            # Scale the coordinates to the sketch coordinates
            preferred_x = template_elem.x / model._template.canvas_width * layout.canvas_width
            preferred_y = template_elem.y / model._template.canvas_height * layout.canvas_height
        else:
            # Compute the distance to the position in the sketch
            preferred_x = element.x
            preferred_y = element.y

        model.addConstr(var.move_x[i] == var.l[i] - round(preferred_x / model._grid_size), name="moveX")
        model.addConstr(var.move_x_abs[i] == abs_(var.move_x[i]), name="moveXAbs")
        model.addConstr(var.move_y[i] == var.t[i] - round(preferred_y / model._grid_size), name="moveY")
        model.addConstr(var.move_y_abs[i] == abs_(var.move_y[i]), name="moveYAbs")

        # TODO: support for locking

        # TODO: try to check that constraints don’t make the model infeasible
        # (e.g. by making the element to be too big or putting it outside the canvas)


        if element.constrainLeft:
            model.addConstr(var.l[i] == element.x, "PrespecifiedXOfElement("+str(i)+")")
        else:
            pass #model.addConstr(var.lg[i] * model._grid_size == var.l[i], "SnapXToGrid("+str(i)+")")

        if element.constrainTop:
            model.addConstr(var.t[i] == element.y, "PrespecifiedYOfElement("+str(i)+")")
        else:
            pass  #model.addConstr(var.tg[i] * model._grid_size == var.t[i], "SnapYToGrid("+str(i)+")")

        if element.constrainRight:
            model.addConstr(var.l[i] + var.w[i] == element.x + element.width, "PrespecifiedROfElement("+str(i)+")")

        if element.constrainBottom:
            model.addConstr(var.t[i] + var.h[i] == element.y + element.height, "PrespecifiedBOfElement("+str(i)+")")

        if element.constrainWidth:
            model.addConstr(var.w[i] == element.width, "PrespecifiedWOfElement("+str(i)+")")
        else:
            pass  #model.addConstr(var.wg[i] * model._grid_size == var.w[i], "SnapWToGrid(" + str(i) + ")")

        if element.constrainHeight:
            model.addConstr(var.h[i] == element.height, "PrespecifiedHOfElement("+str(i)+")")
        else:
            pass  #model.addConstr(var.hg[i] * model._grid_size == var.h[i], "SnapHToGrid("+str(i)+")")
        
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


def get_resize_expr(model: Model):
    layout: Layout = model._layout
    var: Variables = model._var

    element_width_coeffs = [1 / e.width for e in layout.elements]
    element_height_coeffs = [1 / e.height for e in layout.elements]

    resize_expr = LinExpr(0.0)
    for i in range(layout.n):
        resize_expr.addTerms([element_width_coeffs[i]], [var.resize_width_abs[i]])
        resize_expr.addTerms([element_height_coeffs[i]], [var.resize_height_abs[i]])

    return resize_expr


def get_move_expr(model: Model):
    layout: Layout = model._layout
    var: Variables = model._var

    element_width_coeffs = [1 / e.width for e in layout.elements]
    element_height_coeffs = [1 / e.height for e in layout.elements]

    move_expr = LinExpr(0.0)
    for i in range(layout.n):
        move_expr.addTerms([element_width_coeffs[i]], [var.move_x_abs[i]])
        move_expr.addTerms([element_height_coeffs[i]], [var.move_y_abs[i]])

    return move_expr


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
        objective_grid_count.addTerms([1.0, 1.0], [var.lag[i], var.tag[i]])
        objective_grid_count.addTerms([1.0, 1.0], [var.bag[i], var.rag[i]])
    # TODO: Ask Niraj about objective_lt
    objective_lt = LinExpr(0)
    for i in range(layout.n):
        objective_lt.addTerms([1, 1, 2, 2, -1, -1],
                              [var.t[i], var.l[i], var.b[i], var.r[i], var.w[i], var.h[i]])
    full_objective = LinExpr(0)
    # EXPL: LinExpr.add( expr, mult=1.0 ): Add one linear expression into another.
    full_objective.add(objective_grid_count, 1)
    full_objective.add(objective_lt, 0.001)



    # Minimize resizing and moving of elements
    obj_resize = get_resize_expr(model)
    obj_move = get_move_expr(model)

    full_objective.add(obj_move, 10)
    full_objective.add(obj_resize, 100)

    # EXPL: Maximum number of grid lines is at minimum something
    model.addConstr(objective_grid_count >= (compute_minimum_grid(layout.n)))
    # EXPL: Minimize grid line count
    model.setObjective(full_objective, GRB.MINIMIZE)
    return objective_grid_count, objective_lt


def set_variable_bounds(layout: Layout, var: Variables, grid_size: int):


    for i, element in enumerate(layout.elements):

        var.l[i].LB = 0
        var.l[i].UB = (layout.canvas_width - element.minWidth) / grid_size

        var.r[i].LB = math.ceil(element.minWidth / grid_size)
        var.r[i].UB = layout.canvas_width / grid_size

        var.t[i].LB = 0
        var.t[i].UB = (layout.canvas_height - element.minHeight) / grid_size

        var.b[i].LB = math.ceil(element.minHeight / grid_size)
        var.b[i].UB = layout.canvas_height / grid_size

        var.w[i].LB = math.ceil(element.minWidth / grid_size)
        var.w[i].UB = math.ceil(element.maxWidth / grid_size)

        var.h[i].LB = math.ceil(element.minHeight / grid_size)
        var.h[i].UB = math.ceil(element.maxHeight / grid_size)

        var.v_lag[i].LB = 0
        var.v_lag[i].UB = layout.canvas_width / grid_size - 1

        var.v_rag[i].LB = 1
        var.v_rag[i].UB = layout.canvas_width / grid_size

        var.v_tag[i].LB = 0
        var.v_tag[i].UB = layout.canvas_height / grid_size - 1

        var.v_bag[i].LB = 1
        var.v_bag[i].UB = layout.canvas_height / grid_size


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

        t = model.cbGet(GRB.Callback.RUNTIME)


        obj_value = math.floor(obj_value * 10000) / 10000.0
        print("Tapped into Solution No",
              model._solution_number,
              "of objective value ",
              obj_value,
              "with lower bound at",
              lower_bound,
              'at runtime',
              t, time.time(), model._start_time)
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

