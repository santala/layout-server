from itertools import combinations, product
from math import factorial, floor, sqrt

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError

from .classes import Layout, Element

def solve(layout: Layout):

    n = layout.n
    m = Model('GLayoutQuality')
    m._grid_size = 8

    elem_indices = range(n)
    edge_indices = ['x0', 'y0', 'x1', 'y1']

    try:

        # VARIABLES

        # Element coordinates (in multiples of grid size)
        #edge_coord = m.addVars(edge_indices, elem_indices, vtype=GRB.INTEGER, name='EdgeCoord')
        x0 = m.addVars(n, vtype=GRB.INTEGER, name='X0')
        y0 = m.addVars(n, vtype=GRB.INTEGER, name='Y0')
        x1 = m.addVars(n, vtype=GRB.INTEGER, name='X1')
        y1 = m.addVars(n, vtype=GRB.INTEGER, name='Y1')

        m.addConstrs((x0[i] <= x1[i] - 1 for i in range(n)), name='X0X1Sanity')
        m.addConstrs((y0[i] <= y1[i] - 1 for i in range(n)), name='X0X1Sanity')
        m.addConstrs((x1[i] <= round(layout.canvas_width / m._grid_size) for i in range(n)), name='LimitToCanvasW')
        m.addConstrs((y1[i] <= round(layout.canvas_height / m._grid_size) for i in range(n)), name='LimitToCanvasH')


        # Element size (in multiples of grid size)
        w = m.addVars(n, lb=1, vtype=GRB.INTEGER, name='W')
        h = m.addVars(n, lb=1, vtype=GRB.INTEGER, name='H')
        # Bind width and height to the coordinates
        m.addConstrs((x1[i] - x0[i] == w[i] for i in range(n)), 'X1-X0=W')
        m.addConstrs((y1[i] - y0[i] == h[i] for i in range(n)), 'Y1-Y0=H')





        for i, element in enumerate(layout.elements):
            break
            x0[i].start = round(element.x / m._grid_size)
            y0[i].start = round(element.y / m._grid_size)
            w[i].start = round(element.width / m._grid_size)
            h[i].start = round(element.height / m._grid_size)


        # SCALING OF ELEMENTS

        resize_width = m.addVars(n, lb=-GRB.INFINITY, vtype=GRB.INTEGER, name='ResizeW')
        m.addConstrs((
            resize_width[i] == w[i] - round(element.width / m._grid_size) for i, element in enumerate(layout.elements)
        ), name='LinkResizeW')
        resize_width_abs = m.addVars(n, vtype=GRB.INTEGER, name='ResizeWAbs')
        m.addConstrs((
            resize_width_abs[i] == abs_(resize_width[i]) for i in range(n)
        ), name='LinkResizeWAbs')
        resize_height = m.addVars(n, lb=-GRB.INFINITY, vtype=GRB.INTEGER, name='ResizeH')
        m.addConstrs((
            resize_height[i] == h[i] - round(element.height / m._grid_size) for i, element in enumerate(layout.elements)
        ), name='LinkResizeH')
        resize_height_abs = m.addVars(n, vtype=GRB.INTEGER, name='ResizeHAbs')
        m.addConstrs((
            resize_height_abs[i] == abs_(resize_height[i]) for i in range(n)
        ), name='LinkResizeHAbs')

        # MOVEMENT OF ELEMENTS

        move_x = m.addVars(n, lb=-GRB.INFINITY, vtype=GRB.INTEGER, name='MoveX')
        m.addConstrs((
            move_x[i] == x0[i] - round(element.x0 / m._grid_size) for i, element in enumerate(layout.elements)
        ), name='LinkMoveX')
        move_x_abs = m.addVars(n, vtype=GRB.INTEGER, name='MoveXAbs')
        m.addConstrs((
            move_x_abs[i] == abs_(move_x[i]) for i in range(n)
        ), name='LinkMoveXAbs')

        move_y = m.addVars(n, lb=-GRB.INFINITY, vtype=GRB.INTEGER, name='MoveY')
        m.addConstrs((
            move_y[i] == y0[i] - round(element.y0 / m._grid_size) for i, element in enumerate(layout.elements)
        ), name='LinkMoveY')
        move_y_abs = m.addVars(n, vtype=GRB.INTEGER, name='MoveYAbs')
        m.addConstrs((
            move_y_abs[i] == abs_(move_y[i]) for i in range(n)
        ), name='LinkMoveYAbs')

        # ABOVE & ON LEFT

        above = m.addVars(n, n, vtype=GRB.BINARY, name='Above')
        m.addConstrs((
            (above[i1, i2] == 1) >> (y1[i1] <= y0[i2])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkAbove1')
        m.addConstrs((
            (above[i1, i2] == 0) >> (y1[i1] >= y0[i2] + 1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkAbove2')
        m.addConstrs((
            above[i1, i2] + above[i2, i1] <= 1
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='AboveSanity') # TODO: check if sanity checks are necessary

        on_left = m.addVars(n, n, vtype=GRB.BINARY, name='OnLeft')
        m.addConstrs((
            (on_left[i1, i2] == 1) >> (x1[i1] <= x0[i2])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkOnLeft1')
        m.addConstrs((
            (on_left[i1, i2] == 0) >> (x1[i1] >= x0[i2] + 1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkOnLeft2')
        m.addConstrs((
            on_left[i1, i2] + on_left[i2, i1] <= 1
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='OnLeftSanity')


        # OVERLAP

        element_pairs = [(i1, i2) for i1, i2 in product(range(n), range(n)) if i1 != i2]

        h_overlap = m.addVars(element_pairs, vtype=GRB.BINARY, name='HorizontalOverlap')
        m.addConstrs((
            h_overlap[i1, i2] == 1 - (on_left[i1, i2] + on_left[i2, i1])
            for i1, i2 in element_pairs
        ), name='LinkHorizontalOverlap')

        v_overlap = m.addVars(n, n, vtype=GRB.BINARY, name='VerticalOverlap')
        m.addConstrs((
            v_overlap[i1, i2] == 1 - (above[i1, i2] + above[i2, i1])
            for i1, i2 in element_pairs
        ), name='LinkVerticalOverlap')

        overlap = m.addVars(n, n, vtype=GRB.BINARY, name='Overlap')
        m.addConstrs((
            overlap[i1, i2] == and_(h_overlap[i1, i2], v_overlap[i1, i2])
            for i1, i2 in element_pairs
        ), name='LinkOverlap')

        # EXPL: the existence of below variable speeds up the optimizer, even if the variable isn’t used
        edge_diff = m.addVars(edge_indices, elem_indices, elem_indices, lb=-GRB.INFINITY, vtype=GRB.INTEGER,
                              name='EdgeDistance')
        for edge, edge_var in zip(edge_indices, [x0, y0, x1, y1]):
            m.addConstrs((
                edge_diff[edge, i1, i2] == edge_var[i1] - edge_var[i2]
                for i1, i2 in product(elem_indices, elem_indices)
            ), name='Link' + str(edge) + 'Diff')


        # IN PREV COL

        x0_diff, y0_diff, x1_diff, y1_diff = [
            m.addVars(n, n, lb=-GRB.INFINITY, vtype=GRB.INTEGER, name=name)
            for name in ['X0Diff', 'Y0Diff', 'X1Diff', 'Y1Diff']
        ]
        for diff, var in zip([x0_diff, y0_diff, x1_diff, y1_diff], [x0, y0, x1, y1]):
            m.addConstrs((
                diff[i1, i2] == var[i1] - var[i2]
                for i1, i2 in product(range(n), range(n))
            ))


        x0_less_than = m.addVars(n, n, vtype=GRB.BINARY, name='X0LessThan')
        m.addConstrs((
            (x0_less_than[i1, i2] == 1) >> (x0_diff[i1, i2] <= -1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkX0LessThan1')
        m.addConstrs((
            (x0_less_than[i1, i2] == 0) >> (x0_diff[i1, i2] >= 0)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkX0LessThan2')

        y0_less_than = m.addVars(n, n, vtype=GRB.BINARY, name='Y0LessThan')
        m.addConstrs((
            (y0_less_than[i1, i2] == 1) >> (y0_diff[i1, i2] <= -1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkY0LessThan1')
        m.addConstrs((
            (y0_less_than[i1, i2] == 0) >> (y0_diff[i1, i2] >= 0)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkY0LessThan2')

        x1_less_than = m.addVars(n, n, vtype=GRB.BINARY, name='X1LessThan')
        m.addConstrs((
            (x1_less_than[i1, i2] == 1) >> (x1_diff[i1, i2] <= -1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkX1LessThan1')
        m.addConstrs((
            (x1_less_than[i1, i2] == 0) >> (x1_diff[i1, i2] >= 0)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkX1LessThan2')

        y1_less_than = m.addVars(n, n, vtype=GRB.BINARY, name='Y1LessThan')
        m.addConstrs((
            (y1_less_than[i1, i2] == 1) >> (y1_diff[i1, i2] <= -1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkY1LessThan1')
        m.addConstrs((
            (y1_less_than[i1, i2] == 0) >> (y1_diff[i1, i2] >= 0)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkY1LessThan2')



        # ALT NUMBER OF GROUPS
        x0_group = m.addVars(n, lb=1, ub=n, vtype=GRB.INTEGER, name='X0Group')
        y0_group = m.addVars(n, lb=1, ub=n, vtype=GRB.INTEGER, name='Y0Group')
        x1_group = m.addVars(n, lb=1, ub=n, vtype=GRB.INTEGER, name='X1Group')
        y1_group = m.addVars(n, lb=1, ub=n, vtype=GRB.INTEGER, name='Y1Group')
        m.addConstrs((
            (x0_less_than[i1, i2] == 1) >> (x0_group[i1] <= x0_group[i2] - 1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkX0Group1')
        m.addConstrs((
            (x0_less_than[i1, i2] == 0) >> (x0_group[i1] >= x0_group[i2])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkX0Group2')
        m.addConstrs((
            (y0_less_than[i1, i2] == 1) >> (y0_group[i1] <= y0_group[i2] - 1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkY0Group1')
        m.addConstrs((
            (y0_less_than[i1, i2] == 0) >> (y0_group[i1] >= y0_group[i2])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkY0Group2')
        m.addConstrs((
            (x1_less_than[i1, i2] == 1) >> (x1_group[i1] <= x1_group[i2] - 1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkX1Group1')
        m.addConstrs((
            (x1_less_than[i1, i2] == 0) >> (x1_group[i1] >= x1_group[i2])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkX1Group2')
        m.addConstrs((
            (y1_less_than[i1, i2] == 1) >> (y1_group[i1] <= y1_group[i2] - 1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkY1Group1')
        m.addConstrs((
            (y1_less_than[i1, i2] == 0) >> (y1_group[i1] >= y1_group[i2])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkY1Group2')

        x0_group_count = m.addVar(lb=1, ub=n, vtype=GRB.INTEGER, name='X0GroupCount')
        y0_group_count = m.addVar(lb=1, ub=n, vtype=GRB.INTEGER, name='Y0GroupCount')
        x1_group_count = m.addVar(lb=1, ub=n, vtype=GRB.INTEGER, name='X1GroupCount')
        y1_group_count = m.addVar(lb=1, ub=n, vtype=GRB.INTEGER, name='Y1GroupCount')
        m.addConstr(x0_group_count == max_(x0_group))
        m.addConstr(y0_group_count == max_(y0_group))
        m.addConstr(y0_group_count == max_(x1_group))
        m.addConstr(y1_group_count == max_(y1_group))











        # OBJECTIVES

        # BALANCE

        elem_areas = [QuadExpr(w[i] * h[i]) for i in range(n)]
        elem_d_x = [LinExpr(0.5 * (w[i] * m._grid_size + layout.canvas_width) - x0[i] * m._grid_size) for i in range(n)]
        elem_d_y = [LinExpr(0.5 * (h[i] * m._grid_size + layout.canvas_height) - y0[i] * m._grid_size) for i in range(n)]


        # B = 1 - (Bh + Bv) / 2
        # Bh = |Wr-Wl|/max(|Wl|,|Wr|)
        # Wr = sum( ar*d )
        # Wl = sum( al*-d ) = -sum( al*d )
        # |Wr-Wl| = |Wr+Wl| = sum(a*d)

        # EXPL: If the objective is to maximize Balance, it helps to maximize the denominator, i.e. max(|Wl|,|Wr|)
        # EXPL: therefore, it should be enough to define: denominator <= [-Wl, Wl, -Wr, Wr]

        # Minimize scaling of elements

        element_width_coeffs = [1 / e.width for e in layout.elements]
        element_height_coeffs = [1 / e.height for e in layout.elements]

        resize_expr = LinExpr(0.0)
        for i in range(layout.n):
            resize_expr.addTerms([element_width_coeffs[i]], [resize_width_abs[i]])
            resize_expr.addTerms([element_height_coeffs[i]], [resize_height_abs[i]])

        move_expr = LinExpr(0.0)
        for i in range(layout.n):
            move_expr.addTerms([element_width_coeffs[i]], [move_x_abs[i]])
            move_expr.addTerms([element_height_coeffs[i]], [move_y_abs[i]])

        # Minimize number of grid lines
        # Number of elements sharing an edge

        h_overlap_expr = h_overlap.sum()
        v_overlap_expr = v_overlap.sum()
        overlap_expr = overlap.sum()



        number_of_groups_expr = LinExpr()
        number_of_groups_expr.add(x0_group_count)
        number_of_groups_expr.add(y0_group_count)
        number_of_groups_expr.add(x1_group_count)
        number_of_groups_expr.add(y1_group_count)
        m.addConstr(number_of_groups_expr >= compute_minimum_grid(n), name='PreventOvertOptimization')


        obj_expr = LinExpr()

        # MINIMIZE OVERLAP
        #obj_expr.add(overlap_expr, 100)

        # PREVENT OVERLAP
        #m.addConstr(overlap_expr == 0)

        # PRESERVE OVERLAP
        if True:

            m.addConstrs((
                overlap[i1, i2] == do_overlap(layout.elements[i1], layout.elements[i2])
                for i1, i2 in element_pairs
            ), name='PreserveOverlap')

            for i1, i2 in element_pairs:

                e1: Element = layout.elements[i1]
                e2: Element = layout.elements[i2]
                print(e1.elementType, e2.elementType, do_overlap(e1, e2))
                if do_overlap(e1, e2):
                    # x0
                    if e1.x0 < e2.x0:
                        m.addConstr(x0[i1] <= x0[i2] - 1, 'DontAlignOverlappingX0'+str(i1)+','+str(i2))
                    elif e1.x0 > e2.x0:
                        m.addConstr(x0[i1] >= x0[i2] + 1, 'DontAlignOverlappingX0'+str(i1)+','+str(i2))
                    # x1
                    if e1.x1 < e2.x1:
                        m.addConstr(x1[i1] <= x1[i2] - 1, 'DontAlignOverlappingX1' + str(i1) + ',' + str(i2))
                    elif e1.x1 > e2.x1:
                        m.addConstr(x1[i1] >= x1[i2] + 1, 'DontAlignOverlappingX1' + str(i1) + ',' + str(i2))
                    # y0
                    if e1.y0 < e2.y0:
                        m.addConstr(y0[i1] <= y0[i2] - 1, 'DontAlignOverlappingY'+str(i1)+','+str(i2))
                    elif e1.y0 > e2.y0:
                        m.addConstr(y0[i1] >= y0[i2] + 1, 'DontAlignOverlappingY'+str(i1)+','+str(i2))
                    # y1
                    if e1.y1 < e2.y1:
                        m.addConstr(y1[i1] <= y1[i2] - 1, 'DontAlignOverlappingY1' + str(i1) + ',' + str(i2))
                    elif e1.y1 > e2.y1:
                        m.addConstr(y1[i1] >= y1[i2] + 1, 'DontAlignOverlappingY1' + str(i1) + ',' + str(i2))

        # MINIMIZE LAYOUT COLUMNS
        # TODO: what about preferred number of columns?


        # MINIMIZE DIFFERENCE BETWEEN LEFT AND RIGHT MARGINS
        left_margin = m.addVar(lb=0, vtype=GRB.INTEGER, name='LeftMargin')
        m.addConstr(left_margin == min_(x0), name='LinkLeftMargin')

        max_x1 = m.addVar(lb=0, vtype=GRB.INTEGER, name='MaxX1')
        m.addConstr(max_x1 == max_(x1), name='LinkMaxX1')

        right_margin = m.addVar(lb=0, vtype=GRB.INTEGER, name='MaxX1')
        m.addConstr(right_margin == floor(layout.canvas_width / 8) - max_x1, name='LinkRightMargin')

        margin_diff_loose_abs = m.addVar(lb=0, vtype=GRB.INTEGER, name='MarginDiffLooseAbs')
        m.addConstr(margin_diff_loose_abs >= left_margin - right_margin, name='LinkMarginDiffLooseAbs1')
        m.addConstr(margin_diff_loose_abs >= right_margin - left_margin, name='LinkMarginDiffLooseAbs2')

        margin_diff_abs_expr = LinExpr()
        margin_diff_abs_expr.add(margin_diff_loose_abs)

        obj_expr.add(margin_diff_abs_expr)

        obj_expr.add(number_of_groups_expr)

        obj_expr.add(move_expr, 1)
        obj_expr.add(resize_expr, 1)

        m.setObjective(obj_expr, GRB.MINIMIZE)

        # https://www.gurobi.com/documentation/8.1/refman/mip_models.html

        m.Params.MIPFocus = 1
        m.Params.TimeLimit = 20

        m.Params.PoolSearchMode = 2
        m.Params.PoolSolutions = 1
        # model.Params.MIPGap = 0.01

        m.Params.MIPGapAbs = 0.97
        m.Params.OutputFlag = 1

        m.write("output/SimoPracticeModel.lp")

        m.optimize()


        if m.Status in [GRB.Status.OPTIMAL, GRB.Status.INTERRUPTED, GRB.Status.TIME_LIMIT]:
            print('Number of groups', number_of_groups_expr.getValue(), 'Minimum:', compute_minimum_grid(n))
            print('Overlap', overlap_expr.getValue(), h_overlap_expr.getValue(), v_overlap_expr.getValue())

            elements = [
                {
                    'id': element.id,
                    'x': int(x0[i].X) * m._grid_size,  # ‘X’ is the value of the variable in the current solution
                    'y': int(y0[i].X) * m._grid_size,
                    'width': int(w[i].X) * m._grid_size,
                    'height': int(h[i].X) * m._grid_size,
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
            if m.Status == GRB.Status.INFEASIBLE:
                m.computeIIS()
                m.write("output/SimoPracticeModel.ilp")
            print('Non-optimal status:', m.Status)
            print(n, factorial(n))
            return {'status': 1}

    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ": " + str(e))
        raise e

def overlap_width(e1: Element, e2: Element):
    return (e1.width + e2.width) - (max(e1.x0 + e1.width, e2.x0 + e2.width) - min(e1.x0, e2.x0))

def overlap_height(e1: Element, e2: Element):
    return (e1.height + e2.height) - (max(e1.y0 + e1.height, e2.y0 + e2.height) - min(e1.y0, e2.y0))

def overlap_area(e1: Element, e2: Element):
    return overlap_width(e1, e2) * overlap_height(e1, e2)

def do_overlap(e1: Element, e2: Element):
    return overlap_width(e1, e2) > 0 and overlap_height(e1, e2) > 0

def compute_minimum_grid(n: int) -> int:
    min_grid_width = int(sqrt(n))
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