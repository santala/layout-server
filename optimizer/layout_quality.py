from itertools import combinations, product
from math import factorial, sqrt

from gurobipy import GRB, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError

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
        edge_coord = m.addVars(edge_indices, elem_indices, ub=n, vtype=GRB.INTEGER, name='EdgeCoord')
        m.addConstrs((
            edge_coord['x0', i] <= edge_coord['x1', i] - 1
            for i in elem_indices
        ), name='X0X1Sanity')
        m.addConstrs((
            edge_coord['y0', i] <= edge_coord['y1', i] - 1
            for i in elem_indices
        ), name='Y0Y1Sanity')
        '''
        x0 = m.addVars(n, vtype=GRB.INTEGER, name='X0')
        y0 = m.addVars(n, vtype=GRB.INTEGER, name='Y0')
        x1 = m.addVars(n, vtype=GRB.INTEGER, name='X1')
        y1 = m.addVars(n, vtype=GRB.INTEGER, name='Y1')
        '''
        for i in range(n):
            edge_coord['x0', i].start = i
            edge_coord['y0', i].start = i

        # Element size (in multiples of grid size)
        w = m.addVars(n, lb=1, ub=n, vtype=GRB.INTEGER, name='W')
        h = m.addVars(n, lb=1, ub=n, vtype=GRB.INTEGER, name='H')
        # Bind width and height to the coordinates
        m.addConstrs((edge_coord['x1', i] - edge_coord['x0', i] == w[i] for i in range(n)), 'X1-X0=W')
        m.addConstrs((edge_coord['y1', i] - edge_coord['y0', i] == h[i] for i in range(n)), 'Y1-Y0=H')


        # ABOVE & ON LEFT

        above = m.addVars(n, n, vtype=GRB.BINARY, name='Above')
        m.addConstrs((
            (above[i1, i2] == 1) >> (edge_coord['y1', i1] <= edge_coord['y0', i2])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkAbove1')
        m.addConstrs((
            (above[i1, i2] == 0) >> (edge_coord['y1', i1] >= edge_coord['y0', i2] - 1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkAbove2')

        on_left = m.addVars(n, n, vtype=GRB.BINARY, name='OnLeft')
        m.addConstrs((
            (on_left[i1, i2] == 1) >> (edge_coord['x1', i1] <= edge_coord['x0', i2])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkOnLeft1')
        m.addConstrs((
            (on_left[i1, i2] == 0) >> (edge_coord['x1', i1] >= edge_coord['x0', i2] - 1)
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkOnLeft2')

        # OVERLAP

        h_overlap = m.addVars(n, n, vtype=GRB.BINARY, name='HorizontalOverlap')
        m.addConstrs((
            h_overlap[i1, i2] == 1 - (on_left[i1, i2] + on_left[i2, i1])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkHorizontalOverlap')

        v_overlap = m.addVars(n, n, vtype=GRB.BINARY, name='VerticalOverlap')
        m.addConstrs((
            v_overlap[i1, i2] == 1 - (above[i1, i2] + above[i2, i1])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkVerticalOverlap')

        overlap = m.addVars(n, n, vtype=GRB.BINARY, name='Overlap')
        m.addConstrs((
            overlap[i1, i2] == and_(h_overlap[i1, i2], v_overlap[i1, i2])
            for i1, i2 in product(range(n), range(n)) if i1 != i2
        ), name='LinkOverlap')

        # EXPL: OLD
        # TODO: 4x *ag [n BOOL]: alignment group enabled
        # TODO: 4x v_*ag [n INT]: position of alignment group
        # TODO: 4x at_*ag [n*n BOOL]: whether element is using alignment group

        # EXPL: ALT
        # 4x sharing_*ag [n*n BOOL]: whether corresponding edges of two elements align
        # 3D matrix with shape 4*n*n


        # VAR element edge distance 4*n*n
        edge_diff = m.addVars(edge_indices, elem_indices, elem_indices, lb=-GRB.INFINITY, vtype=GRB.INTEGER, name='EdgeDistance')
        m.addConstrs((
            edge_diff[edge, i1, i2] == edge_coord[edge, i1] - edge_coord[edge, i2]
            for edge, i1, i2 in product(edge_indices, elem_indices, elem_indices)
        ), name='LinkEdgeDistance')


        edge_diff_abs = m.addVars(edge_indices, elem_indices, elem_indices, vtype=GRB.INTEGER, name='EdgeDistanceAbs')
        m.addConstrs((
            edge_diff_abs[edge, i1, i2] == abs_(edge_diff[edge, i1, i2])
            for edge, i1, i2 in product(edge_indices, elem_indices, elem_indices)
        ), name='LinkEdgeDistanceAbs')
        edges_dont_align = m.addVars(edge_indices, elem_indices, elem_indices, vtype=GRB.BINARY, name='EdgesAlign')
        m.addConstrs((
            edges_dont_align[edge, i1, i2] == min_(1, edge_diff_abs[edge, i1, i2])
            for edge, i1, i2 in product(edge_indices, elem_indices, elem_indices)
        ), name='LinkEdgesAlign')

        # 0 if element is aligned with another element that comes before it, else 1
        is_elem_first_in_group = m.addVars(edge_indices, elem_indices, vtype=GRB.BINARY, name='IsElemFirstInGroup')
        m.addConstrs((
            # Forces variable to zero if it is aligned with any previous element
            is_elem_first_in_group[edge, i2] <= edges_dont_align[edge, i1, i2]
            for edge, (i1, i2) in product(edge_indices, combinations(elem_indices, 2))
        ), name='LinkIsElemFirstInGroup1')
        m.addConstrs((
            # Ensures that if variable is zero, the element aligns at least one previous element
            (is_elem_first_in_group[edge, i] == 0) >> (LinExpr([1]*i, [edges_dont_align[edge, prev, i] for prev in range(i)]) <= i - 1)
            for edge, i in product(edge_indices, elem_indices)
        ), name='LinkIsElemFirstInGroup2')

        number_of_groups = m.addVars(edge_indices, vtype=GRB.INTEGER, name='NumberOfGroups')
        m.addConstrs((
            number_of_groups[edge] == is_elem_first_in_group.sum(edge, '*')
            for edge in edge_indices
        ), name='LinkNumberOfGroups')

        # OBJECTIVES

        # Minimize number of grid lines
        # Number of elements sharing an edge

        h_overlap_expr = h_overlap.sum()
        v_overlap_expr = v_overlap.sum()
        overlap_expr = overlap.sum()

        m.addConstr(overlap_expr == 0)
        #m.setObjective(overlap_expr, GRB.MINIMIZE)


        number_of_groups_expr = LinExpr()
        number_of_groups_expr.add(number_of_groups.sum())
        m.addConstr(number_of_groups_expr >= compute_minimum_grid(n), name='PreventOvertOptimization')
        m.setObjective(number_of_groups_expr, GRB.MINIMIZE)

        # https://www.gurobi.com/documentation/8.1/refman/mip_models.html

        m.Params.MIPFocus = 1
        m.Params.TimeLimit = 30

        m.Params.PoolSearchMode = 2
        m.Params.PoolSolutions = 1
        # model.Params.MIPGap = 0.01

        m.Params.MIPGapAbs = 0.97
        m.Params.OutputFlag = 1

        m.write("output/SimoPracticeModel.lp")

        m.optimize()

        print('Number of groups', number_of_groups_expr.getValue(), 'Minimum:', compute_minimum_grid(n))
        print('Overlap', overlap_expr.getValue(), h_overlap_expr.getValue(), v_overlap_expr.getValue())

        if m.Status in [GRB.Status.OPTIMAL, GRB.Status.INTERRUPTED, GRB.Status.TIME_LIMIT]:

            elements = [
                {
                    'id': element.id,
                    'x': int(edge_coord['x0', i].X) * m._grid_size,  # ‘X’ is the value of the variable in the current solution
                    'y': int(edge_coord['y0', i].X) * m._grid_size,
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