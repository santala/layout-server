from itertools import product
from math import factorial, sqrt

from gurobipy import GRB, LinExpr, Model, tupledict, max_, min_

from .classes import Layout, Element

def solve(layout: Layout):

    n = layout.n
    m = Model('GLayoutQuality')
    m._grid_size = 8

    elem_indices = range(n)
    edge_indices = ['x0', 'y0', 'x1', 'y1']

    # VARIABLES

    # Element coordinates (in multiples of grid size)
    edge_coord = m.addVars(edge_indices, elem_indices, vtype=GRB.INTEGER, name='EdgeCoord')
    '''
    x0 = m.addVars(n, vtype=GRB.INTEGER, name='X0')
    y0 = m.addVars(n, vtype=GRB.INTEGER, name='Y0')
    x1 = m.addVars(n, vtype=GRB.INTEGER, name='X1')
    y1 = m.addVars(n, vtype=GRB.INTEGER, name='Y1')
    '''

    # Element size (in multiples of grid size)
    w = m.addVars(n, vtype=GRB.INTEGER, name='W')
    h = m.addVars(n, vtype=GRB.INTEGER, name='H')
    # Bind width and height to the coordinates
    m.addConstrs((edge_coord['x1', i] - edge_coord['x0', i] == w[i] for i in range(n)), 'X1-X0=W')
    m.addConstrs((edge_coord['y1', i] - edge_coord['y0', i] == h[i] for i in range(n)), 'Y1-Y0=H')

    # EXPL: OLD
    # TODO: 4x *ag [n BOOL]: alignment group enabled
    # TODO: 4x v_*ag [n INT]: position of alignment group
    # TODO: 4x at_*ag [n*n BOOL]: whether element is using alignment group

    # EXPL: ALT
    # 4x sharing_*ag [n*n BOOL]: whether corresponding edges of two elements align
    # 3D matrix with shape 4*n*n
    m.addVars(edge_indices, elem_indices, elem_indices, vtype=GRB.BINARY, name='ElementEdgesAlign')
    m.addConstrs((
        edge_coord[edge, i1] == edge_coord[edge, i2]
        for edge, i1, i2
        in product(edge_indices, elem_indices, elem_indices)
    ), name='LinkElementEdgesAlign')

    # Number of elements sharing an edge
    count_aligned_elements = m.addVars(edge_indices, elem_indices, vtype=GRB.INTEGER, name='AlignedElementCount')

    # OBJECTIVES

    # Minimize number of grid lines

    # TODO: check if this needs to multiplied by factorial(n) to keep it as an integer
    inv_count_aligned_elements = m.addVars(edge_indices, elem_indices, vtype=GRB.CONTINUOUS, name='InverseOfAlignedElementCount')
    '''m.addConstrs((
    inv_count_aligned_elements[edge, i] * count_aligned_elements[edge, i] == 1
    for edge, i
    in product(edge_indices, elem_indices)
), name='InvertAlignedElementCount')'''

    count_grid_lines = m.addVars(edge_indices, vtype=GRB.INTEGER, name='GridLineCount')
    m.addConstrs((count_grid_lines[edge] == inv_count_aligned_elements.sum(edge, '*') for edge in edge_indices), name='LinkGridLineCount')

    total_grid_lines_expr = count_grid_lines.sum()
    # Set a minimum constraint to prevent the optimizer from trying to over-optimize
    m.addConstr(total_grid_lines_expr >= compute_minimum_grid(layout.n))

    # Minimize overlap
    # TODO: consider trying to preserve existing overlaps
    left_overlap = m.addVars(elem_indices, elem_indices, vtype=GRB.INTEGER, name='LeftOverlap')
    m.addConstrs((
        left_overlap[i1, i2] == edge_coord['x1', i2] - edge_coord['x0', i1]
        for i1, i2 in product(elem_indices, elem_indices)
    ), name='LinkLeftOverlap')
    horizontal_overlap = m.addVars(elem_indices, elem_indices, vtype=GRB.INTEGER, name='HorizontalOverlap')
    m.addConstrs((
        horizontal_overlap[i1, i2] == max_(0, left_overlap[i1, i2], left_overlap[i2, i1])
        for i1, i2 in product(elem_indices, elem_indices)
    ), name='LinkHorizontalOverlap')
    top_overlap = m.addVars(elem_indices, elem_indices, vtype=GRB.INTEGER, name='TopOverlap')
    m.addConstrs((
        top_overlap[i1, i2] == edge_coord['y1', i2] - edge_coord['y0', i1]
        for i1, i2 in product(elem_indices, elem_indices)
    ), name='LinkLeftOverlap')
    vertical_overlap = m.addVars(elem_indices, elem_indices, vtype=GRB.INTEGER, name='VerticalOverlap')
    m.addConstrs((
        vertical_overlap[i1, i2] == max_(0, top_overlap[i1, i2], top_overlap[i2, i1])
        for i1, i2 in product(elem_indices, elem_indices)
    ), name='LinkHorizontalOverlap')
    overlap = m.addVars(elem_indices, elem_indices, vtype=GRB.INTEGER, name='Overlap')
    m.addConstrs((
        overlap[i1, i2] == horizontal_overlap[i1, i2] + vertical_overlap[i1, i2]
        for i1, i2 in product(elem_indices, elem_indices)
    ), name='LinkOverlap')

    total_overlap_expr = overlap.sum()

    obj_expr = LinExpr()
    obj_expr.add(total_grid_lines_expr)
    obj_expr.add(total_overlap_expr)

    m.setObjective(obj_expr, GRB.MINIMIZE)

    # https://www.gurobi.com/documentation/8.1/refman/mip_models.html

    m.Params.MIPFocus = 1
    m.Params.TimeLimit = 10

    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions = 1
    # model.Params.MIPGap = 0.01

    m.Params.MIPGapAbs = 0.97
    m.Params.OutputFlag = 1

    m.write("output/SimoPracticeModel.lp")

    m.optimize()

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
        return {'status': 1}


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