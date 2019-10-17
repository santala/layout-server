import math

from itertools import permutations, product
from typing import List

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError


from .classes import Layout, Element


def solve(layout: Layout, base_unit: int=8, time_out: int=30, number_of_solutions: int=1):

    m = Model('LayoutGuidelines')

    m.Params.MIPFocus = 1
    m.Params.TimeLimit = time_out

    print('Time out:', time_out)

    # https://www.gurobi.com/documentation/8.1/refman/poolsearchmode.html#parameter:PoolSearchMode
    m.Params.PoolSearchMode = 0 # Use 2 to find high quality solutions
    m.Params.PoolSolutions = 1 # Number of solutions to be saved
    # model.Params.MIPGap = 0.01

    m.Params.MIPGapAbs = 10
    m.Params.OutputFlag = 1

    # TODO
    # For each group:
        # Align edge elements
        # Constrain rest to the remaining area (minus padding)
            # E.g. y0 limit must be max value of top edge elements y1
        # Define a grid (if reasonable)

    elem_ids = [element.id for element in layout.elements]

    m._base_unit = base_unit
    m._min_gutter_width = 1
    m._max_gutter_width = 4

    # Layout dimensions in base units
    m._layout_width = int(layout.canvas_width / m._base_unit)
    m._layout_height = int(layout.canvas_height / m._base_unit)

    # Element width in base units
    elem_width = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='ElementWidth')
    # Element height in base units
    elem_height = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementHeight')

    # Width of the gutter (i.e. the space between adjacent columns)
    gutter_width = m.addVar(lb=m._min_gutter_width, ub=m._max_gutter_width, vtype=GRB.INTEGER, name='GutterWidth')

    available_width = m.addVar(vtype=GRB.INTEGER, lb=m._layout_width, ub=m._layout_width, name='AvailableWidth')
    available_height = m.addVar(vtype=GRB.INTEGER, lb=m._layout_height, ub=m._layout_height, name='AvailableHeight')

    col_count, row_count, col_width, row_height, gutter_width, col_start, row_start, width_error, height_error, gap_count \
        = build_grid(m, layout.elements, available_width, available_height, elem_width, elem_height, gutter_width)

    # OBJECTIVES

    obj = LinExpr()


    # Element scaling

    # Difference of the element width to the original in base units
    # Note, the constraints below define this variable to be *at least* the actual difference,
    # i.e. the variable may take a larger value. However, we will define an objective of minimizing
    # the difference, so it won’t be a problem. This is faster than defining an absolute value constraint.

    min_width_diff = m.addVars(elem_ids, vtype=GRB.INTEGER, name='MinWidthDifferenceToOriginal')
    m.addConstrs((
        min_width_diff[element.id] >= elem_width[element.id] - round(element.width/base_unit)
        for element in layout.elements
    ), name='LinkWidthDiff1')
    m.addConstrs((
        min_width_diff[element.id] >= round(element.width/base_unit) - elem_width[element.id]
        for element in layout.elements
    ), name='LinkWidthDiff2')


    min_height_diff = m.addVars(elem_ids, vtype=GRB.INTEGER, name='MinHeightDifferenceToOriginal')
    m.addConstrs((
        min_height_diff[element.id] >= elem_height[element.id] - round(element.height/base_unit)
        for element in layout.elements
    ), name='LinkHeightDiff1')
    m.addConstrs((
        min_height_diff[element.id] >= round(element.height/base_unit) - elem_height[element.id]
        for element in layout.elements
    ), name='LinkHeightDiff2')

    # Minimize size difference

    obj.add(min_width_diff.sum(), 1)
    obj.add(min_height_diff.sum(), 1)

    # Aim for best fit of grid

    # TODO: add penalty if error is an odd number (i.e. prefer symmetry)
    obj.add(width_error, 1)

    obj.add(height_error, 1)


    # TODO: test which one is better, hard or soft constraint
    m.addConstr(gap_count == 0)
    #obj.add(gap_count, 10)



    m.setObjective(obj, GRB.MINIMIZE)



    try:

        m.optimize()

        if m.Status in [GRB.Status.OPTIMAL, GRB.Status.INTERRUPTED, GRB.Status.TIME_LIMIT]:
            # ‘X’ is the value of the variable in the current solution

            print('Grid Width Error', width_error.getValue())
            print('Gap Count', gap_count.getValue())
            print('Column Count', col_count.X)
            print('Column Width', col_width.X)
            print('Row Count', row_count.X)
            print('Row Height', row_height.X)
            print('Resize Error', min_width_diff.sum().getValue(), min_height_diff.sum().getValue())


            elements = [
                {
                    'id': e,
                    'x': (col_start[e].X - 1) * col_width.X * base_unit,
                    'y': (row_start[e].X - 1) * row_height.X * base_unit,
                    'width': elem_width[e].X * base_unit,
                    'height': elem_height[e].X * base_unit,
                } for e in elem_ids
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

    except GurobiError as e:
        print('Gurobi Error code ' + str(e.errno) + ": " + str(e))
        raise e


def build_grid(m: Model, elements: List[Element], available_width, available_height, elem_width, elem_height, gutter_width):
    '''

    :param m: The Gurobi model
    :param elements: List of elements to align into a grid
    :param available_width: Variable for the maximum width for the grid to take (not including left/right margins)
    :param available_height: Variable for the maximum height for the grid to take (not including left/right margins)
    :return:
    '''
    elem_count = len(elements)
    elem_ids = [element.id for element in elements]

    # Parameters


    min_col_width = 1  # Not including gutter
    min_row_height = 1  # Not including gutter


    # TODO: margins

    # Maximum number of columns that can fit on the layout
    max_col_count = int((m._layout_width + m._min_gutter_width) / (min_col_width + m._min_gutter_width))
    max_row_count = int((m._layout_height + m._min_gutter_width) / (min_row_height + m._min_gutter_width))

    col_counts = range(1, max_col_count + 1)
    row_counts = range(1, max_row_count + 1)

    # Number of columns
    col_count = m.addVar(lb=1, ub=max_col_count, vtype=GRB.INTEGER, name='ColumnCount')

    # Number of rows
    row_count = m.addVar(vtype=GRB.INTEGER, lb=1, ub=elem_count, name='RowCount')

    col_count_selected = m.addVars(col_counts, vtype=GRB.BINARY, name='SelectedColumnCount')
    m.addConstr(col_count_selected.sum() == 1, name='SelectOneColumnCount')  # One option must always be selected
    m.addConstrs((
        # TODO compare performance:
        # (col_count_selected[n] == 1) >> (col_count == n)
        # col_count_selected[n] * n == col_count_selected[n] * col_count
        col_count_selected[n] * (col_count - n) == 0
        for n in col_counts
    ), name='LinkColumnCountToSelection')

    row_count_selected = m.addVars(row_counts, vtype=GRB.BINARY, name='SelectedRowCount')
    m.addConstr(row_count_selected.sum() == 1, name='SelectOneRowCount')  # One option must always be selected
    m.addConstrs((
        # TODO compare performance:
        # (row_count_selected[n] == 1) >> (row_count == n)
        # row_count_selected[n] * n == row_count_selected[n] * row_count
        row_count_selected[n] * (row_count - n) == 0
        for n in row_counts
    ), name='LinkRowCountToSelection')


    # Width of a single column (including gutter width)
    col_width = m.addVar(lb=min_col_width + m._min_gutter_width, vtype=GRB.INTEGER, name='ColumnWidth')
    m.addConstr(col_width <= available_width, name='FitColumnIntoAvailableSpace')
    m.addConstr(col_width - gutter_width >= min_col_width, name='AdjustMinColumnWidthAccordingToGutterWidth')

    # Actual width of the grid (not including left/right margins)
    actual_width = m.addVar(vtype=GRB.INTEGER, name='ActualWidth')
    m.addConstr(actual_width <= available_width, name='FitGridIntoAvailableSpace')
    m.addConstrs((
        # TODO compare performance:
        # (col_count_selected[n] == 1) >> (actual_width == n * col_width - gutter_width)
        # col_count_selected[n] * n * col_width == col_count_selected[n] * (gutter_width + actual_width)
        col_count_selected[n] * (n * col_width - gutter_width - actual_width) == 0
        for n in col_counts
    ), name='LinkActualWidthToColumnCount')

    # Row height (fixed to one base unit)
    row_height = m.addVar(vtype=GRB.INTEGER, lb=1, name='RowHeight')
    m.addConstr(row_height <= available_height, name='FitRowIntoAvailableSpace')
    m.addConstr(row_height - gutter_width >= min_row_height, name='AdjustMinRowHeightAccordingToGutterWidth')

    # Actual height of the grid (not including top/bottom margins)
    actual_height = m.addVar(vtype=GRB.INTEGER, name='ActualHeight')
    m.addConstr(actual_height <= available_height, name='FitGridIntoAvailableSpace')
    m.addConstrs((
        # TODO compare performance:
        # (row_count_selected[n] == 1) >> (actual_height == n * row_height - gutter_width)
        row_count_selected[n] * (n * row_height - gutter_width - actual_height) == 0
        for n in row_counts
    ), name='LinkActualHeightToRowCount')

    # Element coordinates
    col_start = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='StartColumn')
    col_end = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='EndColumn')
    m.addConstrs((
        col_end[e] >= col_start[e]
        for e in elem_ids
    ), name='ColumnStartEndOrder')
    col_span = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='ColumnSpan')
    m.addConstrs((
        col_span[e] <= col_count
        for e in elem_ids
    ), name='ColumnSpanCannotExceedColumnCount')
    m.addConstrs((
        col_span[e] == col_end[e] - col_start[e] + 1
        for e in elem_ids
    ), name='LinkColumnSpan')
    col_span_selected = m.addVars(product(elem_ids, col_counts), vtype=GRB.BINARY, name='SelectedColumnSpan')
    m.addConstrs((
        col_span_selected.sum(e, '*') == 1
        for e in elem_ids
    ), name='SelectOneColumnSpan')
    m.addConstrs((
        # TODO compare performance
        # (col_span_selected[e, n] == 1) >> (col_span[e] == n)
        # col_span_selected[e, n] * n == col_span_selected[e, n] * col_span[e]
        col_span_selected[e, n] * (n - col_span[e]) == 0
        for e, n in product(elem_ids, col_counts)
    ), name='LinkColumnSpanToSelection')

    m.addConstrs((
        (col_span_selected[e, n] == 1) >> (elem_width[e] == n * col_width - gutter_width)
        for e, n in product(elem_ids, col_counts)
    ), name='LinkElementWidthToColumnSpan')

    row_start = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='StartRow')
    row_end = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='EndRow')
    m.addConstrs((
        row_end[e] >= row_start[e]
        for e in elem_ids
    ), name='RowStartEndOrder')
    row_span = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='RowSpan')
    m.addConstrs((
        row_span[e] == row_end[e] - row_start[e] + 1
        for e in elem_ids
    ), name='LinkRowSpan')
    row_span_selected = m.addVars(product(elem_ids, row_counts), vtype=GRB.BINARY, name='SelectedColumnSpan')
    m.addConstrs((
        row_span_selected.sum(e, '*') == 1
        for e in elem_ids
    ), name='SelectOneColumnSpan')
    m.addConstrs((
        # TODO compare performance
        # (row_span_selected[e, n] == 1) >> (row_span[e] == n)
        # row_span_selected[n] * n == row_span_selected[n] * col_span[e]
        row_span_selected[e, n] * (n - row_span[e]) == 0
        for e, n in product(elem_ids, row_counts)
    ), name='LinkRowSpanToSelection')

    m.addConstrs((
        (row_span_selected[e, n] == 1) >> (elem_height[e] == n * row_height - gutter_width)
        for e, n in product(elem_ids, row_counts)
    ), name='LinkElementHeightToRowSpan')

    # TODO: col_span–elem_height relationship
    # TODO: col_start–elem_y relationship
    # TODO: adjacent rows > gutter_width apart

    # Integer: The number of rows between element start rows
    row_start_diff = m.addVars(permutations(elem_ids, 2), vtype=GRB.INTEGER, lb=-GRB.INFINITY,
                               name='StartRowDifference')
    m.addConstrs((
        row_start_diff[e1, e2] == row_start[e1] - row_start[e2]
        for e1, e2 in permutations(elem_ids, 2)
    ))
    # Integer: The number of rows between element start rows
    row_end_diff = m.addVars(permutations(elem_ids, 2), vtype=GRB.INTEGER, lb=-GRB.INFINITY, name='EndRowDifference')
    m.addConstrs((
        row_end_diff[e1, e2] == row_end[e1] - row_end[e2]
        for e1, e2 in permutations(elem_ids, 2)
    ))

    # Binary: whether e1 starts on a any row before e2
    start_row_before = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='StartRowLessThan')
    m.addConstrs((
        # TODO compare performance
        # (start_row_before[e1, e2] == 1) >> (row_start_diff[e1, e2] <= -1)
        start_row_before[e1, e2] * (row_start_diff[e1, e2] + 1) <= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkStartRowLessThan1')
    m.addConstrs((
        # TODO compare performance
        # (start_row_before[e1, e2] == 0) >> (row_start_diff[e1, e2] >= 0)
        (1 - start_row_before[e1, e2]) * row_start_diff[e1, e2] >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkStartRowLessThan2')

    # Binary: whether e1 ends on a any row before e2
    end_row_before = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='StartRowLessThan')
    m.addConstrs((
        # TODO compare performance
        # (end_row_before[e1, e2] == 1) >> (row_end_diff[e1, e2] <= -1)
        end_row_before[e1, e2] * (row_end_diff[e1, e2] + 1) <= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkEndRowOrder1')
    m.addConstrs((
        # TODO compare performance
        # (end_row_before[e1, e2] == 0) >> (row_end_diff[e1, e2] >= 0)
        (1 - end_row_before[e1, e2]) * row_end_diff[e1, e2] >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkEndRowOrder2')

    # At least one element must start at the first column/row
    min_col_start = m.addVar(vtype=GRB.INTEGER, lb=1, ub=1, name='MinStartColumn')
    m.addConstr(min_col_start == min_(col_start), name='EnsureElementInFirstColumn')
    min_row_start = m.addVar(vtype=GRB.INTEGER, lb=1, ub=1, name='MinStartRow')
    m.addConstr(min_row_start == min_(row_start), name='EnsureElementOnFirstRow')

    # Bind column/row count
    # TODO compare performance
    m.addConstr(col_count == max_(col_end), name='LinkColumnCountToElementCoordinates')
    m.addConstr(row_count == max_(row_end), name='LinkRowCountToElementCoordinates')
    '''
    m.addConstrs((
        col_count >= col_end[e]
        for e in elem_ids
    ), name='LinkColumnCountToElementCoordinates')
    m.addConstrs((
        row_count >= row_end[e]
        for e in elem_ids
    ), name='LinkRowCountToElementCoordinates')
    '''

    # The area of the grid in terms of cells, i.e. col_count * row_count
    grid_cell_count = m.addVar(vtype=GRB.INTEGER, lb=1, name='GridCellCount')
    m.addConstrs((
        (col_count_selected[n] == 1) >> (grid_cell_count == n * row_count)
        for n in col_counts
    ), name='LinkGridCellCountToColumnCount')

    # The area of the elements in terms of cells, i.e. col_span * row_span
    # cell_coverage: row_span_equals[e,n] >> cell_coverage == n * row_span
    elem_cell_count = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='ElemCellCount')
    m.addConstrs((
        # Using indicator constraint to avoid quadratic constraints
        # TODO compare performance
        (col_span_selected[e, n] == 1) >> (elem_cell_count[e] == n * row_span[e])
        for e, n in product(elem_ids, col_counts)
    ), name='LinkElemCellCountToColumnCount')

    # Directional relationships

    above = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='Above')
    '''
    m.addConstrs((
        # TODO compare performance
        above[e1, e2] * (row_start[e2] - row_end[e1] - 1) + (1 - above[e1, e2]) * (row_end[e1] - row_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove1')
    '''
    m.addConstrs((
        # TODO compare performance
        (above[e1, e2] == 1) >> (row_end[e1] + 1 <= row_start[e2])
        # above[e1, e2] * (row_start[e2] - row_end[e1] - 1) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove1')
    m.addConstrs((
        # TODO compare performance
        (above[e1, e2] == 0) >> (row_end[e1] >= row_start[e2])
        # (1 - above[e1, e2]) * (row_end[e1] - row_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove2')

    m.addConstrs((
        above[e1, e2] + above[e2, e1] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='AboveSanity')  # TODO: check if sanity checks are necessary

    on_left = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='OnLeft')
    m.addConstrs((
        # TODO compare performance
        (on_left[e1, e2] == 1) >> (col_end[e1] + 1 <= col_start[e2])
        # on_left[e1, e2] * (col_start[e2] - col_end[e1] - 1) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOnLeft1')
    m.addConstrs((
        # TODO compare performance
        (on_left[e1, e2] == 0) >> (col_end[e1] >= col_start[e2])
        # (1 - on_left[e1, e2]) * (col_end[e1] - col_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOnLeft2')
    m.addConstrs((
        on_left[e1, e2] + on_left[e2, e1] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='OnLeftSanity')

    # Overlap of elements

    h_overlap = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='HorizontalOverlap')
    m.addConstrs((
        h_overlap[e1, e2] == 1 - (on_left[e1, e2] + on_left[e2, e1])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkHorizontalOverlap')

    v_overlap = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='VerticalOverlap')
    m.addConstrs((
        v_overlap[e1, e2] == 1 - (above[e1, e2] + above[e2, e1])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkVerticalOverlap')

    '''
    overlap = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='Overlap')
    m.addConstrs((
        # TODO test alternatives for performance
        overlap[e1, e2] == and_(h_overlap[e1, e2], v_overlap[e1, e2])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOverlap')

    m.addConstr(overlap.sum() == 0)
    '''
    # EXPL: this may be faster way to constrain overlap
    m.addConstrs((
        # Prevent overlap
        h_overlap[e1, e2] + v_overlap[e1, e2] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='PreventOverlap')

    # Starting values

    col_count.Start = max_col_count
    row_count.Start = max_row_count
    gutter_width.Start = m._min_gutter_width
    col_width.Start = min_col_width + m._min_gutter_width
    row_height.Start = min_row_height + m._min_gutter_width

    for element in elements:
        # (elem_width[e] + gutter_width) / col_width == col_span
        col_span[element.id].Start = round(
            (round(element.width / m._base_unit) + m._min_gutter_width) / (min_col_width + m._min_gutter_width))
        row_span[element.id].Start = round(
            (round(element.height / m._base_unit) + m._min_gutter_width) / (min_row_height + m._min_gutter_width))
        # elem_width[element.id].Start = round(element.width / base_unit)
        # elem_height[element.id].Start = round(element.height / base_unit)

    # Expressions
    width_error = available_width - actual_width
    height_error = available_height - actual_height

    # Aim for best coverage/packing, i.e. minimize gaps in the grid
    gap_count = grid_cell_count - elem_cell_count.sum()
    m.addConstr(gap_count >= 0, name='GapCountSanity')


    return col_count, row_count, col_width, row_height, gutter_width, col_start, row_start, width_error, height_error, gap_count