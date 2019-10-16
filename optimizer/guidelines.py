from itertools import permutations, product

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError


from .classes import Layout, Element


def solve(layout: Layout, base_unit: int=8, time_out: int=30, number_of_solutions: int=1, max_col_count: int=None):

    m = Model('LayoutGuidelines')
    m.Params.MIPFocus = 1
    m.Params.TimeLimit = time_out

    print('Time out:', time_out)

    m.Params.PoolSearchMode = 2
    m.Params.PoolSolutions = 1
    # model.Params.MIPGap = 0.01

    m.Params.MIPGapAbs = 0.97
    m.Params.OutputFlag = 1

    # For each group:
        # Align edge elements
        # Constrain rest to the remaining area (minus padding)
            # E.g. y0 limit must be max value of top edge elements y1
        # Define a grid (if reasonable)
    elem_count = len(layout.elements)
    elem_ids = [e.id for e in layout.elements]

    # Parameters

    layout_width = int(layout.canvas_width / base_unit)
    layout_height = int(layout.canvas_height / base_unit)

    min_col_width = 1 # Not including gutter

    min_gutter_width = 1
    max_gutter_width = 4

    if max_col_count is None:
        # Maximum number of columns that can fit on the layout
        max_col_count = int((layout_width + min_gutter_width) / (min_col_width + min_gutter_width))

    col_counts = range(1, max_col_count + 1)

    # Number of columns
    col_count = m.addVar(lb=1, ub=max_col_count, vtype=GRB.INTEGER, name='ColumnCount')
    # TODO: col_count == max(col_end)

    col_count_selected = m.addVars(col_counts, vtype=GRB.BINARY, name='SelectedColumnCount')
    m.addConstr(col_count_selected.sum() == 1, name='SelectOneColumnCount') # One option must always be selected
    m.addConstrs((
        # (col_count_selected[n] == 1) >> (col_count == n)
        # TODO compare performance:
        col_count_selected[n] * n == col_count_selected[n] * col_count
        for n in col_counts
    ), name='LinkColumnCountToSelection')

    # Maximum width for the grid to take (not including left/right margins)
    available_width = m.addVar(vtype=GRB.INTEGER, name='AvailableWidth')
    m.addConstr(available_width == layout_width)
    # Maximum height for the grid to take (not including left/right margins)
    available_height = m.addVar(vtype=GRB.INTEGER, name='AvailableHeight')
    m.addConstr(available_height == layout_height)

    # Width of the gutter (i.e. the space between adjacent columns)
    gutter_width = m.addVar(lb=min_gutter_width, ub=max_gutter_width, vtype=GRB.INTEGER, name='GutterWidth')

    # Width of a single column (including gutter width)
    col_width = m.addVar(lb=min_col_width+min_gutter_width, vtype=GRB.INTEGER, name='ColumnWidth')
    m.addConstr(col_width <= available_width, name='FitColumnIntoAvailableSpace')
    m.addConstr(col_width - gutter_width >= min_col_width, name='AdjustMinColumnWidthAccordingToGutterWidth')

    # Actual width of the grid (not including left/right margins)
    actual_width = m.addVar(vtype=GRB.INTEGER, name='ActualWidth')
    m.addConstrs((
        (col_count_selected[n] == 1) >> (actual_width == n * col_width - gutter_width)
        for n in col_counts
    ), name='LinkActualWidthToColumnCount')
    m.addConstr(actual_width <= available_width, name='FitGridIntoAvailableSpace')

    # Actual height of the grid (not including top/bottom margins)
    actual_height = m.addVar(vtype=GRB.INTEGER, name='ActualHeight')
    m.addConstr(actual_height <= available_width, name='FitGridIntoAvailableSpace')


    # Number of rows
    row_count = m.addVar(lb=1, ub=elem_count, vtype=GRB.INTEGER, name='RowCount')

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
        (col_span_selected[e, n] == 1) >> (col_span[e] == n)
        # TODO compare performance: col_span_selected[n] * n == col_span_selected[n] * col_span[e]
        for e, n in product(elem_ids, col_counts)
    ), name='LinkColumnCountToSelection')
    # Element width in base units
    elem_width = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='ElementWidth')
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


    # Element y coordinates in base units
    elem_y0 = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=0, name='ElementY0')
    elem_y1 = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=0, name='ElementY1')
    m.addConstrs((
        elem_y0[e] <= elem_y1[e] + 1
        for e in elem_ids
    ), name='Y0Y1OrderAndMinHeight')
    m.addConstrs((
        elem_y1[e] <= available_height
        for e in elem_ids
    ), name='Y1Sanity')

    # Element height in base units
    elem_height = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='ElementHeight')
    m.addConstrs((
        elem_height[e] == elem_y1[e] - elem_y0[e]
        for e in elem_ids
    ), name='LinkElementHeight')

    elem_height_diff = m.addVars(permutations(elem_ids, 2), vtype=GRB.INTEGER, lb=-GRB.INFINITY, name='HeightDifference')
    m.addConstrs((
        elem_height_diff[e1, e2] == elem_height[e1] - elem_height[e2]
        for e1, e2 in permutations(elem_ids, 2)
    ))
    row_span_diff = m.addVars(permutations(elem_ids, 2), vtype=GRB.INTEGER, lb=-GRB.INFINITY, name='RowSpanDifference')
    m.addConstrs((
        row_span_diff[e1, e2] == row_span[e1] - row_span[e2]
        for e1, e2 in permutations(elem_ids, 2)
    ))
    # Binary: whether e1 is taller than e2
    is_taller = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='IsTaller')
    m.addConstrs((
        (is_taller[e1, e2] == 1) >> (row_span_diff[e1, e2] >= 0)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkIsTaller1')
    m.addConstrs((
        (is_taller[e1, e2] == 0) >> (row_span_diff[e1, e2] <= -1)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkIsTaller2')
    m.addConstrs((
        (is_taller[e1, e2] == 1) >> (elem_height_diff[e1, e2] >= 0)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkIsTaller3')
    m.addConstrs((
        (is_taller[e1, e2] == 0) >> (elem_height_diff[e1, e2] <= -1)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkIsTaller4')


    # TODO: col_span–elem_height relationship
    # TODO: col_start–elem_y relationship
    # TODO: adjacent rows > gutter_width apart

    # Integer: The number of rows between element start rows
    row_start_diff = m.addVars(permutations(elem_ids, 2), vtype=GRB.INTEGER, lb=-GRB.INFINITY, name='StartRowDifference')
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
    # Integer: The number of base units between element Y0 coordinates
    elem_y0_diff = m.addVars(permutations(elem_ids, 2), vtype=GRB.INTEGER, lb=-GRB.INFINITY, name='YDifference')
    m.addConstrs((
        elem_y0_diff[e1, e2] == elem_y0[e1] - elem_y0[e2]
        for e1, e2 in permutations(elem_ids, 2)
    ))
    # Integer: The number of base units between element Y1 coordinates
    elem_y1_diff = m.addVars(permutations(elem_ids, 2), vtype=GRB.INTEGER, lb=-GRB.INFINITY, name='YDifference')
    m.addConstrs((
        elem_y1_diff[e1, e2] == elem_y1[e1] - elem_y1[e2]
        for e1, e2 in permutations(elem_ids, 2)
    ))

    # Binary: whether e1 starts on a any row before e2
    start_row_before = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='StartRowLessThan')
    m.addConstrs((
        (start_row_before[e1, e2] == 1) >> (row_start_diff[e1, e2] <= -1)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkStartRowLessThan1')
    m.addConstrs((
        (start_row_before[e1, e2] == 0) >> (row_start_diff[e1, e2] >= 0)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkStartRowLessThan2')

    # Binary: whether e1 ends on a any row before e2
    end_row_before = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='StartRowLessThan')
    m.addConstrs((
        (end_row_before[e1, e2] == 1) >> (row_end_diff[e1, e2] <= -1)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkEndRowOrder1')
    m.addConstrs((
        (end_row_before[e1, e2] == 0) >> (row_end_diff[e1, e2] >= 0)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkEndRowOrder2')

    # Element Y coordinates must follow the order of the start/end rows
    m.addConstrs((
        (start_row_before[e1, e2] == 1) >> (elem_y0_diff[e1, e2] <= -1)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkYOrder1')
    m.addConstrs((
        (start_row_before[e1, e2] == 0) >> (elem_y0_diff[e1, e2] >= 0)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkYOrder2')
    m.addConstrs((
        (end_row_before[e1, e2] == 1) >> (elem_y1_diff[e1, e2] <= -1)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkYPlusHeightOrder1')
    m.addConstrs((
        (end_row_before[e1, e2] == 0) >> (elem_y1_diff[e1, e2] >= 0)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkYPlusHeightOrder2')


    '''
    row_end_diff = m.addVars(permutations(elem_ids, 2), vtype=GRB.INTEGER, lb=-GRB.INFINITY, name='EndRowDifference')
    m.addConstrs((
        row_end_diff[e1, e2] == row_end[e1] - row_end[e2]
        for e1, e2 in permutations(elem_ids, 2)
    ))

    start_row_before = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='StartRowLessThan')
    m.addConstrs((
        (start_row_before[e1, e2] == 1) >> (row_start_diff[e1, e2] <= -1)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkStartRowLessThan1')
    m.addConstrs((
        (start_row_before[e1, e2] == 0) >> (row_start_diff[e1, e2] >= 0)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkStartRowLessThan2')

    end_row_before = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='EndRowLessThan')
    m.addConstrs((
        (end_row_before[e1, e2] == 1) >> (row_end_diff[e1, e2] <= -1)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkEndRowLessThan1')
    m.addConstrs((
        (end_row_before[e1, e2] == 0) >> (row_end_diff[e1, e2] >= 0)
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkEndRowLessThan2')

    y0_group = m.addVars(elem_ids, lb=1, ub=elem_count, vtype=GRB.INTEGER, name='StartRowGroup')
    y1_group = m.addVars(elem_ids, lb=1, ub=elem_count, vtype=GRB.INTEGER, name='EndRowGroup')

    m.addConstrs((
        (start_row_before[i1, i2] == 1) >> (y0_group[i1] <= y0_group[i2] - 1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY0Group1')
    m.addConstrs((
        (start_row_before[i1, i2] == 0) >> (y0_group[i1] >= y0_group[i2])
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY0Group2')
    m.addConstrs((
        (end_row_before[i1, i2] == 1) >> (y1_group[i1] <= y1_group[i2] - 1)
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY1Group1')
    m.addConstrs((
        (end_row_before[i1, i2] == 0) >> (y1_group[i1] >= y1_group[i2])
        for i1, i2 in permutations(elem_ids, 2)
    ), name='LinkY1Group2')
    '''


    # At least one element must start at the first column/row
    min_col_start = m.addVar(vtype=GRB.INTEGER, lb=1, ub=1, name='MinStartColumn')
    m.addConstr(min_col_start == min_(col_start), name='EnsureElementInFirstColumn')
    min_row_start = m.addVar(vtype=GRB.INTEGER, lb=1, ub=1, name='MinStartRow')
    m.addConstr(min_row_start == min_(row_start), name='EnsureElementOnFirstRow')
    # Bind column/row count
    m.addConstr(col_count == max_(col_end), name='LinkColumnCountToElementCoordinates')
    m.addConstr(row_count == max_(row_end), name='LinkRowCountToElementCoordinates')

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
        (col_span_selected[e, n] == 1) >> (elem_cell_count[e] == n * row_span[e])
        for e, n in product(elem_ids, col_counts)
    ), name='LinkElemCellCountToColumnCount')


    # Directional relationships

    above = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='Above')
    m.addConstrs((
        (above[e1, e2] == 1) >> (row_end[e1] + 1 <= row_start[e2])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove1')
    m.addConstrs((
        (above[e1, e2] == 0) >> (row_end[e1] >= row_start[e2])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove2')
    m.addConstrs((
        (above[e1, e2] == 1) >> (elem_y1[e1] + 1 <= elem_y0[e2])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove1')
    m.addConstrs((
        (above[e1, e2] == 0) >> (elem_y1[e1] >= elem_y0[e2])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove2')
    m.addConstrs((
        above[e1, e2] + above[e2, e1] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='AboveSanity') # TODO: check if sanity checks are necessary

    on_left = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='OnLeft')
    m.addConstrs((
        (on_left[e1, e2] == 1) >> (col_end[e1] + 1 <= col_start[e2])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOnLeft1')
    m.addConstrs((
        (on_left[e1, e2] == 0) >> (col_end[e1] >= col_start[e2])
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

    overlap = m.addVars(permutations(elem_ids, 2), vtype=GRB.BINARY, name='Overlap')
    m.addConstrs((
        overlap[e1, e2] == and_(h_overlap[e1, e2], v_overlap[e1, e2])
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOverlap')

    m.addConstr(overlap.sum() == 0)

    # OBJECTIVES

    obj = LinExpr()


    # Element scaling

    # Difference of the element width to the original in pixels (not in base units)
    # Note, the constraints below define this variable to be *at least* the actual difference,
    # i.e. the variable may take a larger value. However, we will define an objective of minimizing
    # the difference, so it won’t be a problem. This is faster than defining an absolute value constraint.

    min_width_diff_px = m.addVars(elem_ids, vtype=GRB.INTEGER, name='MinWidthDifferenceToOriginal')
    m.addConstrs((
        min_width_diff_px[element.id] >= elem_width[element.id] - round(element.width/base_unit)
        for element in layout.elements
    ), name='LinkWidthDiff1')
    m.addConstrs((
        min_width_diff_px[element.id] >= round(element.width/base_unit) - elem_width[element.id]
        for element in layout.elements
    ), name='LinkWidthDiff2')


    min_height_diff_px = m.addVars(elem_ids, vtype=GRB.INTEGER, name='MinHeightDifferenceToOriginal')
    m.addConstrs((
        min_height_diff_px[element.id] >= elem_height[element.id] - round(element.height/base_unit)
        for element in layout.elements
    ), name='LinkHeightDiff1')
    m.addConstrs((
        min_height_diff_px[element.id] >= round(element.height/base_unit) - elem_height[element.id]
        for element in layout.elements
    ), name='LinkHeightDiff2')

    # Minimize size difference

    obj.add(min_width_diff_px.sum(), .1)
    obj.add(min_height_diff_px.sum(), .1)

    # Aim for best fit of grid
    width_error = available_width - actual_width
    # TODO: add penalty if error is an odd number (i.e. prefer symmetry)

    obj.add(width_error)

    # Aim for best coverage/packing, i.e. minimize gaps in the grid
    gap_count = grid_cell_count - elem_cell_count.sum()
    m.addConstr(gap_count >= 0, name='GapCountSanity')

    obj.add(gap_count)



    m.setObjective(obj, GRB.MINIMIZE)



    try:

        m.optimize()

        if m.Status in [GRB.Status.OPTIMAL, GRB.Status.INTERRUPTED, GRB.Status.TIME_LIMIT]:
            # ‘X’ is the value of the variable in the current solution

            print('Width Error', width_error.getValue())
            print('Gap Count', gap_count.getValue())
            print('Area', grid_cell_count.X, elem_cell_count.sum().getValue())
            print('Column Count', col_count.X)
            print('Column Width', col_width.X)
            print('Row Count', row_count.X)


            elements = [
                {
                    'id': e,
                    'x': (col_start[e].X - 1) * col_width.X * base_unit,
                    'y': elem_y0[e].X * base_unit,
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

# GRID





# w = column_span * column_width - gutter_width

# w >= column_width - gutter_width



# VARIABLES

# General:
    # margin_left
    # margin_right
    # cell_count; col_count_equals[n] >> cell_count == n * row_count

# Per element:
    # col_start
    # col_end >= col_start
    # col_span == col_end - col_start + 1
    # row_start
    # row_end >= row_start
    # row_span == row_end - row_start + 1
    # row_span_equals[e,n] * row_span[e] == row_span_equals[e,n] * n
    # width: row_span_equals[e,n] >> width == n * column_width
    # height: row_span[e1] == row_span[e2] >> height[e1] == height[e2]
    # cell_coverage: row_span_equals[e,n] >> cell_coverage == n * row_span

# CONSTRAINTS

# All columns must fit within the available space
# layout_width + gutter_width >= col_width * col_count

# At least one element must start at the first column/row
# min(col_start) == 1
# min(row_start) == 1

# TODO One element must be in top left corner
# TODO If elements are on adjacent rows, distance should be gutter_width

# OBJECTIVES

# Aim for best fit of grid
# minimize( layout_width + gutter_width - col_width * col_count )

# Minimize gaps in the grid (or maximize coverage)
# minimize( cell_count - sum( cell_coverage ) )

# TODO minimize resizing of elements
# TODO width difference, height difference
# TODO (aim to fill available height)