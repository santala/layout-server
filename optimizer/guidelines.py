import math

from functools import reduce
from itertools import permutations, product
from typing import List

from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError


from .classes import Layout, Element, Edge



def solve(layout: Layout, base_unit: int=8, time_out: int=5, number_of_solutions: int=1):

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


    # TODO edge elements
    # Compute the main content area by subtracting the horizontally aligned edge element widths from the convas width
    # and do the same in the vertical direction. Be careful to account for margins as well.
    # TODO: decide how to deal with the margin between content area and edge elements with defined margins
    # I.e. compute the effect of a margin differently depending on whether the element is the inner most of the edge
    # elements

    m._base_unit = base_unit
    m._min_gutter_width = 1
    m._max_gutter_width = 4
    # Layout dimensions in base units
    m._layout_width = int(layout.canvas_width / m._base_unit)
    m._layout_height = int(layout.canvas_height / m._base_unit)

    elem_ids = [element.id for element in layout.element_list]

    # ELEMENT GROUPS

    # We divide elements into groups based on their containers. I.e. elements that are contained within the same element
    # (or are not contained by any element, but rather the canvas) are treated as groups.
    # On the top level group, if there are elements with affinity for certain edges, those elements are placed
    # deterministically according to their priority. The widths, heights, and paddings of the edge elements are,
    # however, optimized to provide the best remaining content area for the content elements.
    # The rest of the elements in the top level group are then aligned in a grid.

    # For lower level groups (i.e. elements that are contained within other elements), elements are simply aligned with
    # each other, without any specific dimensions.



    # TODO groups
    # TODO in each group: edge elements and non-edge elements
    # TODO if edge elements exist, compute widths or heights, as well as content area
    # TODO for other elements, decide type of alignment, and do it

    # TODO optimise
    # TODO compute sizes and positions


    # Element width in base units
    elem_width = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='ElementWidth')
    # Element height in base units
    elem_height = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementHeight')

    # Width of the gutter (i.e. the space between adjacent columns)
    gutter_width = m.addVar(lb=m._min_gutter_width, ub=m._max_gutter_width, vtype=GRB.INTEGER, name='GutterWidth')

    groups = {}

    for element in layout.element_list:
        if element.parent_id not in groups:
            groups[element.parent_id] = []
        groups[element.parent_id].append(element)


    group_ids = groups.keys()

    # TODO implement support for scrolling (i.e., infinite width or height)
    group_full_width = m.addVars(group_ids, vtype=GRB.INTEGER, lb=1, ub=m._layout_width, name='GroupFullWidth')
    group_full_height = m.addVars(group_ids, vtype=GRB.INTEGER, lb=1, ub=m._layout_height, name='GroupFullHeight')
    group_edge_width = m.addVars(group_ids, vtype=GRB.INTEGER, lb=1, ub=m._layout_width, name='GroupEdgeWidth')
    group_edge_height = m.addVars(group_ids, vtype=GRB.INTEGER, lb=1, ub=m._layout_width, name='GroupEdgeHeight')
    group_content_width = m.addVars(group_ids, vtype=GRB.INTEGER, lb=1, ub=m._layout_width, name='GroupContentWidth')
    group_content_height = m.addVars(group_ids, vtype=GRB.INTEGER, lb=1, ub=m._layout_height, name='GroupContentHeight')

    m.addConstrs((
        group_full_width[g] == group_edge_width[g] + group_content_width[g]
        for g in group_ids
    ), name='LinkGroupWidth')
    m.addConstrs((
        group_full_height[g] == group_edge_height[g] + group_content_height[g]
        for g in group_ids
    ), name='LinkGroupHeight')


    obj = LinExpr()

    # Try to retain directional relationships
    relationship_change = LinExpr()


    get_rel_coord = {}

    def get_abs_coord(element_id):
        x, y, w, h = get_rel_coord[element_id](element_id)
        parent_id = layout.element_dict[element_id].parent_id
        if parent_id is not None:
            px, py, *rest = get_abs_coord(parent_id)
            x += px
            y += py
        return x, y, w, h


    for parent_id, elements in groups.items():
        print('P:', parent_id, 'nE', len(elements))

        content_width = group_content_width[parent_id]
        content_height = group_content_height[parent_id]

        enable_grid = layout.enable_grid if parent_id is None else layout.element_dict[parent_id].enable_grid

        edge_elements = [e for e in elements if e.snap_to_edge is not Edge.NONE]

        if len(edge_elements) > 0:
            # TODO compute space required for the edge elements and constrain the content width/height accordingly
            pass

        content_elements = [e for e in elements if e.snap_to_edge is Edge.NONE]
        print('ce',len(content_elements))
        if len(content_elements) > 0:
            # TODO align other elements within the content area
            if enable_grid:
                get_rel_xywh, width_error, height_error, gap_count, on_left, above \
                    = build_grid(m, content_elements, content_width, content_height, elem_width, elem_height, gutter_width)

                # TODO: test which one is better, hard or soft constraint
                m.addConstr(gap_count == 0)
                # obj.add(gap_count, 10)

                # TODO: add penalty if error is an odd number (i.e. prefer symmetry)
                obj.add(width_error, 1)
                # obj.add(height_error, 1)

            else:
                get_rel_xywh, on_left, above \
                    = improve_alignment(m, content_elements, content_width, content_height, elem_width, elem_height)
            # TODO alignment function should take in:
            # TODO gutter/min.margin
            # TODO alignment function should return:
            # TODO objective functions, function to compute x/y/w/h

            for element in content_elements:
                get_rel_coord[element.id] = get_rel_xywh

            for element, other in permutations(content_elements, 2):
                if element.is_above(other):
                    relationship_change.add(1 - above[element.id, other.id])
                if element.is_on_left(other):
                    relationship_change.add(1 - on_left[element.id, other.id])


    print('__', len(get_rel_coord.keys()))
    for k in get_rel_coord.keys():
        print('_', k)
    for element in layout.element_list:
        if element.id not in get_rel_coord.keys():
            print('???', element.id, element.component_name)
        else:
            print('!!!')



    grid_containers = groups[None]

    #get_rel_xywh, width_error, height_error, gap_count, on_left, above \
    #    = build_grid(m, grid_containers, m._layout_width, m._layout_height, elem_width, elem_height, gutter_width)

    '''
    child_coordinates = {}

    for parent_id, elements in groups.items():
        if parent_id is not None:
            aw = elem_width[parent_id]
            ah = elem_height[parent_id]
            child_coordinates[parent_id] = improve_alignment(m, elements, aw, ah, elem_width, elem_height)
    '''
    # OBJECTIVES




    obj.add(relationship_change)

    # Element scaling

    # Decrease of the element width compared to the original in base units
    # Note, the constraints below define this variable to be *at least* the actual difference,
    # i.e. the variable may take a larger value. However, we will define an objective of minimizing the difference,
    # so the solver will minimize it for us. This is faster than defining an absolute value constraint.

    min_width_loss = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=0, name='MinWidthLossFromOriginal')
    m.addConstrs((
        min_width_loss[element.id] >= round(element.width / base_unit) - elem_width[element.id]
        for element in layout.element_list
    ), name='LinkWidthLoss')

    min_height_loss = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=0, name='MinHeightLossFromOriginal')
    m.addConstrs((
        min_height_loss[element.id] >= round(element.height / base_unit) - elem_height[element.id]
        for element in layout.element_list
    ), name='LinkHeightLoss')

    # Minimize size difference

    obj.add(min_width_loss.sum())
    obj.add(min_height_loss.sum())

    # Aim for best fit of grid




    m.setObjective(obj, GRB.MINIMIZE)



    try:

        m.optimize()

        if m.Status in [GRB.Status.OPTIMAL, GRB.Status.INTERRUPTED, GRB.Status.TIME_LIMIT]:
            # ‘X’ is the value of the variable in the current solution

            #print('Resize Error', min_width_diff.sum().getValue(), min_height_diff.sum().getValue())
            print('Resize Loss', min_width_loss.sum().getValue(), min_height_loss.sum().getValue())


            elements = []

            for e in elem_ids:
                x, y, w, h = get_abs_coord(e)
                elements.append({
                    'id': e,
                    'x': x * base_unit,
                    'y': y * base_unit,
                    'width': w * base_unit,
                    'height': h * base_unit,
                })
            '''
            for parent_id, children in groups.items():
                if parent_id is not None:
                    for child in children:
                        x, y, w, h = get_coord[child.id]()

                        elements.append({
                            'id': child.id,
                            'x': x * base_unit,
                            'y': y * base_unit,
                            'width': w * base_unit,
                            'height': h * base_unit,
                        })
            '''

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


def improve_alignment(m: Model, elements: List[Element], available_width, available_height, elem_width, elem_height):

    elem_ids = [element.id for element in elements]

    print('ia', elem_ids)

    elem_x0 = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementX0')
    elem_y0 = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementY0')
    elem_x1 = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementX1')
    elem_y1 = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementY1')

    m.addConstrs((
        elem_x0[e] + elem_width[e] == elem_x1[e]
        for e in elem_ids
    ), name='LinkX1ToWidth')
    m.addConstrs((
        elem_y0[e] + elem_height[e] == elem_y1[e]
        for e in elem_ids
    ), name='ContainElementToAvailableHeight')
    m.addConstrs((
        elem_x1[e] <= available_width
        for e in elem_ids
    ), name='ContainElementToAvailableWidth')
    m.addConstrs((
        elem_y1[e] <= available_height
        for e in elem_ids
    ), name='ContainElementToAvailableHeight')

    on_left, above = get_directional_relationships(m, elem_ids, elem_x0, elem_x1, elem_y0, elem_y1)

    prevent_overlap(m, elem_ids, on_left, above)

    def get_rel_xywh(element_id):
        # Returns the element position (relative to the parent top left corner)

        # Attribute Xn refers to the variable value in the solution selected using SolutionNumber parameter.
        # When SolutionNumber equals 0 (default), Xn refers to the variable value in the best solution.
        # https://www.gurobi.com/documentation/8.1/refman/xn.html#attr:Xn
        print(element_id in elem_ids, element_id)
        print(elem_ids)
        x = elem_x0[element_id].Xn
        y = elem_y0[element_id].Xn
        w = elem_width[element_id].Xn
        h = elem_height[element_id].Xn

        return x, y, w, h

    return get_rel_xywh, on_left, above

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

    # At least one element must start at the first column/row
    min_col_start = m.addVar(vtype=GRB.INTEGER, lb=1, ub=1, name='MinStartColumn')
    m.addConstr(min_col_start == min_(col_start), name='EnsureElementInFirstColumn')
    min_row_start = m.addVar(vtype=GRB.INTEGER, lb=1, ub=1, name='MinStartRow')
    m.addConstr(min_row_start == min_(row_start), name='EnsureElementOnFirstRow')

    # Bind column/row count
    # TODO compare performance
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
        # Using indicator constraint to avoid quadratic constraints
        # TODO compare performance
        (col_span_selected[e, n] == 1) >> (elem_cell_count[e] == n * row_span[e])
        for e, n in product(elem_ids, col_counts)
    ), name='LinkElemCellCountToColumnCount')

    # Directional relationships

    on_left, above = get_directional_relationships(m, elem_ids, col_start, col_end, row_start, row_end)

    prevent_overlap(m, elem_ids, on_left, above)

    # Starting values
    # TODO test starting values
    if False:
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

    def get_rel_xywh(element_id):
        # Returns the element position (relative to the grid top left corner)

        # Attribute Xn refers to the variable value in the solution selected using SolutionNumber parameter.
        # When SolutionNumber equals 0 (default), Xn refers to the variable value in the best solution.
        # https://www.gurobi.com/documentation/8.1/refman/xn.html#attr:Xn
        x = (col_start[element_id].Xn - 1) * col_width.Xn
        y = (row_start[element_id].Xn - 1) * row_height.Xn
        w = elem_width[element_id].Xn
        h = elem_height[element_id].Xn

        return x, y, w, h

    return get_rel_xywh, width_error, height_error, gap_count, on_left, above

def get_directional_relationships(m: Model, elem_ids: List[str], x0: tupledict, x1: tupledict, y0: tupledict, y1: tupledict):

    # TODO: use consistent x1 and y1 coordinates, i.e. choose whether they are inclusive or exclusive

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
        (above[e1, e2] == 1) >> (y1[e1] + 1 <= y0[e2])
        # above[e1, e2] * (row_start[e2] - row_end[e1] - 1) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkAbove1')
    m.addConstrs((
        # TODO compare performance
        (above[e1, e2] == 0) >> (y1[e1] >= y0[e2])
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
        (on_left[e1, e2] == 1) >> (x1[e1] + 1 <= x0[e2])
        # on_left[e1, e2] * (col_start[e2] - col_end[e1] - 1) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOnLeft1')
    m.addConstrs((
        # TODO compare performance
        (on_left[e1, e2] == 0) >> (x1[e1] >= x0[e2])
        # (1 - on_left[e1, e2]) * (col_end[e1] - col_start[e2]) >= 0
        for e1, e2 in permutations(elem_ids, 2)
    ), name='LinkOnLeft2')
    m.addConstrs((
        on_left[e1, e2] + on_left[e2, e1] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='OnLeftSanity')

    return on_left, above


def prevent_overlap(m: Model, elem_ids: List[str], on_left: tupledict, above: tupledict):
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

    m.addConstrs((
        h_overlap[e1, e2] + v_overlap[e1, e2] <= 1
        for e1, e2 in permutations(elem_ids, 2)
    ), name='PreventOverlap')