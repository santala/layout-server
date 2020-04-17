import math

from collections import namedtuple
from enum import Enum
from functools import reduce
from itertools import permutations, product
from typing import List

from gurobi import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError


from .classes import Layout, Element, Edge

from .alignment import improve_alignment
from .grid2 import equal_width_columns


BBox = namedtuple('BBox', 'x y w h')
Padding = namedtuple('Padding', 'top right bottom left')


def solve(layout: Layout, base_unit: int=8, time_out: int=30, number_of_solutions: int=5):

    m = Model('LayoutGuidelines')

    m.Params.MIPFocus = 1
    m.Params.TimeLimit = time_out

    obj_i = 1

    print('Time out:', time_out)

    # https://www.gurobi.com/documentation/8.1/refman/poolsearchmode.html#parameter:PoolSearchMode
    m.Params.PoolSearchMode = 1 # Use 2 to find high quality solutions
    m.Params.PoolSolutions = number_of_solutions # Number of solutions to be saved
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

    m._layout = layout

    m._base_unit = base_unit
    m._min_gutter_width = 1

    if layout.canvas_width < 1280:      # widescreen
        m._max_gutter_width = 4
    elif layout.canvas_width < 1280:    # desktop 24px
        m._max_gutter_width = 3
    elif layout.canvas_width < 599:     # tablet 16px
        m._max_gutter_width = 2
    else:                               # mobile 8px
        m._max_gutter_width = 1

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


    # Element width in base units
    elem_width = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='ElementWidth')
    # Element height in base units
    elem_height = m.addVars(elem_ids, vtype=GRB.INTEGER, name='ElementHeight')

    # TODO: theoretically, this might lead to infeasible model if there are too many fixed size elements and too little space
    for element in layout.element_list:
        if element.fixed_width is not None:
            m.addConstr(elem_width[element.id] == element.fixed_width)
        if element.fixed_height is not None:
            m.addConstr(elem_height[element.id] == element.fixed_height)

    # Width of the gutter (i.e. the space between adjacent columns)
    gutter_width = m.addVar(lb=m._min_gutter_width, ub=m._max_gutter_width, vtype=GRB.INTEGER, name='GutterWidth')

    groups = {}

    for element in layout.element_list:
        if element.get_parent_id() not in groups:
            groups[element.get_parent_id()] = []
        groups[element.get_parent_id()].append(element)



    group_ids = groups.keys()

    # TODO implement support for scrolling (i.e., infinite width or height)
    group_full_width = m.addVars(group_ids, vtype=GRB.INTEGER, lb=0, ub=m._layout_width, name='GroupFullWidth')
    group_full_height = m.addVars(group_ids, vtype=GRB.INTEGER, lb=0, ub=m._layout_height, name='GroupFullHeight')
    group_edge_width = m.addVars(group_ids, vtype=GRB.INTEGER, lb=0, ub=m._layout_width, name='GroupEdgeWidth')
    group_edge_height = m.addVars(group_ids, vtype=GRB.INTEGER, lb=0, ub=m._layout_height, name='GroupEdgeHeight')
    group_content_width = m.addVars(group_ids, vtype=GRB.INTEGER, lb=0, ub=m._layout_width, name='GroupContentWidth')
    group_content_height = m.addVars(group_ids, vtype=GRB.INTEGER, lb=0, ub=m._layout_height, name='GroupContentHeight')

    group_padding_top = m.addVars(group_ids, vtype=GRB.INTEGER, name='GroupPaddingTop')
    group_padding_right = m.addVars(group_ids, vtype=GRB.INTEGER, name='GroupPaddingRight')
    group_padding_bottom = m.addVars(group_ids, vtype=GRB.INTEGER, name='GroupPaddingBottom')
    group_padding_left = m.addVars(group_ids, vtype=GRB.INTEGER, name='GroupPaddingLeft')

    m.addConstrs((
        group_full_width[g] == group_edge_width[g] + group_content_width[g] + group_padding_left[g] + group_padding_right[g]
        for g in group_ids
    ), name='LinkGroupEdgeAndContentWidth')
    m.addConstrs((
        group_full_height[g] == group_edge_height[g] + group_content_height[g] + group_padding_top[g] + group_padding_bottom[g]
        for g in group_ids
    ), name='LinkGroupEdgeAndContentHeight')
    m.addConstrs((
        group_full_width[g] == elem_width[g]
        for g in group_ids if g is not layout.id
    ), name='LinkGroupFullWidth')
    m.addConstrs((
        group_full_height[g] == elem_height[g]
        for g in group_ids if g is not layout.id
    ), name='LinkGroupFullHeight')
    m.addConstr(group_full_width[layout.id] == m._layout_width, name='LinkLayoutFullWidth')
    m.addConstr(group_full_height[layout.id] == m._layout_height, name='LinkLayoutFullHeight')

    get_rel_coord = {}

    def get_content_offset(element_id):
        if element_id in group_ids:
            return group_padding_left[element_id].Xn, group_padding_top[element_id].Xn
        else:
            return 0, 0

    def get_abs_coord(element_id):
        x, y, w, h = get_rel_coord[element_id](element_id)
        parent_id = layout.element_dict[element_id].get_parent_id()
        if parent_id is not layout.id:
            px, py, *rest = get_abs_coord(parent_id)
            x += px
            y += py
            px, py = get_content_offset(parent_id)
            x += px
            y += py
        return x, y, w, h


    # Loop through all groups and return
    # * grid fitness to available space
    # * number of gaps in grids (for grid alignment)
    # * group count (for basic alignment)
    # * element relationship expression

    '''
    for group_id, elements in groups.items():
        # divide elements into two categories: edge elements and content elements
        # align edge elements
        # compute content area bbox
        # align content elements
        # (maybe) add objective for group alignment, priority from hierarchy depth
    '''

    layout_quality_obj_index = 10

    for container_id, elements in groups.items():
        layout_quality_obj_index += 1

        layout_quality = LinExpr()
        width_error_sum = LinExpr()
        height_error_sum = LinExpr()

        if container_id in layout.element_dict:
            container = layout.element_dict[container_id]
            group_priority = layout.depth - layout.element_dict[container_id].get_ancestor_count()
        else:
            container = None
            group_priority = layout.depth

        # TODO: test which one is better, hard or soft constraint
        m.setObjectiveN(layout_quality, index=layout_quality_obj_index, priority=group_priority, weight=1)

        # Optimize for grid fitness within available space
        m.setObjectiveN(width_error_sum, index=7, priority=group_priority, weight=1, name='MinimizeWidthError')
        m.setObjectiveN(height_error_sum, index=8, priority=group_priority, weight=.5, name='MinimizeWidthError')

        if container is not None and container.element_type == 'component' and 'Card' in container.component_name:
            m.addConstr(group_padding_top[container_id] == 10)
            m.addConstr(group_padding_bottom[container_id] == 4)
            m.addConstr(group_padding_left[container_id] == 4)
            m.addConstr(group_padding_right[container_id] == 4)
        else:
            m.addConstr(group_padding_top[container_id] == gutter_width)
            m.addConstr(group_padding_bottom[container_id] == gutter_width)
            m.addConstr(group_padding_left[container_id] == gutter_width)
            m.addConstr(group_padding_right[container_id] == gutter_width)

        edge_elements = []
        content_elements = []

        for e in elements:
            if e.snap_to_edge is Edge.NONE:
                content_elements.append(e)
            else:
                edge_elements.append(e)

        if len(edge_elements) > 0:
            # TODO compute space required for the edge elements and constrain the content width/height accordingly

            get_rel_xywh, edge_top_height, edge_right_width, edge_bottom_height, edge_left_width \
                = align_edge_elements(m, edge_elements, group_full_width[container_id], group_full_height[container_id], elem_width, elem_height)

            for element in edge_elements:
                get_rel_coord[element.id] = get_rel_xywh
        else:
            edge_top_height = LinExpr(0)
            edge_right_width = LinExpr(0)
            edge_bottom_height = LinExpr(0)
            edge_left_width = LinExpr(0)

        m.addConstr(group_edge_width[container_id] == edge_left_width + edge_right_width)
        m.addConstr(group_edge_height[container_id] == edge_top_height + edge_bottom_height)

        if len(content_elements) > 0:
            enable_grid = layout.enable_grid if container_id is layout.id else layout.element_dict[container_id].enable_grid

            # TODO align other elements within the content area
            if enable_grid:

                offset_x = edge_left_width + group_padding_left[container_id]
                offset_y = edge_top_height + group_padding_top[container_id]
                get_rel_xywh, width_error, height_error, layout_quality\
                    = equal_width_columns(m, content_elements, group_content_width[container_id], group_content_height[container_id], elem_width, elem_height, gutter_width, offset_x, offset_y)

                # TODO: add penalty if error is an odd number (i.e. prefer symmetry)
                width_error_sum.add(width_error)
                height_error_sum.add(height_error)

            else:
                get_rel_xywh, layout_quality \
                    = improve_alignment(m, content_elements, group_content_width[container_id], group_content_height[container_id], elem_width, elem_height)

            # TODO alignment function should take in:
            # TODO gutter/min.margin
            # TODO alignment function should return:
            # TODO objective functions, function to compute x/y/w/h

            for element in content_elements:
                get_rel_coord[element.id] = get_rel_xywh

    # Element scaling

    # Decrease of the element width compared to the original in base units
    # Note, the constraints below define this variable to be *at least* the actual difference,
    # i.e. the variable may take a larger value. However, we will define an objective of minimizing the difference,
    # so the solver will minimize it for us. This is faster than defining an absolute value constraint.

    width_loss, height_loss = get_size_loss(m, layout.element_list, elem_width, elem_height)

    # Minimize total downscaling
    m.setObjectiveN(width_loss, index=3, priority=layout.depth+1, weight=1, name='MinimizeElementWidthLoss')
    m.setObjectiveN(height_loss, index=4, priority=layout.depth+1, weight=1, name='MinimizeElementWidthLoss')

    try:
        m.Params.ModelSense = GRB.MINIMIZE
        m.Params.Presolve = -1 # -1=auto, 0=off, 1=conservative, 2=aggressive

        m.optimize()

        if m.Status in [GRB.Status.OPTIMAL, GRB.Status.INTERRUPTED, GRB.Status.TIME_LIMIT]:

            layouts = []

            for s in range(m.SolCount):
                m.Params.SolutionNumber = s

                elements = []

                for e in elem_ids:
                    x, y, w, h = get_abs_coord(e)
                    elements.append({
                        'id': e,
                        'x': int(round(x)) * base_unit,
                        'y': int(round(y)) * base_unit,
                        'width': int(round(w)) * base_unit,
                        'height': int(round(h)) * base_unit,
                    })

                layouts.append({
                    'solutionNumber': s,
                    'canvasWidth': layout.canvas_width,
                    'canvasHeight': layout.canvas_height,
                    'elements': elements
                })

            try:
                print('Width loss', width_loss.getValue())
                print('Height loss', width_loss.getValue())
            except e:
                print(e)

            return {
                'status': 0,
                'layouts': layouts
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


def get_size_loss(m: Model, elements: List[Element], width: tupledict, height: tupledict):
    elem_ids = [e.id for e in elements]
    max_width = max([e.width for e in elements])
    max_height = max([e.height for e in elements])

    width_loss = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=0, name='MinWidthLossFromOriginal')
    m.addConstrs((
        width_loss[e.id] >= (math.ceil(e.width / m._base_unit) - width[e.id])
        for e in elements
    ), name='LinkWidthLoss')

    height_loss = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=0, name='MinHeightLossFromOriginal')
    m.addConstrs((
        height_loss[e.id] >= (math.ceil(e.height / m._base_unit) - height[e.id])
        for e in elements
    ), name='LinkHeightLoss')

    weighted_width_loss = LinExpr(0)
    weighted_height_loss = LinExpr(0)
    for e in elements:
        # The smaller the element, the larger the weight
        # Weight for the element with the largest width/height equals 1
        weighted_width_loss.add(width_loss[e.id], max_width / e.width)
        weighted_height_loss.add(height_loss[e.id], max_height / e.height)

    return weighted_width_loss, weighted_height_loss



def align_edge_elements(m: Model, elements: List[Element], available_width, available_height, elem_width, elem_height):
    #edge_width = LinExpr(0)
    #edge_height = LinExpr(0)



    edge_top_height = LinExpr()
    edge_right_width = LinExpr()
    edge_bottom_height = LinExpr()
    edge_left_width = LinExpr()

    for element in elements:
        higher_priority_elements = [other for other in elements if other.snap_priority < element.snap_priority]
        if element.snap_to_edge in [Edge.TOP, Edge.BOTTOM]:
            if element.snap_to_edge == Edge.TOP:
                edge_top_height.add(elem_height[element.id])
            else:
                edge_bottom_height.add(elem_height[element.id])


            # Edge elements will span the whole edge, except when there are higher priority elements
            # on adjacent edges
            higher_priority_elements_on_adjacent_edges = [
                other for other in higher_priority_elements
                if other.snap_to_edge in [Edge.LEFT, Edge.RIGHT]
            ]
            higher_priority_elements_on_adjadent_edges_width = LinExpr(0)

            # TODO support for margins

            for other in higher_priority_elements_on_adjacent_edges:
                higher_priority_elements_on_adjadent_edges_width.add(elem_width[other.id])

            m.addConstr(elem_width[element.id] == available_width - higher_priority_elements_on_adjadent_edges_width)

        else:  # Left or right
            if element.snap_to_edge == Edge.LEFT:
                edge_left_width.add(elem_width[element.id])
            else:
                edge_right_width.add(elem_width[element.id])

            higher_priority_elements_on_adjacent_edges = [
                other for other in higher_priority_elements
                if other.snap_to_edge in [Edge.TOP, Edge.BOTTOM]
            ]

            higher_priority_elements_on_adjadent_edges_height = LinExpr(0)

            # TODO support for margins

            for other in higher_priority_elements_on_adjacent_edges:
                higher_priority_elements_on_adjadent_edges_height.add(elem_height[other.id])

            m.addConstr(elem_height[element.id] == available_height - higher_priority_elements_on_adjadent_edges_height)

    def get_rel_xywh(elem_id):
        element = m._layout.element_dict[elem_id]
        x_expr = LinExpr()
        y_expr = LinExpr()


        if element.snap_to_edge in [Edge.TOP, Edge.BOTTOM]:
            higher_priority_elements = [
                other for other in elements
                if other.snap_priority < element.snap_priority
            ]

            for other in higher_priority_elements:
                if other.snap_to_edge == Edge.LEFT:
                    x_expr.add(elem_width[other.id])
                if element.snap_to_edge == other.snap_to_edge:
                    y_expr.add(elem_height[other.id])

        else:
            higher_priority_elements = [
                other for other in elements
                if other.snap_priority < element.snap_priority
            ]
            for other in higher_priority_elements:
                if other.snap_to_edge == Edge.TOP:
                    y_expr.add(elem_height[other.id])
                if element.snap_to_edge == other.snap_to_edge:
                    x_expr.add(elem_width[other.id])

        # TODO: this is untested
        if element.snap_to_edge == Edge.RIGHT:
            x_expr = available_width - x_expr - elem_width[elem_id]
        if element.snap_to_edge == Edge.BOTTOM:
            y_expr = available_height - y_expr - elem_height[elem_id]

        x = x_expr.getValue()
        y = y_expr.getValue()
        w = elem_width[elem_id].Xn
        h = elem_height[elem_id].Xn
        return x, y, w, h

    return get_rel_xywh, edge_top_height, edge_right_width, edge_bottom_height, edge_left_width


def build_grid(m: Model, elements: List[Element], available_width, available_height, elem_width, elem_height, gutter_width, offset_x, offset_y):
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

    directional_relationships = get_directional_relationships(m, elem_ids, col_start, col_end, row_start, row_end)

    prevent_overlap(m, elem_ids, directional_relationships)

    # Prevent relationship change
    for element, other in permutations(elements, 2):
        if element.is_above(other):
            m.addConstr(directional_relationships.above[element.id, other.id] == 1)
        if element.is_on_left(other):
            m.addConstr(directional_relationships.on_left[element.id, other.id] == 1)

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
        x = offset_x.getValue() + (col_start[element_id].Xn - 1) * col_width.Xn
        y = offset_y.getValue() + (row_start[element_id].Xn - 1) * row_height.Xn
        w = elem_width[element_id].Xn
        h = elem_height[element_id].Xn

        return x, y, w, h

    return get_rel_xywh, width_error, height_error, gap_count



def get_inner_bbox(outer: BBox, padding: Padding):
    return BBox(
        x=outer.x + padding.left,
        y=outer.y + padding.top,
        w=outer.w - padding.left - padding.right,
        h=outer.h - padding.top - padding.bottom
    )