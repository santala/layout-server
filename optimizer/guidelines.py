from gurobipy import GRB, GenExpr, LinExpr, Model, tupledict, abs_, and_, max_, min_, QuadExpr, GurobiError


def optimize():

    # For each group:
        # Align edge elements
        # Constrain rest to the remaining area (minus padding)
            # E.g. y0 limit must be max value of top edge elements y1
        # Define a grid (if reasonable)

    elem_ids = ['a', 'b', 'c']


    m = Model('GLayoutQuality')

    col_count = m.addVar(name='ColumnCount')
    col_width = m.addVar(name='ColumnWidth')
    gutter_width = m.addVar(name='GutterWidth')
    grid_width = m.addVar(name='GridWidth')
    row_count = m.addVar(name='RowCount')


    col_start = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='StartColumn')
    col_end = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='EndColumn')
    m.addConstr((
        col_end[e] >= col_start[e]
        for e in elem_ids
    ), name='ColumnStartEndOrder')

    row_start = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='RowColumn')
    row_end = m.addVars(elem_ids, vtype=GRB.INTEGER, lb=1, name='RowColumn')
    m.addConstr((
        row_end[e] >= row_start[e]
        for e in elem_ids
    ), name='RowStartEndOrder')

    # At least one element must start at the first column/row
    m.addConstr(min_(col_start) == 1)
    m.addConstr(min_(row_start) == 1)
    # Bind column/row count
    m.addConstr(col_count == max_(col_end))
    m.addConstr(row_count == max_(row_end))

    # All columns must fit within the available space
    m.addConstr(grid_width + gutter_width >= col_width * col_count)

    # OBJECTIVES

    # Aim for best fit of grid
    # minimize( layout_width + gutter_width - column_width * column_count )

    # Minimize gaps in the grid (or maximize coverage)
    # minimize( column_count * row_count - sum( col_span * row_span ) )

    pass


# GRID

# column_count
# column_width
# gutter_width
# layout_width

# layout_width + gutter_width = column_width * column_count

# x = column_index * column_width
# w = column_span * column_width - gutter_width

# w >= column_width - gutter_width



# VARIABLES

# General:
    # column_count == max(col_end)
    # column_width
    # gutter_width
    # layout_width
    # margin_left
    # margin_right
    # row_count == max(row_end)

# Per element:
    # col_start
    # col_end >= col_start
    # col_span == col_end - col_start + 1
    # row_start
    # row_end >= row_start
    # row_span == row_end - row_start + 1

# CONSTRAINTS

# All columns must fit within the available space
# layout_width + gutter_width >= column_width * column_count

# At least one element must start at the first column/row
# min(col_start) == 1
# min(row_start) == 1

# TODO One element must be in top left corner

# OBJECTIVES

# Aim for best fit of grid
# minimize( layout_width + gutter_width - column_width * column_count )

# Minimize gaps in the grid (or maximize coverage)
# minimize( column_count * row_count - sum( col_span * row_span ) )