

def optimize():

    # For each group:
        # Align edge elements
        # Constrain rest to the remaining area (minus padding)
            # E.g. y0 limit must be max value of top edge elements y1
        # Define a grid (if reasonable)
            # Check divisors

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
    # column_count
    # column_width
    # gutter_width
    # layout_width
    # margin_left
    # margin_right

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

# At least one element must end at the last column
# max(col_end) == column_count

# OBJECTIVES

# Aim for best fit of grid
# minimize( layout_width + gutter_width - column_width * column_count )