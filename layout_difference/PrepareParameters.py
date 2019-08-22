from . import Layout


def resolve_individual_layout_parameters(layout: Layout):
    for element in layout.elements:
        element.area = element.width * element.height
        layout.x_sum = layout.x_sum + element.x
        layout.y_sum = layout.y_sum + element.y
        layout.w_sum = layout.w_sum + element.width
        layout.h_sum = layout.h_sum + element.height
        layout.area_sum = layout.area_sum + element.area


def build_layout_parameters(layout_a: Layout, layout_b: Layout) -> list:
    penalty_assignment = []

    resolve_individual_layout_parameters(layout_a)
    resolve_individual_layout_parameters(layout_b)

    # EXPL: Penalty of being skipped is the relative size of the element
    for a_elem in layout_a.elements:
        a_elem.PenaltyIfSkipped = a_elem.area / layout_a.area_sum

    for b_elem in layout_b.elements:
        b_elem.PenaltyIfSkipped = b_elem.area / layout_b.area_sum

    # EXPL: loop through possible element pairs and build a matrix (2D list) of penalty incurred if they are paired up
    for a_elem in layout_a.elements:
        local_penalty = []

        for b_elem in layout_b.elements:
            # TODO: would the calculation be different if we used relative coordinates instead, and how would that work
            # TODO: with layouts of different aspect ratio?
            delta_x = abs(a_elem.x - b_elem.x)
            delta_y = abs(a_elem.y - b_elem.y)
            delta_w = abs(a_elem.width - b_elem.width)
            delta_h = abs(a_elem.height - b_elem.height)

            # EXPL: movement penalty equals relative movement, multiplied by the elements’ relative size
            # EXPL: i.e. bigger elements incur more penalty for movement
            # TODO: consider, if the formula makes sense, e.g. for differently sized canvases
            penalty_to_move = ((delta_x / (layout_a.x_sum + layout_b.x_sum)) + (delta_y / (layout_a.y_sum + layout_b.y_sum))) \
                            * ((a_elem.area + b_elem.area) / (layout_a.area_sum + layout_b.area_sum))
            # EXPL: resize penalty equals the relative resize, multipled by the element’s relative size
            penalty_to_resize = ((delta_w / (layout_a.w_sum + layout_b.w_sum)) + (delta_h / (layout_a.h_sum + layout_b.h_sum))) \
                              * ((a_elem.area + b_elem.area) / (layout_a.area_sum + layout_b.area_sum))
            # TODO: is there a penalty for changing element type? # EXPL: if not, consider places to add that (and test)

            local_penalty.append(penalty_to_move + penalty_to_resize)

        penalty_assignment.append(local_penalty)

    return penalty_assignment
