from typing import List

from tools.JSONLoader import Layout

def compute_penalty_assignment(layout1: Layout, layout2: Layout) -> List[List[float]]:
    penalty_assignment = []

    # EXPL: loop through possible element pairs and build a matrix (2D list) of penalty incurred if they are paired up
    for element1 in layout1.elements:
        local_penalty = []

        for element2 in layout2.elements:
            # TODO: would the calculation be different if we used relative coordinates instead, and how would that work
            # TODO: with layouts of different aspect ratio?
            delta_x = abs(element1.x - element2.x)
            delta_y = abs(element1.y - element2.y)
            delta_w = abs(element1.width - element2.width)
            delta_h = abs(element1.height - element2.height)

            # EXPL: movement penalty equals relative movement, multiplied by the elements’ relative size
            # EXPL: i.e. bigger elements incur more penalty for movement
            # TODO: consider, if the formula makes sense, e.g. for differently sized canvases
            penalty_to_move = ((delta_x / (layout1.x_sum + layout2.x_sum)) + (delta_y / (layout1.y_sum + layout2.y_sum))) \
                            * ((element1.area + element2.area) / (layout1.area_sum + layout2.area_sum))
            # EXPL: resize penalty equals the relative resize, multipled by the element’s relative size
            penalty_to_resize = ((delta_w / (layout1.w_sum + layout2.w_sum)) + (delta_h / (layout1.h_sum + layout2.h_sum))) \
                              * ((element1.area + element2.area) / (layout1.area_sum + layout2.area_sum))
            # TODO: is there a penalty for changing element type? # EXPL: if not, consider places to add that (and test)

            '''
            if penalty_to_move < 0:
                print('penalty to move')
                print(layout1.x_sum, layout2.x_sum, layout1.y_sum, layout2.y_sum, element1.area, element2.area, layout1.area_sum, layout2.area_sum)
            elif penalty_to_resize < 0:
                print('penalty to resize')
                print(layout1.w_sum, layout2.w_sum, layout1.h_sum, layout2.h_sum, element1.area, element2.area, layout1.area_sum, layout2.area_sum)
            '''


            local_penalty.append(penalty_to_move + penalty_to_resize)

        penalty_assignment.append(local_penalty)

    return penalty_assignment
