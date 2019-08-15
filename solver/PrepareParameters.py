from model import Layout



def resolveIndividualLayoutParameters(layout:Layout):
    for element in layout.elements:
        element.area = element.width * element.height
        layout.Xsum = layout.Xsum + element.X
        layout.Ysum = layout.Ysum + element.Y
        layout.Wsum = layout.Wsum + element.width
        layout.Hsum = layout.Hsum + element.height
        layout.AreaSum = layout.AreaSum + element.area


def buildLayoutParameters(firstLayout: Layout, secondLayout: Layout):
    PenaltyAssignment = []

    resolveIndividualLayoutParameters(firstLayout)
    resolveIndividualLayoutParameters(secondLayout)

    # EXPL: Penalty of being skipped is the relative size of the element
    for firstElement in firstLayout.elements:
        firstElement.PenaltyIfSkipped = firstElement.area / firstLayout.AreaSum

    for secondElement in secondLayout.elements:
        secondElement.PenaltyIfSkipped = secondElement.area / secondLayout.AreaSum

    # EXPL: loop through possible element pairs and build a matrix (2D list) of penalty incurred if they are paired up
    for firstElement in firstLayout.elements:
        localPenalty = []

        for secondElement in secondLayout.elements:
            # TODO: would the calculation be different if we used relative coordinates instead, and how would that work
            # TODO: with layouts of different aspect ratio?
            deltaX = abs(firstElement.X - secondElement.X)
            deltaY = abs(firstElement.Y - secondElement.Y)
            deltaW = abs(firstElement.width - secondElement.width)
            deltaH = abs(firstElement.height - secondElement.height)

            # EXPL: movement penalty equals relative movement, multiplied by the elements’ relative size
            # EXPL: i.e. bigger elements incur more penalty for movement
            # TODO: consider, if the formula makes sense, e.g. for differently sized canvases
            PenaltyToMove = ((deltaX / (firstLayout.Xsum + secondLayout.Xsum)) + (deltaY / (firstLayout.Ysum + secondLayout.Ysum))) \
                            * ((firstElement.area + secondElement.area) / (firstLayout.AreaSum + secondLayout.AreaSum))
            # EXPL: resize penalty equals the relative resize, multipled by the element’s relative size
            PenaltyToResize = ((deltaW / (firstLayout.Wsum + secondLayout.Wsum)) + (deltaH / (firstLayout.Hsum + secondLayout.Hsum))) \
                              * ((firstElement.area + secondElement.area) / (firstLayout.AreaSum + secondLayout.AreaSum))
            # TODO: is there a penalty for changing element type? # EXPL: if not, consider places to add that (and test)

            localPenalty.append(PenaltyToMove + PenaltyToResize)

        PenaltyAssignment.append(localPenalty)

    return PenaltyAssignment


def prepare(firstLayout:Layout, secondLayout:Layout):

    return buildLayoutParameters(firstLayout, secondLayout)
