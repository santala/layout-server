from typing import List

class SolutionInstance:
    def __init__(self, objVal: float, X: List[int], Y: List[int], W: List[int], H: List[int]):
        self.X = X
        self.Y = Y
        self.W = W
        self.H = H
        self.objVal = objVal


# TODO: check if code below is useful for anything
'''
def compute_index(n: int, var: Variables):
    index = 0
    for element in range(1, n+1):
        for other in range(1, n+1):
            index = index + (element*other*other*var.ABOVE[element,other])+(element*other*element*var.LEFT[element,other])
    return index
'''