from layout_engine.SolutionInstance import *
from tools.JSonExportUtility import *
from tools.PlotUtility import *

from gurobipy import Model
import tools.GurobiUtils
from tools.GurobiUtils import *
from tools.JSONLoader import *

from typing import List

def build_new_solution(model: Model, objValue: float, Lval: List[int], Tval: List[int], Wval: List[int], Hval: List[int], hash_to_solution: dict):
    layout: Layout = model._layout
    var: Variables = model._var

    solution = SolutionInstance(objValue, Lval,Tval, Wval, Hval)
    solution_hash = str(Lval)+str(Tval)+ str(Wval)+str(Hval)
    if solution_hash in hash_to_solution:
        print("** Neglecting a repeat solution **")
        return
    else:
        hash_to_solution[solution_hash] = solution
        SaveToJSon(layout.n, layout.canvas_width, layout.canvas_height,
                   Lval, Tval, Wval, Hval, model._solution_number, layout, objValue)
        DrawPlotOnPage(layout.n, layout.canvas_width, layout.canvas_height, Lval, Tval, Wval, Hval, model._solution_number)

