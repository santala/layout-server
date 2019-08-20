import layout_engine.SolutionInstance
from layout_engine.SolutionInstance import *
from tools.JSonExportUtility import *
from tools.PlotUtility import *

hashToSolution = dict()

def buildNewSolution(objValue, Lval,Tval, Wval, Hval):
    solution = SolutionInstance(objValue, Lval,Tval, Wval, Hval)
    hash = str(Lval)+str(Tval)+ str(Wval)+str(Hval)
    if hash in hashToSolution:
        print("** Neglecting a repeat solution **")
        return
    else:
        hashToSolution[hash] = solution
        useSolution(objValue, Lval,Tval, Wval, Hval)


def useSolution(objValue, Lval,Tval, Wval, Hval):
    SaveToJSon(tools.GurobiUtils.data.N, tools.GurobiUtils.data.canvasWidth, tools.GurobiUtils.data.canvasHeight, Lval,Tval, Wval, Hval, tools.GurobiUtils.solNo, tools.GurobiUtils.data, objValue)
    DrawPlotOnPage(tools.GurobiUtils.data.N, tools.GurobiUtils.data.canvasWidth, tools.GurobiUtils.data.canvasHeight, Lval,Tval, Wval, Hval, tools.GurobiUtils.solNo)
