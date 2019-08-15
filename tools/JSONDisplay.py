import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from model import Layout
import os
import threading
from pathlib import Path

def actualDisplay(layout: Layout, name: str):
    Lval=[]
    Tval =[]
    Wval =[]
    Hval = []
    for index in range(layout.N):
        Lval.append(layout.elements[index].X)
        Tval.append(layout.elements[index].Y)
        Wval.append(layout.elements[index].width)
        Hval.append(layout.elements[index].height)
    
    displayJSON(layout.N, layout.canvasWidth, layout.canvasHeight, Lval, Tval, Wval, Hval, name)

def displayJSON(N:int , CanvasSize_W:int, CanvasSize_H:int, Lval, Tval, Wval, Hval, name: str):
    fig, ax = plt.subplots()
    
    rectangles = []
    for x in range(N):
        myRect = mpatch.Rectangle((Lval[x], Tval[x]), Wval[x], Hval[x], edgecolor='0.5')
        rectangles.append(myRect)

    x = 0
    for r in rectangles:
        ax.add_artist(r)
        rx, ry = r.get_xy()
        cx = rx + r.get_width() / 2.0
        cy = ry + r.get_height() / 2.0
        ax.annotate(str(x), (cx, cy), color='black', weight='bold', fontsize=6, ha='center', va='center')
        x=x+1
    ax.set_xlim((0, CanvasSize_W))
    ax.set_ylim((0, CanvasSize_H))
    ax.set_aspect('equal')
    #plt.title(name)

    plt.axis([0, CanvasSize_W, 0, CanvasSize_H])
    plt.grid(False)  # set the grid
    ax = plt.gca()  # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    ax.xaxis.tick_top()  # and move the X-Axis
    #plt.show()
    plt.savefig(str(Path("{}/../Output/{}.png".format(os.path.dirname(__file__), name))))
    plt.close()
    t = threading.Thread(target=openImageInNewThread, args = [name])
    t.start()
    
def openImageInNewThread(name:str):
    os.system(str(Path("{}/../Output/{}.png".format(os.path.dirname(__file__), name))))
