# Plotting
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import matplotlib.colors as mcolors
import random


def DrawPlotOnPage(N, CanvasSize_W, CanvasSize_H, Lval, Tval, Wval, Hval, solNo):
    #print("plotter called")
    fig, ax = plt.subplots()

    rectangles = []
    for x in range(N):
        myRect = mpatch.Rectangle((Lval[x], Tval[x]), Wval[x], Hval[x], edgecolor='0.5')
        rectangles.append(myRect)

    #print("Rectangles are:",rectangles)

    x = 0
    for r in rectangles:
        #print("X is ",x,"At rectange",r)
        ax.add_artist(r)
        rx, ry = r.get_xy()
        cx = rx + r.get_width() / 2.0
        cy = ry + r.get_height() / 2.0
        ax.annotate(str(x), (cx, cy), color='black', weight='bold', fontsize=6, ha='center', va='center')
        x=x+1
    ax.set_xlim((0, CanvasSize_W))
    ax.set_ylim((0, CanvasSize_H))
    ax.set_aspect('equal')
    #plt.title("")

    ## New start
    plt.axis([0, CanvasSize_W, 0, CanvasSize_H])
    plt.grid(False)  # set the grid

    ax = plt.gca()  # get the axis
    ax.set_ylim(ax.get_ylim()[::-1])  # invert the axis
    ax.xaxis.tick_top()  # and move the X-Axis

    plt.savefig("output/Test"+(str(solNo)+".png"))
    plt.close()
    plt.show()

