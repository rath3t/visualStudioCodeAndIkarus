import matplotlib
from matplotlib import pyplot
from matplotlib.collections import PolyCollection
import numpy as np
from numpy import amin, amax, linspace, linalg, random
_addPlot = True

def plotDeformedGrid(grid,gf, gridLines="black"):

    figure = pyplot.figure()

    lf = gf.localFunction()

    dispEles =[]
    for e in grid.elements:
        corners = e.geometry.corners
        lf.bind(e)
        c0 = corners[0] + lf([0,0])
        c1 = corners[1] + lf([1,0])
        c2 = corners[3] + lf([1,1])
        c3 = corners[2] + lf([0,1])

        array= np.matrix((c0,c1,c2,c3))
        dispEles.append(array)


    coll = PolyCollection(dispEles, facecolor='none', edgecolor=gridLines, linewidth=0.5, zorder=2)
    pyplot.gca().add_collection(coll)


    figure.gca().set_aspect('equal')
    figure.gca().autoscale()
   