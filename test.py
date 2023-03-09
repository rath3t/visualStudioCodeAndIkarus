# SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-3.0-or-later
import matplotlib
import dune.grid
import dune.functions
import pyikarus as iks
import pyikarus.finite_elements
import pyikarus.utils
import pyikarus.assembler
import pyikarus.dirichletValues
import numpy as np
import scipy as sp
import time

from dune.vtk import vtkUnstructuredGridWriter, vtkWriter, RangeTypes, FieldInfo

if __name__ == "__main__":
    # help(iks)
    tic = time.perf_counter()
    lowerLeft = []
    upperRight = []


    elements = []
    for i in range(2):
        lowerLeft.append(-1)
        upperRight.append(1)
        elements.append(3)

    req= pyikarus.FErequirements()
    req.addAffordance(iks.ScalarAffordances.mechanicalPotentialEnergy)

    grid = dune.grid.structuredGrid(lowerLeft,upperRight,elements)
    grid.hierarchicalGrid.globalRefine(6)

    basisLagrange1 = dune.functions.defaultGlobalBasis(grid, dune.functions.Power(dune.functions.Lagrange(order=1),2))
    print('We have {} dofs.'.format(len(basisLagrange1)))
    print('We have {} vertices.'.format(grid.size(2)))
    print('We have {} elements.'.format(grid.size(0)))
    d = np.zeros(len(basisLagrange1))

    lambdaLoad = iks.ValueWrapper(3.0)
    req.insertParameter(iks.FEParameter.loadfactor,lambdaLoad)

    req.insertGlobalSolution(iks.FESolutions.displacement,d)

    def volumeLoad(x,lambdaVal) :
        return np.array([lambdaVal*x[0]*2, 2*lambdaVal*x[1]*0])

    def neumannLoad(x,lambdaVal) :
        return np.array([lambdaVal*0, lambdaVal])

    neumannVertices = np.zeros(grid.size(2)*2, dtype=bool)

    basisLagrange1.interpolate(neumannVertices, lambda x :  True  if x[1]>0.9 else False)

    boundaryPatch = iks.utils.boundaryPatch(grid,neumannVertices)

    fes = []
    for e in grid.elements:
        fes.append(iks.finite_elements.linearElasticElement(basisLagrange1,e,1000,0.2,volumeLoad,boundaryPatch,neumannLoad))

    forces = np.zeros(8)
    stiffness = np.zeros((8,8))
    fes[0].calculateVector(req,forces)
    fes[0].calculateMatrix(req,stiffness)
    np.set_printoptions(precision=3)
    print('Forces:\n {}'.format(forces))
    print('Stiffness:\n {}'.format(stiffness))

    dirichletValues = iks.dirichletValues.dirichletValues(basisLagrange1) 

    def fixLeftHandEdge(vec,localIndex,localView,intersection):
        if (intersection.geometry.center[1]<-0.9):
            vec[localView.index(localIndex)]= True

    dirichletValues.fixBoundaryDOFsUsingLocalViewAndIntersection(fixLeftHandEdge)

    assembler = iks.assembler.sparseFlatAssembler(fes,dirichletValues)
    assemblerDense = iks.assembler.denseFlatAssembler(fes,dirichletValues)

    Msparse = assembler.getMatrix(req)
    forces = assembler.getVector(req)
    print(Msparse)

    x = sp.sparse.linalg.spsolve(Msparse,-forces)
    fx = basisLagrange1.asFunction(x)
    #grid.plot()
    writer = vtkWriter( grid, "nameTest",  pointData   = {( "displacement",(0,1)):fx})
    toc = time.perf_counter()
    print(f"The whole code took {toc - tic:0.4f} seconds to excecute")