# SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-3.0-or-later

import dune.grid
import dune.functions
import pyikarus as iks
import pyikarus.finite_elements
import pyikarus.utils
import pyikarus.assembler
import pyikarus.dirichletValues
import numpy as np
import scipy as sp


from dune.vtk import vtkUnstructuredGridWriter, vtkWriter, RangeTypes, FieldInfo

if __name__ == "__main__":
    print("test")
    # help(iks)
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
    grid. hierarchicalGrid . globalRefine(6)
    basisLagrange1 = dune.functions.defaultGlobalBasis(grid, dune.functions.Power(dune.functions.Lagrange(order=1),2))

    d = np.zeros(len(basisLagrange1))
    d[0]=0.0

    lambdaLoad = iks.ValueWrapper(3.0)
    req.insertParameter(iks.FEParameter.loadfactor,lambdaLoad)

    assert req.getParameter(iks.FEParameter.loadfactor) == lambdaLoad
    req.insertGlobalSolution(iks.FESolutions.displacement,d)

    d2= req.getGlobalSolution(iks.FESolutions.displacement)

    assert ('{}'.format(hex(d2.__array_interface__['data'][0]))) == ('{}'.format(hex(d.__array_interface__['data'][0])))
    assert len(d2)== len(d)
    assert (d2== d).all()
    fes = []
    forces = np.zeros(8)
    stiffness = np.zeros((8,8))

    def volumeLoad(x,lambdaVal) :
        return np.array([lambdaVal*x[0]*2, 2*lambdaVal*x[1]*0])

    def neumannLoad(x,lambdaVal) :
        return np.array([lambdaVal*0, lambdaVal])


    neumannVertices = np.zeros(grid.size(2)*2, dtype=bool)
    assert len(neumannVertices)== len(basisLagrange1)

    basisLagrange1.interpolate(neumannVertices, lambda x :  True  if x[1]>0.9 else False)

    boundaryPatch = iks.utils.boundaryPatch(grid,neumannVertices)

    # the following should throw
    try:
        iks.finite_elements.linearElasticElement(basisLagrange1,grid.elements[0],1000,0.2,volumeLoad,boundaryPatch)
        assert False
    except TypeError :
        assert True

    for e in grid.elements:
        fes.append(iks.finite_elements.linearElasticElement(basisLagrange1,e,1000,0.2,volumeLoad,boundaryPatch,neumannLoad))

    fes[0].calculateVector(req,forces)
    fes[0].calculateMatrix(req,stiffness)
    fes[0].localView()

    dirichletValues = iks.dirichletValues.dirichletValues(basisLagrange1) # TODO make second redundant

    def fixFirstIndex(vec,globalIndex):
        vec[0]= True

    def fixAnotherVertex(vec,localIndex,localView):
        localView.index(localIndex)
        vec[1]= True

    def fixLeftHandEdge(vec,localIndex,localView,intersection):
        if (intersection.geometry.center[1]<-0.9):
            vec[localView.index(localIndex)]= True

    dirichletValues.fixBoundaryDOFs(fixFirstIndex)
    dirichletValues.fixBoundaryDOFsUsingLocalView(fixAnotherVertex)
    dirichletValues.fixBoundaryDOFsUsingLocalViewAndIntersection(fixLeftHandEdge)

    assembler = iks.assembler.sparseFlatAssembler(fes,dirichletValues)
    assemblerDense = iks.assembler.denseFlatAssembler(fes,dirichletValues)

    Msparse = assembler.getMatrix(req)
    forces = assembler.getVector(req)
    # Mdense = assemblerDense.getMatrix(req)
    # assert (Msparse == Mdense).all()
    # print(Mdense)
    # print("F:",forces)

    x = sp.sparse.linalg.spsolve(Msparse,-forces)
    # print(x)
    fx = basisLagrange1.asFunction(x)
    grid.plot()
    # grid.writeVTK(pointdata={"fx":fx},name="nameTest",number=1)
    writer = vtkWriter( grid, "nameTest",
                    pointData   = {( "displacement",(0,1)):fx}

                    )
