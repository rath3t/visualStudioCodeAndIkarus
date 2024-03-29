# SPDX-FileCopyrightText: 2022 The Ikarus Developers mueller@ibb.uni-stuttgart.de
# SPDX-License-Identifier: LGPL-3.0-or-later

# import setpath
# setpath.set_path()
import pyikarus as iks
import pyikarus.finite_elements
import pyikarus.utils
import pyikarus.assembler
import pyikarus.dirichletValues
import numpy as np
import scipy as sp
from scipy.optimize import minimize

import dune.grid
import dune.functions
from dune.vtk import vtkUnstructuredGridWriter, vtkWriter, RangeTypes, FieldInfo

if __name__ == "__main__":
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
    grid. hierarchicalGrid.globalRefine(8)
    basisLagrange1 = dune.functions.defaultGlobalBasis(grid, dune.functions.Power(dune.functions.Lagrange(order=1),2))


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

    svk = iks.StVenantKirchhoff(emodul=1000,nu=0.3)

    psNH = svk.asPlainStress()
    fes = []
    for e in grid.elements:
        fes.append(iks.finite_elements.nonLinearElasticElement(basisLagrange1,e,psNH,volumeLoad,boundaryPatch,neumannLoad))

    dirichletValues = iks.dirichletValues(basisLagrange1)

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

    dRed = np.zeros(assembler.reducedSize())

    lambdaLoad = iks.ValueWrapper(3.0)
    def energy(dRedInput) :
        global assembler

        reqL= pyikarus.FErequirements()
        reqL.addAffordance(iks.ScalarAffordances.mechanicalPotentialEnergy)
        reqL.insertParameter(iks.FEParameter.loadfactor,lambdaLoad)

        dBig = assembler.createFullVector(dRedInput)
        reqL.insertGlobalSolution(iks.FESolutions.displacement,dBig)
        return assembler.getScalar(reqL)
    def gradient(dRedInput) :
        global assembler

        reqL= pyikarus.FErequirements()
        reqL.addAffordance(iks.VectorAffordances.forces)
        reqL.insertParameter(iks.FEParameter.loadfactor,lambdaLoad)

        dBig = assembler.createFullVector(dRedInput)
        reqL.insertGlobalSolution(iks.FESolutions.displacement,dBig)
        return assembler.getReducedVector(reqL)
    def hess(dRedInput) :
        global assembler

        reqL= pyikarus.FErequirements()
        reqL.addAffordance(iks.MatrixAffordances.stiffness)
        reqL.insertParameter(iks.FEParameter.loadfactor,lambdaLoad)

        dBig = assembler.createFullVector(dRedInput)
        reqL.insertGlobalSolution(iks.FESolutions.displacement,dBig)
        return assembler.getReducedMatrix(reqL).todense()
    
    from numpy.linalg import norm
    maxiter = 100
    abs_tolerance = 1e-8
    d = np.zeros(assembler.reducedSize())
    for k in range(maxiter):
        R, K = gradAndhess(d)
        r_norm = norm(R)
    
        deltad = sp.sparse.linalg.spsolve(K, R)
        d -= deltad
        print(k,r_norm,norm(deltad),energy(d))
        if r_norm < abs_tolerance:
            break

    print("Energy at equilibrium: ",energy(d)) 

    print("energy(dRed):",energy(dRed))
    print("energyafer")
    #resultd = minimize(energy,x0=dRed,options={"disp": True},tol=1e-14)
    #resultd2 = minimize(energy,x0=dRed,jac=gradient,options={"disp": True},tol=1e-14)
    #resultd3 = minimize(energy,method="trust-constr",x0=dRed,jac=gradient,hess=hess,options={ 'disp': True})
    #resultd4 = sp.optimize.root(gradient,jac=hess,x0=dRed,options={ 'disp': True},tol=1e-10)
    # print(assembler.createFullVector(resultd.g))
    np.set_printoptions(precision=3)

    assert(np.allclose(resultd.x,resultd2.x,atol=1e-6))
    assert(np.allclose(resultd3.x,resultd4.x))
    assert(np.all(abs(resultd3.grad)<1e-8))
    assert(np.all(abs(resultd4.fun)<1e-8))
