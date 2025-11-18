import quimb as qu
import quimb.tensor as qtn
import numpy as np
import functools
import math
import cmath
import scipy
from operator import add
from autoray import do, dag
from numpy import linalg as LA
from scipy.linalg import expm, sinm, cosm

import functools
from operator import add
from autoray import do, dag
from scipy.sparse import coo_matrix, bmat
from scipy.sparse import csr_matrix
from scipy.integrate import odeint
from scipy.integrate import complex_ode
import pickle
import math # noqa
import random # noqa
import itertools # noqa
import functools # noqa
import collections # noqa
import tqdm # noqa
import numpy as np # noqa
import matplotlib as mpl # noqa
import matplotlib.pyplot as plt # noqa
import quimb as qu # noqa
import xyzpy as xyz # noqa
import autoray as ar # noqa
import quimb.tensor as qtn # noqa
from tqdm import tqdm
import torch
from quimb.tensor.tensor_1d_compress import *

XI = qu.pauli('X') & qu.pauli('I')
IX = qu.pauli('I') & qu.pauli('X')
YI = qu.pauli('Y') & qu.pauli('I')
IY = qu.pauli('I') & qu.pauli('Y')
IZ = qu.pauli('I') & qu.pauli('Z')
ZI = qu.pauli('Z') & qu.pauli('I')

Z = qu.pauli('Z')
Y = qu.pauli('Y')
X = qu.pauli('X')

def to_backend(x):
    return torch.tensor(x, dtype=torch.complex64, device="cuda")

#build the MPO form of the Hamiltonian
def build_H_MPO(pxx, pyy, pzz, px, py, pz):
    
    buildersg = qtn.SpinHam1D(S=1/2)
    for i in range(L-1):
      buildersg[i,i+1] += pxx, X, X
      buildersg[i,i+1] += pyy, Y, Y
      buildersg[i,i+1] += pzz, Z, Z
      buildersg[i] += px, X
      buildersg[i] += py, Y
      buildersg[i] += pz, Z

    buildersg[L-1] += px, X
    buildersg[L-1] += py, Y
    buildersg[L-1] += pz, Z
    
    mpo_hamsg = buildersg.build_mpo(L=L)
    
    return mpo_hamsg

#double MPO for TEBD Hamiltonian evolution
def build_double_MPO(pxx, pyy, pzz, px, py, pz):
    
    builder_sys = qtn.SpinHam1D(S=3/2)

    for i in range(L-1):
        builder_sys[i,i+1] += -pxx, XI, XI
        builder_sys[i,i+1] += pxx, IX, IX
        builder_sys[i,i+1] += -pyy, YI, YI
        builder_sys[i,i+1] += pyy, IY, IY
        builder_sys[i,i+1] += -pzz, ZI, ZI
        builder_sys[i,i+1] += pzz, IZ, IZ
        builder_sys[i] += -px, XI
        builder_sys[i] += px, IX
        builder_sys[i] += -py, YI
        builder_sys[i] += py, IY
        builder_sys[i] += -pz, ZI
        builder_sys[i] += pz, IZ

    builder_sys[L-1] += -px, XI
    builder_sys[L-1] += px, IX
    builder_sys[L-1] += -py, YI
    builder_sys[L-1] += py, IY
    builder_sys[L-1] += -pz, ZI
    builder_sys[L-1] += pz, IZ
    
    return  builder_sys.build_local_ham(L)
    
#compute the gap of the Hamiltonian
def computegap(Hamiltonian):
    
    eigenergy, eigenstates=scipy.sparse.linalg.eigsh(Hamiltonian, k=4)
    print("Eigenergies = ",eigenergy)
    minE=np.min(eigenergy)
    eigenergy[np.argmin(eigenergy)]=0
    minE2=np.min(eigenergy)
    gap=minE2-minE
    
    return(gap)

#filter function
def fsig(t):
    
  result = (np.exp(-da**2*t**2/4)*cmath.exp(1j*a*t)-np.exp(-(gap/4)**2*t**2/4)*cmath.exp(1j*(gap/4)*t)+(a-gap/4)*0.0001j)/(2*np.pi*1j*t+2*np.pi*0.0001j)
  
  return result

#MPO contraction
def contract_mpos(AA, BB, CC):
    
    DD = AA.gate_lower_with_op_lazy(BB).gate_lower_with_op_lazy(CC)

    return DD

#turn MPOs to GPU arrays
def applyGPU():
    
    opKx.apply_to_arrays(to_backend)
    opKy.apply_to_arrays(to_backend)
    opKz.apply_to_arrays(to_backend)
    opKx_H.apply_to_arrays(to_backend)
    opKy_H.apply_to_arrays(to_backend)
    opKz_H.apply_to_arrays(to_backend)
    opKxe.apply_to_arrays(to_backend)
    opKye.apply_to_arrays(to_backend)
    opKze.apply_to_arrays(to_backend)
    opKxe_H.apply_to_arrays(to_backend)
    opKye_H.apply_to_arrays(to_backend)
    opKze_H.apply_to_arrays(to_backend)
    sys_ground.apply_to_arrays(to_backend)
    rho.apply_to_arrays(to_backend)
    Ham_MPO.apply_to_arrays(to_backend)
    decay.apply_to_arrays(to_backend)
    

#compute jump operators
def Computejump(site, pauli):
    
    Precision = 20
    
    ID = qtn.MPO_identity(L=L)
    mps_array = np.empty(NT, dtype=object)
    pauli_arrayx = np.array([0.0, 1.0, 1.0, 0.0])
    pauli_arrayy = np.array([0.0, -1.0j, 1.0j, 0.0])
    pauli_arrayz = np.array([1.0, 0.0, 0.0, -1.0])

    A = np.zeros((L,4),dtype=np.complex128)
    A[:,0] = 1.0
    A[:,3] = 1.0
    if pauli==1:
        A[site,:] = pauli_arrayx
    if pauli==2:
        A[site,:] = pauli_arrayy
    if pauli==3:
        A[site,:] = pauli_arrayz


    psi_mpdo = qtn.MPS_product_state(A)
    tebd = qtn.TEBD(psi_mpdo, ham_total, dt = dT)
    tebd.split_opts = {'max_bond': maxbondevo}

    mps_array[(NT-1)//2] = psi_mpdo
    numt = (NT-1)//2
    for psit in tebd.at_times(ts):

      ID=qtn.MPO_identity(L=L)
      
      for j in range(L):
        ID[j].modify(
            data=psit[j].unfuse({'k'+str(j):('k'+str(j),'b'+str(j))},{'k'+str(j):(2,2)}).data, 
            inds=psit[j].unfuse({'k'+str(j):('k'+str(j),'b'+str(j))},{'k'+str(j):(2,2)}).inds,
          )

      mps_array[numt] = ID
      numt = numt-1

    psi_mpdo = qtn.MPS_product_state(A)
    tebd = qtn.TEBD(psi_mpdo, ham_total_minus, dt = dT)
    tebd.split_opts = {'max_bond': maxbondevo}

    mps_array[(NT-1)//2] = psi_mpdo
    numt = (NT-1)//2
    for psit in tebd.at_times(ts):

      ID=qtn.MPO_identity(L=L)
      
      for j in range(L):
        ID[j].modify(
            data=psit[j].unfuse({'k'+str(j):('k'+str(j),'b'+str(j))},{'k'+str(j):(2,2)}).data, 
            inds=psit[j].unfuse({'k'+str(j):('k'+str(j),'b'+str(j))},{'k'+str(j):(2,2)}).inds,
          )

      mps_array[numt] = ID
      numt = numt+1


    opK = qtn.MPO_zeros(L)
    for i in range(NT):
      opK = opK.add_MPO(dT*fsig(timepts[i])*mps_array[NT-1-i])
      if (i % 20 ==0):
        opK.compress(max_bond=maxbondop)

    opK.compress(max_bond = maxbondop)

    #opK_H is the Hermitian conjugate of opK
    opK_H = qtn.MPO_zeros(L)
    for i in range(NT):
      opK_H = opK_H.add_MPO(dT*np.conjugate(fsig(timepts[i]))*mps_array[NT-1-i])
      if (i % 20 ==0):
        opK_H.compress(max_bond=maxbondop)

    opK_H.compress(max_bond = maxbondop)
    
    print("Maxbond_evo = ", mps_array[0].max_bond())
    
    print("Maxbond_jump = ", opK.max_bond())
    
    return opK, opK_H

#turn MPO arrays to GPU arrays 
def applyGPU():
    
    for i in range(L):
        opKx[i].apply_to_arrays(to_backend)
    for i in range(L):
        opKx_H[i].apply_to_arrays(to_backend)
    for i in range(L):
        opKy[i].apply_to_arrays(to_backend)
    for i in range(L):
        opKy_H[i].apply_to_arrays(to_backend)
    for i in range(L):
        opKz[i].apply_to_arrays(to_backend)
    for i in range(L):
        opKz_H[i].apply_to_arrays(to_backend)
    
    sys_ground.apply_to_arrays(to_backend)
    rho.apply_to_arrays(to_backend)
    Ham_MPO.apply_to_arrays(to_backend)
    

def Eulerforward(rho):

    rho1 = qtn.MPO_zeros(L)
    rho1.apply_to_arrays(to_backend)

    mpo_array = [qtn.MPO_zeros(L) for _ in range(L)]
    for i in range(L):
        mpo_array[i].apply_to_arrays(to_backend)

    for i in range(L):
        mpo_array[i] = qtn.tensor_1d_compress.tensor_network_1d_compress_fit({dTime*contract_mpos(opKx[i],rho,opKx_H[i]),
                                                                              dTime*contract_mpos(opKy[i],rho,opKy_H[i]),
                                                                              dTime*contract_mpos(opKz[i],rho,opKz_H[i]),
                                                                              -dTime/2*contract_mpos(opKx_H[i],opKx[i],rho),
                                                                              -dTime/2*contract_mpos(opKy_H[i],opKy[i],rho),
                                                                              -dTime/2*contract_mpos(opKz_H[i],opKz[i],rho),
                                                                              -dTime/2*contract_mpos(rho,opKx_H[i],opKx[i]),
                                                                              -dTime/2*contract_mpos(rho,opKy_H[i],opKy[i]),
                                                                              -dTime/2*contract_mpos(rho,opKz_H[i],opKz[i])}, max_bond = maxbond, tn_fit="zipup")

    rho1 = qtn.tensor_1d_compress.tensor_network_1d_compress_fit({mpo_array[i] for i in range(L)}|{rho,
                                                                  -1j*dTime*Ham_MPO.gate_lower_with_op_lazy(rho),
                                                                  1j*dTime*rho.gate_lower_with_op_lazy(Ham_MPO)},max_bond = maxbond, tn_fit="zipup")

    return rho1


inputL = input("System Size:")
inputbond = input("Max bond dim:")
    
J = 1
g = 1.5
Delta = 0.1
#TFIM Hamiltonian
L = int(inputL)
pxx = -J
pyy = -Delta
pzz = -Delta
px = 0.0
py = 0.0
pz = g
print("L =",L)

#Hamiltonian=qu.ham_heis(n = L, j = (4.0*pxx, 4.0*pyy, 4.0*pzz), b=(2.0*px, 2.0*py, 2.0*pz),sparse=True)

Ham_MPO = build_H_MPO(pxx = pxx, pyy = pyy, pzz = pzz, px = px, py = py, pz = pz)
dmrg = qtn.DMRG2(Ham_MPO)
dmrg.solve(max_sweeps=80, verbosity=0, cutoffs=1e-10)
sys_ground = dmrg.state
sys_ground /= 1.0

minE=qtn.expec_TN_1D(sys_ground.H, Ham_MPO, sys_ground)
print("Ground state norm = ", sys_ground.H @ sys_ground)
print("Ground state energy = ", minE)

#gap = computegap(Hamiltonian = Hamiltonian)
#print("Gap = ", gap)
gap = 1

#parameter settings
a = 2.5 * (-minE) 
da = a/5 
dT =  3.14/(4*a)
Ss = 2/gap
NT = int(np.round(2*Ss/dT))
NT = 2*(NT//2)+1
dT = 2*Ss/NT
timepts = np.linspace(-Ss, Ss, NT)
ts = np.linspace(0, Ss, (NT-1)//2+1)
print("Number of time points = ", NT)
print("Gap = ", gap)

ham_total = build_double_MPO(pxx = pxx, pyy = pyy, pzz = pzz, px = px, py = py, pz = pz)
ham_total_minus = build_double_MPO(pxx = -pxx, pyy = -pyy, pzz = -pzz, px = -px, py = -py, pz = -pz)

maxbondevo = int(inputbond)
maxbondop = int(inputbond)
maxbond = int(inputbond)

opKx = [qtn.MPO_zeros(L) for _ in range(L)]
opKx_H = [qtn.MPO_zeros(L) for _ in range(L)]
opKy = [qtn.MPO_zeros(L) for _ in range(L)]
opKy_H = [qtn.MPO_zeros(L) for _ in range(L)]
opKz = [qtn.MPO_zeros(L) for _ in range(L)]
opKz_H = [qtn.MPO_zeros(L) for _ in range(L)]

#Compute the jump operators
for i in range(L):
    opKx[i], opKx_H[i] = Computejump(site = i, pauli = 1)
    opKy[i], opKy_H[i] = Computejump(site = i, pauli = 2)
    opKz[i], opKz_H[i] = Computejump(site = i, pauli = 3)


rho = (1/2**L) * qtn.MPO_identity(L=L)
nrun = 80      #number of time step
dTime = 0.025  #time step
tdis = np.zeros((nrun))
data1 = np.zeros(nrun)
data2 = np.zeros(nrun)
applyGPU()

#compute the dissipative dynamics
for i in range(nrun):
  
  data1[i] = np.real(qtn.expec_TN_1D(sys_ground.H, rho, sys_ground))
  data2[i] = np.real((rho.apply(Ham_MPO)).trace())
  
  print("Step = ", i,"Fidelity = ", data1[i], "Energy = ", data2[i])
  
  rho = Eulerforward(rho)

