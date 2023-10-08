"""
Lindblad based method for ground state preparation

List of funcionts:

filter_freq: construct the filter function f in the fourier domain (\hat{f})
filter_time: construct the filter function f in the time domain
construct_jump_exact: construct the exact jump operator in matrix form
time_contour: construct discrete points for approximating integral
step_Lindblad: one step simulation for Lindbladian
Lindblad_simulation: Lindblad simulation
"""

import numpy as np # generic math functions
import scipy.sparse
import scipy.linalg as la
import scipy.io
from scipy.special import erf
from scipy.linalg import expm
from numpy import pi
from numpy.fft import fft
from time import time

class Lindblad:
    def __init__(self, H_op, A_op, filter_params):
        """
        Initialize Lindblad object.
        
        Parameters:
        H_op (Operator): The Hamiltonian operator.
        A_op (Operator): The coupling operator.
        filter_params (dict): The parameters for defining the filter
        """
        self.H_op = H_op
        self.A_op = A_op
        self.Ns = H_op.shape[0]
        self.filter_a = filter_params['a']
        self.filter_b = filter_params['b']
        self.filter_da = filter_params['da']
        self.filter_db = filter_params['db']

    def filter_time(self, t): 
        """Define the function f"""
        a = self.filter_a
        b = self.filter_b
        da = self.filter_da
        db = self.filter_db
        if np.abs(t)<1e-10:
            return (-b+a)/(2.0*pi)
        else:
            return (np.exp(-(da*t)**2/4)*np.exp(1j*a*t)-\
                np.exp(-(db*t)**2/4)*np.exp(1j*b*t)) / (1j*2*np.pi*t)

    def filter_freq(self, x): 
        """Define the fourier transform \hat{f}"""
        a = self.filter_a
        b = self.filter_b
        da = self.filter_da
        db = self.filter_db
        return 0.5 * (erf((x+a)/da) - erf((x+b)/db))
        
    def construct_jump_exact(self): 
        """Construct jump operator in frequency domain 
        (use for exact Lindblad solver)."""
        H_mat = self.H_op
        A_mat = self.A_op
        E_H, psi_H = la.eigh(H_mat)
        A_ovlp = psi_H.conj().T @ (A_mat.dot(psi_H))
        Ns = self.Ns
        A_jump = np.zeros((Ns, Ns))

        for i in range(Ns):
            for j in range(Ns):
                A_jump += self.filter_freq(E_H[i]-E_H[j]) * A_ovlp[i,j] * \
                        np.outer(psi_H[:,i], psi_H[:,j].conj())
        
        self.A_jump = A_jump

    def time_contour(self, S_s, M_s, isreverse=True):
        """
        Construct the time contour for propagating the Kraus operator in
        time domain.
        2M_s+1 grid points (include s=0)
        """
        tau_s = S_s / M_s
        tgrid = np.zeros(2*M_s+1)
        tgrid = -S_s + np.arange(2*M_s+1) * tau_s

        if isreverse: 
            return np.append(tgrid,tgrid[::-1])# reverse
        else:
            return tgrid
    
    def step_Lindblad(self, psi, tau, num_t, num_segment, num_rep, S_s, M_s, dice, intorder):
        """
        Propagate one step of the dilated jump operator in a batch.
        """
        num_batch = psi.shape[1]
        if not intorder in {1,2}:
            raise ValueError('intorder must be 1 or 2.')
        #Simulation preparation
        # first order method does not require reversing the grid
        isreverse = (intorder > 1);
        tau_s = S_s / M_s
        s_contour = self.time_contour(S_s,M_s, 
                isreverse = isreverse) #discrete s point 
        Ns_contour = s_contour.shape[0] 
        F_contour = np.zeros((Ns_contour), dtype=complex) #discrete F value
        VF_contour = np.zeros((Ns_contour,2,2), dtype=complex) #discrete dilated F value
        tau_scal = np.ones(num_rep)*np.sqrt(tau) / num_segment #rescaled tau (for discrete Lindblad)
        eHt = self.eHt
        eHT = self.eHT
        E_A=self.E_A #eigenvalue of A
        psi_A =self.psi_A #eigenvector of A
        Ns = self.Ns #dimension of the system
        ZA_dilate = np.zeros((Ns_contour, 2*Ns, num_rep), dtype=complex) #local jump operator 
        #for discrete integral point
        #---Construct local \widetilde{A}_l----
        for i in range(Ns_contour):
            if (s_contour[i]==np.min(s_contour)) or (s_contour[i]==np.max(s_contour)):
                F_contour[i] = self.filter_time(s_contour[i])/2
            else: 
                F_contour[i] = self.filter_time(s_contour[i])

            fac = np.exp(1j*np.angle(F_contour[i]))
            VF_contour[i,:,:] = 1.0/np.sqrt(2) * np.array([[1,-1], [fac,fac]])
            if intorder == 1: #first order
                expZA = np.exp(-1j*tau_s*np.abs(F_contour[i])*np.outer(E_A,
                    tau_scal))
            else: #second order
                expZA = np.exp(-1j*0.5*tau_s*np.abs(F_contour[i])*np.outer(E_A,
                    tau_scal))
            ZA_dilate[i,:Ns,:] = expZA
            ZA_dilate[i,Ns:,:] = expZA.conj()
        #---start simulation
        psi_t_batch = np.zeros((2*Ns, num_batch), dtype=complex)
        psi_t_batch.fill(0j)
        psi_t_batch[:Ns,:] = psi
        for iseg in range(num_segment):
            if isreverse: #second order
                for i in range(int(Ns_contour/2)):#left-ordered product
                    VK = np.kron(VF_contour[i,:,:], psi_A) 
                    psi_t_batch = VK.conj().T @ psi_t_batch  
                    # pointwise multiplication
                    psi_t_batch *= ZA_dilate[i,:,:]
                    psi_t_batch = VK @ psi_t_batch         
                    psi_t_batch=np.kron(np.identity(2),eHt) @ psi_t_batch
                for i in range(int(Ns_contour/2)):#right-ordered product
                    psi_t_batch=np.kron(np.identity(2),eHt.conj().T) @ psi_t_batch
                    VK = np.kron(VF_contour[i+int(Ns_contour/2),:,:], psi_A) 
                    psi_t_batch = VK.conj().T @ psi_t_batch  
                    # pointwise multiplication
                    psi_t_batch *= ZA_dilate[i+int(Ns_contour/2),:,:]
                    psi_t_batch = VK @ psi_t_batch                                               
            else: #first order
                # only #left-ordered product
                for i in range(int(Ns_contour)):
                    VK = np.kron(VF_contour[i,:,:], psi_A) 
                    psi_t_batch = VK.conj().T @ psi_t_batch  
                    # pointwise multiplication
                    psi_t_batch *= ZA_dilate[i,:,:]
                    psi_t_batch = VK @ psi_t_batch         
                    psi_t_batch=np.kron(np.identity(2),eHt) @ psi_t_batch
                # rewind the time. This seems quite important in
                # practice, which is consistent with the (unexplained)
                # importance of adding the coherent contribution.
                is_rewind = True
                if is_rewind:
                    psi_t_batch=np.kron(np.identity(2),self.eHT.conj().T) @ psi_t_batch
                    psi_t_batch=np.kron(np.identity(2),self.eHT.conj().T) @ psi_t_batch
        for ir in range(num_batch):
            prob = la.norm(psi_t_batch[Ns:,ir])**2
            if( dice[ir] <= prob ):
                # flip the |1>| state
                psi[:,ir] = psi_t_batch[Ns:,ir]
            else:
                # keep the |0> state
                psi[:,ir] = psi_t_batch[:Ns,ir]
    
            # normalize
            psi[:,ir] /= la.norm(psi[:,ir])

        return psi


    def Lindblad_simulation(self, T, num_t, num_segment, psi0,
            num_rep, S_s, M_s, psi_GS = [], intorder = 2):
        """
        Lindblad simulation

        This uses the deterministic propagation with first or second
        order Trotter (intorder).  In particular, the first order Trotter method
        enables propagation with positive time.
        """
        H = self.H_op
        #Simulation parameter
        tau = T / num_t
        time_series = np.arange(num_t+1) * tau
        Ns = psi0.shape[0] #length of the state
        tau_s = S_s / M_s #time step for integral discretization
        eHtau = la.expm(-1j*tau*H) 
        self.eHt = la.expm(-1j*tau_s*self.H_op) #short time Hamiltonian simulation
        self.eHT = la.expm(-1j*S_s*self.H_op)
        self.E_A , self.psi_A = la.eigh(self.A_op) #diagonalize A for later implementation
        #Output Storage
        time_H=np.zeros(num_t+1) #List of total Hamiltonian simulation time
        avg_energy_hist = np.zeros((num_t+1, num_rep))
        avg_energy_hist[0,:].fill(np.vdot(psi0, H @ psi0).real) #List of energy
        avg_pGS_hist = np.zeros((num_t+1, num_rep)) #List of overlap with ground state
        if len(psi_GS) == 0:
            psi_GS = np.zeros_like(psi0)
        avg_pGS_hist[0,:].fill(np.abs(np.vdot(psi0, psi_GS))**2) #initial overlap
        # this randomness is introduced for modeling the tracing out
        # operation. Cannot be derandomized.
        np.random.seed(seed=1)
        flip_dice  = np.random.rand(num_t,num_rep) #used for simulating tracing out in quantum circuit
        rho_hist   = np.zeros((Ns,Ns,num_t+1), dtype=complex) #\rho_n
        psi_all    = np.zeros((Ns,num_rep), dtype=complex) #List of psi_n
        for i in range(num_rep):
            psi_all[:,i] = self.eHT.conj().T @ psi0.copy()
        rho_hist[:,:,0] = np.outer(psi_all[:,0],psi_all[:,0].conj().T)
        for it in range(num_t):
            psi_all = eHtau @ psi_all
            time_H[it+1]=time_H[it]+tau
            psi_all = self.step_Lindblad(psi_all, tau,  num_t, 
                    num_segment, num_rep, S_s, M_s, flip_dice[it,:], intorder)
            rho_hist[:,:,it+1] = np.einsum('in,jn->ij',psi_all,
                    psi_all.conj()) / num_rep #taking average to get \rho_n
            time_H[it+1]=time_H[it+1]+2*num_segment*S_s #Calculating total H simulation time
            # measurement
            avg_energy_hist[it+1,:] = np.einsum('in,in->n', psi_all.conj(),
                    H @ psi_all).real #Calculating energy
            avg_pGS_hist[it+1,:] = np.abs( np.einsum('in,i->n', psi_all.conj(),
                    psi_GS) )**2 #Calculating overlap
        avg_energy = np.mean(avg_energy_hist,axis=1)
        avg_pGS = np.mean(avg_pGS_hist,axis=1)
        return time_series, avg_energy, avg_pGS, time_H, rho_hist