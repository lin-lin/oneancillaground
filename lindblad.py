"""
Lindblad based method for ground state preparation

Authors: Zhiyan Ding
         Lin Lin

Last revision: 10/07/2023
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

        # For time localized filter. Not used for now.
        self.filter_t_supp = 2.0/filter_params['b']
        self.filter_t_iden = 1.0/filter_params['b']

    def filter_time(self, t):
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
        a = self.filter_a
        b = self.filter_b
        da = self.filter_da
        db = self.filter_db
        return 0.5 * (erf((x+a)/da) - erf((x+b)/db))

    # def filter_freq_localized(self, x):
        # """ Fourier transform of the time localized filter """
        # a = self.filter_a
        # b = self.filter_b
        # da = self.filter_da
        # db = self.filter_db
        # t = np.linspace(-self.filter_t_supp, self.filter_t_supp, num=int(1000/filter_params['b']))
        # f=np.zeros(len(t),'complex')
        # for n in range(len(t)):
            # f[n]=self.filter_time_localized(t[n])
        # return np.real(np.dot(f, np.exp(1j*x*t))*(t[1]-t[0]))
    # 
    # def filter_time_localized(self, t):
        # """ Localized filter which is compacted supported in time """
        # a = self.filter_a
        # b = self.filter_b
        # da = self.filter_da
        # db = self.filter_db
        # r0 = self.filter_t_iden
        # r1 = self.filter_t_supp
        # r = np.abs(t)
        # if r <= r0:
          # lc=1
        # elif r >= r1:
          # lc=0
        # else:
          # lc=np.exp(-r1/(r1 -r)) /(np.exp(-r1/(r-r0)) + np.exp(-r1/(r1-r)))
        # return self.filter_time(t)*lc
        
    def construct_kraus_coherent_freq(self):
        """Coherently construct Kraus operator in frequency domain."""
        H_mat = self.H_op
        A_mat = self.A_op
        E_H, psi_H = la.eigh(H_mat)
        A_ovlp = psi_H.conj().T @ (A_mat.dot(psi_H))
        Ns = self.Ns
        A_kraus = np.zeros((Ns, Ns))

        for i in range(Ns):
            for j in range(Ns):
                A_kraus += self.filter_freq(E_H[i]-E_H[j]) * A_ovlp[i,j] * \
                        np.outer(psi_H[:,i], psi_H[:,j].conj())
        
        self.A_kraus = A_kraus

    def construct_kraus_coherent_time(self, T_kraus, Nt_kraus):
        """
        Coherently construct Kraus operator in time domain using
        quadrature
        2Nt_kraus grid points (not include s=0)
        """
        H_mat = self.H_op
        A_mat = self.A_op
        E_H, psi_H = la.eigh(H_mat)
        Ns = self.Ns
        tgrid = np.linspace(-T_kraus, T_kraus, 2*int(Nt_kraus)+1)
        dt = tgrid[1]-tgrid[0]
        A_kraus = np.zeros((Ns, Ns),dtype=complex)
        for t in tgrid:            
            eHt = psi_H @ np.diag(np.exp(-1j*t*E_H)) @ psi_H.conj().T
            A_kraus += dt * self.filter_time(t) * eHt.conj().T @ A_mat.dot(eHt)
        eHt = psi_H @ np.diag(np.exp(-1j*T_kraus*E_H)) @ psi_H.conj().T    
        A_kraus=A_kraus-dt * self.filter_time(T_kraus) * eHt.conj().T @ A_mat.dot(eHt)/2
        eHt = psi_H @ np.diag(np.exp(-1j*(-T_kraus)*E_H)) @ psi_H.conj().T 
        A_kraus=A_kraus-dt * self.filter_time(-T_kraus) * eHt.conj().T @ A_mat.dot(eHt)/2
        self.A_kraus = A_kraus

    def time_contour(self, T_kraus, Nt_kraus, isreverse=True):
        """
        Construct the time contour for propagating the Kraus operator in
        time domain.
        2Nt_kraus+1 grid points (include s=0)
        """
        tau_kraus = T_kraus / Nt_kraus
        tgrid = np.zeros(2*Nt_kraus+1)
        tgrid = -T_kraus + np.arange(2*Nt_kraus+1) * tau_kraus  

        if isreverse:
            return np.append(tgrid,tgrid[::-1])# reverse
        else:
            return tgrid

    

    def step_kraus_coherent(self, psi0, eta, dice,it):
        """
        Propagate one step of the coherently formed dilated Kraus
        operator using randomized implementation
        """
        Ns = self.Ns
        psi_t = np.zeros((2*Ns), dtype=complex)
        psi_t.fill(0j)
        psi_t[:Ns] = psi0
        psi_t = np.exp(-1j*eta*self.E_K_dilate) * (self.V_K_dilate_dag @ psi_t)
        psi_t = self.V_K_dilate @ psi_t
        # decide whether to flip
        prob = la.norm(psi_t[Ns:])**2
        if( dice <= prob ):
            # flip the |1>| state
            psi = psi_t[Ns:]
        else:
            # keep the |0> state
            psi = psi_t[:Ns]

        # normalize
        psi = psi/la.norm(psi)

        return psi


    def step_kraus_coherent_matvec(self, H, U_A, Vh_A, psi0, eta, dice, it):
        """
        Propagate one step of the coherently formed dilated Kraus
        operator using the quatum jump method.

        This assumes K, K^dag have the low rank representation as given
        in lowrank_A_kraus. 

        This routine assumes the time step tau=eta^2
        """
    
        tau = eta**2
        Ns = self.Ns
        Kpsi = self.apply_lowrank_A(U_A, Vh_A, psi0)
        Kpsi_norm = la.norm(Kpsi,2)
        p_jump = Kpsi_norm**2 * tau 
        # Check for quantum jump
        if dice < p_jump:
            psi_new = Kpsi / Kpsi_norm
        else:
            Hpsi = H @ psi0
            psi_update = -1j * Hpsi - 0.5 * \
                    self.apply_lowrank_Adag(U_A, Vh_A, Kpsi)
            psi_new = psi0 + tau * psi_update
            psi_new = psi_new / la.norm(psi_new,2)

        return psi_new

        


    def evolve_kraus_coherent(self, T, num_t, psi0, num_rep, psi_GS=[],
            is_etarandom = False):
        # start from a pure state psi0, stochastically evolve the Lindblad
        # equation
        H = self.H_op
        tau = T / num_t
        ts = np.arange(num_t+1) * tau
        Ns = psi0.shape[0]
        avg_H_hist = np.zeros((num_t+1, num_rep))
        avg_H_hist[0,:].fill(np.vdot(psi0, H @ psi0).real)
        
        avg_pGS_hist = np.zeros((num_t+1, num_rep))
        if len(psi_GS) == 0:
            psi_GS = np.zeros_like(psi0)
        avg_pGS_hist[0,:].fill(np.abs(np.vdot(psi0, psi_GS))**2)
        eHt = la.expm(-1j*tau*H)
        K_dilate = np.zeros((2*Ns,2*Ns), dtype=complex)
        K_dilate[:Ns,Ns:] = self.A_kraus.conj().T
        K_dilate[Ns:,:Ns] = self.A_kraus
        self.E_K_dilate, self.V_K_dilate = la.eigh(K_dilate) 
        self.V_K_dilate_dag = self.V_K_dilate.conj().T
        if is_etarandom:
            # random propagation
            np.random.seed(seed=1)
            eta = np.random.randn(num_t,num_rep)*np.sqrt(tau)
        else:
            # deterministic propagation
            eta = np.ones((num_t,num_rep))*np.sqrt(tau)
        np.random.seed(seed=1)
        flip_dice = np.random.rand(num_t,num_rep)
        rho_hist   = np.zeros((Ns,Ns,num_t+1), dtype=complex)
        for ir in range(num_rep):
            # number of repetition
            if num_rep<10:
              print(f'repetition {ir}')
            psi   = np.zeros((Ns), dtype=complex)
            psi[:] = psi0
            rho_hist[:,:,0]+=np.outer(psi0,psi0.conj().T)
            for it in range(num_t):
                psi = eHt @ psi
                psi = self.step_kraus_coherent(psi, eta[it,ir],flip_dice[it,ir],it)
                rho_hist[:,:,it+1]+=np.outer(psi,psi.conj().T)
                # measurement
                avg_H_hist[it+1,ir] = np.vdot(psi, H @ psi).real
                avg_pGS_hist[it+1,ir] = np.abs(np.vdot(psi, psi_GS))**2
        rho_hist=rho_hist/num_rep            
        avg_H = np.mean(avg_H_hist,axis=1)
        avg_pGS = np.mean(avg_pGS_hist,axis=1)
        return ts, avg_H, avg_pGS, rho_hist
   

    def evolve_kraus_coherent_matvec(self, T, num_t, psi0, U_A, Vh_A, num_rep, psi_GS=[]):
        # start from a pure state psi0, stochastically evolve the Lindblad
        # equation
        H = self.H_op
        tau = T / num_t
        ts = np.arange(num_t+1) * tau
        Ns = psi0.shape[0]
        avg_H_hist = np.zeros((num_t+1, num_rep))
        avg_H_hist[0,:].fill(np.vdot(psi0, H @ psi0).real)
        
        avg_pGS_hist = np.zeros((num_t+1, num_rep))
        if len(psi_GS) == 0:
            psi_GS = np.zeros_like(psi0)
        avg_pGS_hist[0,:].fill(np.abs(np.vdot(psi0, psi_GS))**2)
        # deterministic propagation
        eta = np.ones((num_t,num_rep))*np.sqrt(tau)

        np.random.seed(seed=1)
        flip_dice = np.random.rand(num_t,num_rep)
        rho_hist   = np.zeros((Ns,Ns,num_t+1), dtype=complex)
        for ir in range(num_rep):
            # number of repetition
            if num_rep<10:
              print(f'repetition {ir}')
            psi   = np.zeros((Ns), dtype=complex)
            psi[:] = psi0
            rho_hist[:,:,0]+=np.outer(psi0,psi0.conj().T)
            for it in range(num_t):
                psi = self.step_kraus_coherent_matvec(H, U_A, Vh_A, psi, eta[it,ir],flip_dice[it,ir],it)
                rho_hist[:,:,it+1]+=np.outer(psi,psi.conj().T)
                # measurement
                avg_H_hist[it+1,ir] = np.vdot(psi, H @ psi).real
                avg_pGS_hist[it+1,ir] = np.abs(np.vdot(psi, psi_GS))**2
        rho_hist=rho_hist/num_rep            
        avg_H = np.mean(avg_H_hist,axis=1)
        avg_pGS = np.mean(avg_pGS_hist,axis=1)
        return ts, avg_H, avg_pGS, rho_hist


    def step_kraus_decoherent_batch(self, psi_batch, eta, num_segment, num_rep,
            T_kraus, Nt_kraus, intorder, dice, it):
        """
        Propagate one step of the dilated Kraus operator using
        randomized decoherent implementation with a second order
        propagator in a batch.
        """
        num_batch = psi_batch.shape[1]
        if not intorder in {1,2}:
            raise ValueError('intorder must be 1 or 2.')
        
        # first order method does not require reversing the grid
        isreverse = (intorder > 1);

        t_contour = self.time_contour(T_kraus, Nt_kraus, 
                isreverse = isreverse)

        Nt_contour = t_contour.shape[0]
        F_contour = np.zeros((Nt_contour), dtype=complex)
        VF_contour = np.zeros((Nt_contour,2,2), dtype=complex)
        tau_kraus = T_kraus / Nt_kraus
        eta_scal = eta / num_segment
        eHt = self.eHt
        eHT = self.eHT
        E_A=self.E_A 
        psi_A =self.psi_A
        Ns = self.Ns
        ZA_dilate = np.zeros((Nt_contour, 2*Ns, num_rep), dtype=complex)
        # t_start = time()
        for i in range(Nt_contour):
            if (t_contour[i]==np.min(t_contour)) or (t_contour[i]==np.max(t_contour)):
                F_contour[i] = self.filter_time(t_contour[i])/2
            else: 
                F_contour[i] = self.filter_time(t_contour[i])

            fac = np.exp(1j*np.angle(F_contour[i]))
            VF_contour[i,:,:] = 1.0/np.sqrt(2) * np.array([[1,-1], [fac,fac]])
            if intorder == 1:
                expZA = np.exp(-1j*tau_kraus*np.abs(F_contour[i])*np.outer(E_A,
                    eta_scal))
            else:
                expZA = np.exp(-1j*0.5*tau_kraus*np.abs(F_contour[i])*np.outer(E_A,
                    eta_scal))

            ZA_dilate[i,:Ns,:] = expZA
            ZA_dilate[i,Ns:,:] = expZA.conj()

        # t_end = time()
        # print('    Time for preparation   = ', t_end - t_start, ' s')
        # t_start = time()
        psi_t_batch = np.zeros((2*Ns, num_batch), dtype=complex)
        psi_t_batch.fill(0j)
        psi_t_batch[:Ns,:] = psi_batch
        for iseg in range(num_segment):
            if isreverse:
                for i in range(int(Nt_contour/2)):#left-ordered product
                    VK = np.kron(VF_contour[i,:,:], psi_A) 
                    psi_t_batch = VK.conj().T @ psi_t_batch  
                    # pointwise multiplication
                    psi_t_batch *= ZA_dilate[i,:,:]
                    psi_t_batch = VK @ psi_t_batch         
                    psi_t_batch=np.kron(np.identity(2),eHt) @ psi_t_batch
                for i in range(int(Nt_contour/2)):#right-ordered product
                    psi_t_batch=np.kron(np.identity(2),eHt.conj().T) @ psi_t_batch
                    VK = np.kron(VF_contour[i+int(Nt_contour/2),:,:], psi_A) 
                    psi_t_batch = VK.conj().T @ psi_t_batch  
                    # pointwise multiplication
                    psi_t_batch *= ZA_dilate[i+int(Nt_contour/2),:,:]
                    psi_t_batch = VK @ psi_t_batch                                               
            else:
                # only #left-ordered product
                for i in range(int(Nt_contour)):
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


        # t_end = time()
        # print('    Time for propagation = ', t_end - t_start, ' s')
        # decide whether to flip
        for ir in range(num_batch):
            prob = la.norm(psi_t_batch[Ns:,ir])**2
            if( dice[ir] <= prob ):
                # flip the |1>| state
                psi_batch[:,ir] = psi_t_batch[Ns:,ir]
            else:
                # keep the |0> state
                psi_batch[:,ir] = psi_t_batch[:Ns,ir]
    
            # normalize
            psi_batch[:,ir] /= la.norm(psi_batch[:,ir])

        return psi_batch


    def evolve_kraus_decoherent_batch(self, T, num_t, num_segment, psi0,
            num_rep, T_kraus, Nt_kraus, psi_GS = [], intorder=2,
            is_etarandom = False):
        """
        Accelerated Lindblad dynamics in a batch.

        This uses the deterministic propagation with first or second
        order Trotter.  In particular, the first order Trotter method
        enables propagation with positive time.
        """
        H = self.H_op
        tau = T / num_t
        ts = np.arange(num_t+1) * tau
        Ns = psi0.shape[0]
        time_H=np.zeros(num_t+1)
        avg_H_hist = np.zeros((num_t+1, num_rep))
        avg_H_hist[0,:].fill(np.vdot(psi0, H @ psi0).real)
        avg_pGS_hist = np.zeros((num_t+1, num_rep))
        if len(psi_GS) == 0:
            psi_GS = np.zeros_like(psi0)
        avg_pGS_hist[0,:].fill(np.abs(np.vdot(psi0, psi_GS))**2)
        eHtau = la.expm(-1j*tau*H)
        tau_kraus = T_kraus / Nt_kraus
        self.eHt = la.expm(-1j*tau_kraus*self.H_op)
        self.eHT = la.expm(-1j*T_kraus*self.H_op)
        self.E_A , self.psi_A = la.eigh(self.A_op)
        if is_etarandom:
            # random propagation
            np.random.seed(seed=1)
            eta = np.random.randn(num_t,num_rep)*np.sqrt(tau)
        else:
            # deterministic propagation
            eta = np.ones((num_t,num_rep))*np.sqrt(tau)

        # this randomness is introduced for modeling the tracing out
        # operation. Cannot be derandomized.
        np.random.seed(seed=1)
        flip_dice  = np.random.rand(num_t,num_rep)
        rho_hist   = np.zeros((Ns,Ns,num_t+1), dtype=complex)
        psi_all    = np.zeros((Ns,num_rep), dtype=complex)
        for i in range(num_rep):
            psi_all[:,i] = self.eHT.conj().T @ psi0.copy()
        rho_hist[:,:,0] = np.outer(psi_all[:,0],psi_all[:,0].conj().T)
        for it in range(num_t):
            psi_all = eHtau @ psi_all
            time_H[it+1]=time_H[it]+tau
            psi_all = self.step_kraus_decoherent_batch(psi_all, eta[it,:],
                    num_segment, num_rep, T_kraus, Nt_kraus, intorder,
                    flip_dice[it,:],it)
            rho_hist[:,:,it+1] = np.einsum('in,jn->ij',psi_all,
                    psi_all.conj()) / num_rep
            time_H[it+1]=time_H[it+1]+2*num_segment*T_kraus
            # measurement
            avg_H_hist[it+1,:] = np.einsum('in,in->n', psi_all.conj(),
                    H @ psi_all).real
            avg_pGS_hist[it+1,:] = np.abs( np.einsum('in,i->n', psi_all.conj(),
                    psi_GS) )**2
        avg_H = np.mean(avg_H_hist,axis=1)
        avg_pGS = np.mean(avg_pGS_hist,axis=1)
        return ts, avg_H, avg_pGS, time_H, rho_hist

    def apply_lowrank_A(self, U_A, Vh_A, psi):
        Ns_left = U_A.shape[1]
        Ns_right = Vh_A.shape[1]

        psi_reshape = psi.reshape([Ns_left, Ns_right])

        return np.einsum('ijk,kl,iml->jm', 
                U_A, psi_reshape, Vh_A).reshape(-1)

    def apply_lowrank_Adag(self, U_A, Vh_A, psi):
        Ns_left = U_A.shape[1]
        Ns_right = Vh_A.shape[1]

        psi_reshape = psi.reshape([Ns_left, Ns_right])

        return np.einsum('ikj,kl,ilm->jm', 
                U_A.conj(), psi_reshape, Vh_A.conj()).reshape(-1)


    def lowrank_A_kraus(self, A, Nk, l, tau):
        """
        Compute the low rank SVD of matrix A.

        This implements Alg. 5.1 of 

        Halko, N., Martinsson, P.-G., & Tropp, J. A. (2011). Finding
        Structure with Randomness: Probabilistic Algorithms for
        Constructing Approximate Matrix Decompositions. SIAM Review,
        53(2), 217–288.
        
        Parameters:
        - A: array_like, shape (Ns, Ns)
            The input matrix.
        - Nk: int
            should be larger than the bond dimension.
        - l: int
            bond cut at the l-th site. (0<l<L), where L is the total
            number of sites. Left: sites 0..l-1, right: l..L-1
        - tau: float
            singular value threshold 
        
        Returns:
        - U_A,  Vh_A: The approximate SVD of A.
        """
        Ns = A.shape[0]
        Ns_left = 2**l
        Ns_right = Ns//Ns_left
        assert(Ns_left * Ns_right == Ns)
        A_reshape = A.reshape([Ns_left, Ns_right, Ns_left,
            Ns_right]).transpose([0,2,1,3]).reshape([Ns_left**2,
                Ns_right**2])
        # Fully random. This works
        # Omega = np.random.randn(Ns_right**2, Nk) + \
                # 1j * np.random.randn(Ns_right**2, Nk)
        
        # Dicier: use Kronecker product form that allows us to use
        # matrix vector multiplication of A instead of A_reshape
        Omega = np.zeros((Ns_right*Ns_right, Nk), dtype=complex)
        for k in range(Nk):
            vec1 = np.random.randn(Ns_right) +  1j * np.random.randn(Ns_right)
            vec2 = np.random.randn(Ns_right) +  1j * np.random.randn(Ns_right)
            Omega[:, k] = np.kron(vec1, vec2)

        Y = A_reshape @ Omega

        Q, _ = la.qr(Y, mode='economic')

        B = Q.conj().T @ A_reshape
        U_B, S_B, Vh_B = la.svd(B, full_matrices=False)
        print("singular values = ", S_B)

        # truncate the singular values
        indices = np.where(S_B > tau)[0]
        Nr = len(indices)
        print("Number of significant singular values= ", Nr)
        U_B = U_B[:, indices]
        S_B = S_B[indices]
        Vh_B = Vh_B[indices,:]

        U = Q @ U_B
        
        # absorb the singular value into Vh
        #
        Vh_B = np.diag(S_B) @ Vh_B

        # FIXME move the test routine elsewhere
        # test 2-norm
        print("2-norm of error = ", 
                la.norm(U @ Vh_B - A_reshape, 2))

        U_A = U.reshape([Ns_left, Ns_left, Nr]).transpose([2,0,1])
        Vh_A = Vh_B.reshape([Nr, Ns_right, Ns_right])

        
        # test matrix vector multiplication
        psi = np.random.randn(Ns) + 1j*np.random.randn(Ns)
        psi = psi / la.norm(psi,2)
        A_psi = A @ psi
        A_lowrank_psi = self.apply_lowrank_A(U_A, Vh_A, psi)

        print("A psi matvec error = ", 
                la.norm(A_psi - A_lowrank_psi, 2))

        Adag_psi = A.conj().T @ psi
        Adag_lowrank_psi = self.apply_lowrank_Adag(U_A, Vh_A, psi)

        print("Adag psi matvec error = ", 
                la.norm(Adag_psi - Adag_lowrank_psi, 2))


        return U_A, Vh_A
        

if __name__ == "__main__":
    from quspin.operators import hamiltonian # Hamiltonians and operators
    from quspin.basis import spin_basis_1d
    import matplotlib.pyplot as plt
    from qutip import Qobj, mesolve
    
    plotting = True
    is_decoherent = True

    ##### define model parameters #####
    L=6 # system size
    J=1.0 # spin zz interaction
    g=1.2 # z magnetic field strength
    	
    ##### define spin model
    # site-coupling lists (PBC for both spin inversion sectors)
    h_field=[[-g,i] for i in range(L)]
    J_zz=[[-J,i,i+1] for i in range(L-1)] # no PBC
    # define spin static and dynamic lists
    static =[["zz",J_zz],["x",h_field]] # static part of H
    dynamic=[] # time-dependent part of H
    # construct spin basis in pos/neg spin inversion sector depending on APBC/PBC
    spin_basis = spin_basis_1d(L=L)
    # build spin Hamiltonians
    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    H=hamiltonian(static,dynamic,basis=spin_basis,dtype=np.float64,**no_checks)
    # calculate spin energy levels
    E_GS,psi_GS = H.eigsh(k=1,which="SA")
    psi_GS = psi_GS.flatten()
    print('E_GS = ', E_GS)
    
    H_mat = np.array(H.todense())
    E_H , psi_H = la.eigh(H_mat)
    gap = E_H[1] - E_H[0]
    print('gap = ', gap)
    print('||H|| = ', la.norm(H_mat,2))

    a = 8.0
    da = 1.0
    b  = gap
    db = gap
    
    filter_params = {'a': a, 'b': b, 'da': da, 'db': db} 

    T_kraus = 4.0 / np.min([da, db])
    Nt_kraus = 100
    # Nt_kraus = int(5/db/(2*np.pi/(4*a)))

    A = hamiltonian([ ['z',[[1.0,0]]] ],[],basis=spin_basis,dtype=np.float64,**no_checks)
    A_mat = np.array(A.todense())

    lb = Lindblad(H_mat, A_mat, filter_params)

    if plotting and False:
        tgrid = np.linspace(-T_kraus, T_kraus, 2000)
        # floc_tgrid = np.array([lb.filter_time_localized(t) for t in tgrid])
        f_tgrid    = np.array([lb.filter_time(t) for t in tgrid])
        # plt.plot(tgrid, floc_tgrid, 'b-o')
        plt.plot(tgrid, f_tgrid, 'r-.')
        plt.xlabel('t')
        plt.ylabel('f(t)')
        plt.show()

        wgrid = np.linspace(-2*lb.filter_a, 2*lb.filter_a, 100)

        # floc_wgrid = np.array([lb.filter_freq_localized(w) for w in wgrid])
        f_wgrid = np.array([lb.filter_freq(w) for w in wgrid])
        # plt.plot(wgrid, floc_wgrid.real, 'b-o')
        plt.plot(wgrid, f_wgrid.real, 'r-.')
        plt.xlabel('w')
        plt.ylabel(r'$\hat f(w)$')
        plt.show()
        
    # fix the random seed
    np.random.seed(seed=1)
    vt = np.random.randn(lb.Ns)
    # worst case in some sense: make psi0 and psi_GS orthogonal
    psi0 = vt.copy()
    psi0 -= psi_GS * np.vdot(psi_GS,psi0)
    psi0 = psi0 / la.norm(psi0)

#    psi0 = psi_GS.reshape(-1)
    print('|<psi0|psiGS>| = ', np.abs(np.vdot(psi_GS,psi0)))
    T = 100
    num_t = int(T)
    # FIXME. Consolidate this later
    num_segment = 3
    num_rep = 100
    times = np.arange(num_t+1) * (T/num_t)
    is_etarandom = False
    intorder = 2

    H_obj = Qobj(H_mat)
    zero_obj = Qobj(np.zeros_like(H_mat))

    rho_GS_obj = Qobj(np.outer(psi_GS, psi_GS.conj()))

    mode = 'mesolve'
    t_start = time()
    lb.construct_kraus_coherent_freq()

    # low rank representation
    U_A, Vconj_A = lb.lowrank_A_kraus(lb.A_kraus, 5, L//2, 1e-5)
    # stop 


    result = mesolve(H_obj, Qobj(psi0), times, [Qobj(lb.A_kraus)],
                [H_obj, rho_GS_obj])
    avg_H_e = result.expect[0]
    avg_pGS_e = result.expect[1]
    t_end = time()
    print('Frequency domain         = ', t_end - t_start, ' s')
    print('Overlap  = ', avg_pGS_e[-1])
    print('Energy   = ', avg_H_e[-1])

    mode = 'coherent'
    t_start = time()
    lb.construct_kraus_coherent_time(T_kraus, Nt_kraus)
    times, avg_H_c, avg_pGS_c, _= lb.evolve_kraus_coherent(T, num_t, psi0,
            num_rep, psi_GS, is_etarandom = is_etarandom)
    t_end = time()
    print('Time domain (coherent)   = ', t_end - t_start, ' s')
    print('Overlap  = ', avg_pGS_c[-1])
    print('Energy   = ', avg_H_c[-1])
   
    if is_decoherent:
        mode = 'decoherent'
        t_start = time()
        times, avg_H_de, avg_pGS_de, _, _= lb.evolve_kraus_decoherent_batch(T, num_t,
                num_segment, psi0, num_rep, T_kraus, Nt_kraus,
                psi_GS, intorder=intorder, is_etarandom = is_etarandom)
        t_end = time()
        print('Time domain (decoherent, batch) = ', t_end - t_start, ' s')
        print('Overlap  = ', avg_pGS_de[-1])
        print('Energy   = ', avg_H_de[-1])
    
    # mode = 'decoherent'
    # t_start = time()
    # times, avg_H_de, avg_pGS_de, _= lb.evolve_kraus_decoherent(T, num_t,
            # num_segment, psi0, num_rep, T_kraus, Nt_kraus,
            # psi_GS, intorder=2)
    # t_end = time()
    # print('Time domain (decoherent) = ', t_end - t_start, ' s')



    if plotting:
        plt.figure(figsize=(12,10))
        plt.plot(times, avg_H_e, 'g--', label='Lindblad', linewidth=3, markersize=10)
        plt.plot(times, avg_H_c, 'r-o', label='Coherent', linewidth=3, markersize=10)
        if is_decoherent:
            plt.plot(times, avg_H_de,'b-*', label='Decoherent', linewidth=3, markersize=10)
        plt.plot(times, np.ones_like(times)*E_GS, 'p-', label='exact', linewidth=3,markersize=10)
        plt.legend()
        plt.xlabel('t',fontsize=25)
        plt.ylabel('<E>',fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize=30)
        # plt.savefig('TFIM_6_energy.pdf')
        plt.show()
        
        plt.figure(figsize=(12,10))
        plt.plot(times, avg_pGS_e, 'g--', label='Lindblad', linewidth=3, markersize=10)
        plt.plot(times, avg_pGS_c, 'r-o', label='Coherent', linewidth=3, markersize=10)
        if is_decoherent:
            plt.plot(times, avg_pGS_de,'b-*', label='Decoherent', linewidth=3, markersize=10)
        plt.legend()
        plt.xlabel('t',fontsize=25)
        plt.ylabel('<p0>',fontsize=25)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.legend(fontsize=30)
        #plt.savefig('TFIM_6_overlap.pdf')
        plt.show()
