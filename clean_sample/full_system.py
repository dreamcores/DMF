import numpy as np
import kwant
from numpy import sqrt,pi,cos,sin,kron
from scipy.sparse.linalg import eigsh


#pauli matrix
sigma_0 = np.array([[1, 0],[0, 1]])
sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])

tau_0 = sigma_0
tau_x = sigma_x
tau_y = sigma_y
tau_z = sigma_z


def sys_bands(params):
    
    '''params: dictionary of parameters'''
    L = params["L"]
    alpha = params["alpha"]
    Bw = params["Bw"]
    mu_w = params["mu_w"]
    tw = params["tw"]

    Bs = params["Bs"]
    mu_s = params["mu_s"]
    ts = params["ts"]
    delta = params["delta"]
    T = params["T"]
    
    BdG = params["BdG"]
    PBC = params["PBC"]

    if PBC:
        sys = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
    else:
        sys = kwant.Builder()

    # Define lattice based on BdG formalism
    norbs = 4 if BdG else 2
    lat_w = kwant.lattice.square(a=1, norbs=norbs, name='W')
    lat_s = kwant.lattice.square(a=1, norbs=norbs, name='SC')

    def W_region(pos):
        x, y = pos
        return 0 <= x <= L and y == 0
    
    def SC_region(pos):
        x, y = pos
        return 0 <= x <= L and y == -1
    
    '''define hamiltonian'''
    # Wire Hamiltonian terms
    def W_onsite(site):
        if BdG:
            return (2 * tw - mu_w) * kron(tau_z, sigma_0) + Bw * kron(tau_0, sigma_x)
        return (2 * tw - mu_w) * sigma_0 + Bw * sigma_x
    def W_hop_x(site1, site2):
        if BdG:
            return -tw * kron(tau_z, sigma_0) + 1j * alpha / 2 * kron(tau_z, sigma_y)
        return -tw * sigma_0 + 1j * alpha / 2 * sigma_y
    
    # Superconductor Hamiltonian terms
    def SC_onsite(site):
        if BdG:
            return (2 * ts - mu_s) * kron(tau_z, sigma_0) + delta * kron(tau_x, sigma_0) + Bs * kron(tau_0, sigma_x)
        return (2 * ts - mu_s) * sigma_0 + Bs * sigma_x
    def SC_hop_x(site1, site2):
        return -ts * (kron(tau_z, sigma_0) if BdG else sigma_0)
    

    sys[lat_s.shape(SC_region, (0, -1))] = SC_onsite
    sys[kwant.builder.HoppingKind((1, 0), lat_s, lat_s)] = SC_hop_x

    sys[lat_w.shape(W_region, (0, 0))] = W_onsite
    sys[kwant.builder.HoppingKind((1, 0), lat_w, lat_w)] = W_hop_x

    # Add coupling between wire and superconductor
    coupling_term = T * (kron(tau_z, sigma_0) if BdG else sigma_0)
    sys[kwant.builder.HoppingKind((0, 1), lat_w, lat_s)] = coupling_term
    return sys


def sys_conds(params):
    '''params: dictionary of parameters'''
    L = params["L"]
    alpha = params["alpha"]
    Bw = params["Bw"]
    mu_w = params["mu_w"]
    tw = params["tw"]

    Bs = params["Bs"]
    mu_s = params["mu_s"]
    ts = params["ts"]
    delta = params["delta"]
    T = params["T"]
    
    t_lead = tw
    mu_lead = 2*tw

    sys = kwant.Builder()
    wire_lat = kwant.lattice.square(a=1,norbs = 4, name = 'wire')
    sc_lat = kwant.lattice.square(a=1,norbs = 4, name = 'sc')

    #wire onsite
    sys[ (wire_lat(i,0) for i in range(L+1))] = (2 * tw - mu_w) * kron(tau_z, sigma_0) + Bw * kron(tau_0, sigma_x)

    #wire hopping
    sys[kwant.builder.HoppingKind((1, 0), wire_lat)] = -tw * kron(tau_z, sigma_0) + 1j * alpha / 2 * kron(tau_z, sigma_y)


    #left lead
    left = kwant.Builder(kwant.TranslationalSymmetry((-1,0)))
    left[wire_lat(0, 0)] =  (2 * t_lead - mu_lead) * kron(tau_z,sigma_0)
    left[kwant.builder.HoppingKind((-1, 0), wire_lat)] = -t_lead * kron(tau_z,sigma_0)
    sys.attach_lead(left)

    #rignt lead
    rignt = kwant.Builder(kwant.TranslationalSymmetry((1,0)))
    rignt[wire_lat(L, 0)] =  (2 * t_lead - mu_lead) * kron(tau_z,sigma_0)
    rignt[kwant.builder.HoppingKind((-1, 0), wire_lat)] = -t_lead * kron(tau_z,sigma_0)
    sys.attach_lead(rignt)

    #sc
    sys[ (sc_lat(i,-1) for i in range(L+1))] = (2 * ts - mu_s) * kron(tau_z, sigma_0) + delta * kron(tau_x, sigma_0) + Bs * kron(tau_0, sigma_x)
    sys[kwant.builder.HoppingKind((1,0),sc_lat)] = -ts * kron(tau_z, sigma_0) 

    #tunneling 
    sys[kwant.builder.HoppingKind((0,1),wire_lat,sc_lat)] = T * kron(tau_z,sigma_0)
    return sys


def local_cond(S):
    ree = np.sum(np.abs(S[:len(S)//2, :len(S)//2])**2)
    reh = np.sum(np.abs(S[:len(S)//2, len(S)//2:])**2)
    return (len(S)//2 - ree + reh)


def nonlocal_cond(S):
    ree = np.sum(np.abs(S[:len(S)//2, :len(S)//2])**2)
    reh = np.sum(np.abs(S[:len(S)//2, len(S)//2:])**2)
    return (ree - reh)


def cal_conds(sys,energys):
    smatrix = kwant.smatrix(sys,energy=energys,check_hermiticity=True)
    GLL = local_cond(np.array(smatrix.submatrix(0,0)))
    GLR = nonlocal_cond(np.array(smatrix.submatrix(0,1)))
    return GLL, GLR




