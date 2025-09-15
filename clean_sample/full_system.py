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

def make_system_1D(L=600, delta = 1, tw = 12, ts = 10, mu_w = 0, mu_s = 4, 
                   Zw = 1.5, Zs = 0, alpha =1.5, T = 1.5, BdG=False, PBC=False, Addleads=False):
    """
    Create a 1D quantum system with a wire (W) and a superconductor (SC).

    Parameters:

        L (int): Length of the wire and superconductor in x direction.
        delta (float): Pairing potential in the superconductor.
        tw_x, ts_x (float): Hopping strengths in the wire and superconductor.
        mu_w, mu_s (float): Chemical potentials for the wire and superconductor.
        Zw, Zs (float): Zeeman splitting strengths in the wire and superconductor.
        alpha_w (float): Spin-orbit coupling strength in the wire.
        T (float): Coupling strength between wire and superconductor.
        
        BdG (bool): Use Bogoliubov-de Gennes (BdG) formalism if True.
        PBC (bool): Apply periodic boundary conditions along the x-direction if True.
        Addleads (bool): Include leads in the system if True.

    Notes:
        All default values are taken from:
        "Dressed Majorana Fermion in a Hybrid Nanowire" Phys. Rev. Lett. 133, 266605 (2024)
        DOI: https://doi.org/10.1103/PhysRevLett.133.266605
        
    Returns:
        kwant.Builder: The defined quantum system.
    """

    if PBC:
        sys = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
    else:
        sys = kwant.Builder()

    if PBC and Addleads:
        raise ValueError(" Cannot add leads when PBC(periodic boundary condition) is enabled.")

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
            return (2 * tw - mu_w) * kron(tau_z, sigma_0) + Zw * kron(tau_0, sigma_x)
        return (2* tw - mu_w) * sigma_0 + Zw * sigma_x

    def W_hop_x(site1, site2):
        if BdG:
            return -tw * kron(tau_z, sigma_0) + 1j * alpha / 2 * kron(tau_z, sigma_y)
        return -tw * sigma_0 - 1j * alpha * sigma_y

    
    # Superconductor Hamiltonian terms
    def SC_onsite(site):
        if BdG:
            return (2 * ts - mu_s) * kron(tau_z, sigma_0) + delta * kron(tau_x, sigma_0) + Zs * kron(tau_0, sigma_x)
        return (2 * ts - mu_s) * sigma_0 + Zs * sigma_x

    def SC_hop_x(site1, site2):
        return -ts * (kron(tau_z, sigma_0) if BdG else sigma_0)

    '''add onsite and hopping'''
    # Add sites and hoppings for superconductor
    sys[lat_s.shape(SC_region, (0, -1))] = SC_onsite
    sys[kwant.builder.HoppingKind((1, 0), lat_s, lat_s)] = SC_hop_x

    # Add sites and hoppings for wire
    sys[lat_w.shape(W_region, (0, 0))] = W_onsite
    sys[kwant.builder.HoppingKind((1, 0), lat_w, lat_w)] = W_hop_x

    # Add coupling between wire and superconductor
    coupling_term = T * (kron(tau_z, sigma_0) if BdG else sigma_0)
    sys[kwant.builder.HoppingKind((0, 1), lat_w, lat_s)] = coupling_term

    
    '''add leads'''
    if Addleads:
        lead_left = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
        lead_right = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))

        t_lead = tw
        mu_lead = 2 * tw
        
        def lead_W_region(pos):
            x, y = pos
            return y == 0

        lead_onsite = (t_lead - mu_lead) * (kron(tau_z, sigma_0) if BdG else sigma_0)
        lead_hop = -t_lead * (kron(tau_z, sigma_0) if BdG else sigma_0)

        # Add sites and hoppings to the lead
        lead_left[lat_w.shape(lead_W_region, (0, 0))] = lead_onsite
        lead_left[kwant.builder.HoppingKind((1, 0), lat_w, lat_w)] = lead_hop

        lead_right[lat_w.shape(lead_W_region, (0, 0))] = lead_onsite
        lead_right[kwant.builder.HoppingKind((1, 0), lat_w, lat_w)] = lead_hop

        # Attach leads to the system
        sys.attach_lead(lead_left)
        sys.attach_lead(lead_right)
    return sys




