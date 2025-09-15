import numpy as np
import kwant
from numpy import kron

#pauli matrix
sigma_0 = np.array([[1, 0],[0, 1]])
sigma_x = np.array([[0, 1],[1, 0]])
sigma_y = np.array([[0, -1j],[1j, 0]])
sigma_z = np.array([[1, 0],[0, -1]])

tau_0 = sigma_0
tau_x = sigma_x
tau_y = sigma_y
tau_z = sigma_z

L = 300
delta = 1
tw = 12
ts = 6
mu_s = 0
Vs = 0
alpha = 0.8
T = 0.4



def BdG_conds_1D(Vw, mu_w, Vdis, Pairdis, t_lead = tw, mu_lead = 2*tw):
    sys = kwant.Builder()
    wire_lat = kwant.lattice.square(a=1,norbs = 4, name = 'wire')
    sc_lat = kwant.lattice.square(a=1,norbs = 4,name = 'sc')

    #wire onsite
    for i in range(L+1):
        sys[ wire_lat(i, 0) ] = (2 * tw - mu_w + Vdis[i]) * kron(tau_z, sigma_0) + Vw * kron(tau_0, sigma_x)


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
    for i in range(L+1):
        sys[ sc_lat(i,-1) ] = (2 * ts - mu_s) * kron(tau_z, sigma_0) + (delta + Pairdis[i]) * kron(tau_x, sigma_0) + Vs * kron(tau_0, sigma_x)

    sys[kwant.builder.HoppingKind((1,0),sc_lat)] = -ts * kron(tau_z, sigma_0) 

    #tunneling 
    sys[kwant.builder.HoppingKind((0,1),wire_lat,sc_lat)] = T * kron(tau_z,sigma_0)
    return sys

def local_cond(S):
    ree = np.sum(np.abs(S[:len(S)//2, :len(S)//2])**2)
    reh = np.sum(np.abs(S[:len(S)//2, len(S)//2:])**2)
    # print(np.shape(S)[0]//2,ree,reh,ree+reh,2-ree+reh)
    return (len(S)//2 - ree + reh)

def nonlocal_cond(S):
    ree = np.sum(np.abs(S[:len(S)//2, :len(S)//2])**2)
    reh = np.sum(np.abs(S[:len(S)//2, len(S)//2:])**2)
    # print(np.shape(S)[0]//2,ree,reh)
    return (ree - reh)
