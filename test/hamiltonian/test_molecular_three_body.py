import numpy as np
from fermi_operators import construct_fermi_operators
import pytenet as ptn


def test_molecular_hamiltonian_three_body_mpo():

    rng = np.random.default_rng()
    for nsites in range(3, 8):
        
        tkin = ptn.crandn(2 * (nsites,), rng)
        vint = ptn.crandn(4 * (nsites,), rng)
        wint = ptn.crandn(6 * (nsites,), rng)
        
        h_ref = construct_three_body_molecular_hamiltonian(tkin, vint, wint)
        h_mpo = ptn.molecular_hamiltonian_three_body_mpo(tkin, vint, wint)
        assert np.allclose(h_mpo.to_matrix(), h_ref.todense()), \
            "matrix representation of MPO and reference Hamiltonian must match"

def construct_three_body_molecular_hamiltonian(tkin, vint, wint):
    nmodes = tkin.shape[0]
    assert tkin.shape == (nmodes, nmodes)
    assert vint.shape == (nmodes, nmodes, nmodes, nmodes)
    assert wint.shape == (nmodes, nmodes, nmodes, nmodes, nmodes, nmodes)

    clist, alist, _ = construct_fermi_operators(nmodes)
 ### check the third output of construct_fermi_operators 
    
    hamiltonian = \
        sum(tkin[i, j] * (clist[i] @ alist[j])
            for i in range(nmodes)
            for j in range(nmodes)) + \
        sum(0.5 * vint[i, j, k, l] * (clist[i] @ clist[j] @ alist[l] @ alist[k])
            for i in range(nmodes)
            for j in range(nmodes)
            for k in range(nmodes)
            for l in range(nmodes)) + \
        sum((1.0/6) * wint[i, j, k, l, m, n]
                * (clist[i] @ clist[j] @ clist[k] @ alist[n] @ alist[m] @ alist[l])
            for i in range(nmodes)
            for j in range(nmodes)
            for k in range(nmodes)
            for l in range(nmodes)
            for m in range(nmodes)
            for n in range(nmodes))
    hamiltonian.eliminate_zeros()
    return hamiltonian
