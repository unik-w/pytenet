import os
import sys
import numpy as np
# ensure that 'test_molecular_three_body' is found when calling this test from a parent folder
sys.path.append(os.path.dirname(__file__))
from test_molecular_three_body import construct_three_body_molecular_hamiltonian
import pytenet as ptn


def test_spin_molecular_hamiltonian_three_body_mpo():

    rng = np.random.default_rng()
    for nsites in range(3, 5):

        tkin = ptn.crandn(2 * (nsites,), rng)
        vint = ptn.crandn(4 * (nsites,), rng)
        wint = ptn.crandn(6 * (nsites,), rng)

        h_ref = construct_spin_molecular_hamiltonian_three_body(tkin, vint, wint)
        h_mpo = ptn.spin_molecular_hamiltonian_three_body_mpo(tkin, vint, wint)
        assert np.allclose(h_mpo.to_matrix(), h_ref.todense()), \
            "matrix representation of MPO and reference Hamiltonian must match"


def construct_spin_molecular_hamiltonian_three_body(tkin, vint, wint):
    """
    Construct a three-body molecular Hamiltonian for a spin orbital basis as sparse matrix.
    """
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    wint = np.asarray(wint)

    nsites = tkin.shape[0]
    assert tkin.shape == (nsites, nsites)
    assert vint.shape == 4 * (nsites,)
    assert wint.shape == 6 * (nsites,)

    # enlarge the single-, two- and three-particle integral tensors
    # from an orbital basis without spin to a spin orbital basis

    # single-particle integrals
    tkin_spin = np.kron(tkin, np.identity(2))

    # two-particle integrals
    tmp = np.zeros((2*nsites, nsites, 2*nsites, nsites), dtype=vint.dtype)
    for i in range(nsites):
        for j in range(nsites):
            tmp[:, i, :, j] = np.kron(vint[:, i, :, j], np.identity(2))
    vint_spin = np.zeros((2*nsites, 2*nsites, 2*nsites, 2*nsites), dtype=vint.dtype)
    for i in range(2*nsites):
        for j in range(2*nsites):
            vint_spin[i, :, j, :] = np.kron(tmp[i, :, j, :], np.identity(2))

    # three-particle integrals: pair creation index i with annihilation index l,
    # j with m and k with n (matching spin for each pair)
    tmp1 = np.zeros((2*nsites, nsites, nsites, 2*nsites, nsites, nsites), dtype=wint.dtype)
    for j in range(nsites):
        for k in range(nsites):
            for m in range(nsites):
                for n in range(nsites):
                    tmp1[:, j, k, :, m, n] = np.kron(wint[:, j, k, :, m, n], np.identity(2))
    tmp2 = np.zeros((2*nsites, 2*nsites, nsites, 2*nsites, 2*nsites, nsites), dtype=wint.dtype)
    for i in range(2*nsites):
        for k in range(nsites):
            for l in range(2*nsites):
                for n in range(nsites):
                    tmp2[i, :, k, l, :, n] = np.kron(tmp1[i, :, k, l, :, n], np.identity(2))
    wint_spin = np.zeros(6 * (2*nsites,), dtype=wint.dtype)
    for i in range(2*nsites):
        for j in range(2*nsites):
            for l in range(2*nsites):
                for m in range(2*nsites):
                    wint_spin[i, j, :, l, m, :] = np.kron(tmp2[i, j, :, l, m, :], np.identity(2))

    return construct_three_body_molecular_hamiltonian(tkin_spin, vint_spin, wint_spin)
