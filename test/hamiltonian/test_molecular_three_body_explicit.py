import itertools
import numpy as np
from fermi_operators import construct_fermi_operators
import pytenet as ptn


def test_molecular_hamiltonian_three_body_mpo_explicit():

    rng = np.random.default_rng()
    for nsites in range(3, 8):

        tkin = ptn.crandn(2 * (nsites,), rng)
        vint = ptn.crandn(4 * (nsites,), rng)
        wint = ptn.crandn(6 * (nsites,), rng)

        h_ref = construct_three_body_molecular_hamiltonian(tkin, vint, wint)
        h_mpo = ptn.molecular_hamiltonian_three_body_mpo_explicit(tkin, vint, wint)
        assert np.allclose(h_mpo.to_matrix(), h_ref.todense()), \
            "matrix representation of MPO and reference Hamiltonian must match"

def construct_three_body_molecular_hamiltonian(tkin, vint, wint):
    nmodes = tkin.shape[0]
    assert tkin.shape == (nmodes, nmodes)
    assert vint.shape == (nmodes, nmodes, nmodes, nmodes)
    assert wint.shape == (nmodes, nmodes, nmodes, nmodes, nmodes, nmodes)

    clist, alist, _ = construct_fermi_operators(nmodes)

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


def test_molecular_hamiltonian_three_body_mpo_explicit_single_tensor_entries():
    failures = []
    nsites = 5
    rng = np.random.default_rng(42)
    zeros2 = np.zeros((nsites, nsites))
    zeros4 = np.zeros((nsites,) * 4)
    clist, alist, _ = construct_fermi_operators(nsites)
    triples = list(itertools.combinations(range(nsites), 3))
    entries = list(itertools.product(triples, repeat=2))
    sample_indices = rng.choice(len(entries), size=16, replace=False)
    report = []

    for sample_index in sample_indices:
        creators, annihilators = entries[sample_index]
        i, j, k = creators
        l, m, n = annihilators
        indices = (i, j, k, l, m, n)
        wint = np.zeros((nsites,) * 6)
        wint[indices] = 1
        h_mpo = ptn.molecular_hamiltonian_three_body_mpo_explicit(
            zeros2, zeros4, wint)
        h_ref = (1 / 6) * (clist[i] @ clist[j] @ clist[k]
                            @ alist[n] @ alist[m] @ alist[l])
        actual = h_mpo.to_matrix()
        expected = h_ref.todense()
        if not np.allclose(actual, expected):
            max_error = np.max(np.abs(actual - expected))
            entry_report = f"indices={indices}: MISMATCH (max_error={max_error})"
            failures.append(entry_report)
        else:
            entry_report = f"indices={indices}: OK"
        report.append(entry_report)

    print("\n".join(report))

    assert not failures, "Single-entry 3-body mismatches:\n" + "\n".join(failures)
