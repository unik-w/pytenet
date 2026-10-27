import sys
import numpy as np
sys.path.insert(0, ".")
from fermi_operators import construct_fermi_operators
import pytenet as ptn
from test_molecular_three_body_explicit import construct_three_body_molecular_hamiltonian


def run(label, tkin_fn, vint_fn, wint_fn):
    rng = np.random.default_rng(42)
    print(f"--- {label} ---")
    for nsites in range(3, 7):
        tkin = tkin_fn(nsites, rng)
        vint = vint_fn(nsites, rng)
        wint = wint_fn(nsites, rng)
        try:
            h_ref = construct_three_body_molecular_hamiltonian(tkin, vint, wint)
            h_mpo = ptn.molecular_hamiltonian_three_body_mpo_explicit(tkin, vint, wint)
            ok = np.allclose(h_mpo.to_matrix(), h_ref.todense())
            print(f"nsites={nsites}: {'OK' if ok else 'MISMATCH'}")
        except Exception as e:
            print(f"nsites={nsites}: EXCEPTION {type(e).__name__}: {e}")


zeros2 = lambda n, rng: np.zeros((n, n))
zeros4 = lambda n, rng: np.zeros(4 * (n,))
zeros6 = lambda n, rng: np.zeros(6 * (n,))
rand2 = lambda n, rng: ptn.crandn(2 * (n,), rng)
rand4 = lambda n, rng: ptn.crandn(4 * (n,), rng)
rand6 = lambda n, rng: ptn.crandn(6 * (n,), rng)

run("a) only kinetic/1-body", rand2, zeros4, zeros6)
run("b) only 2-body", zeros2, rand4, zeros6)
run("c) only 3-body", zeros2, zeros4, rand6)
