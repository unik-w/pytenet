import itertools
import numpy as np
from ..mpo import MPO
from ..opchain import OpChain
from ..opgraph import OpGraph
from ..qnumber import encode_quantum_number_pair
from .molecular import MolecularOID, _molecular_hamiltonian_generate_operator_map
from .spin_molecular import (SpinMolecularOID, SpinOperatorConverter,
                             _spin_molecular_hamiltonian_generate_operator_map)

__all__ = ["molecular_hamiltonian_three_body_mpo", "spin_molecular_hamiltonian_three_body_mpo"]


def _antisymmetrize_2(vint):
    return 0.5 * (vint
                 - np.transpose(vint, (1, 0, 2, 3))
                 - np.transpose(vint, (0, 1, 3, 2))
                 + np.transpose(vint, (1, 0, 3, 2)))


def _antisymmetrize_creation_indices(wint):

    return  (
          wint
        - np.transpose(wint, (1, 0, 2, 3, 4, 5))
        - np.transpose(wint, (0, 2, 1, 3, 4, 5))
        - np.transpose(wint, (2, 1, 0, 3, 4, 5))
        + np.transpose(wint, (1, 2, 0, 3, 4, 5))
        + np.transpose(wint, (2, 0, 1, 3, 4, 5)))


def _antisymmetrize_annihilation_indices(wint):
  
    return (
          wint
        - np.transpose(wint, (0, 1, 2, 4, 3, 5))
        - np.transpose(wint, (0, 1, 2, 3, 5, 4))
        - np.transpose(wint, (0, 1, 2, 5, 4, 3))
        + np.transpose(wint, (0, 1, 2, 4, 5, 3))
        + np.transpose(wint, (0, 1, 2, 5, 3, 4)))


def _antisymmetrize_3(wint):
    return (1/6) * _antisymmetrize_annihilation_indices(_antisymmetrize_creation_indices(wint))


def _molecular_hamiltonian_opchain(oplist, coeff) -> OpChain:
    oplist = sorted(oplist)
    merged = []
    i = 0
    while i < len(oplist):
        site, oid = oplist[i]
        if i + 1 < len(oplist) and oplist[i + 1][0] == site:
            merged.append((site, MolecularOID.N))
            i += 2
        else:
            merged.append((site, oid))
            i += 1
    istart = merged[0][0]
    oids = []
    nops = 0
    prev_site = None
    for site, oid in merged:
        if prev_site is not None:
            filler = MolecularOID.Z if nops % 2 == 1 else MolecularOID.I
            for _ in range(site - prev_site - 1):
                oids.append(filler)
        oids.append(oid)
        nops += 2 if oid == MolecularOID.N else 1
        prev_site = site
    qnums = (len(oids) + 1) * [0]
    return OpChain(oids, qnums, coeff, istart)


def molecular_hamiltonian_three_body_mpo(tkin, vint, wint) -> MPO:

    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    wint = np.asarray(wint)
    nsites = tkin.shape[0]
    assert tkin.shape == (nsites, nsites)
    assert vint.shape == (nsites, nsites, nsites, nsites)
    assert wint.shape == 6 * (nsites,)

    gint = _antisymmetrize_2(vint)
    hint = _antisymmetrize_3(wint)

    nsites = tkin.shape[0]
    opchains = []

    for i in range(nsites):
        for j in range(nsites):
            opchains.append(_molecular_hamiltonian_opchain(
                [(i, MolecularOID.C), (j, MolecularOID.A)], tkin[i, j]))

    for i in range(nsites):
        for j in range(i + 1, nsites):
            for k in range(nsites):
                for l in range(k + 1, nsites):
                    opchains.append(_molecular_hamiltonian_opchain(
                        [(i, MolecularOID.C), (j, MolecularOID.C),
                         (l, MolecularOID.A), (k, MolecularOID.A)], gint[i, j, k, l]))

    for i in range(nsites):
        for j in range(i + 1, nsites):
            for k in range(j + 1, nsites):
                for l in range(nsites):
                    for m in range(l + 1, nsites):
                        for n in range(m + 1, nsites):
                            opchains.append(_molecular_hamiltonian_opchain(
                                [(i, MolecularOID.C), (j, MolecularOID.C), (k, MolecularOID.C),
                                 (n, MolecularOID.A), (m, MolecularOID.A), (l, MolecularOID.A)],
                                hint[i, j, k, l, m, n]))
    graph= OpGraph.from_opchains(opchains, nsites, MolecularOID.I)

   
    if nsites <= 10:
        assert graph.is_consistent()

    opmap = _molecular_hamiltonian_generate_operator_map()
    dtype = complex if (
        np.iscomplexobj(tkin) or np.iscomplexobj(vint) or np.iscomplexobj(wint)
    ) else float
    return MPO.from_opgraph([0, 0], graph, opmap, dtype)


def spin_molecular_hamiltonian_three_body_mpo(tkin, vint, wint) -> MPO:

    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    wint = np.asarray(wint)
    nsites = tkin.shape[0]
    assert tkin.shape == (nsites, nsites)
    assert vint.shape == (nsites, nsites, nsites, nsites)
    assert wint.shape == 6 * (nsites,)
    assert nsites >= 3

    # quantum numbers are disabled for now (see OpChain docstring), so use a trivial physical basis
    qsite = [0, 0, 0, 0]

    gint0 = 0.5 * (vint                             + np.transpose(vint, (1, 0, 3, 2)))
    gint1 = 0.5 * (np.transpose(vint, (1, 0, 2, 3)) + np.transpose(vint, (0, 1, 3, 2)))

    def get_vint_coeff(spatial_idx, spin_idx):
        valid = False
        coeff = 0
        if (spin_idx[0] == spin_idx[2]) and (spin_idx[1] == spin_idx[3]):
            coeff += gint0[spatial_idx]
            valid = True
        if (spin_idx[0] == spin_idx[3]) and (spin_idx[1] == spin_idx[2]):
            coeff -= gint1[spatial_idx]
            valid = True
        return coeff, valid

    hint = _antisymmetrize_3(wint)

    perm3 = list(itertools.permutations(range(3)))

    def _permutation_sign(perm):
        sign = 1
        for a in range(3):
            for b in range(a + 1, 3):
                if perm[a] > perm[b]:
                    sign = -sign
        return sign

    perm3_sign = {perm: _permutation_sign(perm) for perm in perm3}

    def get_wint_coeff(spatial_idx, spin_idx):
        i, j, k, l, m, n = spatial_idx
        sigma, tau, mu, nu, rho, pi_ = spin_idx
        ann_sites = (l, m, n)
        ann_spins = (nu, rho, pi_)
        coeff = 0
        valid = False

        for perm in perm3:
            if (sigma, tau, mu) == tuple(ann_spins[idx] for idx in perm):
                a, b, c = (ann_sites[idx] for idx in perm)
                coeff += perm3_sign[perm] * hint[i, j, k, a, b, c]
                valid = True
        return coeff, valid
    opchains = []
    # kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
    for i in range(2*nsites):
        for j in range(2*nsites):
            if (i - j) % 2 != 0:
                continue
            if i == j:
                opchains.append(SpinOperatorConverter.to_spin_opchain(
                    OpChain([MolecularOID.N], [0, 0], tkin[i//2, i//2], i)))
            else:
                (a, p), (b, q) = sorted([(i, MolecularOID.C), (j, MolecularOID.A)])
                opchains.append(SpinOperatorConverter.to_spin_opchain(
                    OpChain([p] + (b - a - 1)*[MolecularOID.Z] + [q],
                            (b - a + 2)*[0], tkin[i//2, j//2], a)))

    # interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau} v_{i,j,k,l}
    #   a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\tau} a_{k,\sigma}
    for i in range(2*nsites):
        for j in range(i + 1, 2*nsites):
            for k in range(2*nsites):
                for l in range(k + 1, 2*nsites):
                    coeff, valid = get_vint_coeff((i // 2, j // 2, k // 2, l // 2),
                                                  (i %  2, j %  2, k %  2, l %  2))
                    if not valid:
                        continue
                    (a, p), (b, q), (c, r), (d, s) = sorted([(i, MolecularOID.C),
                                                             (j, MolecularOID.C),
                                                             (l, MolecularOID.A),
                                                             (k, MolecularOID.A)])
                    if a == b:
                        assert b < c
                        if c == d:
                            oids  = (  [MolecularOID.N]
                                     + (c - b - 1)*[MolecularOID.I]
                                     + [MolecularOID.N])
                        else:
                            oids  = (  [MolecularOID.N]
                                     + (c - b - 1)*[MolecularOID.I]
                                     + [r]
                                     + (d - c - 1)*[MolecularOID.Z]
                                     + [s])
                    elif b == c:
                        oids  = (  [p]
                                 + (b - a - 1)*[MolecularOID.Z]
                                 + [MolecularOID.N]
                                 + (d - c - 1)*[MolecularOID.Z]
                                 + [s])
                    elif c == d:
                        oids  = (  [p]
                                 + (b - a - 1)*[MolecularOID.Z]
                                 + [q]
                                 + (c - b - 1)*[MolecularOID.I]
                                 + [MolecularOID.N])
                    else:
                        oids  = (  [p]
                                 + (b - a - 1)*[MolecularOID.Z]
                                 + [q]
                                 + (c - b - 1)*[MolecularOID.I]
                                 + [r]
                                 + (d - c - 1)*[MolecularOID.Z]
                                 + [s])
                    # quantum numbers disabled for now (see OpChain docstring)
                    qnums = (len(oids) + 1)*[0]
                    opchains.append(SpinOperatorConverter.to_spin_opchain(
                        OpChain(oids, qnums, coeff, a)))

    # three-body interaction terms 1/6 \sum v_{ijklmn} a^{\dagger}_i a^{\dagger}_j a^{\dagger}_k a_n a_m a_l
    # (spin-orbital indices, creation sites i<j<k, annihilation sites l<m<n);
    # bond quantum numbers are set to zero to disable them for now (see OpChain docstring)
    for i in range(2*nsites):
        for j in range(i + 1, 2*nsites):
            for k in range(j + 1, 2*nsites):
                for l in range(2*nsites):
                    for m in range(l + 1, 2*nsites):
                        for n in range(m + 1, 2*nsites):
                            coeff, valid = get_wint_coeff(
                                (i // 2, j // 2, k // 2, l // 2, m // 2, n // 2),
                                (i %  2, j %  2, k %  2, l %  2, m %  2, n %  2))




                            
                            pass
    graph = OpGraph.from_opchains(opchains, nsites, SpinMolecularOID.I)

    if nsites <= 10:
        assert graph.is_consistent()
    opmap = _spin_molecular_hamiltonian_generate_operator_map()
    dtype = complex if (
        np.iscomplexobj(tkin) or np.iscomplexobj(vint) or np.iscomplexobj(wint)
    ) else float
    return MPO.from_opgraph(qsite, graph, opmap, dtype)


## LATER MANUAL CONSTRUCTION
# def spin_molecular_hamiltonian_three_body_mpo(tkin, vint, wint) -> MPO:
   
#     tkin = np.asarray(tkin)
#     vint = np.asarray(vint)
#     wint = np.asarray(wint)
#     nsites = tkin.shape[0]
#     assert tkin.shape == (nsites, nsites)
#     assert vint.shape == (nsites, nsites, nsites, nsites)
#     assert wint.shape == 6 * (nsites,)
#     assert nsites >= 3

#     # physical particle number and spin quantum numbers (encoded as single integer)
#     qn = [0,  1,  1,  2]
#     qs = [0, -1,  1,  0]
#     qsite = [encode_quantum_number_pair(q[0], q[1]) for q in zip(qn, qs)]

    
#     gint0 = 0.5 * (vint                             + np.transpose(vint, (1, 0, 3, 2)))
#     gint1 = 0.5 * (np.transpose(vint, (1, 0, 2, 3)) + np.transpose(vint, (0, 1, 3, 2)))

#     def get_vint_coeff(spatial_idx, spin_idx):
#         valid = False
#         coeff = 0
#         if (spin_idx[0] == spin_idx[2]) and (spin_idx[1] == spin_idx[3]):
#             coeff += gint0[spatial_idx]
#             valid = True
#         if (spin_idx[0] == spin_idx[3]) and (spin_idx[1] == spin_idx[2]):
#             coeff -= gint1[spatial_idx]
#             valid = True
#         return coeff, valid

   
#     hint = _antisymmetrize_3(wint)

#     perm3 = list(itertools.permutations(range(3)))

#     def _permutation_sign(perm):
#         sign = 1
#         for a in range(3):
#             for b in range(a + 1, 3):
#                 if perm[a] > perm[b]:
#                     sign = -sign
#         return sign

#     perm3_sign = {perm: _permutation_sign(perm) for perm in perm3}

#     def get_wint_coeff(spatial_idx, spin_idx):
#         i, j, k, l, m, n = spatial_idx
#         sigma, tau, mu, nu, rho, pi_ = spin_idx
#         ann_sites = (l, m, n)
#         ann_spins = (nu, rho, pi_)
#         coeff = 0
#         valid = False
       
#         for perm in perm3:
#             if (sigma, tau, mu) == tuple(ann_spins[idx] for idx in perm):
#                 a, b, c = (ann_sites[idx] for idx in perm)
#                 coeff += perm3_sign[perm] * hint[i, j, k, a, b, c]
#                 valid = True
#         return coeff, valid

#     nodes = SpinMolecularOpGraphNodes(nsites)
#     graph = nodes.generate_graph()

#     # kinetic hopping terms \sum_{i,j,\sigma} t_{i,j} a^{\dagger}_{i,\sigma} a_{j,\sigma}
#     for i in range(nsites):
#         for j in range(nsites):
#             for sigma in (0, 1):
#                 _spin_molecular_hamiltonian_graph_add_term(
#                     graph, nodes, [(i, sigma, MolecularOID.C),
#                                    (j, sigma, MolecularOID.A)], tkin[i, j])

#     # interaction terms 1/2 \sum_{i,j,k,l,\sigma,\tau} v_{i,j,k,l}
#     #   a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} a_{l,\tau} a_{k,\sigma}
#     for i, sigma in itertools.product(range(nsites), (0, 1)):
#         for j, tau in itertools.product(range(i, nsites), (0, 1)):
#             if (i, sigma) >= (j, tau):
#                 continue
#             for k, mu in itertools.product(range(nsites), (0, 1)):
#                 for l, nu in itertools.product(range(k, nsites), (0, 1)):
#                     if (k, mu) >= (l, nu):
#                         continue
#                     coeff, valid = get_vint_coeff((i, j, k, l), (sigma, tau, mu, nu))
#                     if not valid:
#                         continue
#                     oplist = [(i, sigma, MolecularOID.C),
#                               (j, tau,   MolecularOID.C),
#                               (l, nu,    MolecularOID.A),
#                               (k, mu,    MolecularOID.A)]
#                     _spin_molecular_hamiltonian_graph_add_term(
#                         graph, nodes, oplist, coeff)

#     for i, sigma in itertools.product(range(nsites), (0, 1)):
#         for j, tau in itertools.product(range(i, nsites), (0, 1)):
#             if (i, sigma) >= (j, tau):
#                 continue
#             for k, mu in itertools.product(range(j, nsites), (0, 1)):
#                 if (j, tau) >= (k, mu):
#                     continue
#                 for l, nu in itertools.product(range(nsites), (0, 1)):
#                     for m, rho in itertools.product(range(l, nsites), (0, 1)):
#                         if (l, nu) >= (m, rho):
#                             continue
#                         for n, pi_ in itertools.product(range(m, nsites), (0, 1)):
#                             if (m, rho) >= (n, pi_):
#                                 continue
#                             coeff, valid = get_wint_coeff(
#                                 (i, j, k, l, m, n), (sigma, tau, mu, nu, rho, pi_))
#                             if not valid:
#                                 continue
#                             oplist = [(i, sigma, MolecularOID.C),
#                                       (j, tau,   MolecularOID.C),
#                                       (k, mu,    MolecularOID.C),
#                                       (n, pi_,   MolecularOID.A),
#                                       (m, rho,   MolecularOID.A),
#                                       (l, nu,    MolecularOID.A)]
#                             #### UNIMPLEMENTED: node list doesnt support 6 op terms
#                             _spin_molecular_hamiltonian_graph_add_term(
#                                 graph, nodes, oplist, coeff)

#     if nsites <= 10:
#         assert graph.is_consistent()
#     opmap = _spin_molecular_hamiltonian_generate_operator_map()
#     dtype = complex if (
#         np.iscomplexobj(tkin) or np.iscomplexobj(vint) or np.iscomplexobj(wint)
#     ) else float
#     mpo = MPO.from_opgraph(qsite, graph, opmap, dtype, compute_nid_map=True)
#     nodes.copy_nids(mpo)

#     return mpo
