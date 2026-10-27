"""
Operator graph nodes for a molecular Hamiltonian including three-body terms,
assuming a spin orbital basis.
"""

import itertools
from ..opgraph import OpGraphNode
from ..qnumber import encode_quantum_number_pair

__all__ = ["SpinMolecularThreeBodyOpGraphNodes"]


class SpinMolecularThreeBodyOpGraphNodes:

    def __init__(self, nsites):

        self.nsites = nsites
        half_site_count = nsites // 2

        nid_next = 0

        self.identity_l = {}
        self.identity_r = {}
        for i in range(nsites):
            self.identity_l[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1
        for i in range(1, nsites + 1):
            self.identity_r[i] = OpGraphNode(nid_next, [], [], 0)
            nid_next += 1

        ###  SINGLE OPERATORS L and R
        # a^{\dagger}_{i,\sigma} operators connected to left terminal
        self.a_dag_l = {}
        for i, sigma in itertools.product(range(nsites - 2), (0, 1)):
            self.a_dag_l[i, sigma] = {}
            for j in range(i + 1, nsites - 1):
                self.a_dag_l[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair( 1, [1, -1][sigma]))
                nid_next += 1
        # a_{i,\sigma} operators connected to left terminal
        self.a_ann_l = {}
        for i, sigma in itertools.product(range(nsites - 2), (0, 1)):
            self.a_ann_l[i, sigma] = {}
            for j in range(i + 1, nsites - 1):
                self.a_ann_l[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair(-1, [-1, 1][sigma]))
                nid_next += 1
        # a^{\dagger}_{i,\sigma} operators connected to right terminal
        self.a_dag_r = {}
        for i, sigma in itertools.product(range(2, nsites), (0, 1)):
            self.a_dag_r[i, sigma] = {}
            for j in range(2, i + 1):
                self.a_dag_r[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair(-1, [-1, 1][sigma]))
                nid_next += 1
        # a_{i,\sigma} operators connected to right terminal
        self.a_ann_r = {}
        for i, sigma in itertools.product(range(2, nsites), (0, 1)):
            self.a_ann_r[i, sigma] = {}
            for j in range(2, i + 1):
                self.a_ann_r[i, sigma][j] = OpGraphNode(
                    nid_next, [], [], encode_quantum_number_pair( 1, [1, -1][sigma]))
                nid_next += 1

        ###  Double operators L and R

        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to left terminal
        self.a_dag_a_dag_l = {}
        for i, sigma in itertools.product(range(nsites - 2), (0, 1)):
            for j, tau in itertools.product(range(i, nsites - 1), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                self.a_dag_a_dag_l[i, sigma, j, tau] = {}
                for k in range(j + 1, nsites):
                    self.a_dag_a_dag_l[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair( 2, [1, -1][sigma] + [1, -1][tau]))
                    nid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to left terminal
        self.a_ann_a_ann_l = {}
        for i, sigma in itertools.product(range(nsites - 2), (0, 1)):
            for j, tau in itertools.product(range(i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                self.a_ann_a_ann_l[i, sigma, j, tau] = {}
                for k in range(i + 1, nsites):
                    self.a_ann_a_ann_l[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair(-2, [-1, 1][sigma] + [-1, 1][tau]))
                    nid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to left terminal
        self.a_dag_a_ann_l = {}
        for i, sigma in itertools.product(range(nsites - 2), (0, 1)):
            for j, tau in itertools.product(range(nsites - 2), (0, 1)):
                self.a_dag_a_ann_l[i, sigma, j, tau] = {}
                for k in range(max(i, j) + 1, nsites):
                    self.a_dag_a_ann_l[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair(0, [1, -1][sigma] + [-1, 1][tau]))
                    nid_next += 1

        # a^{\dagger}_{i,\sigma} a^{\dagger}_{j,\tau} operators connected to right terminal
        self.a_dag_a_dag_r = {}
        for i, sigma in itertools.product(range(2, nsites - 1), (0, 1)):
            for j, tau in itertools.product(range(i, nsites), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                self.a_dag_a_dag_r[i, sigma, j, tau] = {}
                for k in range(2, i + 1):
                    self.a_dag_a_dag_r[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair(-2, [-1, 1][sigma] + [-1, 1][tau]))
                    nid_next += 1
        # a_{i,\sigma} a_{j,\tau} operators connected to right terminal
        self.a_ann_a_ann_r = {}
        for i, sigma in itertools.product(range(2, nsites), (0, 1)):
            for j, tau in itertools.product(range(2, i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                self.a_ann_a_ann_r[i, sigma, j, tau] = {}
                for k in range(2, j + 1):
                    self.a_ann_a_ann_r[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair( 2, [1, -1][sigma] + [1, -1][tau]))
                    nid_next += 1
        # a^{\dagger}_{i,\sigma} a_{j,\tau} operators connected to right terminal
        self.a_dag_a_ann_r = {}
        for i, sigma in itertools.product(range(2, nsites), (0, 1)):
            for j, tau in itertools.product(range(2, nsites), (0, 1)):
                self.a_dag_a_ann_r[i, sigma, j, tau] = {}
                for k in range(2, min(i, j) + 2):
                    self.a_dag_a_ann_r[i, sigma, j, tau][k] = OpGraphNode(
                        nid_next, [], [], encode_quantum_number_pair( 0, [-1, 1][sigma] + [1, -1][tau]))
                    nid_next += 1

        ###  TRIPLE operators

        # triple creation operators connected to left terminal (restricted to the half boundary)
        self.a_dag_a_dag_a_dag_l = {}
        for i, sigma in itertools.product(range(half_site_count - 2), (0, 1)):
            for j, tau in itertools.product(range(i, half_site_count - 1), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                for k, mu in itertools.product(range(j, half_site_count), (0, 1)):
                    if (j, tau) >= (k, mu):
                        continue
                    self.a_dag_a_dag_a_dag_l[i, sigma, j, tau, k, mu] = {}
                    for m in range(k + 1, half_site_count + 1):
                        self.a_dag_a_dag_a_dag_l[i, sigma, j, tau, k, mu][m] = OpGraphNode(
                            nid_next, [], [],
                            encode_quantum_number_pair(3, [1, -1][sigma] + [1, -1][tau] + [1, -1][mu]))
                        nid_next += 1

        # triple annihilation operators connected to left terminal
        self.a_ann_a_ann_a_ann_l = {}
        for i, sigma in itertools.product(range(half_site_count), (0, 1)):
            for j, tau in itertools.product(range(i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                for k, mu in itertools.product(range(j + 1), (0, 1)):
                    if (j, tau) <= (k, mu):
                        continue
                    self.a_ann_a_ann_a_ann_l[i, sigma, j, tau, k, mu] = {}
                    for m in range(i + 1, half_site_count + 1):
                        self.a_ann_a_ann_a_ann_l[i, sigma, j, tau, k, mu][m] = OpGraphNode(
                            nid_next, [], [],
                            encode_quantum_number_pair(-3, [-1, 1][sigma] + [-1, 1][tau] + [-1, 1][mu]))
                        nid_next += 1

        # two creation, one annihilation operators connected to left terminal
        # (key: (i, sigma, j, tau, a, alpha) with (i, sigma) < (j, tau))
        self.a_dag_a_dag_a_ann_l = {}
        for i, sigma in itertools.product(range(half_site_count), (0, 1)):
            for j, tau in itertools.product(range(i, half_site_count), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                for a, alpha in itertools.product(range(half_site_count), (0, 1)):
                    self.a_dag_a_dag_a_ann_l[i, sigma, j, tau, a, alpha] = {}
                    for m in range(max(i, j, a) + 1, half_site_count + 1):
                        self.a_dag_a_dag_a_ann_l[i, sigma, j, tau, a, alpha][m] = OpGraphNode(
                            nid_next, [], [],
                            encode_quantum_number_pair(1, [1, -1][sigma] + [1, -1][tau] + [-1, 1][alpha]))
                        nid_next += 1
        # one creation, two annihilation operators connected to left terminal
        # (key: (c, gamma, i, sigma, j, tau) with (i, sigma) < (j, tau))
        self.a_dag_a_ann_a_ann_l = {}
        for c, gamma in itertools.product(range(half_site_count), (0, 1)):
            for i, sigma in itertools.product(range(half_site_count), (0, 1)):
                for j, tau in itertools.product(range(i, half_site_count), (0, 1)):
                    if (i, sigma) >= (j, tau):
                        continue
                    self.a_dag_a_ann_a_ann_l[c, gamma, i, sigma, j, tau] = {}
                    for m in range(max(c, i, j) + 1, half_site_count + 1):
                        self.a_dag_a_ann_a_ann_l[c, gamma, i, sigma, j, tau][m] = OpGraphNode(
                            nid_next, [], [],
                            encode_quantum_number_pair(-1, [1, -1][gamma] + [-1, 1][sigma] + [-1, 1][tau]))
                        nid_next += 1

        # triple creation operators connected to right terminal (restricted to the half boundary)
        self.a_dag_a_dag_a_dag_r = {}
        for i, sigma in itertools.product(range(half_site_count + 1, nsites - 2), (0, 1)):
            for j, tau in itertools.product(range(i, nsites - 1), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                for k, mu in itertools.product(range(j, nsites), (0, 1)):
                    if (j, tau) >= (k, mu):
                        continue
                    self.a_dag_a_dag_a_dag_r[i, sigma, j, tau, k, mu] = {}
                    for m in range(half_site_count + 1, i + 1):
                        self.a_dag_a_dag_a_dag_r[i, sigma, j, tau, k, mu][m] = OpGraphNode(
                            nid_next, [], [],
                            encode_quantum_number_pair(-3, [-1, 1][sigma] + [-1, 1][tau] + [-1, 1][mu]))
                        nid_next += 1
        # triple annihilation operators connected to right terminal
        self.a_ann_a_ann_a_ann_r = {}
        for i, sigma in itertools.product(range(half_site_count + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(half_site_count + 1, i + 1), (0, 1)):
                if (i, sigma) <= (j, tau):
                    continue
                for k, mu in itertools.product(range(half_site_count + 1, j + 1), (0, 1)):
                    if (j, tau) <= (k, mu):
                        continue
                    self.a_ann_a_ann_a_ann_r[i, sigma, j, tau, k, mu] = {}
                    for m in range(half_site_count + 1, k + 1):
                        self.a_ann_a_ann_a_ann_r[i, sigma, j, tau, k, mu][m] = OpGraphNode(
                            nid_next, [], [],
                            encode_quantum_number_pair(3, [1, -1][sigma] + [1, -1][tau] + [1, -1][mu]))
                        nid_next += 1
        # two creation, one annihilation operators connected to right terminal
        self.a_dag_a_dag_a_ann_r = {}
        for i, sigma in itertools.product(range(half_site_count + 1, nsites), (0, 1)):
            for j, tau in itertools.product(range(i, nsites), (0, 1)):
                if (i, sigma) >= (j, tau):
                    continue
                for a, alpha in itertools.product(range(half_site_count + 1, nsites), (0, 1)):
                    self.a_dag_a_dag_a_ann_r[i, sigma, j, tau, a, alpha] = {}
                    for m in range(half_site_count + 1, min(i, j, a) + 1):
                        self.a_dag_a_dag_a_ann_r[i, sigma, j, tau, a, alpha][m] = OpGraphNode(
                            nid_next, [], [],
                            encode_quantum_number_pair(-1, [-1, 1][sigma] + [-1, 1][tau] + [1, -1][alpha]))
                        nid_next += 1
        # one creation, two annihilation operators connected to right terminal
        self.a_dag_a_ann_a_ann_r = {}
        for c, gamma in itertools.product(range(half_site_count + 1, nsites), (0, 1)):
            for i, sigma in itertools.product(range(half_site_count + 1, nsites), (0, 1)):
                for j, tau in itertools.product(range(i, nsites), (0, 1)):
                    if (i, sigma) >= (j, tau):
                        continue
                    self.a_dag_a_ann_a_ann_r[c, gamma, i, sigma, j, tau] = {}
                    for m in range(half_site_count + 1, min(c, i, j) + 1):
                        self.a_dag_a_ann_a_ann_r[c, gamma, i, sigma, j, tau][m] = OpGraphNode(
                            nid_next, [], [],
                            encode_quantum_number_pair(1, [-1, 1][gamma] + [1, -1][sigma] + [1, -1][tau]))
                        nid_next += 1
