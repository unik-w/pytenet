

from collections.abc import Sequence
import autoray as ar
from autoray import numpy as np
from ..mpo import MPO
from ..opgraph import OpGraphNode, OpGraphEdge, OpGraph
from .molecular import MolecularOID, _molecular_hamiltonian_generate_operator_map
from .molecular_three_body import _antisymmetrize_2, _antisymmetrize_3

__all__ = ["MolecularThreeBodyOpGraphNodes", "molecular_hamiltonian_three_body_mpo_explicit"]


class MolecularThreeBodyOpGraphNodes:

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
        # a^{\dagger}_i operators connected to left terminal
        self.a_dag_l = {}
        for i in range(nsites - 2):
            self.a_dag_l[i] = {}
            for j in range(i + 1, nsites - 1):
                self.a_dag_l[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1
        # a_i operators connected to left terminal
        self.a_ann_l = {}
        for i in range(nsites - 2):
            self.a_ann_l[i] = {}
            for j in range(i + 1, nsites - 1):
                self.a_ann_l[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # a^{\dagger}_i operators connected to right terminal
        self.a_dag_r = {}
        for i in range(2, nsites):
            self.a_dag_r[i] = {}
            for j in range(2, i + 1):
                self.a_dag_r[i][j] = OpGraphNode(nid_next, [], [], -1)
                nid_next += 1
        # a_i operators connected to right terminal
        self.a_ann_r = {}
        for i in range(2, nsites):
            self.a_ann_r[i] = {}
            for j in range(2, i + 1):
                self.a_ann_r[i][j] = OpGraphNode(nid_next, [], [], 1)
                nid_next += 1

        ###  Double operators L and R

        # a^{\dagger}_i a^{\dagger}_j operators connected to left terminal
        self.a_dag_a_dag_l = {}
        for i in range(nsites - 2):
            for j in range(i + 1, nsites - 1):
                self.a_dag_a_dag_l[i, j] = {}
                for k in range(j + 1, nsites):
                    self.a_dag_a_dag_l[i, j][k] = OpGraphNode(nid_next, [], [], 2)
                    nid_next += 1
        # a_i a_j operators connected to left terminal
        self.a_ann_a_ann_l = {}
        for i in range(nsites - 2):
            for j in range(i):
                self.a_ann_a_ann_l[i, j] = {}
                for k in range(i + 1, nsites):
                    self.a_ann_a_ann_l[i, j][k] = OpGraphNode(nid_next, [], [], -2)
                    nid_next += 1
         # a^{\dagger}_i a_j  and a_i a^{\dagger}_j operators connected to left terminal
        self.a_dag_a_ann_l = {}
        for i in range(nsites ):
            for j in range(nsites ):
                self.a_dag_a_ann_l[i, j] = {}
                for k in range(max(i, j) + 1, nsites):
                    self.a_dag_a_ann_l[i, j][k] = OpGraphNode(nid_next, [], [], 0)
                    nid_next += 1

        # a^{\dagger}_i a^{\dagger}_j operators connected to right terminal
        self.a_dag_a_dag_r = {}
        for i in range(2, nsites - 1):
            for j in range(i + 1, nsites):
                self.a_dag_a_dag_r[i, j] = {}
                for k in range(2, i + 1):
                    self.a_dag_a_dag_r[i, j][k] = OpGraphNode(nid_next, [], [], -2)
                    nid_next += 1
        # a_i a_j operators connected to right terminal
        self.a_ann_a_ann_r = {}
        for i in range(2, nsites):
            for j in range(2, i):
                self.a_ann_a_ann_r[i, j] = {}
                for k in range(2, j + 1):
                    self.a_ann_a_ann_r[i, j][k] = OpGraphNode(nid_next, [], [], 2)
                    nid_next += 1
        # a^{\dagger}_i a_j and a_i a^{\dagger}_j operators connected to right terminal
        self.a_dag_a_ann_r = {}
        for i in range(2, nsites):
            for j in range(2, nsites):
                self.a_dag_a_ann_r[i, j] = {}
                for k in range(2, min(i, j) + 2):
                    self.a_dag_a_ann_r[i, j][k] = OpGraphNode(nid_next, [], [], 0)
                    nid_next += 1


       
        ###  TRIPLE operators


        # triple operators connected to left terminal (restricted to the
        # half boundary
        self.a_dag_a_dag_a_dag_l = {}
        for i in range(half_site_count - 2):
            for j in range(i + 1, half_site_count - 1):
                for k in range(j + 1, half_site_count):
                    self.a_dag_a_dag_a_dag_l[i, j, k] = {}
                    for m in range(k + 1, half_site_count + 1):
                        self.a_dag_a_dag_a_dag_l[i, j, k][m] = OpGraphNode(nid_next,[],[],3)
                        nid_next += 1

        self.a_ann_a_ann_a_ann_l = {}
        for i in range(half_site_count):
            for j in range(i):
                for k in range(j):
                    self.a_ann_a_ann_a_ann_l[i, j, k] = {}
                    for m in range(i + 1, half_site_count + 1):
                        self.a_ann_a_ann_a_ann_l[i, j, k][m] = OpGraphNode(nid_next,[],[],-3)
                        nid_next += 1

        # two creation, one annihilation (key: (c1, c2, a) with c1 < c2)
        self.a_dag_a_dag_a_ann_l = {}
        for i in range(half_site_count):
            for j in range(i + 1, half_site_count):
                for a in range(half_site_count):
                    
                    self.a_dag_a_dag_a_ann_l[i, j, a] = {}
                    for m in range(max(i, j, a) + 1, half_site_count + 1):
                        self.a_dag_a_dag_a_ann_l[i, j, a][m] = OpGraphNode(nid_next,[],[],1)
                        nid_next += 1
        # one creation, two annihilation (key: (c, a1, a2) with a1 < a2)
        self.a_dag_a_ann_a_ann_l = {}
        for c in range(half_site_count):
            for i in range(half_site_count):
                for j in range(i + 1, half_site_count):
                    
                    self.a_dag_a_ann_a_ann_l[c, i, j] = {}
                    for m in range(max(c, i, j) + 1, half_site_count + 1):
                        self.a_dag_a_ann_a_ann_l[c, i, j][m] = OpGraphNode(nid_next,[],[],-1)
                        nid_next += 1

        # triple operators connected to right terminal (restricted to the
        # half boundary)
        self.a_dag_a_dag_a_dag_r = {}
        for i in range(half_site_count + 1, nsites-2):
            for j in range(i + 1, nsites - 1):
                for k in range(j + 1, nsites):
                    self.a_dag_a_dag_a_dag_r[i, j, k] = {}
                    for m in range(half_site_count + 1, i+1):
                        self.a_dag_a_dag_a_dag_r[i, j, k][m] = OpGraphNode(nid_next,[],[],-3)
                        nid_next += 1
        self.a_ann_a_ann_a_ann_r = {}
        for i in range(half_site_count + 1, nsites):
            for j in range(half_site_count + 1, i):
                for k in range(half_site_count + 1, j):
                    self.a_ann_a_ann_a_ann_r[i, j, k] = {}
                    for m in range(half_site_count + 1, k+1):
                        self.a_ann_a_ann_a_ann_r[i, j, k][m] = OpGraphNode(nid_next,[],[],3)
                        nid_next += 1
        self.a_dag_a_dag_a_ann_r = {}
        for i in range(half_site_count + 1, nsites):
            for j in range(i + 1, nsites):
                for a in range(half_site_count + 1, nsites):
                   
                    self.a_dag_a_dag_a_ann_r[i, j, a] = {}
                    for m in range(half_site_count + 1, min(i, j, a) + 1):
                        self.a_dag_a_dag_a_ann_r[i, j, a][m] = OpGraphNode(nid_next,[],[],-1)
                        nid_next += 1
        self.a_dag_a_ann_a_ann_r = {}
        for c in range(half_site_count + 1, nsites):
            for i in range(half_site_count + 1, nsites):
                for j in range(i + 1, nsites):
                    
                    self.a_dag_a_ann_a_ann_r[c, i, j] = {}
                    for m in range(half_site_count + 1, min(c, i, j) + 1):
                        self.a_dag_a_ann_a_ann_r[c, i, j][m] = OpGraphNode(nid_next,[],[],1)
                        nid_next += 1

    def get(self, oplist: Sequence[tuple], connection):
        """
        Retrieve nodes corresponding to a single, pair, or triple of creation and annihilation operators.
        """
        if len(oplist) == 1:
            (i, oid) = oplist[0]
            if oid == MolecularOID.C:
                return self.a_dag_l[i] if connection == "left" else self.a_dag_r[i]
            if oid == MolecularOID.A:
                return self.a_ann_l[i] if connection == "left" else self.a_ann_r[i]
            raise KeyError(f"nodes for OID {oid} and i == {i} do not exist")
        if len(oplist) == 2:
            (i, oid0), (j, oid1) = oplist
            if (oid0, oid1) == (MolecularOID.C, MolecularOID.C):
                i, j = sorted((i, j))
                return self.a_dag_a_dag_l[i, j] if connection == "left" \
                    else self.a_dag_a_dag_r[i, j]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.A):
                i, j = sorted((i, j), reverse=True)
                return self.a_ann_a_ann_l[i, j] if connection == "left" \
                    else self.a_ann_a_ann_r[i, j]
            if (oid0, oid1) == (MolecularOID.C, MolecularOID.A):
                return self.a_dag_a_ann_l[i, j] if connection == "left" \
                    else self.a_dag_a_ann_r[i, j]
            if (oid0, oid1) == (MolecularOID.A, MolecularOID.C):
                return self.a_dag_a_ann_l[j, i] if connection == "left" \
                    else self.a_dag_a_ann_r[j, i]
            raise KeyError(f"nodes for OIDs ({oid0}, {oid1}) and (i, j) == ({i}, {j}) do not exist")
        if len(oplist) == 3:
            cs = [op for op in oplist if op[1] == MolecularOID.C]
            as_ = [op for op in oplist if op[1] == MolecularOID.A]
            if len(cs) == 3:
                i, j, k = sorted(op[0] for op in cs)
                return self.a_dag_a_dag_a_dag_l[i, j, k] if connection == "left" \
                    else self.a_dag_a_dag_a_dag_r[i, j, k]
            if len(as_) == 3:
                i, j, k = sorted((op[0] for op in as_), reverse=True)
                return self.a_ann_a_ann_a_ann_l[i, j, k] if connection == "left" \
                    else self.a_ann_a_ann_a_ann_r[i, j, k]
            if len(cs) == 2:
                i, j = sorted(op[0] for op in cs)
                a = as_[0][0]
                return self.a_dag_a_dag_a_ann_l[i, j, a] if connection == "left" \
                    else self.a_dag_a_dag_a_ann_r[i, j, a]
            if len(as_) == 2:
                i, j = sorted(op[0] for op in as_)
                c = cs[0][0]
                return self.a_dag_a_ann_a_ann_l[c, i, j] if connection == "left" \
                    else self.a_dag_a_ann_a_ann_r[c, i, j]
            raise KeyError(f"nodes for operator list {oplist} do not exist")
        raise KeyError(f"nodes for operator list of length {len(oplist)} do not exist")

    def generate_graph(self) -> OpGraph:
        """
        Create and initialize an operator graph with corresponding nodes and edges.
        """
        nsites = self.nsites
        h = nsites // 2

        # initialize graph with nodes
        graph = OpGraph(list(self.identity_l.values()) +
                        list(self.identity_r.values()) +
                        [node for nodes in self.a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_a_dag_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_a_ann_l.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_a_dag_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_ann_a_ann_a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_dag_a_ann_r.values() for node in nodes.values()] +
                        [node for nodes in self.a_dag_a_ann_a_ann_r.values() for node in nodes.values()],
                        [], [self.identity_l[0].nid, self.identity_r[nsites].nid])

        eid_next = 0

        def add_edge(nid0, nid1, oid, weight=1.):
            nonlocal eid_next
            graph.add_connect_edge(OpGraphEdge(eid_next, [nid0, nid1], [(oid, weight)]))
            eid_next += 1

        # identities connected to left and right terminals
        for i in range(nsites - 1):
            add_edge(self.identity_l[i].nid, self.identity_l[i + 1].nid, MolecularOID.I)
        for i in range(1, nsites):
            add_edge(self.identity_r[i].nid, self.identity_r[i + 1].nid, MolecularOID.I)

        # single operators connected to left terminal
        for i in range(nsites - 2):
            add_edge(self.identity_l[i].nid, self.a_dag_l[i][i + 1].nid, MolecularOID.C)
            add_edge(self.identity_l[i].nid, self.a_ann_l[i][i + 1].nid, MolecularOID.A)
            for j in range(i + 1, nsites - 2):
                add_edge(self.a_dag_l[i][j].nid, self.a_dag_l[i][j + 1].nid, MolecularOID.Z)
                add_edge(self.a_ann_l[i][j].nid, self.a_ann_l[i][j + 1].nid, MolecularOID.Z)
        # single operators connected to right terminal
        for i in range(2, nsites):
            for j in range(2, i):
                add_edge(self.a_dag_r[i][j].nid, self.a_dag_r[i][j + 1].nid, MolecularOID.Z)
                add_edge(self.a_ann_r[i][j].nid, self.a_ann_r[i][j + 1].nid, MolecularOID.Z)
            add_edge(self.a_dag_r[i][i].nid, self.identity_r[i + 1].nid, MolecularOID.C)
            add_edge(self.a_ann_r[i][i].nid, self.identity_r[i + 1].nid, MolecularOID.A)

        # pair operators connected to left terminal (full range)
        for i in range(nsites - 2):
            for j in range(i + 1, nsites - 1):
                add_edge(self.a_dag_l[i][j].nid, self.a_dag_a_dag_l[i, j][j + 1].nid, MolecularOID.C)
                for k in range(j + 1, nsites - 1):
                    add_edge(self.a_dag_a_dag_l[i, j][k].nid,
                             self.a_dag_a_dag_l[i, j][k + 1].nid, MolecularOID.I)
        for i in range(nsites - 2):
            for j in range(i):
                add_edge(self.a_ann_l[j][i].nid, self.a_ann_a_ann_l[i, j][i + 1].nid, MolecularOID.A)
                for k in range(i + 1, nsites - 1):
                    add_edge(self.a_ann_a_ann_l[i, j][k].nid,
                             self.a_ann_a_ann_l[i, j][k + 1].nid, MolecularOID.I)
        for i in range(h):
            for j in range(h):
                if i < j:
                    add_edge(self.a_dag_l[i][j].nid, self.a_dag_a_ann_l[i, j][j + 1].nid, MolecularOID.A)
                elif i == j:
                    add_edge(self.identity_l[i].nid, self.a_dag_a_ann_l[i, j][i + 1].nid, MolecularOID.N)
                else:
                    add_edge(self.a_ann_l[j][i].nid, self.a_dag_a_ann_l[i, j][i + 1].nid, MolecularOID.C)
                for k in range(max(i, j) + 1, h):
                    add_edge(self.a_dag_a_ann_l[i, j][k].nid,
                             self.a_dag_a_ann_l[i, j][k + 1].nid, MolecularOID.I)

        # pair operators connected to right terminal (full range)
        for i in range(2, nsites - 1):
            for j in range(i + 1, nsites):
                for k in range(2, i):
                    add_edge(self.a_dag_a_dag_r[i, j][k].nid,
                             self.a_dag_a_dag_r[i, j][k + 1].nid, MolecularOID.I)
                add_edge(self.a_dag_a_dag_r[i, j][i].nid, self.a_dag_r[j][i + 1].nid, MolecularOID.C)
        for i in range(2, nsites):
            for j in range(2, i):
                for k in range(2, j):
                    add_edge(self.a_ann_a_ann_r[i, j][k].nid,
                             self.a_ann_a_ann_r[i, j][k + 1].nid, MolecularOID.I)
                add_edge(self.a_ann_a_ann_r[i, j][j].nid, self.a_ann_r[i][j + 1].nid, MolecularOID.A)
        for i in range(2, nsites):
            for j in range(2, nsites):
                for k in range(2, min(i, j)):
                    add_edge(self.a_dag_a_ann_r[i, j][k].nid,
                             self.a_dag_a_ann_r[i, j][k + 1].nid, MolecularOID.I)
                if i < j:
                    add_edge(self.a_dag_a_ann_r[i, j][i].nid, self.a_ann_r[j][i + 1].nid, MolecularOID.C)
                elif i == j:
                    add_edge(self.a_dag_a_ann_r[i, j][i].nid, self.identity_r[i + 1].nid, MolecularOID.N)
                else:
                    add_edge(self.a_dag_a_ann_r[i, j][j].nid, self.a_dag_r[i][j + 1].nid, MolecularOID.A)

        # triple operators connected to left terminal (restricted to half)
        for i in range(h - 2):
            for j in range(i + 1, h - 1):
                for k in range(j + 1, h):
                    add_edge(self.a_dag_a_dag_l[i, j][k].nid,
                             self.a_dag_a_dag_a_dag_l[i, j, k][k + 1].nid, MolecularOID.C)
                    for m in range(k + 1, h):
                        add_edge(self.a_dag_a_dag_a_dag_l[i, j, k][m].nid,
                                 self.a_dag_a_dag_a_dag_l[i, j, k][m + 1].nid, MolecularOID.I)
        for i in range(h):
            for j in range(i):
                for k in range(j):
                    add_edge(self.a_ann_a_ann_l[j, k][i].nid,
                             self.a_ann_a_ann_a_ann_l[i, j, k][i + 1].nid, MolecularOID.A)
                    for m in range(i + 1, h):
                        add_edge(self.a_ann_a_ann_a_ann_l[i, j, k][m].nid,
                                 self.a_ann_a_ann_a_ann_l[i, j, k][m + 1].nid, MolecularOID.I)
       
        for i in range(h):
            for j in range(i + 1, h):
                for a in range(h):
                    m0 = max(i, j, a) + 1
                    if a > j:
                        add_edge(self.a_dag_a_dag_l[i, j][a].nid,
                                 self.a_dag_a_dag_a_ann_l[i, j, a][m0].nid, MolecularOID.A)
                    elif a < i:
                        add_edge(self.a_dag_a_ann_l[i, a][j].nid,
                                 self.a_dag_a_dag_a_ann_l[i, j, a][m0].nid, MolecularOID.C)
                    elif a == j:
                        add_edge(self.a_dag_l[i][j].nid,
                                 self.a_dag_a_dag_a_ann_l[i, j, a][m0].nid, MolecularOID.N)
                    elif a == i:  # merges into a number operator
                        add_edge(self.a_dag_a_ann_l[i, a][j].nid,
                                 self.a_dag_a_dag_a_ann_l[i, j, a][m0].nid, MolecularOID.C)
                    else:  # i < a < j
                        add_edge(self.a_dag_a_ann_l[i, a][j].nid,
                                 self.a_dag_a_dag_a_ann_l[i, j, a][m0].nid, MolecularOID.C)

                    for m in range(m0, h):
                        add_edge(self.a_dag_a_dag_a_ann_l[i, j, a][m].nid,
                                 self.a_dag_a_dag_a_ann_l[i, j, a][m + 1].nid, MolecularOID.I)
       
        for c in range(h):
            for i in range(h):
                for j in range(i + 1, h):
                    m0 = max(c, i, j) + 1
                    if c > j:
                        add_edge(self.a_ann_a_ann_l[j, i][c].nid,
                                 self.a_dag_a_ann_a_ann_l[c, i, j][m0].nid, MolecularOID.C)
                    elif c == i:  # merges into a number operator
                        add_edge(self.a_dag_a_ann_l[c, i][j].nid,
                                 self.a_dag_a_ann_a_ann_l[c, i, j][m0].nid, MolecularOID.A)
                    elif c == j:  # merges into a number operator
                        add_edge(self.a_ann_l[i][j].nid,
                                 self.a_dag_a_ann_a_ann_l[c, i, j][m0].nid, MolecularOID.N)
                    else:  # i < c < j
                        add_edge(self.a_dag_a_ann_l[c, i][j].nid,
                                 self.a_dag_a_ann_a_ann_l[c, i, j][m0].nid, MolecularOID.A)

                    for m in range(m0, h):
                        add_edge(self.a_dag_a_ann_a_ann_l[c, i, j][m].nid,
                                 self.a_dag_a_ann_a_ann_l[c, i, j][m + 1].nid, MolecularOID.I)

        
        for i in range(h + 1, nsites - 2):
            for j in range(i + 1, nsites - 1):
                for k in range(j + 1, nsites):
                    for m in range(h + 1, i):
                        add_edge(self.a_dag_a_dag_a_dag_r[i, j, k][m].nid,
                                 self.a_dag_a_dag_a_dag_r[i, j, k][m + 1].nid, MolecularOID.I)
                    add_edge(self.a_dag_a_dag_a_dag_r[i, j, k][i].nid,
                             self.a_dag_a_dag_r[j, k][i + 1].nid, MolecularOID.C)
        for i in range(h + 1, nsites):
            for j in range(h + 1, i):
                for k in range(h + 1, j):
                    for m in range(h + 1, k):
                        add_edge(self.a_ann_a_ann_a_ann_r[i, j, k][m].nid,
                                 self.a_ann_a_ann_a_ann_r[i, j, k][m + 1].nid, MolecularOID.I)
                    add_edge(self.a_ann_a_ann_a_ann_r[i, j, k][k].nid,
                             self.a_ann_a_ann_r[i, j][k + 1].nid, MolecularOID.A)
       
        for i in range(h + 1, nsites):
            for j in range(i + 1, nsites):
                for a in range(h + 1, nsites):
                    m1 = min(i, j, a)
                    for m in range(h + 1, m1):
                        add_edge(self.a_dag_a_dag_a_ann_r[i, j, a][m].nid,
                                 self.a_dag_a_dag_a_ann_r[i, j, a][m + 1].nid, MolecularOID.I)

                    if a < i:
                        add_edge(self.a_dag_a_dag_a_ann_r[i, j, a][m1].nid,
                                    self.a_dag_a_dag_r[i, j][m1 + 1].nid, MolecularOID.A)
                    elif a > j:
                        add_edge(self.a_dag_a_dag_a_ann_r[i, j, a][m1].nid,
                                    self.a_dag_a_ann_r[j, a][m1 + 1].nid, MolecularOID.C)
                    elif a == i:
                        add_edge(self.a_dag_a_dag_a_ann_r[i, j, a][m1].nid,
                                    self.a_dag_r[j][m1 + 1].nid, MolecularOID.N)
                    elif a == j:
                        add_edge(self.a_dag_a_dag_a_ann_r[i, j, a][m1].nid,
                                    self.a_dag_a_ann_r[j, j][m1 + 1].nid, MolecularOID.C)
                    else:  # i < a < j
                        add_edge(self.a_dag_a_dag_a_ann_r[i, j, a][m1].nid,
                                    self.a_dag_a_ann_r[j, a][m1 + 1].nid, MolecularOID.C)

        # one creation (c), two annihilation (i < j); mirrors the left-side
        # wiring above.
        for c in range(h + 1, nsites):
            for i in range(h + 1, nsites):
                for j in range(i + 1, nsites):
                    m1 = min(c, i, j)
                    for m in range(h + 1, m1):
                        add_edge(self.a_dag_a_ann_a_ann_r[c, i, j][m].nid,
                                 self.a_dag_a_ann_a_ann_r[c, i, j][m + 1].nid, MolecularOID.I)

                    if c < i:
                        add_edge(self.a_dag_a_ann_a_ann_r[c, i, j][m1].nid,
                                 self.a_ann_a_ann_r[j, i][m1 + 1].nid, MolecularOID.C)
                    elif c > j:
                        add_edge(self.a_dag_a_ann_a_ann_r[c, i, j][m1].nid,
                                 self.a_dag_a_ann_r[c, j][m1 + 1].nid, MolecularOID.A)
                    elif c == i:
                        add_edge(self.a_dag_a_ann_a_ann_r[c, i, j][m1].nid,
                                 self.a_ann_r[j][m1 + 1].nid, MolecularOID.N)
                    elif c == j:
                        add_edge(self.a_dag_a_ann_a_ann_r[c, i, j][m1].nid,
                                 self.a_dag_a_ann_r[c, c][m1 + 1].nid, MolecularOID.A)
                    else:  # i < c < j
                        add_edge(self.a_dag_a_ann_a_ann_r[c, i, j][m1].nid,
                                 self.a_dag_a_ann_r[c, j][m1 + 1].nid, MolecularOID.A)

        return graph


def _molecular_hamiltonian_graph_add_term(graph: OpGraph, nodes: MolecularThreeBodyOpGraphNodes,
                                          oplist: Sequence[tuple], coeff: float):
    """
    Add an operator term (operator string of creation and annihilation operators)
    to the operator graph describing a molecular Hamiltonian, including three-body terms.
    """
    eid_next = max(graph.edges.keys()) + 1

    nsites = nodes.nsites

    # sort by site (orbital) index
    oplist = sorted(oplist)

    if len(oplist) == 2:
        (i, oid0), (j, oid1) = oplist
        if i == j:
            # expecting number operator
            assert (oid0, oid1) == (MolecularOID.A, MolecularOID.C)
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes.identity_r[i + 1].nid],
                            [(MolecularOID.N, coeff)]))
        else:
            assert i < j
            if j <= nsites//2:
                nodes_l = nodes.get([(i, oid0)], "left")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes.identity_r[j + 1].nid],
                                [(oid1, coeff)]))
            elif i >= nsites//2:
                nodes_r = nodes.get([(j, oid1)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[i].nid, nodes_r[i + 1].nid],
                                [(oid0, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid0)], "left")
                nodes_r = nodes.get([(j, oid1)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[nsites//2].nid, nodes_r[nsites//2 + 1].nid],
                                [(MolecularOID.Z, coeff)]))
    elif len(oplist) == 4:
        (i, oid0), (j, oid1), (k, oid2), (l, oid3) = oplist
        if j == k:
            # expecting number operator
            assert (oid1, oid2) == (MolecularOID.A, MolecularOID.C)
            nodes_l = nodes.get([(i, oid0)], "left")
            nodes_r = nodes.get([(l, oid3)], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid],
                            [(MolecularOID.N, coeff)]))
        elif k <= nsites//2:
            nodes_l = nodes.get([(i, oid0), (j, oid1)], "left")
            if k == l:
                # expecting number operator
                assert (oid2, oid3) == (MolecularOID.A, MolecularOID.C)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes.identity_r[k + 1].nid],
                                [(MolecularOID.N, coeff)]))
            else:
                nodes_r = nodes.get([(l, oid3)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes_r[k + 1].nid], [(oid2, coeff)]))
        elif j >= nsites//2:
            nodes_r = nodes.get([(k, oid2), (l, oid3)], "right")
            if i == j:
                # expecting number operator
                assert (oid0, oid1) == (MolecularOID.A, MolecularOID.C)
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes.identity_l[j].nid, nodes_r[j + 1].nid],
                                [(MolecularOID.N, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid0)], "left")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid], [(oid1, coeff)]))
        else:
            nodes_l = nodes.get([(i, oid0), (j, oid1)], "left")
            nodes_r = nodes.get([(k, oid2), (l, oid3)], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[nsites//2].nid, nodes_r[nsites//2 + 1].nid],
                            [(MolecularOID.I, coeff)]))
    elif len(oplist) == 6:

        (i, oid0), (j, oid1), (k, oid2), (l, oid3), (m, oid4), (n, oid5) = oplist
        h = nsites // 2

        if k == l:
            assert (oid2, oid3) == (MolecularOID.A, MolecularOID.C)
            nodes_l = nodes.get([(i, oid0), (j, oid1)], "left")
            nodes_r = nodes.get([(m, oid4), (n, oid5)], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[k].nid, nodes_r[k + 1].nid],
                            [(MolecularOID.N, coeff)]))
            
        elif k >= h:
            if j==k:
                assert (oid1, oid2) == (MolecularOID.A, MolecularOID.C)
                nodes_l = nodes.get([(i, oid0)], "left")
                nodes_r = nodes.get([(l, oid3), (m, oid4), (n, oid5)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[j].nid, nodes_r[j + 1].nid],
                                [(MolecularOID.N, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid0), (j, oid1)], "left")
                nodes_r = nodes.get([(l, oid3), (m, oid4), (n, oid5)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[k].nid, nodes_r[k + 1].nid],
                                [(oid2, coeff)]))

        elif l <= h:
            if l==m:
                assert (oid3, oid4) == (MolecularOID.A, MolecularOID.C)
                nodes_l = nodes.get([(i, oid0), (j, oid1), (k, oid2)], "left")
                nodes_r = nodes.get([(n, oid5)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[l].nid, nodes_r[l + 1].nid],
                                [(MolecularOID.N, coeff)]))
            else:
                nodes_l = nodes.get([(i, oid0), (j, oid1), (k, oid2)], "left")
                nodes_r = nodes.get([(m, oid4), (n, oid5)], "right")
                graph.add_connect_edge(
                    OpGraphEdge(eid_next, [nodes_l[l].nid, nodes_r[l + 1].nid],
                                [(oid3, coeff)]))
                    
        else:
            nodes_l = nodes.get([(i, oid0), (j, oid1), (k, oid2)], "left")
            nodes_r = nodes.get([(l, oid3), (m, oid4), (n, oid5)], "right")
            graph.add_connect_edge(
                OpGraphEdge(eid_next, [nodes_l[h].nid, nodes_r[h + 1].nid],
                            [(MolecularOID.I, coeff)]))
    else:
        raise NotImplementedError


def molecular_hamiltonian_three_body_mpo_explicit(tkin, vint, wint) -> MPO:
    
    tkin = np.asarray(tkin)
    vint = np.asarray(vint)
    wint = np.asarray(wint)
    nsites = tkin.shape[0]
    assert tkin.shape == (nsites, nsites)
    assert vint.shape == (nsites, nsites, nsites, nsites)
    assert wint.shape == 6 * (nsites,)
    assert nsites >= 3

    # can anti-commute fermionic operators such that i < j (< k) and k < l (< m < n)
    gint = _antisymmetrize_2(vint)
    hint = _antisymmetrize_3(wint)

    nodes = MolecularThreeBodyOpGraphNodes(nsites)
    graph = nodes.generate_graph()
    # kinetic hopping terms \sum_{i,j} t_{i,j} a^{\dagger}_i a_j
    for i in range(nsites):
        for j in range(nsites):
            _molecular_hamiltonian_graph_add_term(
                graph, nodes, [(i, MolecularOID.C), (j, MolecularOID.A)], tkin[i, j])
    # interaction terms 1/2 \sum_{i,j,k,l} v_{i,j,k,l} a^{\dagger}_i a^{\dagger}_j a_l a_k
    for i in range(nsites):
        for j in range(i + 1, nsites):  # i < j
            for k in range(nsites):
                for l in range(k + 1, nsites):  # k < l
                    oplist = [(i, MolecularOID.C),
                              (j, MolecularOID.C),
                              (l, MolecularOID.A),
                              (k, MolecularOID.A)]
                    _molecular_hamiltonian_graph_add_term(
                        graph, nodes, oplist, gint[i, j, k, l])
    # three-body interaction terms 1/6 \sum_{i,j,k,l,m,n} w_{i,j,k,l,m,n}
    # a^{\dagger}_i a^{\dagger}_j a^{\dagger}_k a_n a_m a_l
    for i in range(nsites):
        for j in range(i + 1, nsites):  # i < j
            for k in range(j + 1, nsites):  # j < k
                for l in range(nsites):
                    for m in range(l + 1, nsites):  # l < m
                        for n in range(m + 1, nsites):  # m < n
                            oplist = [(i, MolecularOID.C),
                                      (j, MolecularOID.C),
                                      (k, MolecularOID.C),
                                      (n, MolecularOID.A),
                                      (m, MolecularOID.A),
                                      (l, MolecularOID.A)]
                            _molecular_hamiltonian_graph_add_term(
                                graph, nodes, oplist, hint[i, j, k, l, m, n])

    # skip consistency check for larger nsites (would take very long)
    if nsites <= 5:
        assert graph.is_consistent()
    opmap = _molecular_hamiltonian_generate_operator_map()
    dtype = complex if "complex" in ar.get_dtype_name(tkin) \
                    or "complex" in ar.get_dtype_name(vint) \
                    or "complex" in ar.get_dtype_name(wint) else float
    return MPO.from_opgraph([0, 1], graph, opmap, dtype)

