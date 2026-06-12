import numpy as np
import networkx as nx

class SpectralFilter:
    """
    Negato-Spectral BPE with Stochastic Boundary Tunneling.
    Allows merging across structural boundaries under high Social Pressure (Pi).
    """
    def __init__(self, gap_threshold=0.01, pi=0.1):
        self.boundaries = set()
        self.spectral_gap = 1.0
        self.gap_threshold = gap_threshold
        self.pi = pi # Social Pressure

    def update_boundaries(self, ids, vocab_size):
        G = nx.Graph()
        for i in range(len(ids) - 1):
            u, v = ids[i], ids[i+1]
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)

        if len(G) < 3:
            return

        try:
            L = nx.laplacian_matrix(G).toarray().astype(float)
            eigenvalues, eigenvectors = np.linalg.eigh(L)

            self.spectral_gap = eigenvalues[1] - eigenvalues[0]
            fiedler_vec = eigenvectors[:, 1]
            nodes = list(G.nodes())

            p1_set = {nodes[i] for i, val in enumerate(fiedler_vec) if val > 0}
            p2_set = {nodes[i] for i, val in enumerate(fiedler_vec) if val <= 0}

            self.boundaries = set()
            for u, v in G.edges():
                if (u in p1_set and v in p2_set) or (u in p2_set and v in p1_set):
                    self.boundaries.add((u, v))
                    self.boundaries.add((v, u))
        except Exception:
            pass

    def get_penalty(self, pair):
        """
        Stochastic Tunneling: Returns a probability-weighted penalty.
        If pair in boundaries, penalty is high unless Pi is high.
        """
        if pair not in self.boundaries:
            return 0.0

        # P_merge proportional to exp(-1/Pi)
        # We return a penalty: 1 - P_merge
        p_merge = np.exp(-1.0 / (self.pi + 1e-5))
        return 1.0 - p_merge

    def is_forbidden(self, pair):
        # Hard block if spectral gap collapsed
        if self.spectral_gap < self.gap_threshold:
            return True
        return False # We use get_penalty for boundaries instead of hard boolean
