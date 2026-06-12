import numpy as np
import networkx as nx

class SpectralFilter:
    """
    Negato-Spectral BPE with Renormalization Group (RG) Flow monitoring.
    Monitors the Spectral Gap (lambda_2 - lambda_1) to prevent over-tokenization.
    """
    def __init__(self, gap_threshold=0.01):
        self.boundaries = set()
        self.spectral_gap = 1.0
        self.gap_threshold = gap_threshold

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

            # lambda_1 is usually 0. lambda_2 is the Fiedler eigenvalue.
            lambda_1 = eigenvalues[0]
            lambda_2 = eigenvalues[1]
            self.spectral_gap = lambda_2 - lambda_1

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

    def is_forbidden(self, pair):
        # Merge is forbidden if it crosses a boundary OR if the spectral gap collapsed
        return (pair in self.boundaries) or (self.spectral_gap < self.gap_threshold)

    def get_rg_state(self):
        return self.spectral_gap
