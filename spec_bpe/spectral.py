import numpy as np
import networkx as nx

class SpectralFilter:
    """
    Negato-Spectral BPE (Exclusion via Eigenvalues)
    Uses the Laplacian Spectrum of the co-occurrence graph to identify
    structural boundaries.
    """
    def __init__(self):
        self.boundaries = set()

    def update_boundaries(self, ids, vocab_size):
        # Construct co-occurrence graph
        G = nx.Graph()
        for i in range(len(ids) - 1):
            u, v = ids[i], ids[i+1]
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)

        if len(G) < 2:
            return

        # Compute Laplacian Spectrum
        # We look for the Fiedler vector (eigenvector of the second smallest eigenvalue)
        # to find the "cut" or boundary.
        try:
            L = nx.laplacian_matrix(G).toarray().astype(float)
            eigenvalues, eigenvectors = np.linalg.eigh(L)

            # Second smallest eigenvalue index is usually 1 (0 is always 0)
            fiedler_vec = eigenvectors[:, 1]
            nodes = list(G.nodes())

            # Partition nodes based on sign of Fiedler vector
            part1 = [nodes[i] for i, val in enumerate(fiedler_vec) if val > 0]
            part2 = [nodes[i] for i, val in enumerate(fiedler_vec) if val <= 0]

            # Boundaries are edges between partitions
            self.boundaries = set()
            p1_set = set(part1)
            p2_set = set(part2)

            for u, v in G.edges():
                if (u in p1_set and v in p2_set) or (u in p2_set and v in p1_set):
                    self.boundaries.add((u, v))
                    self.boundaries.add((v, u))
        except Exception:
            # Fallback if spectral decomposition fails
            pass

    def is_forbidden(self, pair):
        return pair in self.boundaries
