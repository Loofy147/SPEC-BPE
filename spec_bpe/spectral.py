import numpy as np
import networkx as nx
from ripser import ripser

class TypologicalAtlas:
    """
    A library of pre-computed persistent homologies for diverse language types.
    """
    def __init__(self):
        # Mock priors for demonstration
        # In a real system, these would be loaded from disk or pre-calculated
        self.priors = {
            "isolating": {"dim": 0.5, "barcode_signature": 0.1, "entropy_target": 0.8},
            "agglutinative": {"dim": 0.9, "barcode_signature": 0.6, "entropy_target": 0.4},
            "inflectional": {"dim": 0.7, "barcode_signature": 0.3, "entropy_target": 0.6}
        }

    def get_closest_prior(self, fractal_dim, persistence_signature):
        best_match = "isolating"
        min_dist = float('inf')
        for name, prior in self.priors.items():
            dist = (prior["dim"] - fractal_dim)**2 + (prior["barcode_signature"] - persistence_signature)**2
            if dist < min_dist:
                min_dist = dist
                best_match = name
        return best_match, self.priors[best_match]

class PersistentHomologyScorer:
    """
    Identifies structural invariants that persist across scales.
    Uses Ripser for persistent homology calculations.
    """
    def __init__(self):
        self.atlas = TypologicalAtlas()
        self.current_prior = self.atlas.priors["isolating"]
        self.current_topology_name = "isolating"

    def calculate_fractal_dimension(self, ids):
        """
        Estimate fractal dimension using unique token growth rate (surrogate).
        High growth -> complex/agglutinative.
        """
        if len(ids) < 10: return 0.5
        unique_counts = []
        # Sample points in the sequence
        steps = np.linspace(5, len(ids), 5, dtype=int)
        for s in steps:
            unique_counts.append(len(set(ids[:s])))

        # log(N) = D * log(L)
        try:
            coeffs = np.polyfit(np.log(steps), np.log(unique_counts), 1)
            return float(np.clip(coeffs[0], 0.1, 1.5))
        except:
            return 0.5

    def get_persistence_signature(self, G):
        """
        Compute persistence diagram from the co-occurrence graph.
        """
        if len(G) < 3: return 0.0

        # Shortest path distances on the graph
        try:
            path_lengths = nx.floyd_warshall_numpy(G, weight='weight')
            path_lengths = np.nan_to_num(path_lengths, nan=1e3)
            # Clip to avoid infinite distances
            path_lengths[path_lengths > 1e3] = 1e3

            dgms = ripser(path_lengths, distance_matrix=True)['dgms']

            # Signature based on H1 (cycles) persistence
            if len(dgms) > 1 and len(dgms[1]) > 0:
                lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
                lifetimes = lifetimes[np.isfinite(lifetimes)]
                if len(lifetimes) > 0:
                    return float(np.mean(lifetimes))
        except:
            pass
        return 0.1

    def probe_typology(self, ids, G):
        f_dim = self.calculate_fractal_dimension(ids)
        p_sig = self.get_persistence_signature(G)
        name, prior = self.atlas.get_closest_prior(f_dim, p_sig)
        self.current_topology_name = name
        self.current_prior = prior
        return name

class SpectralFilter:
    """
    Negato-Spectral BPE with Stochastic Boundary Tunneling.
    Now enhanced with Typological Awareness.
    """
    def __init__(self, gap_threshold=0.01, pi=0.1):
        self.boundaries = set()
        self.spectral_gap = 1.0
        self.gap_threshold = gap_threshold
        self.pi = pi # Social Pressure
        self.ph_scorer = PersistentHomologyScorer()
        self.co_occurrence_graph = nx.Graph()

    def update_boundaries(self, ids, vocab_size):
        G = nx.Graph()
        for i in range(len(ids) - 1):
            u, v = ids[i], ids[i+1]
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)

        self.co_occurrence_graph = G

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
        if pair not in self.boundaries:
            return 0.0

        # Adjust tunneling based on typology
        # For agglutinative languages, we might be more permissive with boundaries
        # within words (detected as small-scale PH features).
        tunnel_factor = 1.0
        if self.ph_scorer.current_topology_name == "agglutinative":
            tunnel_factor = 1.5

        p_merge = np.exp(-1.0 / (self.pi * tunnel_factor + 1e-5))
        return 1.0 - p_merge

    def is_forbidden(self, pair):
        if self.spectral_gap < self.gap_threshold:
            return True
        return False
