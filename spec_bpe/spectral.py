import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from ripser import ripser

class TypologicalAtlas:
    def __init__(self):
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
    def __init__(self):
        self.atlas = TypologicalAtlas()
        self.current_prior = self.atlas.priors["isolating"]
        self.current_topology_name = "isolating"

    def calculate_fractal_dimension(self, ids):
        if len(ids) < 10: return 0.5
        unique_counts = []
        steps = np.linspace(5, len(ids), 5, dtype=int)
        for s in steps:
            unique_counts.append(len(set(ids[:s])))
        try:
            coeffs = np.polyfit(np.log(steps), np.log(unique_counts), 1)
            return float(np.clip(coeffs[0], 0.1, 1.5))
        except:
            return 0.5

    def get_persistence_signature(self, G):
        if len(G) < 3: return 0.0
        try:
            path_lengths = nx.floyd_warshall_numpy(G, weight='weight')
            path_lengths = np.nan_to_num(path_lengths, nan=1e3)
            path_lengths[path_lengths > 1e3] = 1e3
            dgms = ripser(path_lengths, distance_matrix=True)['dgms']
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
    def __init__(self, gap_threshold=0.01, pi=0.1, update_threshold=0.05):
        self.boundaries = set()
        self.spectral_gap = 1.0
        self.gap_threshold = gap_threshold
        self.pi = pi
        self.ph_scorer = PersistentHomologyScorer()
        self.co_occurrence_graph = nx.Graph()
        self.last_edge_count = 0
        self.update_threshold = update_threshold

    def update_boundaries(self, ids, vocab_size, force=False):
        G = nx.Graph()
        for i in range(len(ids) - 1):
            u, v = ids[i], ids[i+1]
            if G.has_edge(u, v):
                G[u][v]['weight'] += 1
            else:
                G.add_edge(u, v, weight=1)

        current_edges = G.number_of_edges()

        # Lazy Update Check
        if not force and self.last_edge_count > 0:
            change = abs(current_edges - self.last_edge_count) / self.last_edge_count
            if change < self.update_threshold:
                # Still need to store the graph for PH probing
                self.co_occurrence_graph = G
                return

        self.co_occurrence_graph = G
        if len(G) < 3:
            return

        try:
            L = nx.laplacian_matrix(G).astype(float)
            k = min(2, len(G) - 1)
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')
            self.spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 1.0
            fiedler_vec = eigenvectors[:, 1] if len(eigenvalues) > 1 else eigenvectors[:, 0]
            nodes = list(G.nodes())
            p1_set = {nodes[i] for i, val in enumerate(fiedler_vec) if val > 0}
            p2_set = {nodes[i] for i, val in enumerate(fiedler_vec) if val <= 0}
            self.boundaries = set()
            for u, v in G.edges():
                if (u in p1_set and v in p2_set) or (u in p2_set and v in p1_set):
                    self.boundaries.add((u, v))
                    self.boundaries.add((v, u))
            self.last_edge_count = current_edges
        except Exception:
            pass

    def get_penalty(self, pair):
        if pair not in self.boundaries:
            return 0.0
        tunnel_factor = 1.5 if self.ph_scorer.current_topology_name == "agglutinative" else 1.0
        p_merge = np.exp(-1.0 / (self.pi * tunnel_factor + 1e-5))
        return 1.0 - p_merge

    def is_forbidden(self, pair):
        return self.spectral_gap < self.gap_threshold
