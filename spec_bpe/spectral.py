import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from ripser import ripser

class TypologicalAtlas:
    """
    A library of pre-computed persistent homologies for diverse language types.
    """
    def __init__(self):
        # Refined priors to better capture linguistic differences
        self.priors = {
            "isolating": {"dim": 0.4, "barcode_signature": 0.05, "entropy_target": 0.9},
            "agglutinative": {"dim": 0.85, "barcode_signature": 0.45, "entropy_target": 0.3},
            "inflectional": {"dim": 0.65, "barcode_signature": 0.25, "entropy_target": 0.5}
        }

    def get_closest_prior(self, fractal_dim, persistence_signature):
        best_match = "isolating"
        min_dist = float('inf')
        for name, prior in self.priors.items():
            # Normalized Euclidean distance
            dist = ((prior["dim"] - fractal_dim)/0.5)**2 + ((prior["barcode_signature"] - persistence_signature)/0.5)**2
            if dist < min_dist:
                min_dist = dist
                best_match = name
        return best_match, self.priors[best_match]

class PersistentHomologyScorer:
    def __init__(self, seed=42):
        self.atlas = TypologicalAtlas()
        self.current_prior = self.atlas.priors["isolating"]
        self.current_topology_name = "isolating"
        self.warp_factor = 1.0 # Diffeomorphic Warping
        self.rng = np.random.default_rng(seed)

    def calculate_fractal_dimension(self, ids):
        if len(ids) < 15: return 0.4
        unique_counts = []
        # More granular steps for better estimation
        num_steps = 10
        steps = np.linspace(min(10, len(ids)), len(ids), num_steps, dtype=int)
        for s in steps:
            unique_counts.append(len(set(ids[:s])))

        try:
            # Use log-log fit to estimate box-counting dimension proxy
            x = np.log(steps)
            y = np.log(unique_counts)
            coeffs = np.polyfit(x, y, 1)
            return float(np.clip(coeffs[0], 0.1, 1.2))
        except:
            return 0.4

    def get_persistence_signature(self, G):
        if len(G) < 5: return 0.02
        try:
            nodes = list(G.nodes())
            if len(nodes) > 120:
                # Deterministic sampling for reproducibility
                sampled_nodes = self.rng.choice(nodes, 120, replace=False)
                path_lengths_dict = {}
                for source in sampled_nodes:
                    lengths = nx.single_source_dijkstra_path_length(G, source, weight='weight')
                    path_lengths_dict[source] = {target: lengths.get(target, 1e4) for target in sampled_nodes}

                size = len(sampled_nodes)
                path_lengths = np.zeros((size, size))
                for i, u in enumerate(sampled_nodes):
                    for j, v in enumerate(sampled_nodes):
                        path_lengths[i, j] = path_lengths_dict[u][v]
            else:
                path_lengths = nx.floyd_warshall_numpy(G, weight='weight')
                path_lengths = np.nan_to_num(path_lengths, nan=1e4)

            path_lengths[path_lengths > 1e4] = 1e4
            # Fill diagonal with 0 if not already
            np.fill_diagonal(path_lengths, 0)

            dgms = ripser(path_lengths, distance_matrix=True)['dgms']
            if len(dgms) > 1 and len(dgms[1]) > 0:
                # Use max lifetime for more distinct signatures
                lifetimes = dgms[1][:, 1] - dgms[1][:, 0]
                lifetimes = lifetimes[np.isfinite(lifetimes)]
                if len(lifetimes) > 0:
                    return float(np.max(lifetimes))
        except:
            pass
        return 0.05

    def probe_typology(self, ids, G):
        f_dim = self.calculate_fractal_dimension(ids)
        p_sig = self.get_persistence_signature(G)
        name, prior = self.atlas.get_closest_prior(f_dim, p_sig)

        dist = np.sqrt(((prior["dim"] - f_dim)/0.5)**2 + ((prior["barcode_signature"] - p_sig)/0.5)**2)
        # Damped warping factor
        self.warp_factor = 1.0 + np.tanh(dist)

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
        if not force and self.last_edge_count > 0:
            change = abs(current_edges - self.last_edge_count) / self.last_edge_count
            if change < self.update_threshold:
                # Still update current graph but don't recompute spectrum
                self.co_occurrence_graph = G
                return

        self.co_occurrence_graph = G
        if len(G) < 5:
            return

        try:
            # Add small self-loops for spectral stability
            for node in G.nodes():
                if not G.has_edge(node, node):
                    G.add_edge(node, node, weight=0.1)

            L = nx.laplacian_matrix(G).astype(float)
            k = min(3, len(G) - 1)
            eigenvalues, eigenvectors = eigsh(L, k=k, which='SM')

            # Sort eigenvalues and vectors
            idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            self.spectral_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 1.0

            # Use second smallest eigenvector (Fiedler vector)
            fiedler_vec = eigenvectors[:, 1] if len(eigenvalues) > 1 else eigenvectors[:, 0]
            nodes = list(G.nodes())
            median_val = np.median(fiedler_vec)
            p1_set = {nodes[i] for i, val in enumerate(fiedler_vec) if val > median_val}
            p2_set = {nodes[i] for i, val in enumerate(fiedler_vec) if val <= median_val}

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

        tunnel_prior = 2.0 if self.ph_scorer.current_topology_name == "agglutinative" else 1.0
        tunnel_factor = tunnel_prior * self.ph_scorer.warp_factor

        p_merge = np.exp(-1.0 / (self.pi * tunnel_factor + 1e-5))
        return 1.0 - p_merge

    def is_forbidden(self, pair):
        # Hard exclusion only if gap is extremely small (disconnected components)
        return self.spectral_gap < (self.gap_threshold * 0.1)
