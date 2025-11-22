import networkx as nx
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyvis.network import Network
import plotly.graph_objects as go
from typing import List
import os


class StanfordWebGraph:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.graph: nx.DiGraph | None = None
        self.metrics: dict[str, dict[int, float]] = {}

    def load_graph(self, max_edges: int = 50_000) -> nx.DiGraph:

        G = nx.DiGraph()
        edge_count = 0
        
        with open(self.data_path, "r") as f:
            for line in tqdm(f, desc="Leyendo aristas"):
                line = line.strip()
                # Saltar líneas vacias o comentarios
                if not line or line.startswith("#"):
                    continue
                
                parts = line.split()
                if len(parts) != 2:
                    continue

                src, dst = map(int, parts)

                #evitar self-loops
                if src == dst:
                    continue

                G.add_edge(src, dst)
                edge_count += 1

                if edge_count >= max_edges:
                    break
        
        self.graph = G
        print(f"\nGrafo cargado:")
        print(f"  - Nodos: {G.number_of_nodes()}")
        print(f"  - Aristas: {G.number_of_edges()}")
        return G


    def describe_graph(self, G: nx.DiGraph | None = None) -> None:
        if G is None:
            G = self.graph
        if G is None:
            raise ValueError("No hay grafo cargado.")

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        degrees = [d for _, d in G.degree()]

        print("\nResumen del grafo:")
        print(f"  - Nodos: {num_nodes}")
        print(f"  - Aristas: {num_edges}")
        print(f"  - Grado medio: {np.mean(degrees):.2f}")
        print(f"  - Grado máximo: {np.max(degrees):.0f}")


    def compute_metrics(self, alpha: float = 0.85) -> pd.DataFrame:

        if self.graph is None:
            raise ValueError("Primero carga el grafo con load_graph().")

        print("\nCalculando métricas (PageRank, grados, HITS)...")

        # PageRank
        pagerank = nx.pagerank(self.graph, alpha=alpha)

        # Grados
        in_deg = dict(self.graph.in_degree())
        out_deg = dict(self.graph.out_degree())

        # HITS (autoridades y hubs)
        hits_auth, hits_hubs = nx.hits(self.graph, max_iter=500, normalized=True)

        # Guardar en self.metrics
        self.metrics = {
            "pagerank": pagerank,
            "in_degree": in_deg,
            "out_degree": out_deg,
            "hits_authorities": hits_auth,
            "hits_hubs": hits_hubs,
        }

        # DataFrame ordenado por PageRank
        df = pd.DataFrame({
            "node": list(pagerank.keys()),
            "pagerank": list(pagerank.values()),
        })
        df["in_degree"] = df["node"].map(in_deg)
        df["out_degree"] = df["node"].map(out_deg)
        df["hits_authority"] = df["node"].map(hits_auth)

        df = df.sort_values("pagerank", ascending=False).reset_index(drop=True)
        self.pagerank_df = df

        print("\nTop-5 nodos por PageRank:")
        print(df.head(5))

        return df


    def build_explanatory_subgraph(
        self,
        pagerank_df: pd.DataFrame | None = None,
        top_n: int = 5,
        max_neighbors_per_top_node: int = 30,
        min_neighbor_in_degree: int = 1,
    ) -> nx.DiGraph:

        if self.graph is None:
            raise ValueError("Primero carga el grafo con load_graph().")

        if pagerank_df is None:
            if self.pagerank_df is None:
                raise ValueError("No hay PageRank calculado. Usa compute_pagerank() primero.")
            pagerank_df = self.pagerank_df

        # Top-N nodos por PageRank
        top_nodes = list(pagerank_df.head(top_n)["node"])
        nodes_sub = set(top_nodes)

        # Mapas auxiliares
        pr_map = dict(zip(pagerank_df["node"], pagerank_df["pagerank"]))
        in_deg = dict(self.graph.in_degree())

        for n in top_nodes:
            # vecinos entrantes y salientes
            neighs = set(self.graph.successors(n)) | set(self.graph.predecessors(n))
            # quitarse a sí mismo
            neighs.discard(n)

            # ordenar vecinos por (in_degree, PageRank) descendente
            neighs_sorted = sorted(
                neighs,
                key=lambda u: (in_deg.get(u, 0), pr_map.get(u, 0.0)),
                reverse=True,
            )

            # filtrar por grado mínimo y tomar solo los más relevantes
            neighs_filtered = [
                u for u in neighs_sorted
                if in_deg.get(u, 0) >= min_neighbor_in_degree
            ]
            selected = neighs_filtered[:max_neighbors_per_top_node]

            nodes_sub.update(selected)

        G_exp = self.graph.subgraph(nodes_sub).copy()

        # quitar nodos aislados (por si acaso)
        isolated = list(nx.isolates(G_exp))
        if isolated:
            G_exp.remove_nodes_from(isolated)

        print(f"\nSubgrafo explicativo compacto:")
        print(f"  - Top-N nodos: {len(top_nodes)}")
        print(f"  - Nodos totales en subgrafo: {G_exp.number_of_nodes()}")
        print(f"  - Aristas totales en subgrafo: {G_exp.number_of_edges()}")

        return G_exp

    def plot_explanatory_subgraph(
        self,
        G_exp: nx.DiGraph,
        pagerank_df: pd.DataFrame | None = None,
        top_n: int = 5,
        figsize: tuple = (10, 8),
    ) -> None:
        if self.graph is None:
            raise ValueError("Primero carga el grafo con load_graph().")

        if pagerank_df is None:
            if self.pagerank_df is None:
                raise ValueError("No hay PageRank calculado. Usa compute_pagerank() primero.")
            pagerank_df = self.pagerank_df

        pr_map = dict(zip(pagerank_df["node"], pagerank_df["pagerank"]))
        top_nodes = set(pagerank_df.head(top_n)["node"].tolist())

        sizes = []
        colors = []
        for n in G_exp.nodes():
            pr = pr_map.get(n, 0.0)
            # tamaños más modestos
            size = 100 + pr * 20_000
            sizes.append(size)

            if n in top_nodes:
                colors.append("tab:red")
            else:
                colors.append("tab:blue")

        plt.figure(figsize=figsize)
        pos = nx.spring_layout(G_exp, k=0.3, iterations=100, seed=42)

        nx.draw_networkx_nodes(G_exp, pos, node_size=sizes, node_color=colors, alpha=0.9)
        nx.draw_networkx_edges(G_exp, pos, arrows=True, alpha=0.15, width=0.5)

        labels_top = {n: str(n) for n in G_exp.nodes() if n in top_nodes}
        nx.draw_networkx_labels(G_exp, pos, labels=labels_top, font_size=9)

        plt.title("Subgrafo explicativo compacto (tamaño = PageRank, rojo = Top-N)")
        plt.axis("off")
        plt.show()


    def analyze_top_nodes(self, pagerank_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:

        top = pagerank_df.head(top_n).copy()

        print(f"\nTop-{top_n} nodos por PageRank:")
        print(top[["node", "pagerank", "in_degree", "out_degree"]])

        print("\nEstadísticas de los Top-N:")
        print("  - PageRank medio:", top["pagerank"].mean())
        print("  - in_degree medio:", top["in_degree"].mean())
        print("  - out_degree medio:", top["out_degree"].mean())

        print("\nCorrelación PageRank con in_degree / out_degree (sobre todo el grafo):")
        print(pagerank_df[["pagerank", "in_degree", "out_degree"]].corr())

        return top


    def visualize_subgraph_interactive(
        self,
        ego_graph: nx.DiGraph,
        top_nodes: List[int],
        metric: str = "pagerank",
        save_path: str | None = None,
    ):

        if not hasattr(self, "metrics") or metric not in self.metrics:
            raise ValueError(
                f"No se encontraron métricas '{metric}'. "
                "Asegúrate de llamar a compute_metrics() antes."
            )

        # Crear red
        net = Network(
            height="750px",
            width="100%", 
            directed=True, 
            bgcolor="#ffffff",
            font_color="black",
        )
        
        # Configurar física
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 100,
              "springConstant": 0.08
            },
            "maxVelocity": 50,
            "solver": "forceAtlas2Based",
            "timestep": 0.35,
            "stabilization": {"iterations": 150}
          }
        }
        """)
        
        # Agregar nodos
        for node in ego_graph.nodes():
            is_top = node in top_nodes
            
            # Tamaño basado en la métrica seleccionada
            raw_value = self.metrics[metric].get(node, 0.0)
            if metric == "pagerank":
                size = 10 + raw_value * 60_000  # escala para que se note
            else:
                size = 10 + raw_value * 2
            size = max(10, min(50, size))  # limitar entre 10 y 50
            
            # Color y etiqueta
            color = "#FF6B6B" if is_top else "#4ECDC4"
            label = f"URL {node}" if is_top else ""
            
            title = f"""
            Node ID: {node}
            PageRank: {self.metrics['pagerank'].get(node, 0):.6f}
            In-Degree: {self.metrics['in_degree'].get(node, 0)}
            Out-Degree: {self.metrics['out_degree'].get(node, 0)}
            HITS Authority: {self.metrics['hits_authorities'].get(node, 0):.6f}
            """
            
            net.add_node(
                node, 
                label=label,
                title=title,
                size=size,
                color=color,
                borderWidth=3 if is_top else 1,
            )
        
        # Agregar aristas
        for source, target in ego_graph.edges():
            net.add_edge(
                source, 
                target, 
                width=0.5, 
                color="rgba(128, 128, 128, 0.3)",
            )
        
        # Guardar
        if save_path is None:
            save_path = "visualizations/ego_graph_interactive.html"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        net.save_graph(save_path)
        
        print(f"\nVisualizacion interactiva guardada en: {save_path}")
        return save_path
