import matplotlib.pyplot as plt
import networkx as nx

from CCPlots.PlotExample import PlotExample
from CCPlots.config import COLOR_PALETTE, OUTPUT_PATH

import math
import random
import itertools


class TravelingSalesmanVisualization(PlotExample):

    def __init__(self, n_cities=10):
        """
        Initialize the Traveling Salesman Problem visualization.

        n_cities        int     Number of cities (nodes) in the graph
        """
        self.n_cities = n_cities
        self.G = self._generate_random_graph()

    def _generate_random_graph(self):
        """Generate a complete graph with random distances between cities."""
        G = nx.complete_graph(self.n_cities)

        for (u, v) in G.edges():
            G.edges[u, v]['weight'] = random.randint(1, 100)

        return G

    def _solve_tsp_brute_force(self):
        """Solve TSP using brute force (only works for small graphs)."""
        nodes = list(self.G.nodes)
        min_path = None
        min_cost = float('inf')

        for path in itertools.permutations(nodes):
            cost = sum(self.G[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
            cost += self.G[path[-1]][path[0]]['weight']  # Return to the starting city

            if cost < min_cost:
                min_cost = cost
                min_path = path

        return min_path, min_cost

    def _plot_tsp_solution(self, path, filename, total_routes, cost):
        """Plot the graph and highlight the TSP path."""
        pos = nx.spring_layout(self.G, seed=42)

        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.set_facecolor('white')

        # Draw all edges
        nx.draw(self.G, pos, with_labels=True, node_color=COLOR_PALETTE['base_colors']['bright_teal'], node_size=500)

        # Highlight the TSP path in a different color
        tsp_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)] + [(path[-1], path[0])]
        nx.draw_networkx_edges(self.G, pos, edgelist=tsp_edges, width=2,
                               edge_color=COLOR_PALETTE['accent_colors']['mint_green'])

        # Add edge labels (distances)
        edge_labels = nx.get_edge_attributes(self.G, 'weight')
        nx.draw_networkx_edge_labels(self.G, pos, edge_labels=edge_labels)

        plt.title(f"Traveling Salesman Problem Solution ({self.n_cities} cities)\n"
                  f"Total routes: {total_routes:,}")
        plt.savefig(OUTPUT_PATH + filename)
        plt.show()

    def main(self):
        """Main entry point to generate and visualize the TSP."""
        total_routes = math.factorial(self.n_cities)

        if self.n_cities <= 20:
            path, cost = self._solve_tsp_brute_force()
            print(f"Optimal path: {path}")
            print(f"Total cost: {cost}")
            print(f"Total possible routes: {total_routes:,}")

            self._plot_tsp_solution(path, f"tsp_small_{self.n_cities}_cities.png", total_routes, cost)
        else:
            # For larger graphs, we only show the graph itself (too expensive to brute force)
            pos = nx.spring_layout(self.G, seed=42)

            plt.figure(figsize=(12, 8))
            nx.draw(self.G, pos, with_labels=True, node_color=COLOR_PALETTE['accent_colors']['mint_green'],
                    node_size=500)
            plt.title(f"TSP Graph ({self.n_cities} cities)\nTotal possible routes: {total_routes:,}")
            plt.savefig(OUTPUT_PATH + f"tsp_large_{self.n_cities}_cities.png")
            plt.show()


if __name__ == "__main__":
    # Example
    TravelingSalesmanVisualization(n_cities=12).main()
