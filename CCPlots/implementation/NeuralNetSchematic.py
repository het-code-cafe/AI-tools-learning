import matplotlib.pyplot as plt
import networkx as nx

from CCPlots.PlotExample import PlotExample
from CCPlots.config import COLOR_PALETTE, OUTPUT_PATH

class NeuralNetSchematic(PlotExample):

    def __init__(self, n_input_nodes=3, hidden_layers: tuple[int] = (5, 5), n_output_nodes=2):
        """
        Initialize the neural network.

        n_input_nodes   int         Number of desired input nodes
        hidden_layers   tuple[int]  List with the number of neurons per hidden layer
        n_output_nodes  int         Number of desired output nodes
        """
        self.layers = [n_input_nodes] + list(hidden_layers) + [n_output_nodes]

    def main(self):
        """ Draws a simple schematic view of a neural network. """
        G: nx.Graph = nx.Graph()
        positions = {}
        neuron_index = 0  # unique index for each neuron across all layers

        layer_count = len(self.layers)

        # Dynamic spacing
        x_spacing = 6 / (layer_count - 1)  # Flexible spacing based on number of layers
        y_spacing = 1.2  # Slightly larger for better separation

        # Colors per layer
        colors = [
            COLOR_PALETTE['accent_colors']['mint_green'],  # Input layer
            *([COLOR_PALETTE['base_colors']['medium_green']] * (layer_count - 2)),  # Hidden layers
            COLOR_PALETTE['base_colors']['bright_yellow']  # Output layer
        ]

        # Build the network graph
        for layer_index, num_neurons in enumerate(self.layers):
            for neuron in range(num_neurons):
                x = layer_index * x_spacing
                y = -neuron * y_spacing + (num_neurons - 1) * y_spacing / 2  # Center vertically

                positions[neuron_index] = (x, y)
                G.add_node(neuron_index, layer=layer_index)

                # Add edges to previous layer neurons
                if layer_index > 0:
                    prev_layer_start = sum(self.layers[:layer_index - 1]) if layer_index > 1 else 0
                    prev_layer_end = prev_layer_start + self.layers[layer_index - 1]
                    for prev_neuron in range(prev_layer_start, prev_layer_end):
                        G.add_edge(prev_neuron, neuron_index)

                neuron_index += 1

        # Plotting
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        ax.set_facecolor('white')

        # Draw edges first so nodes overlay them
        nx.draw_networkx_edges(G, pos=positions, edge_color="gray", width=1.5)

        # Draw nodes by layer (to color them differently)
        neuron_index = 0
        for layer_index, num_neurons in enumerate(self.layers):
            nodes = list(range(neuron_index, neuron_index + num_neurons))
            nx.draw_networkx_nodes(G, pos=positions, nodelist=nodes, node_size=600, node_color=colors[layer_index])
            neuron_index += num_neurons

        # Add layer labels
        for layer_index, num_neurons in enumerate(self.layers):
            x = layer_index * x_spacing
            label = "Input" if layer_index == 0 else "Output" if layer_index == layer_count - 1 else f"Hidden {layer_index}"
            y_bottom = min(y for (xx, y) in positions.values())
            plt.text(x, y_bottom - 0.75, label, ha='center', fontsize=12, fontweight='bold', color='black')

        plt.title("Neural Network Schematic", fontsize=14)
        plt.axis('off')
        plt.savefig(OUTPUT_PATH + "nn_schematic.png")

if __name__ == "__main__":
    # Example: 3 input neurons, 2 hidden layers (5 and 4 neurons), 2 output neurons
    NeuralNetSchematic().main()
