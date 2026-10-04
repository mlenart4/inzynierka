import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def basic_info(G):
    return{ 
        "nodes":  G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "directed": G.is_directed(),
        }