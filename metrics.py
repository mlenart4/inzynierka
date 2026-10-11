import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# **** OGÓLNE FUNKCJE ****
def basic_info(G):
    return{ 
        "nodes":  G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G),
        "directed": G.is_directed(),
        }

def positive_only(data):
    return [k for k in data if k > 0]

def log_binning(values, n_bins):  # values czyli surowa lista  (degree_list() + filter())  
    values = np.array(values)

    x_min = np.min(values)
    x_max = np.max(values)

    bins = np.logspace(np.log10(x_min), np.log10(x_max+1), n_bins)

    counts, edges = np.histogram(values, bins=bins, density=True)
    x = np.sqrt(edges[:-1] * edges[1:])   # srednia geometryczna

    # print(np.sum(counts * np.diff(edges))  )   test czy suma prawd = 1

    mask = counts > 0
    x = x[mask]
    y = counts[mask]

    pairs = []
    for i in range(len(x)):
        pairs.append((x[i], y[i]))

    return pairs


def plot_loglog(data, ax=None, xlabel="", ylabel="", label="", title="", color='purple'):
    if ax is None:
        fig, ax = plt.subplots()

    x_values = []
    y_values = []
    for x, y in data:
        x_values.append(x)
        y_values.append(y)

    #ax.loglog(x_values, y_values, marker='o', linestyle='None', color=color, label=label)
    ax.loglog(x_values, y_values, marker='o', linestyle='-', markersize=4, color=color, label=label)  # linia laczaca punkty
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(title)

    return ax

# **** STOPNIE ****

def degree_list(G):
    k_in = []
    k_out = []
    for node in G.nodes():
        in_degree = G.in_degree(node)
        k_in.append(in_degree)

        out_degree = G.out_degree(node)
        k_out.append(out_degree)
    
    return k_in, k_out

def count_degree(degrees):
    degree_count = {}
    for degree in degrees:
        if degree not in degree_count:
            degree_count[degree] = 0
        degree_count[degree] += 1
    return degree_count

def degree_distribution(G):
    k_in, k_out = degree_list(G)
    k_in_filtered = positive_only(k_in)
    k_out_filtered = positive_only(k_out)

    distribution_in = count_degree(k_in_filtered)
    distribution_out = count_degree(k_out_filtered)

    dist_in = sorted(distribution_in.items())
    dist_out = sorted(distribution_out.items())

    return dist_in, dist_out

def pk_degree_distribution(G):
    count_in, count_out = degree_distribution(G)
    n = G.number_of_nodes()
    dist_in = [(k, c/n) for k, c in count_in]
    dist_out = [(k, c/n) for k, c in count_out]

    return dist_in, dist_out

# **** SIŁY ****

def strength_list(G, weight='weight'):
    s_in = []
    s_out = []
    for node in G.nodes():
        in_strength = G.in_degree(node, weight=weight)
        s_in.append(in_strength)

        out_strength = G.out_degree(node, weight=weight)
        s_out.append(out_strength)
    
    return s_in, s_out

# **** RELACJA STOPIEŃ-SIŁA ****

def degree_strength_pairs(G, weight='weight'):
    k_in, k_out = degree_list(G)
    s_in, s_out = strength_list(G, weight=weight)

    pairs_in = []
    pairs_out = []
    for i in range(len(G.nodes)):
        pairs_in.append((k_in[i], s_in[i]))
        pairs_out.append((k_out[i], s_out[i]))

    pairs_in_filtered = [(k, s) for k, s in pairs_in if k > 0]
    pairs_out_filtered = [(k, s) for k, s in pairs_out if k > 0]

    return pairs_in_filtered, pairs_out_filtered

def average_strength_per_degree(pairs):
    degree_strength = {}

    for k, s in pairs:
        if k not in degree_strength:
            degree_strength[k] = []
        degree_strength[k].append(s)

    
    average = [(k, (sum(s)/len(s))) for k, s in degree_strength.items()]
    return sorted(average)

def binned_average_strength(pairs, n_bins):

    k_values = np.array([k for k, s in pairs])
    s_values = np.array([s for k, s in pairs])

    x_min = np.min(k_values)
    x_max = np.max(k_values)   

    bins = np.logspace(np.log10(x_min), np.log10(x_max+1), n_bins)

    bin_indices = np.digitize(k_values, bins)   # przypisywanie numeru bina do wezla

    x = []
    y = []
    for i in range(1, len(bins)):
        mask = bin_indices == i    # dostajemy tablice t/f gdy wezel w binie i
        if mask.sum() >=5:          # min 5 wezlow, zeby ogon wykresu byl ladny        
        # if np.any(mask):          # pomijamy puste biny
            x.append(np.mean(k_values[mask]))
            y.append(np.mean(s_values[mask]))

    data = []
    for i in range(len(x)):
        data.append((x[i], y[i]))

    # print(mask.sum(), np.mean(k_values[mask]), np.mean(s_values[mask])) # liczenie wezlow w binie bo jakis wykres dziwny

    return data

def linear_regression_loglog(binned_data):
    x = np.log10([k for k, s in binned_data])
    y = np.log10([s for k, s in binned_data])

    l = linregress(x, y)

    return l.slope, l.stderr, l.rvalue**2, l.intercept       # beta, błąd beta, R^2, wyraz wolny (do wykresu)

# **** ROZKŁAD WAG KRAWĘDZI ****
def edge_weights(G, weight='weight'):
    weights = []
    for u, v, data in G.edges(data=True):       #data = True zeby zwracal atrybuty
        weights.append(data[weight])

    return weights