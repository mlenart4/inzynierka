from Graph import G
import networkx as nx
import pandas as pd
import numpy as np

nodes_ids = sorted(G.nodes, key=int)
id_x = {n_id: i for i, n_id in enumerate(nodes_ids)}
# print(id_x['250'])  # test czy dziala przypisywanie numerkow

table = pd.DataFrame({
    "idx": range(len(nodes_ids)),
    "node_id": nodes_ids,
    "name": [G.nodes[n]["firstLastName"] for n in nodes_ids],
    "club": [G.nodes[n]["club"] for n in nodes_ids]
})

# print(table.head())
# print(len(table))

df_edges = nx.to_pandas_edgelist(G)

pairs = pd.DataFrame({
    "nadawca": df_edges["source"].map(id_x),
    "odbiorca": df_edges["target"].map(id_x),
    "p": df_edges["pos"].astype(int),
    "z": df_edges["neu"].astype(int),
    "n": df_edges["neg"].astype(int)
})

print(len(pairs))
assert pairs[["nadawca", "odbiorca"]].notna().all().all()


pairs["znak"] = np.sign(pairs["p"]-pairs["n"])
pairs["znak_wiekszosci"] = pairs["znak"].map({1: "+", -1: "-"})

print(pairs["znak_wiekszosci"].value_counts(dropna=False))
print(pairs.head())

maska = (pairs["p"] == 0) & (pairs["n"] == 0)
liczba_neutralnych =  maska.sum()

print(liczba_neutralnych)

maska_v2 = (pairs["p"]-pairs["n"] == 0) & ((pairs["p"]>0) | (pairs["n"]>0))
liczba_remisow = maska_v2.sum()

print(liczba_remisow)