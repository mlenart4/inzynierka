from Graph import G
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import beta

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

krawedzie = pd.DataFrame({
    "nadawca": df_edges["source"].map(id_x),
    "odbiorca": df_edges["target"].map(id_x),
    "p": df_edges["pos"].astype(int),
    "z": df_edges["neu"].astype(int),
    "n": df_edges["neg"].astype(int)
})

print(len(krawedzie))
assert krawedzie[["nadawca", "odbiorca"]].notna().all().all()


krawedzie["znak"] = np.sign(krawedzie["p"]-krawedzie["n"])
krawedzie["znak_wiekszosci"] = krawedzie["znak"].map({1: "+", -1: "-"})

print(krawedzie["znak_wiekszosci"].value_counts(dropna=False))
print(krawedzie.head())

maska = (krawedzie["p"] == 0) & (krawedzie["n"] == 0)
liczba_neutralnych =  maska.sum()

print(liczba_neutralnych)

maska_v2 = (krawedzie["p"]-krawedzie["n"] == 0) & ((krawedzie["p"]>0) | (krawedzie["n"]>0))
liczba_remisow = maska_v2.sum()

print(liczba_remisow)

krawedzie["a"] = np.maximum(krawedzie["p"], krawedzie["n"])+ 1
krawedzie["b"] = np.minimum(krawedzie["p"], krawedzie["n"])+ 1
krawedzie["beta"] = beta.sf(0.5, krawedzie["a"], krawedzie["b"])

prog = 0.8

warunek = krawedzie["beta"] >= prog

krawedzie["etykieta"] = krawedzie["znak_wiekszosci"].where(warunek)
print(krawedzie["etykieta"].value_counts(dropna=False))

krawedzie["suma_wzmianek"] = krawedzie["p"] + krawedzie["n"]
warunek_v2 = ((krawedzie["suma_wzmianek"] >=3) & (krawedzie["p"] != krawedzie["n"]))
krawedzie["etykieta_min3"] = krawedzie["znak_wiekszosci"].where(warunek_v2)
print(krawedzie["etykieta_min3"].value_counts(dropna=False))