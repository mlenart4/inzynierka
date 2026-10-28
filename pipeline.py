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

# print(len(krawedzie))
assert krawedzie[["nadawca", "odbiorca"]].notna().all().all()


krawedzie["znak"] = np.sign(krawedzie["p"]-krawedzie["n"])
krawedzie["znak_wiekszosci"] = krawedzie["znak"].map({1: "+", -1: "-"})

# print(krawedzie["znak_wiekszosci"].value_counts(dropna=False))
# print(krawedzie.head())

maska = (krawedzie["p"] == 0) & (krawedzie["n"] == 0)
liczba_neutralnych =  maska.sum()

# print(liczba_neutralnych)

maska_v2 = (krawedzie["p"]-krawedzie["n"] == 0) & ((krawedzie["p"]>0) | (krawedzie["n"]>0))
liczba_remisow = maska_v2.sum()

# print(liczba_remisow)

krawedzie["a"] = np.maximum(krawedzie["p"], krawedzie["n"])+ 1
krawedzie["b"] = np.minimum(krawedzie["p"], krawedzie["n"])+ 1
krawedzie["beta"] = beta.sf(0.5, krawedzie["a"], krawedzie["b"])

prog = 0.8

warunek = krawedzie["beta"] >= prog

krawedzie["etykieta"] = krawedzie["znak_wiekszosci"].where(warunek)
# print(krawedzie["etykieta"].value_counts(dropna=False))

krawedzie["suma_wzmianek"] = krawedzie["p"] + krawedzie["n"]
warunek_v2 = ((krawedzie["suma_wzmianek"] >=3) & (krawedzie["p"] != krawedzie["n"]))
krawedzie["etykieta_min3"] = krawedzie["znak_wiekszosci"].where(warunek_v2)
# print(krawedzie["etykieta_min3"].value_counts(dropna=False))

# print(table["club"].value_counts())

table["klub"] = table["club"].replace({"Polska2050-TD": "Polska2050"})
# print(table["klub"].value_counts())

koalicja = {"KO", "PSL-TD", "Polska2050", "Lewica"}
opozycja = {"PiS", "Konfederacja", "Razem", "Republikanie", "niez.", "Konfederacja_KP"}

czy_koalicja = table["klub"].isin(koalicja)

table["strona"] = np.where(czy_koalicja, "koalicja", "opozycja")
print(table["strona"].value_counts())

klub_wg_idx = table.set_index("idx")["klub"] # numer posła -> klub / strona
strona_wg_idx = table.set_index("idx")["strona"]

krawedzie["klub_nadawcy"] = krawedzie["nadawca"].map(klub_wg_idx)
krawedzie["klub_adresata"] = krawedzie["odbiorca"].map(klub_wg_idx)
krawedzie["strona_nadawcy"] = krawedzie["nadawca"].map(strona_wg_idx)
krawedzie["strona_adresata"] = krawedzie["odbiorca"].map(strona_wg_idx)

# print(krawedzie[["nadawca", "klub_nadawcy", "strona_nadawcy",
#                  "odbiorca", "klub_adresata", "strona_adresata"]].head())

wewnatrzklubowa = (krawedzie["klub_nadawcy"] == krawedzie["klub_adresata"]) & (krawedzie["klub_nadawcy"] != "niez.")
koalicyjna = (krawedzie["strona_nadawcy"] == krawedzie["strona_adresata"]) & (krawedzie["strona_nadawcy"] == "koalicja")
opozycyjna = (krawedzie["strona_nadawcy"] == krawedzie["strona_adresata"]) & (krawedzie["strona_nadawcy"] == "opozycja")
rząd_opozycja = (krawedzie["strona_nadawcy"] != krawedzie["strona_adresata"])

typ_pary = [wewnatrzklubowa, koalicyjna, opozycyjna, rząd_opozycja]
typ_pary_wartosc = ["wewnatrzklubowa", "koalicyjna", "opozycja_opozycja", "rząd_opozycja"]

krawedzie["typ_pary"] = np.select(typ_pary, typ_pary_wartosc, default="BRAK")
print(krawedzie["typ_pary"].head())


print(krawedzie["typ_pary"].value_counts())
print(krawedzie.groupby(["typ_pary", "etykieta"]).size())  # typ pary + etykieta
