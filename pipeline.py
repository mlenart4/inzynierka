from Graph import G
import networkx as nx
import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_ODNIESIENIA = pd.Timestamp("2026-10-01")

nodes_ids = sorted(G.nodes, key=int)
id_x = {n_id: i for i, n_id in enumerate(nodes_ids)}

table = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
table = table.loc[nodes_ids]               # kolejnosc zgodna z idx
table.index.name = "node_id"
table = table.reset_index()
table.insert(0, "idx", range(len(table)))
table["name"] = table["firstLastName"]

df_edges = nx.to_pandas_edgelist(G)

krawedzie = pd.DataFrame({
    "nadawca": df_edges["source"].map(id_x),
    "odbiorca": df_edges["target"].map(id_x),
    "p": df_edges["pos"].astype(int),
    "z": df_edges["neu"].astype(int),
    "n": df_edges["neg"].astype(int)
})

krawedzie["znak"] = np.sign(krawedzie["p"]-krawedzie["n"])
krawedzie["znak_wiekszosci"] = krawedzie["znak"].map({1: "+", -1: "-"})


maska = (krawedzie["p"] == 0) & (krawedzie["n"] == 0)
liczba_neutralnych =  maska.sum()


maska_v2 = (krawedzie["p"]-krawedzie["n"] == 0) & ((krawedzie["p"]>0) | (krawedzie["n"]>0))
liczba_remisow = maska_v2.sum()


krawedzie["a"] = np.maximum(krawedzie["p"], krawedzie["n"])+ 1
krawedzie["b"] = np.minimum(krawedzie["p"], krawedzie["n"])+ 1
krawedzie["beta"] = beta.sf(0.5, krawedzie["a"], krawedzie["b"])

prog = 0.8

warunek = krawedzie["beta"] >= prog

krawedzie["etykieta"] = krawedzie["znak_wiekszosci"].where(warunek)

krawedzie["suma_wzmianek"] = krawedzie["p"] + krawedzie["n"]
warunek_v2 = ((krawedzie["suma_wzmianek"] >=3) & (krawedzie["p"] != krawedzie["n"]))
krawedzie["etykieta_min3"] = krawedzie["znak_wiekszosci"].where(warunek_v2)


table["klub"] = table["club"].replace({"Polska2050-TD": "Polska2050"})

koalicja = {"KO", "PSL-TD", "Polska2050", "Lewica", "Centrum"}

czy_koalicja = table["klub"].isin(koalicja)

table["strona"] = np.where(czy_koalicja, "koalicja", "opozycja")

klub_wg_idx = table.set_index("idx")["klub"] # numer posła -> klub / strona
strona_wg_idx = table.set_index("idx")["strona"]

krawedzie["klub_nadawcy"] = krawedzie["nadawca"].map(klub_wg_idx)
krawedzie["klub_adresata"] = krawedzie["odbiorca"].map(klub_wg_idx)
krawedzie["strona_nadawcy"] = krawedzie["nadawca"].map(strona_wg_idx)
krawedzie["strona_adresata"] = krawedzie["odbiorca"].map(strona_wg_idx)

wewnatrzklubowa = (krawedzie["klub_nadawcy"] == krawedzie["klub_adresata"]) & (krawedzie["klub_nadawcy"] != "niez.")
koalicyjna = (krawedzie["strona_nadawcy"] == krawedzie["strona_adresata"]) & (krawedzie["strona_nadawcy"] == "koalicja")
opozycyjna = (krawedzie["strona_nadawcy"] == krawedzie["strona_adresata"]) & (krawedzie["strona_nadawcy"] == "opozycja")
rzad_opozycja = (krawedzie["strona_nadawcy"] != krawedzie["strona_adresata"])

typ_pary = [wewnatrzklubowa, koalicyjna, opozycyjna, rzad_opozycja]
typ_pary_wartosc = ["wewnatrzklubowa", "koalicyjna", "opozycja_opozycja", "rzad_opozycja"]

krawedzie["typ_pary"] = np.select(typ_pary, typ_pary_wartosc, default="BRAK")

table["birthDate"] = pd.to_datetime(table["birthDate"])
table["wiek"] = (DATA_ODNIESIENIA - table["birthDate"]).dt.days / 365.25
table["staz"] = table["seniority"].str.split(",").str.len()   # "8, 9, 10" -> 3
 
table["active_num"] = table["active"].astype(float)   # true false na 1 0 

kolumny_tekst = ["club", "klub", "strona", "voivodeship", "districtName",
                 "educationLevel", "profession", "firstLastName"]
table[kolumny_tekst] = table[kolumny_tekst].fillna("brak")

kolumny_klub = ["klub", "strona"]  # bez club bo to przed scaleniem
kolumny_kategorie = ["voivodeship", "districtName", "educationLevel"]   # profession ma osobną grupę
kolumny_liczbowe = ["wiek", "numberOfVotes", "staz"]

preproces = ColumnTransformer([
    ("klub", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kolumny_klub),
    ("kategorie", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kolumny_kategorie),
    ("zawod", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=5, sparse_output=False), ["profession"]),
    ("liczbowe",  StandardScaler(), kolumny_liczbowe),
    ("active",    "passthrough", ["active_num"])
])

X_pelne = preproces.fit_transform(table).astype("float32")
nazwy = preproces.get_feature_names_out()


bez_klubu_maska = ~np.char.startswith(nazwy.astype(str), "klub__")
X_bez_klubu = X_pelne[:, bez_klubu_maska]

bez_klubu_i_funkcji_maska = bez_klubu_maska  # bedzie zmiana po dostarczeniu danych o funkcjach, w nowym zestawie danych
X_bez_klubu_funkcji = X_pelne[:, bez_klubu_i_funkcji_maska]

###### sprawdzamy :)))))
if __name__ == "__main__":
    print("=== 2.1 Etykieta ===")
    print("liczba posłów: ", len(table))
    print("wiersze tabeli par: ", len(krawedzie))
    print("same neutralne p=n=0: ", liczba_neutralnych)
    print("remisy p=n>0: ", liczba_remisow)
    print("pary ze znakiem większości: ", (krawedzie["znak"] != 0).sum())

    print("\n=== 2.2 Próg pewności 0,8 ===")
    print("pary z etykietą:", krawedzie["etykieta"].notna().sum())
    print("etykiety + :", (krawedzie["etykieta"] == "+").sum())
    print("etykiety - :", (krawedzie["etykieta"] == "-").sum())
    print("wariant p+n>=3 :", krawedzie["etykieta_min3"].notna().sum())

    print("\n=== 2.3 Kluby, strony, typy par ===")
    print("liczba klubów po scaleniu :", table["klub"].nunique())
    print("posłowie koalicji :", (table["strona"] == "koalicja").sum())
    print("posłowie opozycji :", (table["strona"] == "opozycja").sum())
    print("typy par :")
    print(krawedzie["typ_pary"].value_counts())
    print("typ pary x etykieta:")
    print(krawedzie.groupby(["typ_pary", "etykieta"]).size())

    z = table["profession"].value_counts()
    ile_klub = sum(table[k].nunique() for k in kolumny_klub)
    ile_kategorie = sum(table[k].nunique() for k in kolumny_kategorie)
    ile_zawod = (z >= 5).sum() + (z < 5).any()   # zawody min. 5 razy + jedna kolumna na rzadkie
    ile_razem = ile_klub + ile_kategorie + ile_zawod + len(kolumny_liczbowe) + 1   # active

    print("\n=== 2.3 Macierze cech (wiersze", len(table), ") ===")
    print("pełne:", X_pelne.shape)
    print("bez klubu:", X_bez_klubu.shape)
    print("bez klubu i roli:", X_bez_klubu_funkcji.shape)
    print("pełne kolumny: ", X_pelne.shape[1], " a oczekiwane: ", ile_razem)
    print("kolumny klubu: ", X_pelne.shape[1] - X_bez_klubu.shape[1], " a oczekiwane: ", ile_klub)
    print("kolumny zawodu: ", sum(n.startswith("zawod__") for n in nazwy), " a oczekiwane: ", ile_zawod)
    print("kolumny funkcji: ", "na razie nie istnieją xd")

