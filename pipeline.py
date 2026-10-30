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
# print(id_x['250'])  # test czy dziala przypisywanie numerkow

# table = pd.DataFrame({
#     "idx": range(len(nodes_ids)),
#     "node_id": nodes_ids,
#     "name": [G.nodes[n]["firstLastName"] for n in nodes_ids],
#     "club": [G.nodes[n]["club"] for n in nodes_ids],
#     "accusativeName": [G.nodes[n]["accusativeName"] for n in nodes_ids],
#     "active": [G.nodes[n]["active"] for n in nodes_ids],
#     "birthDate": [G.nodes[n]["birthDate"] for n in nodes_ids],
#     "birthLocation": [G.nodes[n]["birthLocation"] for n in nodes_ids],
#     "districtName": [G.nodes[n]["districtName"] for n in nodes_ids],
#     "districtNum": [G.nodes[n]["districtNum"] for n in nodes_ids],
#     "educationLevel": [G.nodes[n]["educationLevel"] for n in nodes_ids],
#     "email": [G.nodes[n]["email"] for n in nodes_ids],
#     "firstName": [G.nodes[n]["firstName"] for n in nodes_ids],
#     "genitiveName": [G.nodes[n]["genitiveName"] for n in nodes_ids],
#     "lastFirstName": [G.nodes[n]["lastFirstName"] for n in nodes_ids],
#     "lastName": [G.nodes[n]["lastName"] for n in nodes_ids],
#     "numberOfVotes": [G.nodes[n]["numberOfVotes"] for n in nodes_ids],
#     "profession": [G.nodes[n]["profession"] for n in nodes_ids],
#     "secondName": [G.nodes[n]["secondName"] for n in nodes_ids],
#     "voivodeship": [G.nodes[n]["voivodeship"] for n in nodes_ids],
# })
table = pd.DataFrame.from_dict(dict(G.nodes(data=True)), orient="index")
table = table.loc[nodes_ids]               # kolejnosc zgodna z idx
table.index.name = "node_id"
table = table.reset_index()
table.insert(0, "idx", range(len(table)))
table["name"] = table["firstLastName"]
table["node_id"].astype(int)

# print(table.columns.tolist())

# print(table.isna().sum())
# print(table["active"].value_counts())
# print(table["profession"].nunique(), table["voivodeship"].nunique(), table["educationLevel"].nunique())

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
# print(table["strona"].value_counts())

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
rzad_opozycja = (krawedzie["strona_nadawcy"] != krawedzie["strona_adresata"])

typ_pary = [wewnatrzklubowa, koalicyjna, opozycyjna, rzad_opozycja]
typ_pary_wartosc = ["wewnatrzklubowa", "koalicyjna", "opozycja_opozycja", "rzad_opozycja"]

krawedzie["typ_pary"] = np.select(typ_pary, typ_pary_wartosc, default="BRAK")
# print(krawedzie["typ_pary"].head())


# print(krawedzie["typ_pary"].value_counts())
# print(krawedzie.groupby(["typ_pary", "etykieta"]).size())  # typ pary + etykieta

table["birthDate"] = pd.to_datetime(table["birthDate"])
table["wiek"] = (DATA_ODNIESIENIA - table["birthDate"]).dt.days / 365.25
# print(table["wiek"].describe())
# print(table["wiek"].isna().sum())   # ma byc 0

 
table["active_num"] = table["active"].astype(float)   # true false na 1 0 

kolumny_tekst = ["club", "klub", "strona",
                 "voivodeship", "profession", "educationLevel", "districtName",
                 "birthLocation", "secondName", "inactiveCause", "waiverDesc",
                 "firstName", "lastName", "firstLastName", "lastFirstName",
                 "accusativeName", "genitiveName", "email"]
table[kolumny_tekst] = table[kolumny_tekst].fillna("brak")

# print(table[kolumny_tekst + ["wiek", "numberOfVotes", "districtNum", "active_num"]].isna().sum().sum())  # ma być 0

kolumny_klub = [ "klub", "strona"] # bez club bo to przed scaleniem
kolumny_kategorie = ["voivodeship", "districtName", "educationLevel"]
kolumny_tozsamosc = ["firstName", "secondName", "lastName", "firstLastName", "lastFirstName", "accusativeName", "genitiveName", "email", "birthLocation"]   # bez name bo to kopia firstLastName
kolumny_liczbowe = ["wiek", "numberOfVotes", "districtNum"]
kolumny_rola = ["waiverDesc", "profession", "inactiveCause" ]


preproces = ColumnTransformer([
    ("klub", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kolumny_klub),
    ("kategorie", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kolumny_kategorie),
    ("tozsamosc", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kolumny_tozsamosc),
    ("rola", OneHotEncoder(handle_unknown="ignore", sparse_output=False), kolumny_rola),
    ("liczbowe",  StandardScaler(), kolumny_liczbowe),
    ("active",    "passthrough", ["active_num"])
])

X_pelne = preproces.fit_transform(table).astype("float32")
nazwy = preproces.get_feature_names_out()

# print(X_pelne.shape)
# print(len(nazwy))          # ma być równe X_pelne.shape[1]

bez_klubu_maska = ~np.char.startswith(nazwy.astype(str), "klub__")
X_bez_klubu = X_pelne[:, bez_klubu_maska]

bez_klubu_i_funkcji_maska = bez_klubu_maska & (~np.char.startswith(nazwy.astype(str), "rola__"))
X_bez_klubu_funkcji = X_pelne[:, bez_klubu_i_funkcji_maska]

# print(X_pelne.shape, X_bez_klubu.shape, X_bez_klubu_funkcji.shape)
