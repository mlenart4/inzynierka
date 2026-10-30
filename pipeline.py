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

koalicja = {"KO", "PSL-TD", "Polska2050", "Lewica"}
opozycja = {"PiS", "Konfederacja", "Razem", "Republikanie", "niez.", "Konfederacja_KP"}

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

 
table["active_num"] = table["active"].astype(float)   # true false na 1 0 

kolumny_tekst = ["club", "klub", "strona",
                 "voivodeship", "profession", "educationLevel", "districtName",
                 "birthLocation", "secondName", "inactiveCause", "waiverDesc",
                 "firstName", "lastName", "firstLastName", "lastFirstName",
                 "accusativeName", "genitiveName", "email"]
table[kolumny_tekst] = table[kolumny_tekst].fillna("brak")


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


bez_klubu_maska = ~np.char.startswith(nazwy.astype(str), "klub__")
X_bez_klubu = X_pelne[:, bez_klubu_maska]

bez_klubu_i_funkcji_maska = bez_klubu_maska & (~np.char.startswith(nazwy.astype(str), "rola__"))
X_bez_klubu_funkcji = X_pelne[:, bez_klubu_i_funkcji_maska]

###### sprawdzamy :)))))
if __name__ == "__main__":
    print("=== 2.1 Etykieta ===")
    print("liczba posłów (498):", len(table))
    print("wiersze tabeli par (10373):", len(krawedzie))
    print("same neutralne p=n=0 (3993):", liczba_neutralnych)
    print("remisy p=n>0 (105):", liczba_remisow)
    print("pary ze znakiem większości (6275):", (krawedzie["znak"] != 0).sum())

    print("\n=== 2.2 Próg pewności 0,8 ===")
    print("pary z etykietą (1536):", krawedzie["etykieta"].notna().sum())
    print("etykiety + (451):", (krawedzie["etykieta"] == "+").sum())
    print("etykiety - (1085):", (krawedzie["etykieta"] == "-").sum())
    print("wariant p+n>=3 (783):", krawedzie["etykieta_min3"].notna().sum())

    print("\n=== 2.3 Kluby, strony, typy par ===")
    print("liczba klubów po scaleniu (10):", table["klub"].nunique())
    print("posłowie koalicji (263):", (table["strona"] == "koalicja").sum())
    print("posłowie opozycji (235):", (table["strona"] == "opozycja").sum())
    print("typy par (suma 10373, bez BRAK):")
    print(krawedzie["typ_pary"].value_counts())
    print("typ pary x etykieta (wewnątrzklubowych ujemnych ma być 2):")
    print(krawedzie.groupby(["typ_pary", "etykieta"]).size())

    print("\n=== 2.3 Macierze cech (wiersze 498) ===")
    print("pełne:", X_pelne.shape)
    print("bez klubu:", X_bez_klubu.shape)
    print("bez klubu i roli:", X_bez_klubu_funkcji.shape)
    print("kolumn klubu (pełne - bez klubu, 12):", X_pelne.shape[1] - X_bez_klubu.shape[1])
    print("kolumn roli (bez klubu - bez klubu i roli, 186):", X_bez_klubu.shape[1] - X_bez_klubu_funkcji.shape[1])
    print("są NaN w macierzy pełnej (False):", np.isnan(X_pelne).any())

    idx_num = [i for i, n in enumerate(nazwy.astype(str)) if n.startswith("liczbowe__")]
    print("średnie cech liczbowych (około 0):", X_pelne[:, idx_num].mean(axis=0).round(3))
    print("odchylenia cech liczbowych (około 1):", X_pelne[:, idx_num].std(axis=0).round(3))