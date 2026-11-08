from pipeline import X_pelne, table
from data_split import krawedzie, podzialy_walidacyjne, zaslon_pary, ZIARNA
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import numpy as np  
import pandas as pd

def etykieta_liczbowa(idx):
    return (krawedzie.loc[idx, "etykieta"] == "+").to_numpy().astype(int)

l_plusow = (krawedzie["etykieta"] == "+").sum()
l_minusow = (krawedzie["etykieta"] == "-").sum()

# if(l_plusow < l_minusow):
#     print("dominujacą klasa to -")
# else:
#     print("dominujaca klasa to +") 
# # mega szok, - dominuje xd

suma_p = krawedzie.groupby("odbiorca")["p"].sum()
suma_n = krawedzie.groupby("odbiorca")["n"].sum()    
suma_z = krawedzie.groupby("odbiorca")["z"].sum()
suma = suma_p + suma_n + suma_z
top25_zsuma = suma.nlargest(25)

# sprawdzamy czy dziala top 25
# print("25 posłów z największą liczbą wzmianek (p+n+z):")
# print(top25_zsuma)
# print(table.loc[top25_zsuma.index, ["name", "klub"]])
# print(len(top25_zsuma))

top25 = suma.nlargest(25).index
krawedzie["adresat_top25"] = krawedzie["odbiorca"].isin(top25)

def f_opis_i_y(ziarno, numer):
    p = podzialy_walidacyjne[(ziarno, numer)]
    idx = p["test"]
    opis = krawedzie.loc[idx, ["typ_pary", "klub_nadawcy", "klub_adresata", "adresat_top25"]]
    # print("dlugosc tabeli opis: ", len(opis))
    y = etykieta_liczbowa(idx)
    # print(len(y))   # ok 371 
    # print(y.mean())   # okolo 0,26 bo to ilosc +
    assert len(y) == len(opis) # musza meic ta sama dlugosc
    return opis, y, idx

# print(" ile par w top25: ", opis["adresat_top25"].sum())

def f_decyzje(oceny, prog = 0.5):
    return (oceny >= prog).astype(int)

def f_miary(y, oceny, decyzje):      
    # y - tablica prawdziwych znakow 0 1
    # oceny - tablica ocen modelu (np. prawdopodobienstw)
    # decyzje - tablica 0 1 dla oceny >= prog
    
    if len(np.unique(y)) < 2:
        roc_auc = np.nan
        auc_pr = np.nan # bo tu tez musza byc 2 klasy a nie 1

    else:
        roc_auc = roc_auc_score(y, oceny)
        auc_pr = average_precision_score(y, oceny)
    
    f1 = f1_score(y, decyzje)
    f1_makro = f1_score(y, decyzje, average="macro")

    return {
        "roc_auc": roc_auc,
        "auc_pr": auc_pr,
        "f1": f1,
        "f1_makro": f1_makro,}
    

def ocena(y, oceny, opis, prog):
    czykoalicja = opis["typ_pary"].isin(["koalicyjna"]).to_numpy()
    czywewnatrzklubowa = opis["typ_pary"].isin(["wewnatrzklubowa"]).to_numpy()
    czyopozycyjna = opis["typ_pary"].isin(["opozycja_opozycja"]).to_numpy()
    czyrzad_opozycja = opis["typ_pary"].isin(["rzad_opozycja"]).to_numpy()
    czypozatop25 = (~opis["adresat_top25"]).to_numpy()
    caly_zbior = opis["typ_pary"].notna().to_numpy()

    podgrupy = {
        "caly_test": caly_zbior,
        "koalicyjna": czykoalicja,
        "wewnatrzklubowa": czywewnatrzklubowa,
        "opozycja_opozycja": czyopozycyjna,
        "rzad_opozycja": czyrzad_opozycja,
        "pozatop25": czypozatop25,
        "top25": ~czypozatop25,}

    wyniki =[]
    for nazwa, maska in podgrupy.items():
        y_podgrupa = y[maska]
        oceny_podgrupa = oceny[maska]
        decyzje_podgrupa = f_decyzje(oceny_podgrupa, prog)
        miary = f_miary(y_podgrupa, oceny_podgrupa, decyzje_podgrupa)
        for nazwa_miary, wartosc in miary.items():
            wyniki.append({"podgrupa": nazwa, "miara": nazwa_miary, "wartosc": wartosc})

    # w obrebie klubow 
    tabela = opis[["klub_nadawcy", "klub_adresata"]].copy()
    tabela["y"] = y
    tabela["ocena"] = oceny
    rocauc_pr_klub = []
    wagi = []
    for (klub_n, klub_a), g in tabela.groupby(["klub_nadawcy", "klub_adresata"]):
        y_klub = g["y"].to_numpy()
        if len(np.unique(y_klub)) < 2:
            continue
        roc_auc = roc_auc_score(y_klub, g["ocena"].to_numpy())
        rocauc_pr_klub.append(roc_auc)
        l_plus = (y_klub == 1).sum()
        l_minus = (y_klub == 0).sum()
        waga = l_plus * l_minus
        wagi.append(waga)
    wynik = sum(a * w for a, w in zip(rocauc_pr_klub, wagi)) / sum(wagi)
    wyniki .append({"podgrupa": "kluby", "miara": "roc_auc_wazone", "wartosc": wynik})

        

    return pd.DataFrame(wyniki)

# testujemy funkcje ocena
opis_t, y_t, idx = f_opis_i_y(21, 0)
wynik = ocena(y_t, np.full(len(y_t), 0.5), opis_t, 0.5) # stala ocena 0.5 dla wszystkich
print(wynik) # spodziewamy sie roc-auc 05 i auc-pr dla calosci ok 0,26 (odesetek dodatnich)