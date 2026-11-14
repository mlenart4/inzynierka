import numpy as np
from data_split import krawedzie 

def model_A(p, y_trening, ziarno):      # najczestsza klasa w zbiorze treningowym
    odsetek_dodatnich = y_trening.mean()
    walidacja = np.full(len(p["walidacja"]), odsetek_dodatnich)
    test = np.full(len(p["test"]), odsetek_dodatnich)
    return {"walidacja": walidacja, "test": test}

def model_B(p, y_trening, ziarno):      # para klubow
    tabela = krawedzie.loc[p["trening"], ["klub_nadawcy", "klub_adresata", "typ_pary", "strona_nadawcy", "strona_adresata", "komorka"]].copy()
    tabela["y_trening"] = y_trening
    suma = tabela.groupby(["klub_nadawcy", "klub_adresata"])["y_trening"].sum()
    liczba = tabela.groupby(["klub_nadawcy", "klub_adresata"])["y_trening"].count()
    
    # suma_komorki = tabela.groupby(["komorka"])["y_trening"].sum()
    # liczba_komorki = tabela.groupby(["komorka"])["y_trening"].count()
    # print(tabela["komorka"].value_counts())
    k = 5
    odestki = tabela.groupby(["komorka"])["y_trening"].mean()
    # print(odestki)

    slownik_suma = suma.to_dict()
    slownik_liczba = liczba.to_dict()
    wynik = {}
    for podzial in ["walidacja", "test"]:
        pomocnicza = krawedzie.loc[p[podzial], ["klub_nadawcy", "klub_adresata", "komorka"]]
        pary_klubow = list(zip(pomocnicza["klub_nadawcy"], pomocnicza["klub_adresata"]))
        plusy = np.array([slownik_suma.get(para, 0) for para in pary_klubow])
        ile = np.array([slownik_liczba.get(para, 0) for para in pary_klubow])
        tlo = pomocnicza["komorka"].map(odestki).to_numpy() # odsetek + w kazdej z par
        ocena = (plusy + k * tlo) / (ile + k)
        wynik[podzial] = np.nan_to_num(ocena, nan=y_trening.mean())

    return wynik