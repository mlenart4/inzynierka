import numpy as np
from data_split import krawedzie 
import torch
import torch.nn as nn
from copy import deepcopy
from pipeline import X_pelne

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

def model_C(p, y_trening, ziarno):        # MLP na cechach węzłów
    y_walidacja = (krawedzie.loc[p["walidacja"], "etykieta"] == "+").to_numpy().astype(int)   # musimy miec do uczenia modelu
    
    wejscia = {}
    for podzial in ["trening", "walidacja", "test"]:
        nadawca = krawedzie.loc[p[podzial], "nadawca"].to_numpy()
        adresat = krawedzie.loc[p[podzial], "odbiorca"].to_numpy()
        wejscia[podzial] = np.hstack([X_pelne[nadawca], X_pelne[adresat]])

    # print("trening", wejscia["trening"].shape)
    # print("walidacja", wejscia["walidacja"].shape)
    # print("test", wejscia["test"].shape)

    wejscia["trening"] = torch.as_tensor(wejscia["trening"], dtype=torch.float32)
    wejscia["walidacja"] = torch.as_tensor(wejscia["walidacja"], dtype=torch.float32)
    wejscia["test"] = torch.as_tensor(wejscia["test"], dtype=torch.float32)
    y_trening = torch.as_tensor(y_trening, dtype=torch.float32)
    y_walidacja = torch.as_tensor(y_walidacja, dtype=torch.float32)

    

    return wejscia