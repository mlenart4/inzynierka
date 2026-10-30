import warnings
warnings.filterwarnings("ignore", message="The least populated class")

from pipeline import krawedzie
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path

mniejszy = np.minimum(krawedzie["nadawca"], krawedzie["odbiorca"])
wiekszy = np.maximum(krawedzie["nadawca"], krawedzie["odbiorca"])

krawedzie["grupa"] = krawedzie.groupby([mniejszy, wiekszy]).ngroup()

# print(krawedzie["grupa"].nunique())               # mniej niż 10 373
# print(krawedzie["grupa"].value_counts().max())    # ma być 2


krawedzie["warstwa"] = krawedzie["typ_pary"] + "_" + krawedzie["etykieta"]
# print(krawedzie["warstwa"].value_counts())
# print(krawedzie["warstwa"].notna().sum())   # 1536

z_etykieta_maska = krawedzie["etykieta"].notna()
z_etykieta = krawedzie[z_etykieta_maska]
y = z_etykieta["warstwa"]
grupy = z_etykieta["grupa"]

ZIARNA = [21, 37, 42, 60, 67]

PLIK = Path("podzialy.npz")
if PLIK.exists():
    dane = np.load("podzialy.npz")
    ZIARNA = dane["ziarna"].tolist()

    podzialy_walidacyjne = {}
    for ziarno in ZIARNA:
        for nr in range(5):
            podzialy_walidacyjne[(ziarno, nr)] = {
                czesc: dane[f"z{ziarno}_f{nr}_{czesc}"]
                for czesc in ["trening", "walidacja", "test"]
            }

    print(len(podzialy_walidacyjne))                                                   # 25
    print(np.array_equal(podzialy_walidacyjne[(ZIARNA[0], 0)]["test"], dane[f"z{ZIARNA[0]}_f0_test"]))   # True

else:
    podzialy = {}  # (ziarno, nr faldy)

    for ziarno in ZIARNA:
        
        podzial = StratifiedGroupKFold(n_splits = 5, shuffle= True, random_state=ziarno)

        for numer, (trening, test) in enumerate(podzial.split(z_etykieta, y, grupy)):
            podzialy[(ziarno, numer)] = {
                "trening": z_etykieta.index[trening].to_numpy(),
                "test": z_etykieta.index[test].to_numpy(),}
         

    podzialy_walidacyjne = {}

    for (ziarno, nr), p in podzialy.items():
        treningowa = z_etykieta.loc[p["trening"]]

        podzial_walidacja = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=ziarno)
        for trening_pozycja, walid_pozycja in podzial_walidacja.split(treningowa, treningowa["warstwa"], treningowa["grupa"]):
            break

        podzialy_walidacyjne[(ziarno, nr)] = {
            "trening": treningowa.index[trening_pozycja].to_numpy(),
            "walidacja": treningowa.index[walid_pozycja].to_numpy(),
            "test": p["test"],
        }

    # print([len(p["walidacja"]) for p in podzialy_walidacyjne.values()])
    # print(np.mean([len(p["trening"]) for p in podzialy_walidacyjne.values()]),
    #       np.mean([len(p["walidacja"]) for p in podzialy_walidacyjne.values()]),
    #       np.mean([len(p["test"]) for p in podzialy_walidacyjne.values()]))

    do_zapisu = {}
    for (ziarno, numer), p in podzialy_walidacyjne.items():
        for czesc in ["trening", "walidacja", "test"]:
            do_zapisu[f"z{ziarno}_f{numer}_{czesc}"] = p[czesc]

    print(len(do_zapisu))

    np.savez_compressed(PLIK, ziarna=np.array(ZIARNA), **do_zapisu)

def zaslon_pary(krawedzie, p):
    grupy_walidacji = set( krawedzie.loc[p["walidacja"], "grupa"])
    grupy_testu = set(krawedzie.loc[p["test"], "grupa"])
    zasloniete = grupy_walidacji | grupy_testu

    maska = krawedzie["grupa"].isin(zasloniete)

    return krawedzie[~maska]

p = podzialy_walidacyjne[(ZIARNA[0], 0)]
widoczne = zaslon_pary(krawedzie, p)

# print(len(krawedzie), len(widoczne), len(krawedzie) - len(widoczne))   # różnica ok. 490
