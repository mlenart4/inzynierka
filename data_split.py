import warnings
warnings.filterwarnings("ignore", message="The least populated class")

from pipeline import krawedzie
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

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

ziarna = [21, 37, 42, 60, 67]
podzialy = {}  # (ziarno, nr faldy)

for ziarno in ziarna:
    
    podzial = StratifiedGroupKFold(n_splits = 5, shuffle= True, random_state=ziarno)

    # faldy = []
    for numer, (trening, test) in enumerate(podzial.split(z_etykieta, y, grupy)):
        grupy_test = set(grupy.iloc[test])
        grupy_trening = set(grupy.iloc[trening])
        podzialy[(ziarno, numer)] = {
            "trening": z_etykieta.index[trening].to_numpy(),
            "test": z_etykieta.index[test].to_numpy(),}
        # print(len(faldy))

    # print([len(f["test"]) for f in faldy])                    # rowne powinny byc 
    # print(sum(len(f["test"]) for f in faldy))                 # 1536

# print(len(podzialy))    # 25


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