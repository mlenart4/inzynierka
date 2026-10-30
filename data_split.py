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
krawedzie["warstwa"] = krawedzie["typ_pary"] + "_" + krawedzie["etykieta"]

z_etykieta_maska = krawedzie["etykieta"].notna()
z_etykieta = krawedzie[z_etykieta_maska]
y = z_etykieta["warstwa"]
grupy = z_etykieta["grupa"]

ZIARNA = [21, 37, 42, 60, 67]

PLIK = Path("podzialy.npz")
if PLIK.exists():
    dane = np.load(PLIK)
    ZIARNA = dane["ziarna"].tolist()

    podzialy_walidacyjne = {}
    for ziarno in ZIARNA:
        for nr in range(5):
            podzialy_walidacyjne[(ziarno, nr)] = {
                czesc: dane[f"z{ziarno}_f{nr}_{czesc}"]
                for czesc in ["trening", "walidacja", "test"]
            }

else:
    podzialy = {}  # (ziarno, nr faldy)

    for ziarno in ZIARNA:
        
        podzial = StratifiedGroupKFold(n_splits = 5, shuffle= True, random_state=ziarno)

        for numer, (trening, test) in enumerate(podzial.split(z_etykieta, y, grupy)):
            podzialy[(ziarno, numer)] = {
                "trening": z_etykieta.index[trening].to_numpy(),
                "test": z_etykieta.index[test].to_numpy(),}
         

    podzialy_walidacyjne = {}

    for (ziarno, numer), p in podzialy.items():
        treningowa = z_etykieta.loc[p["trening"]]

        podzial_walidacja = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=ziarno)
        for trening_pozycja, walid_pozycja in podzial_walidacja.split(treningowa, treningowa["warstwa"], treningowa["grupa"]):
            break

        podzialy_walidacyjne[(ziarno, numer)] = {
            "trening": treningowa.index[trening_pozycja].to_numpy(),
            "walidacja": treningowa.index[walid_pozycja].to_numpy(),
            "test": p["test"],
        }

    do_zapisu = {}
    for (ziarno, numer), p in podzialy_walidacyjne.items():
        for czesc in ["trening", "walidacja", "test"]:
            do_zapisu[f"z{ziarno}_f{numer}_{czesc}"] = p[czesc]

    np.savez_compressed(PLIK, ziarna=np.array(ZIARNA), **do_zapisu)

def zaslon_pary(krawedzie, przykladowy_podzial):  # tego bedziemy uzywac do statystyk sieciowych
    grupy_walidacji = set( krawedzie.loc[przykladowy_podzial["walidacja"], "grupa"])
    grupy_testu = set(krawedzie.loc[przykladowy_podzial["test"], "grupa"])
    zasloniete = grupy_walidacji | grupy_testu

    maska = krawedzie["grupa"].isin(zasloniete)

    return krawedzie[~maska]

def zasloniecie_do_epoki(p, krawedzie, widoczne, rng, procent = 0.7):
    grupy_trening = np.array(sorted(set(krawedzie.loc[p["trening"], "grupa"]))) # bez duplikatow, posortowan rosnaco

    losowo = rng.permutation(grupy_trening)
    ile = int(procent * len(losowo))

    grupy_graf = losowo[:ile] # te 70%
    grupy_strat = losowo[ile:] # 30%

    do_nauki = krawedzie.loc[p["trening"]]
    do_nauki = do_nauki[do_nauki["grupa"].isin(grupy_strat)]
    grafowa_tabela = widoczne[~widoczne["grupa"].isin(grupy_strat)]

    return grafowa_tabela, do_nauki, grupy_strat


if __name__ == "__main__":
    print("liczba podziałów (25):", len(podzialy_walidacyjne))
    for ziarno in ZIARNA:
        testy = np.concatenate([podzialy_walidacyjne[(ziarno, nr)]["test"] for nr in range(5)])
        print(f"ziarno {ziarno}: pary w testach (1536) i unikalne (1536):", len(testy), len(set(testy)))
    for c in ["trening", "walidacja", "test"]:
        print(f"średni rozmiar {c} (ok. 1107 / 122 / 307):", np.mean([len(p[c]) for p in podzialy_walidacyjne.values()]).round(1))
    zaslonieto = [len(krawedzie) - len(zaslon_pary(krawedzie, p)) for p in podzialy_walidacyjne.values()]
    print("wiersze zasłaniane z tabeli w fałdzie (ok. 490):", np.mean(zaslonieto).round(1))