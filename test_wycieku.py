from data_split import krawedzie, podzialy_walidacyjne, zaslon_pary
from model_graph import build_graph
from pipeline import X_pelne
import numpy as np

def czy_wycieka (graf, krawedzie, p):
    g = graf["posel"].x.shape[0]

    # widoczne = zaslon_pary(krawedzie, p) # to jednak chyba bez sensu bo sprawdzamy funkcje nią samą xd
    # zasloniete = krawedzie[~krawedzie.index.isin(widoczne.index)]

    zasloniete_grupy = set(krawedzie.loc[p["walidacja"], "grupa"]) | set(krawedzie.loc[p["test"], "grupa"])
    zasloniete = krawedzie[krawedzie["grupa"].isin(zasloniete_grupy)]

    zasloniete_pary = {(min(a, b), max(a, b)) for a, b in zip(zasloniete["nadawca"], zasloniete["odbiorca"])}

    for typ_krawedzi in graf.edge_types:
        indeks = graf[typ_krawedzi].edge_index  # lista krawedzi tego typu
        w_grafie = {(min(a, b), max(a, b)) for a, b in zip(indeks[0].tolist(), indeks[1].tolist())}

        czy_wspolne = zasloniete_pary & w_grafie

        assert not czy_wspolne
    print("nie wycieka")


if __name__ == "__main__":
    for klucz, p in podzialy_walidacyjne.items():
        widoczne = zaslon_pary(krawedzie, p)
        graf = build_graph(widoczne, X_pelne)
        czy_wycieka(graf, krawedzie, p)
