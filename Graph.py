import networkx as nx
import pandas as pd
import numpy as np

OKREGI_WOJEWODZTW = {
    "dolnośląskie": range(1, 4), "kujawsko-pomorskie": range(4, 6), "lubelskie": range(6, 8),
    "lubuskie": range(8, 9), "łódzkie": range(9, 12), "małopolskie": range(12, 16),
    "mazowieckie": range(16, 21), "opolskie": range(21, 22), "podkarpackie": range(22, 24),
    "podlaskie": range(24, 25), "pomorskie": range(25, 27), "śląskie": range(27, 33),
    "świętokrzyskie": range(33, 34), "warmińsko-mazurskie": range(34, 36),
    "wielkopolskie": range(36, 40), "zachodniopomorskie": range(40, 42),
}
WOJEWODZTWO = {nr: woj for woj, okregi in OKREGI_WOJEWODZTW.items() for nr in okregi}

# poprawianie nazw na te ktore juz mamy w kodzie
PRZEKLADANIE = {
    "club": "club",
    "birth_date": "birthDate",
    "votes": "numberOfVotes",
    "education_level": "educationLevel",
    "profession": "profession",
    "constituency_name": "districtName",
    "name": "firstLastName",
    "seniority": "seniority",
}

surowy = nx.read_graphml("sejm10_insider_graphml_2026-11-02.graphml")

poslowie = [n for n, d in surowy.nodes(data=True) if d.get("pl10_mp") is True]

G = nx.DiGraph()
for n in poslowie:
    d = surowy.nodes[n]
    atrybuty = {nowa_nazwa: d.get(stara_nazwa) for stara_nazwa, nowa_nazwa in PRZEKLADANIE.items()}
    atrybuty["voivodeship"] = WOJEWODZTWO[d["constituency_no"]]
    atrybuty["active"] = not d.get("mandate_end")   # True dla mandate_end puste/brak
    G.add_node(n, **atrybuty)                       # id wezla zostaje tekstem

for u, v, d in surowy.subgraph(poslowie).edges(data=True):
    G.add_edge(u, v, pos=d["pos"], neu=d["neu"], neg=d["neg"], weight=d["weight"])

G.graph.update(surowy.graph)  



# G = nx.read_graphml("NET_TERM_10.graphml")
# print(G)
# print(list(G.nodes(data=True))[:5])

# print(G.nodes[next(iter(G.nodes))])
# print(next(iter(G.edges(data=True))))

