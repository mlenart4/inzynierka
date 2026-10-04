import networkx as nx

G = nx.read_graphml("NET_TERM_10.graphml")

print(G)

# print(list(G.nodes(data=True))[:5])
# print(list(G.edges(data=True))[:5])
