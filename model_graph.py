import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import numpy as np
from pipeline import krawedzie, X_pelne

def build_graph(krawedzie, X):
    data = HeteroData()

    data['posel'].x = torch.as_tensor(X, dtype = torch.float32) # niby juz mamy float32 ale tak w razie co

    # kanal pozytywny
    pos = krawedzie[krawedzie["p"] > 0]
    data["posel", "pos", "posel"].edge_index = torch.as_tensor(np.stack([pos["nadawca"].to_numpy(), pos["odbiorca"].to_numpy()]), dtype=torch.long)  # torch.long: tablica na tensor z l. całkowitymi (wymagane dla indeksow)
    data["posel", "pos", "posel"].edge_attr = torch.as_tensor(np.log1p(pos["p"].to_numpy()), dtype=torch.float32).unsqueeze(1)  # unsqueeze bo zeby interperetowalo jako wiele krawedzi z jedna cecha (waga) a nie jedna krawedzi i duzo cech

    # kanal neutralny
    neu = krawedzie[krawedzie["z"] > 0]
    data["posel", "neu", "posel"].edge_index = torch.as_tensor(np.stack([neu["nadawca"].to_numpy(), neu["odbiorca"].to_numpy()]), dtype=torch.long)
    data["posel", "neu", "posel"].edge_attr = torch.as_tensor(np.log1p(neu["z"].to_numpy()), dtype=torch.float32).unsqueeze(1)

    # kanal negatywny
    neg = krawedzie[krawedzie["n"] > 0]
    data["posel", "neg", "posel"].edge_index = torch.as_tensor(np.stack([neg["nadawca"].to_numpy(), neg["odbiorca"].to_numpy()]), dtype=torch.long)
    data["posel", "neg", "posel"].edge_attr = torch.as_tensor(np.log1p(neg["n"].to_numpy()), dtype=torch.float32).unsqueeze(1)

    transformacja = T.ToUndirected(merge=False)
    data = transformacja(data)

    # print(data.edge_types)  # tescik

    assert len(data.edge_types) == 6 

    return data


# # testowe budowanie grafu
# graph = build_graph(krawedzie, X_pelne)
# for et in graph.edge_types:
#     print(et, graph[et].edge_index.shape[1])
# print(graph["posel"].x.shape)