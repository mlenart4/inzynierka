from pipeline import krawedzie
import numpy as np
import pandas as pd

mniejszy = np.minimum(krawedzie["nadawca"], krawedzie["odbiorca"])
wiekszy = np.maximum(krawedzie["nadawca"], krawedzie["odbiorca"])

krawedzie["grupa"] = krawedzie.groupby([mniejszy, wiekszy]).ngroup()

print(krawedzie["grupa"].nunique())               # mniej niż 10 373
print(krawedzie["grupa"].value_counts().max())    # ma być 2


