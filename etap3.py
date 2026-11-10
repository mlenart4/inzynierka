import numpy as np
from data_split import podzialy_walidacyjne
from ocena import etykieta_liczbowa, f_opis_i_y, prog_decyzyjny, ocena
from modele import model_A

ziarno = 21
numer = 0

p = podzialy_walidacyjne[(ziarno, numer)]

y_trening = etykieta_liczbowa(p["trening"])
y_walidacja = etykieta_liczbowa(p["walidacja"])

wyniki_modelu = model_A(p, y_trening, ziarno) # model njaczestszej klasy

prog = prog_decyzyjny(y_walidacja, wyniki_modelu["walidacja"]) # wybor progu
print("prog:", prog) 

opis_t, y_t, idx = f_opis_i_y(ziarno, numer) # opis i etykiety dla zbioru testowego

# ocena na teście
tabela = ocena(y_t, wyniki_modelu["test"], opis_t, prog)
print(tabela)