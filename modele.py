import numpy as np

def model_A(p, y_trening, ziarno):      # najczestsza klasa w zbiorze treningowym
    odsetek_dodatnich = y_trening.mean()
    walidacja = np.full(len(p["walidacja"]), odsetek_dodatnich)
    test = np.full(len(p["test"]), odsetek_dodatnich)
    return {"walidacja": walidacja, "test": test}