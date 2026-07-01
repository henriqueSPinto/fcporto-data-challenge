"""Métrica 2 (independente): índice de momentum/ameaça ao longo do tempo.

Proxy de ameaça transparente (não é um xT treinado, é deliberadamente simples e legível):
cada ação ofensiva vale 'pontos de ameaça' conforme progride para zonas perigosas.
  - remate:                    1.0
  - entrada na grande área:    0.5
  - entrada no último terço:   0.2
Soma por equipa por minuto; `momentum(t) = ameaça(Portugal) menos ameaça(França)`, suavizado.
Positivo = Portugal por cima; negativo = França. É o 'attack momentum' estilo SofaScore.

É independente da hipótese Leão/Ronaldo: descreve o fluxo do jogo por si só, o que a torna
robusta à amostra pequena de um jogo. Cada equipa ataca x=120 no seu frame, por isso as zonas
de ameaça leem-se diretamente.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from ..build import build_all
except ImportError:
    from build import build_all

FINAL_THIRD_X = 80.0
BOX_X = 102.0
BOX_Y = (18.0, 62.0)


def _in_box(x, y):
    return (x > BOX_X) & (y >= BOX_Y[0]) & (y <= BOX_Y[1])


def threat_events(match_id: int = 3942349) -> pd.DataFrame:
    """Devolve os eventos ofensivos com a sua pontuação de ameaça (`threat`)."""
    ev = build_all(match_id)["events"]
    e = ev[ev["type"].isin(["Pass", "Carry", "Shot"]) & (ev["period"] <= 4)].copy()

    # Localização final da ação (fim do passe/condução; local do remate).
    e["ex"] = np.where(e["type"] == "Pass", e["pass_end_x"],
                       np.where(e["type"] == "Carry", e["carry_end_x"], e["location_x"]))
    e["ey"] = np.where(e["type"] == "Pass", e["pass_end_y"],
                       np.where(e["type"] == "Carry", e["carry_end_y"], e["location_y"]))

    end_box = _in_box(e["ex"], e["ey"])
    start_box = _in_box(e["location_x"], e["location_y"])
    end_ft = e["ex"] > FINAL_THIRD_X
    start_ft = e["location_x"] > FINAL_THIRD_X

    box_entry = end_box & (~start_box)
    ft_entry = end_ft & (~start_ft) & (~box_entry)

    e["threat"] = 0.0
    e.loc[ft_entry, "threat"] = 0.2
    e.loc[box_entry, "threat"] = 0.5
    e.loc[e["type"] == "Shot", "threat"] = 1.0
    return e[e["threat"] > 0]


def momentum_series(match_id: int = 3942349, window: int = 5) -> pd.DataFrame:
    """Série por minuto: ameaça de cada equipa + momentum suavizado (Portugal − França)."""
    e = threat_events(match_id)
    per_min = (
        e.groupby(["minute", "team"])["threat"].sum().unstack(fill_value=0)
    )
    for t in ("Portugal", "France"):
        if t not in per_min:
            per_min[t] = 0.0
    per_min = per_min.reindex(range(0, int(e["minute"].max()) + 1), fill_value=0.0)
    per_min["momentum_raw"] = per_min["Portugal"] - per_min["France"]
    per_min["momentum"] = (
        per_min["momentum_raw"].rolling(window, center=True, min_periods=1).mean()
    )
    return per_min.reset_index(names="minute")


if __name__ == "__main__":
    s = momentum_series()
    # Ameaça acumulada por bloco de 15 min (quem esteve por cima).
    s["bloco"] = (s["minute"] // 15) * 15
    blocos = s.groupby("bloco")[["Portugal", "France"]].sum().round(1)
    blocos["quem"] = np.where(blocos["Portugal"] > blocos["France"], "POR",
                              np.where(blocos["France"] > blocos["Portugal"], "FRA", "="))
    print("== Ameaça acumulada por bloco de 15 min ==")
    print(blocos.to_string())
    print(f"\nAmeaça total: Portugal {s['Portugal'].sum():.1f} | França {s['France'].sum():.1f}")
    top = s.reindex(s["momentum"].abs().sort_values(ascending=False).index).head(5)
    print("\nMinutos de maior momentum (|Por−Fra| suavizado):")
    print(top[["minute", "momentum"]].round(2).to_string(index=False))
