"""Métrica 1, o payoff: contributo ofensivo por jogador (o outro lado do trade-off do rest-attack).

O rest-attack tem um custo (a carga cai no lateral e no pivot, ver `pressing.py`) e um benefício:
o avançado que poupa na pressão pode render na criação. Aqui esse benefício mede-se de duas formas
robustas, sem cair no remate (0.13 xG do Leão, a lente errada para um extremo):

  - `threat_by_player`: ameaça gerada (o proxy transparente da Métrica 2), repartida por tipo de
    ação. Mostra *como* se cria perigo (a conduzir vs a passar), não só quanto.
  - `progressive_carries`: conduções progressivas na linha da definição FBref, que já exclui a
    saída de bola dos centrais lá atrás. Volume de conduzir a bola para o perigo.

Reutiliza o motor de ameaça da Métrica 2, portanto não há proxy novo a validar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # como pacote: python -m src.metrics.offense
    from ..build import build_all
    from .momentum import threat_events
except ImportError:  # como script solto
    from build import build_all
    from momentum import threat_events

# Limiar da definição de progressive carry: 10 jardas de avanço para a baliza. As coordenadas
# da StatsBomb já vêm em jardas (grade 120x80), por isso não há conversão a fazer.
PROG_YARDS = 10.0


def threat_by_player(match_id: int = 3942349, team: str = "Portugal") -> pd.DataFrame:
    """Ameaça gerada (xT-lite) por jogador, repartida por tipo de ação (passe/condução/remate)."""
    te = threat_events(match_id)
    t = te[te["team"] == team]
    frag = t.pivot_table(index="player", columns="type", values="threat",
                         aggfunc="sum", fill_value=0.0)
    for col in ["Carry", "Pass", "Shot"]:      # garante colunas mesmo se alguma faltar
        if col not in frag.columns:
            frag[col] = 0.0
    frag["total"] = frag[["Carry", "Pass", "Shot"]].sum(axis=1)
    return frag.sort_values("total", ascending=False)


def progressive_carries(match_id: int = 3942349, team: str = "Portugal") -> pd.Series:
    """Conduções progressivas por jogador (na linha da definição FBref).

    Uma condução conta se aproxima a bola da baliza adversária em pelo menos 10 jardas *ou* entra
    na área, e termina na metade ofensiva (exclui a saída de bola dos centrais lá atrás). A baliza
    adversária está em (120, 40) no frame de cada equipa.
    """
    ev = build_all(match_id)["events"]
    c = ev[(ev["type"] == "Carry") & (ev["team"] == team)
           & ev["carry_end_x"].notna()].copy()

    def dist_to_goal(x, y):
        return np.sqrt((120 - x) ** 2 + (40 - y) ** 2)

    prog = dist_to_goal(c["location_x"], c["location_y"]) - dist_to_goal(c["carry_end_x"], c["carry_end_y"])
    into_box = (c["carry_end_x"] >= 102) & c["carry_end_y"].between(18, 62)
    mask = ((prog >= PROG_YARDS) | into_box) & (c["carry_end_x"] >= 60)
    return c[mask].groupby("player").size().sort_values(ascending=False)


if __name__ == "__main__":
    frag = threat_by_player()
    print("== Ameaça gerada (xT-lite) por tipo, Portugal (top 8) ==")
    print(frag.head(8)[["Carry", "Pass", "Shot", "total"]].round(2).to_string())
    print("\n== Conduções progressivas (FBref), Portugal (top 8) ==")
    print(progressive_carries().head(8).to_string())
