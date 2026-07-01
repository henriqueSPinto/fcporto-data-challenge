"""Métrica 1, Camada A: contributo defensivo individual, ajustado por oportunidade.

'Contest defensivo' = o jogador contesta *ativamente* um adversário com a bola no pé:
`{Pressure, Duel(Tackle), Dribbled Past}`. Fora: duelos aéreos, `Ball Recovery` (bola
solta) e `Clearance`. É *outcome-neutral*: conta contests ganhos e perdidos, porque o que
se quer medir é a *vontade de contestar*, não o êxito.

A taxa por-90 mede volume, mas volume depende de **oportunidade** (quantas vezes a bola
passou pela zona do jogador). Por isso ajusta-se por *exposição*: contests por 100 eventos
de bola da França na zona (corredor) do jogador, enquanto ele está em campo. É a lógica do
PPDA levada ao nível individual.

Nota sobre o 360: esta camada é event-based (identidade limpa). A leitura espacial por zona,
com 360, está em `space.py`, porque os freeze frames não têm identidade de jogador para os
jogadores que não estão a fazer a ação.
"""
from __future__ import annotations

import pandas as pd

try:  # como pacote: python -m src.metrics.pressing
    from ..ingest import load_match
    from ..build import build_all
except ImportError:  # como script solto
    from ingest import load_match
    from build import build_all


# Nomes completos StatsBomb do trio da frente (foco da hipótese).
FRONT_LINE = {
    "Leão": "Rafael Alexandre Conceição Leão",
    "Ronaldo": "Cristiano Ronaldo dos Santos Aveiro",
    "Bernardo": "Bernardo Mota Veiga de Carvalho e Silva",
}


def contest_mask(events: pd.DataFrame) -> pd.Series:
    """Máscara dos 'contest defensivos': Pressure, Duel(Tackle), Dribbled Past."""
    is_press = events["type"] == "Pressure"
    is_dribbled = events["type"] == "Dribbled Past"
    is_tackle = (events["type"] == "Duel") & (events["duel_type"] == "Tackle")
    return is_press | is_dribbled | is_tackle


def player_minutes(raw_events) -> pd.DataFrame:
    """Minutos e intervalo em campo por jogador (Starting XI + Substituições)."""
    match_end = max(e["minute"] for e in raw_events if e.get("period", 1) <= 4)
    start, end = {}, {}
    for e in raw_events:
        t = e["type"]["name"]
        if t == "Starting XI":
            for p in e["tactics"]["lineup"]:
                start[p["player"]["name"]] = 0
        elif t == "Substitution":
            end[e["player"]["name"]] = e["minute"]
            start[e["substitution"]["replacement"]["name"]] = e["minute"]
    rows = [
        {"player": name, "start_min": s, "end_min": end.get(name, match_end),
         "minutes": end.get(name, match_end) - s}
        for name, s in start.items()
    ]
    return pd.DataFrame(rows)


def _canonical_france(events: pd.DataFrame) -> pd.DataFrame:
    """Eventos da França no frame de Portugal (120-x, 80-y), para comparar corredores.

    A StatsBomb normaliza cada equipa a atacar x=120; passar a França para o frame de
    Portugal é uma rotação de 180º. Assim o corredor (y) do avançado e a bola da França
    ficam no mesmo referencial. (Convenção validada empiricamente com os laterais.)
    """
    fra = events[(events["team"] == "France") & events["location_x"].notna()].copy()
    fra["cx"] = 120 - fra["location_x"]
    fra["cy"] = 80 - fra["location_y"]
    return fra


def player_zone(events: pd.DataFrame, player: str, half_width: float = 13.3):
    """Corredor (y-band) do jogador: mediana do seu y ± half_width (frame de Portugal)."""
    p = events[events["player"] == player].dropna(subset=["location_y"])
    ymed = p["location_y"].median()
    return max(0.0, ymed - half_width), min(80.0, ymed + half_width)


def pressing_layer_a(match_id: int = 3942349, focus: dict | None = FRONT_LINE) -> pd.DataFrame:
    """Tabela por jogador: contests, per-90 e taxa ajustada por oportunidade.

    Com `focus`, a taxa por oportunidade (`contests_per100_exp`) é calculada para todos os
    jogadores de Portugal; com `focus=None`, devolve só o volume (contests e per-90).
    """
    raw = load_match(match_id)
    ev = build_all(match_id)["events"]
    ev = ev.assign(is_contest=contest_mask(ev))
    mins = player_minutes(raw["events"]).set_index("player")

    # Base: contests + per-90 para todos os jogadores de campo.
    g = (
        ev[ev["player"].notna()]
        .groupby(["player", "team"], as_index=False)
        .agg(
            pressures=("type", lambda s: int((s == "Pressure").sum())),
            contests=("is_contest", "sum"),
        )
    )
    g = g.merge(mins.reset_index()[["player", "minutes"]], on="player", how="left")
    g["contests_p90"] = (g["contests"] / (g["minutes"] / 90)).round(1)

    if not focus:
        return g.sort_values("contests_p90", ascending=False)

    # Camada ajustada por oportunidade, para todos os jogadores de Portugal (a equipa que defende
    # a França). Exposição = bola da França na faixa-y do jogador, enquanto ele está em campo.
    fra = _canonical_france(ev)
    onball = fra[fra["type"].isin(["Pass", "Carry", "Ball Receipt*"]) & (fra["cx"] < 80)]
    exposure = {}
    for name in g.loc[g["team"] == "Portugal", "player"]:
        if name not in mins.index:
            continue
        lo, hi = player_zone(ev, name)
        s, e = mins.loc[name, "start_min"], mins.loc[name, "end_min"]
        in_zone = onball["cy"].between(lo, hi) & onball["minute"].between(s, e)
        exposure[name] = int(in_zone.sum())
    g["exposure"] = g["player"].map(exposure)
    g["contests_per100_exp"] = (100 * g["contests"] / g["exposure"].replace(0, pd.NA)).round(1)
    return g.sort_values("contests_p90", ascending=False)


if __name__ == "__main__":
    tbl = pressing_layer_a()
    front = tbl[tbl["player"].isin(FRONT_LINE.values())].copy()
    front["quem"] = front["player"].map({v: k for k, v in FRONT_LINE.items()})

    print("== Métrica 1 · Camada A · trio da frente (Portugal) ==")
    cols = ["quem", "minutes", "pressures", "contests", "contests_p90", "exposure", "contests_per100_exp"]
    print(front.sort_values("contests_p90", ascending=False)[cols].to_string(index=False))

    print("\n== Contexto: top-8 contests/90 do jogo (todos os jogadores) ==")
    allp = pressing_layer_a(focus=None)
    print(allp.head(8)[["player", "team", "minutes", "contests", "contests_p90"]].to_string(index=False))
