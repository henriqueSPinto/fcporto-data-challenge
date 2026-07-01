"""Transformação dos dados em bruto em DataFrames tidy, prontos para análise.

Produz cinco estruturas:
- ``events``        : uma linha por evento, achatado (colunas-chave + campos por tipo).
- ``freeze_frames`` : formato *longo*, uma linha por (evento, jogador visível) dos dados 360.
- ``visible_areas`` : dict event_id -> polígono do enquadramento da câmara (dados 360).
- ``players``       : metadados dos jogadores (a partir dos lineups).
- ``context``       : sumário do jogo por equipa (xG, remates, posse).

O JOIN 360 (peça central do exercício): cada entrada dos dados 360 traz um
``event_uuid`` que corresponde ao ``id`` de um evento. Nem todos os eventos têm
freeze frame, a relação é *parcial* (left join). Por isso marca-se ``has_360``
nos eventos e guardam-se os freeze frames numa tabela própria, ligável por
``event_id``. Mantê-los em formato longo (um jogador por linha) é o que torna
trivial, mais à frente, calcular distâncias e contagens de adversários à volta
de uma ação.
"""
from __future__ import annotations

import pandas as pd

try:  # quando corrido como pacote: python -m src.build
    from .ingest import load_match
except ImportError:  # quando corrido como script solto: python src/build.py
    from ingest import load_match


# ---------------------------------------------------------------------------
# Helpers de extração segura de campos aninhados
# ---------------------------------------------------------------------------
def _get(d, *path, default=None):
    """Acede a d[path[0]][path[1]]... devolvendo `default` se algo faltar."""
    cur = d
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
    return cur if cur is not None else default


def _xy(d, key="location"):
    """Devolve (x, y) de um campo de localização [x, y]; (None, None) se ausente."""
    loc = d.get(key) if isinstance(d, dict) else None
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        return loc[0], loc[1]
    return None, None


# ---------------------------------------------------------------------------
# events -> DataFrame tidy
# ---------------------------------------------------------------------------
def build_events(events_json) -> pd.DataFrame:
    """Achata a lista de eventos num DataFrame, projetando só os campos úteis.

    Nota sobre coordenadas: as localizações vêm no referencial 120x80 da
    StatsBomb. A *direção de jogo* (que lado ataca cada equipa) não é
    normalizada aqui de propósito: guarda-se a localização em bruto e o
    `period`, e a direção é tratada na camada de métricas, depois de validada
    empiricamente no EDA.
    """
    rows = []
    for ev in events_json:
        x, y = _xy(ev)
        pass_ex, pass_ey = _xy(ev.get("pass", {}), "end_location")
        carry_ex, carry_ey = _xy(ev.get("carry", {}), "end_location")
        shot_ex, shot_ey = _xy(ev.get("shot", {}), "end_location")
        rows.append({
            "id": ev.get("id"),
            "index": ev.get("index"),
            "period": ev.get("period"),
            "timestamp": ev.get("timestamp"),
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "type": _get(ev, "type", "name"),
            "team": _get(ev, "team", "name"),
            "player": _get(ev, "player", "name"),
            "player_id": _get(ev, "player", "id"),
            "position": _get(ev, "position", "name"),
            "possession": ev.get("possession"),
            "possession_team": _get(ev, "possession_team", "name"),
            "play_pattern": _get(ev, "play_pattern", "name"),
            "location_x": x,
            "location_y": y,
            "duration": ev.get("duration"),
            "under_pressure": bool(ev.get("under_pressure", False)),
            "counterpress": bool(ev.get("counterpress", False)),
            # --- passe ---
            "pass_recipient": _get(ev, "pass", "recipient", "name"),
            "pass_length": _get(ev, "pass", "length"),
            "pass_angle": _get(ev, "pass", "angle"),
            "pass_height": _get(ev, "pass", "height", "name"),
            "pass_end_x": pass_ex,
            "pass_end_y": pass_ey,
            "pass_outcome": _get(ev, "pass", "outcome", "name"),  # None = passe completo
            "pass_type": _get(ev, "pass", "type", "name"),
            # --- remate ---
            "shot_xg": _get(ev, "shot", "statsbomb_xg"),
            "shot_outcome": _get(ev, "shot", "outcome", "name"),
            "shot_type": _get(ev, "shot", "type", "name"),
            "shot_body_part": _get(ev, "shot", "body_part", "name"),
            "shot_end_x": shot_ex,
            "shot_end_y": shot_ey,
            # --- conduções ---
            "carry_end_x": carry_ex,
            "carry_end_y": carry_ey,
            # --- ações defensivas / duelos ---
            "duel_outcome": _get(ev, "duel", "outcome", "name"),
            "duel_type": _get(ev, "duel", "type", "name"),
            "interception_outcome": _get(ev, "interception", "outcome", "name"),
            "ball_recovery_failure": bool(
                _get(ev, "ball_recovery", "recovery_failure", default=False)
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 360 -> freeze frames (formato longo) + áreas visíveis
# ---------------------------------------------------------------------------
def build_freeze_frames(three_sixty_json) -> pd.DataFrame:
    """Explode os freeze frames num DataFrame longo (um jogador visível por linha).

    Flags por jogador:
    - ``teammate``: é colega de quem fez a ação (o "actor")?
    - ``actor``   : é o próprio jogador que executa o evento?
    - ``keeper``  : é guarda-redes?
    """
    rows = []
    for entry in three_sixty_json:
        eid = entry.get("event_uuid")
        for p in entry.get("freeze_frame", []):
            x, y = _xy(p)
            rows.append({
                "event_id": eid,
                "x": x,
                "y": y,
                "teammate": bool(p.get("teammate", False)),
                "actor": bool(p.get("actor", False)),
                "keeper": bool(p.get("keeper", False)),
            })
    return pd.DataFrame(rows)


def build_visible_areas(three_sixty_json) -> dict:
    """dict event_id -> polígono da área visível (lista achatada [x1, y1, x2, y2, ...])."""
    return {e.get("event_uuid"): e.get("visible_area") for e in three_sixty_json}


# ---------------------------------------------------------------------------
# players (a partir dos lineups)
# ---------------------------------------------------------------------------
def build_players(lineups_json) -> pd.DataFrame:
    rows = []
    for team in lineups_json:
        tname = team.get("team_name")
        for p in team.get("lineup", []):
            positions = p.get("positions", [])
            rows.append({
                "player_id": p.get("player_id"),
                "player": p.get("player_name"),
                "nickname": p.get("nickname"),
                "team": tname,
                "jersey": p.get("jersey_number"),
                "primary_position": positions[0]["position"] if positions else None,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Contexto do jogo
# ---------------------------------------------------------------------------
def build_context(events: pd.DataFrame) -> pd.DataFrame:
    """Sumário por equipa: xG, remates, remates enquadrados, passes e posse (proxy).

    Exclui o período 5 (desempate por penáltis). A posse é um *proxy* pela quota
    de passes, não é tempo de bola, mas correlaciona bem e é transparente.
    """
    reg = events[events["period"] <= 4]
    shots = reg[reg["type"] == "Shot"]
    passes = reg[reg["type"] == "Pass"]
    on_target = {"Goal", "Saved", "Saved To Post"}
    total_passes = len(passes)

    rows = []
    for team in sorted(events["team"].dropna().unique()):
        t_shots = shots[shots["team"] == team]
        t_passes = passes[passes["team"] == team]
        rows.append({
            "team": team,
            "xg": round(float(t_shots["shot_xg"].sum()), 2),
            "shots": len(t_shots),
            "shots_on_target": int(t_shots["shot_outcome"].isin(on_target).sum()),
            "passes": len(t_passes),
            "possession_pct": (
                round(100 * len(t_passes) / total_passes, 1) if total_passes else None
            ),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------
def build_all(match_id, data_dir=None) -> dict:
    """Corre o pipeline completo e devolve um dict com todas as estruturas."""
    raw = load_match(match_id) if data_dir is None else load_match(match_id, data_dir)

    events = build_events(raw["events"])
    freeze = build_freeze_frames(raw["three_sixty"])

    # Marca quais os eventos que têm freeze frame (o tal left join).
    ids_360 = set(freeze["event_id"].unique())
    events["has_360"] = events["id"].isin(ids_360)

    return {
        "events": events,
        "freeze_frames": freeze,
        "players": build_players(raw["lineups"]),
        "visible_areas": build_visible_areas(raw["three_sixty"]),
        "context": build_context(events),
    }


if __name__ == "__main__":
    import sys

    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 3942349
    data = build_all(mid)
    ev, fz = data["events"], data["freeze_frames"]

    print(f"events: {ev.shape[0]} linhas x {ev.shape[1]} colunas")
    cov = 100 * ev["has_360"].mean()
    print(f"cobertura 360: {int(ev['has_360'].sum())}/{len(ev)} eventos = {cov:.1f}%")
    n_frames = fz["event_id"].nunique()
    print(
        f"freeze_frames: {len(fz)} linhas (jogador-evento) | "
        f"{n_frames} eventos com frame | "
        f"média {len(fz) / max(n_frames, 1):.1f} jogadores/frame"
    )
    print("\ncontexto:")
    print(data["context"].to_string(index=False))
    print("\ntipos de evento mais comuns:")
    print(ev["type"].value_counts().head(10).to_string())
