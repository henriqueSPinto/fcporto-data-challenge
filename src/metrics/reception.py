"""Métrica 1, o teste do benefício: espaço à receção dos avançados, via freeze frames 360.

O rest-attack promete que o avançado que fica alto e poupa energia é uma **saída fresca em
espaço**. Aqui esse benefício é testado diretamente. Para cada receção (`Ball Receipt*`) com
freeze frame, mede-se:

  - `dist`: distância ao francês de campo mais próximo (o espaço à receção). Maior = mais livre.
  - `near`: quantos franceses a <=5m (proxy de marcação múltipla / gravidade).
  - `x`  : profundidade da receção (frame de Portugal, ataca x=120). Maior = mais alto.

Construído para poder **desconfirmar**: se o Leão recebe alto *mas apertado* (1v1, não cercado),
o mecanismo de "saída livre nas costas" não disparou, e o valor ofensivo dele tem de vir de
outro lado (conduzir a bola, ver `offense.py`). O 360 é load-bearing: sem as posições de todos
os jogadores não há forma de medir o espaço à volta de quem recebe.

Reutiliza a máquina de distância-ao-oponente do `space.py` (o lado do custo), aplicada ao
avançado como *actor* em vez do portador da França, incluindo o mesmo controlo de contenção:
medir só onde a bola cai dentro da área visível da câmara. Sem isso, receções com a bola fora de
imagem devolvem um "mais próximo" de dezenas de metros (só se veem franceses longe) e inflam o espaço.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # como pacote: python -m src.metrics.reception
    from ..build import build_all
    from .pressing import FRONT_LINE
    from .space import _ball_visible, YARD_TO_M
except ImportError:  # como script solto
    from build import build_all
    from pressing import FRONT_LINE
    from space import _ball_visible, YARD_TO_M

# Pares para o mapa de receção: Leão (o caso) e Bernardo (o contraste médio-fundo, lido com controlo
# de profundidade). O Ronaldo fica de fora de propósito: como ponta-de-lança, a marcação apertada dele
# é da posição (marcado pelos centrais), não diz nada sobre rest-attack e confundia a leitura.
RECEPTION_PLAYERS = {k: v for k, v in FRONT_LINE.items() if k != "Ronaldo"}

# Buckets de espaço à receção, em metros (distância convertida da grade StatsBomb). Apertado -> livre.
BUCKETS = [(-np.inf, 2, "<=2m"), (2, 5, "2-5m"), (5, 10, "5-10m"), (10, np.inf, ">=10m")]
NEAR_RADIUS = 5.0   # raio em metros para contar corpos franceses perto (proxy de marcação)
FINAL_THIRD_X = 80  # início do último terço no frame de Portugal


def _bucket(d: float) -> str:
    for lo, hi, lab in BUCKETS:
        if lo <= d < hi:
            return lab
    return BUCKETS[-1][2]


def reception_space(match_id: int = 3942349, players: dict | None = None):
    """Espaço à receção por avançado, via 360.

    Devolve (summary, df):
      - summary: por jogador, n receções, mediana de espaço, x médio, % apertado (<=5m),
        % no último terço, % com 2+ franceses perto.
      - df: uma linha por receção com identidade do jogador (para o mapa espacial).
    """
    players = players or FRONT_LINE
    data = build_all(match_id)
    ev, fz, va = data["events"], data["freeze_frames"], data["visible_areas"]

    # Franceses de campo (oponentes do avançado português que recebe).
    opp = fz[(~fz["teammate"]) & (~fz["keeper"])]
    opp_by_event = {eid: g[["x", "y"]].to_numpy() for eid, g in opp.groupby("event_id")}

    rows = []
    for lab, name in players.items():
        r = ev[(ev["player"] == name) & (ev["type"] == "Ball Receipt*")
               & ev["has_360"] & ev["location_x"].notna()]
        for _, e in r.iterrows():
            pts = opp_by_event.get(e["id"])
            if pts is None or len(pts) == 0:
                continue
            ball = np.array([e["location_x"], e["location_y"]])
            dists = np.sqrt(((pts - ball) ** 2).sum(axis=1)) * YARD_TO_M
            dmin = float(dists.min())
            rows.append({
                "player": lab, "x": float(e["location_x"]), "y": float(e["location_y"]),
                "dist": dmin, "near": int((dists <= NEAR_RADIUS).sum()), "bucket": _bucket(dmin),
                "visible": _ball_visible(va.get(e["id"]), e["location_x"], e["location_y"]),
            })

    df = pd.DataFrame(rows)
    # Controlo de contenção: se a bola cai fora do enquadramento da câmara, os franceses perto podem
    # estar fora de imagem e o "mais próximo" seria lixo (dá 60-90m). Medir só onde é fiável.
    coverage = float(df["visible"].mean())
    df = df[df["visible"]].reset_index(drop=True)
    order = [k for k in players]
    summary = (
        df.groupby("player")
        .agg(n=("dist", "count"),
             espaco_mediana=("dist", "median"),
             x_medio=("x", "mean"),
             pct_apertado=("dist", lambda s: (s <= 5).mean()),
             pct_ult_terco=("x", lambda s: (s >= FINAL_THIRD_X).mean()),
             pct_cercado=("near", lambda s: (s >= 2).mean()))
        .reindex(order)
        .round(2)
    )
    summary.attrs["visible_coverage"] = coverage
    return summary, df


if __name__ == "__main__":
    summary, df = reception_space()
    print("== Métrica 1 · teste do benefício · espaço à receção (360) ==")
    print("(dist = distância ao francês mais próximo ao receber; maior = mais livre)")
    print(summary.to_string())
    print(f"\ncontrolo de contenção (bola dentro da área visível): "
          f"{summary.attrs['visible_coverage']:.1%} das receções medidas")
