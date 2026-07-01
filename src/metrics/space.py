"""Métrica 1, Camada B: espaço concedido por zona, via freeze frames 360.

Para cada evento de bola da França nos dois terços defensivos de Portugal *com* freeze frame,
mede a distância do portador (o `actor`, ~ localização do evento) ao **português de campo
mais próximo** (teammate=False, keeper=False). Maior distância = mais espaço concedido.

O 360 é aqui *load-bearing*: sem as posições de todos os jogadores não há forma de medir
espaço. A medida é **anónima de propósito**: não identifica *quem* está mais perto (isso é
a Camada A, event-based, com identidade). Lidas juntas respondem a perguntas diferentes:
  - Camada A: o *jogador* (Leão) engajou?  (atribuição individual)
  - Camada B: a *zona* concedeu espaço?    (resultado coletivo, quem quer que cubra)

Zonas: cada evento é atribuído ao avançado cuja posição-y típica (mediana) está mais próxima,
um particionamento simples e fiel dos corredores (esq/centro/dir ~ Leão/Ronaldo/Bernardo).

Controlo de contenção: o freeze frame só vê os jogadores no enquadramento da câmara. Mede-se o
espaço só nos eventos com a bola dentro da `visible_area` (o defesa mais próximo está então em
imagem); a cobertura é reportada. Sem isto, um defesa fora de imagem inflacionaria o espaço.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath

try:  # como pacote: python -m src.metrics.space
    from ..build import build_all
    from .pressing import FRONT_LINE
except ImportError:  # como script solto
    from build import build_all
    from pressing import FRONT_LINE

# As coordenadas da StatsBomb vêm em jardas (grade 120x80). As distâncias são reportadas em metros
# (mais intuitivas para leitura espacial), com esta conversão única aplicada na origem do cálculo.
YARD_TO_M = 0.9144


def _front_median_y(events: pd.DataFrame) -> dict:
    """Posição-y típica (mediana) de cada avançado, no frame de Portugal."""
    med = {}
    for lab, name in FRONT_LINE.items():
        p = events[events["player"] == name].dropna(subset=["location_y"])
        med[lab] = float(p["location_y"].median())
    return med


def _assign_zone(cy: float, med: dict) -> str:
    """Zona = avançado com a mediana-y mais próxima da posição (canónica) da bola."""
    return min(med, key=lambda k: abs(cy - med[k]))


def _ball_visible(poly, x: float, y: float) -> bool:
    """A bola cai dentro do polígono da área visível da câmara?

    O freeze frame só regista os jogadores que a câmara vê. Se a bola está bem dentro da área
    visível, o defesa mais próximo também está no enquadramento e a medida de espaço é fiável;
    se a bola está no bordo (ou fora), um defesa pode ficar fora de imagem e a medida sobrestima
    o espaço. Serve de controlo de contenção: medir só onde é de confiança.
    """
    if not poly:
        return False
    return bool(MplPath(np.asarray(poly).reshape(-1, 2)).contains_point((x, y)))


def space_conceded(match_id: int = 3942349):
    """Espaço concedido (distância ao português de campo mais próximo) por zona de avançado."""
    data = build_all(match_id)
    ev, fz = data["events"], data["freeze_frames"]
    va = data["visible_areas"]
    med = _front_median_y(ev)

    # Eventos de bola da França na área defensiva de Portugal, com 360.
    fra = ev[
        (ev["team"] == "France")
        & ev["type"].isin(["Pass", "Carry", "Ball Receipt*"])
        & ev["has_360"]
        & ev["location_x"].notna()
    ].copy()
    fra["cy"] = 80 - fra["location_y"]          # frame canónico (Portugal) só para a zona
    fra["cx"] = 120 - fra["location_x"]
    fra = fra[fra["cx"] < 80]                    # França já no meio-campo/ataque

    # Freeze frames dos portugueses de campo (oponentes do actor França).
    opp = fz[(~fz["teammate"]) & (~fz["keeper"])]
    opp_by_event = {eid: g[["x", "y"]].to_numpy() for eid, g in opp.groupby("event_id")}

    rows = []
    for _, e in fra.iterrows():
        pts = opp_by_event.get(e["id"])
        if pts is None or len(pts) == 0:
            continue
        ball = np.array([e["location_x"], e["location_y"]])   # distância no frame do evento
        dmin = float(np.sqrt(((pts - ball) ** 2).sum(axis=1)).min()) * YARD_TO_M
        rows.append({"zone": _assign_zone(e["cy"], med), "dist_nearest": dmin,
                     "x": float(e["location_x"]), "y": float(e["location_y"]),
                     "visible": _ball_visible(va.get(e["id"]), e["location_x"], e["location_y"])})

    df = pd.DataFrame(rows)
    coverage = float(df["visible"].mean())      # fração medida com a bola dentro do enquadramento
    df = df[df["visible"]].reset_index(drop=True)   # controlo de contenção: só onde é fiável
    order = ["Leão", "Ronaldo", "Bernardo"]
    summary = (
        df.groupby("zone")["dist_nearest"]
        .agg(n="count", dist_media="mean", dist_mediana="median")
        .reindex(order)
        .round(2)
    )
    summary.attrs["visible_coverage"] = coverage
    return summary, df, med


if __name__ == "__main__":
    summary, df, med = space_conceded()
    print("Posição-y típica (mediana) de cada avançado (frame Portugal):")
    for k, v in med.items():
        print(f"  {k:9} y={v:.1f}")
    print("\n== Métrica 1 · Camada B · espaço concedido à França por zona ==")
    print("(dist = distância do portador da França ao português de campo mais próximo; maior = mais espaço)")
    print(summary.to_string())
    print(f"\ncontrolo de contenção (bola dentro da área visível da câmara): "
          f"{summary.attrs['visible_coverage']:.1%} dos eventos medidos")
