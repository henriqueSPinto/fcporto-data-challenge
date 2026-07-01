"""Visualizações. Cada função devolve `(fig, ax)` para embeber na app (Streamlit: st.pyplot)
e pode gravar PNG via `__main__`. Mantidas app-agnósticas de propósito.

- `plot_momentum`         momentum de ameaça ao longo do tempo (a análise de contexto).
- `plot_pressing_ranking` volume de contest por 90 min, trio destacado.
- `plot_press_thirds`     onde cada avançado contesta, por terço (a pressão alta).
- `plot_flank_defense`    quem defende cada flanco: extremo vs lateral.
- `plot_palhinha`         ações defensivas do Palhinha a varrer a largura.
- `plot_space_by_zone`    espaço concedido à França por corredor e profundidade, 360.
- `plot_reception_map`    espaço à receção do Leão no campo, 360 (o teste do benefício).
- `plot_leao_offense`     o payoff ofensivo do Leão (ameaça gerada, conduções progressivas).
- `plot_action_maps`      mapas de ação do trio, para ler cobertura e work-rate.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch

try:
    from .build import build_all
    from .metrics.momentum import momentum_series
    from .metrics.offense import progressive_carries, threat_by_player
    from .metrics.space import space_conceded
    from .metrics.reception import reception_space, RECEPTION_PLAYERS
    from .metrics.pressing import FRONT_LINE, contest_mask, pressing_layer_a
except ImportError:
    from build import build_all
    from metrics.momentum import momentum_series
    from metrics.offense import progressive_carries, threat_by_player
    from metrics.space import space_conceded
    from metrics.reception import reception_space, RECEPTION_PLAYERS
    from metrics.pressing import FRONT_LINE, contest_mask, pressing_layer_a

POR = "#C8102E"   # vermelho Portugal
FRA = "#0055A4"   # azul França
LEAO = "Rafael Alexandre Conceição Leão"
FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"

# Nomes de exibição (o StatsBomb usa o nome completo; para viz usa-se o nome comum).
DISPLAY_NAMES = {
    "João Maria Lobo Alves Palhinha Gonçalves": "Palhinha",
    "Francisco Fernandes Conceição": "Chico Conceição",
    "Nélson Cabral Semedo": "Semedo",
    "Bruno Miguel Borges Fernandes": "Bruno Fernandes",
    "João Pedro Cavaco Cancelo": "Cancelo",
    "Kléper Laveran Lima Ferreira": "Pepe",
    "Vitor Machado Ferreira": "Vitinha",
    "Rúben Santos Gato Alves Dias": "Rúben Dias",
    "Diogo Meireles Costa": "Diogo Costa",
    "Rafael Alexandre Conceição Leão": "Leão",
    "Cristiano Ronaldo dos Santos Aveiro": "Ronaldo",
    "Bernardo Mota Veiga de Carvalho e Silva": "Bernardo Silva",
}


def plot_momentum(match_id: int = 3942349, ax=None):
    """Área temporal do momentum de ameaça (Portugal menos França), suavizado."""
    s = momentum_series(match_id)
    x = s["minute"].to_numpy()
    m = s["momentum"].to_numpy()

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.2))
    else:
        fig = ax.figure

    ax.fill_between(x, m, 0, where=(m >= 0), color=POR, alpha=0.85,
                    interpolate=True, label="Portugal por cima")
    ax.fill_between(x, m, 0, where=(m < 0), color=FRA, alpha=0.85,
                    interpolate=True, label="França por cima")
    ax.axhline(0, color="#333333", lw=0.8)

    for t, lab in [(45, "intervalo"), (90, "fim 90'"), (105, "meio prol.")]:
        ax.axvline(t, color="#999999", ls="--", lw=0.7)
        ax.text(t + 0.4, ax.get_ylim()[0] * 0.92, lab, fontsize=7, color="#777777")

    # Os subs de pernas frescas (Chico, Semedo) entram aos 73', logo antes de o momentum virar.
    # É o único marco editorial: os restantes são eventos estruturais (intervalo, fim 90', prol.).
    ax.axvline(73, color=POR, ls="--", lw=0.9, alpha=0.7)
    ax.text(73.3, ax.get_ylim()[1] * 0.86, "subs 73'\n(Chico, Semedo)", fontsize=7, color=POR)

    ax.set_xlabel("minuto")
    ax.set_ylabel("momentum  (ameaça de Portugal menos França)")
    ax.set_title("Momentum de ameaça: Portugal 0-0 França (Euro 2024, 1/4 final)",
                 fontsize=12, weight="bold")
    ax.set_xlim(0, x.max())
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.9)
    ax.grid(axis="y", alpha=0.15)
    fig.tight_layout()
    return fig, ax


def plot_pressing_ranking(match_id: int = 3942349, team: str = "Portugal",
                          min_minutes: int = 45, ax=None):
    """Ranking do volume de contest defensivo por 90 min, com o trio da frente destacado.

    Situa cada avançado face a toda a equipa. É volume bruto, não ajustado por oportunidade;
    o ajuste por exposição está na Camada A (`pressing.py`).
    """
    data = build_all(match_id)
    players = data["players"]
    nick = dict(zip(players["player"], players["nickname"].fillna(players["player"])))
    trio = set(FRONT_LINE.values())

    t = pressing_layer_a(match_id, focus=None)
    t = t[(t["team"] == team) & (t["minutes"] >= min_minutes)].sort_values("contests_p90")
    labels = [DISPLAY_NAMES.get(p, nick.get(p, p)) for p in t["player"]]
    colors = [POR if p in trio else "#c9ccd1" for p in t["player"]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    ax.barh(labels, t["contests_p90"], color=colors, edgecolor="white")
    for y, v in enumerate(t["contests_p90"]):
        ax.text(v + 0.3, y, f"{v:.1f}", va="center", fontsize=8, color="#444444")
    ax.set_xlabel("contest defensivo por 90 min")
    ax.set_title(f"Volume de contest por 90 min, {team}\n(a vermelho o trio da frente; jogadores com {min_minutes}+ min)",
                 fontsize=12, weight="bold")
    ax.margins(x=0.12)
    fig.tight_layout()
    return fig, ax


def plot_space_by_zone(match_id: int = 3942349, ax=None):
    """Espaço concedido à França por corredor e profundidade, medido no 360.

    Barras inequívocas (sem orientação de campo a confundir). Para cada corredor de avançado,
    a mediana da distância do portador da França ao português mais próximo, separada entre a
    construção (alto, longe da baliza de Portugal) e junto à baliza. Responde à pergunta direta: a
    zona do Leão foi mais aberta? (Não; junto à baliza aperta em todos, e na construção é a do
    Bernardo a mais solta.)
    """
    _, df, med = space_conceded(match_id)
    df = df.copy()
    df["fase"] = np.where(df["x"] < 72, "construção (alto)",
                          np.where(df["x"] >= 88, "junto à baliza", "meio"))
    order = ["Leão", "Ronaldo", "Bernardo"]
    fases = ["construção (alto)", "junto à baliza"]
    piv = (df[df["fase"] != "meio"]
           .pivot_table(index="zone", columns="fase", values="dist_nearest", aggfunc="median")
           .reindex(order)[fases].round(1))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 5))
    else:
        fig = ax.figure
    x = np.arange(len(order))
    w = 0.38
    colors = {"construção (alto)": "#f4a900", "junto à baliza": "#0055A4"}
    for i, f in enumerate(fases):
        bars = ax.bar(x + (i - 0.5) * w, piv[f].to_numpy(), w, label=f, color=colors[f])
        ax.bar_label(bars, fmt="%.1f", fontsize=9, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"corredor do {o}" for o in order])
    ax.set_ylabel("espaço mediano concedido (m)")
    ax.set_title("Espaço concedido à França por corredor e profundidade (360)\n"
                 "junto à baliza aperta em todos; na construção o mais solto é o do Bernardo",
                 fontsize=12, weight="bold")
    ax.legend(fontsize=9)
    ax.margins(y=0.18)
    fig.tight_layout()
    return fig, ax


# Buckets de espaço à receção, do apertado (marcado) ao livre. Rampa vermelho->verde: verde = livre.
BUCKET_ORDER = ["<=2m", "2-5m", "5-10m", ">=10m"]
BUCKET_COLORS = {"<=2m": "#c0392b", "2-5m": "#e67e22", "5-10m": "#f1c40f", ">=10m": "#27ae60"}


def plot_reception_map(match_id: int = 3942349):
    """Espaço à receção do Leão no campo, o teste espacial 360 do benefício do rest-attack.

    Painel esquerdo (o herói): as receções do Leão no seu local real, coloridas pelo espaço ao
    francês mais próximo (verde = livre, vermelho = apertado). Cada ponto vem de um freeze frame
    360, portanto é 360 no espaço, não uma agregação em bandas. Painel direito: a distribuição por
    bucket, o Leão contra o Bernardo, um médio que recebe mais fundo, para separar o aperto do
    Leão do efeito de estar alto (mesmo a igual profundidade o Leão recebe mais marcado). A leitura:
    o Leão fica alto mas recebe apertado (1v1, não cercado), portanto a "saída livre nas
    costas" que o rest-attack promete quase não disparou. Portugal ataca para a direita.
    """
    summary, df = reception_space(match_id, players=RECEPTION_PLAYERS)
    order_players = list(RECEPTION_PLAYERS)
    leao = df[df["player"] == "Leão"]

    fig, (axP, axB) = plt.subplots(1, 2, figsize=(14, 6.2),
                                   gridspec_kw={"width_ratios": [1.5, 1]})
    pitch = Pitch(pitch_type="statsbomb", line_color="#9aa0a6", pitch_color="white", linewidth=1.1)
    pitch.draw(ax=axP)
    for b in BUCKET_ORDER:
        sub = leao[leao["bucket"] == b]
        pitch.scatter(sub["x"], sub["y"], ax=axP, s=72, color=BUCKET_COLORS[b],
                      edgecolors="white", linewidth=0.4, alpha=0.9, label=b, zorder=3)
    axP.legend(title="espaço ao receber", loc="lower left", fontsize=8, title_fontsize=8,
               framealpha=0.9)
    axP.annotate("Portugal ataca →", xy=(0.5, 0.03), xycoords="axes fraction", ha="center",
                 fontsize=9, color="#555555")
    axP.set_title(f"As {len(leao)} receções do Leão, coloridas pelo espaço ao francês mais próximo (360)\n"
                  f"alto (x médio {summary.loc['Leão', 'x_medio']:.0f}), apertado "
                  f"(mediana {summary.loc['Leão', 'espaco_mediana']:.1f}m), marcado 1v1",
                  fontsize=10.5, weight="bold", pad=10)

    # Painel direito: fração das receções por bucket, os três (100% empilhado, do apertado ao livre).
    piv = (df.groupby(["player", "bucket"]).size()
           .unstack(fill_value=0).reindex(order_players).reindex(columns=BUCKET_ORDER, fill_value=0))
    piv = piv.div(piv.sum(axis=1), axis=0)
    ypos = np.arange(len(order_players))[::-1]        # Leão no topo
    left = np.zeros(len(order_players))
    for b in BUCKET_ORDER:
        axB.barh(ypos, piv[b].to_numpy(), left=left, color=BUCKET_COLORS[b],
                 edgecolor="white", height=0.62)
        left = left + piv[b].to_numpy()
    axB.set_yticks(ypos)
    axB.set_yticklabels(order_players)
    for tick in axB.get_yticklabels():
        if tick.get_text() == "Leão":
            tick.set_color(POR)
            tick.set_fontweight("bold")
    for y, pl in zip(ypos, order_players):
        axB.text(1.02, y, f"med {summary.loc[pl, 'espaco_mediana']:.1f}m", va="center",
                 fontsize=8, color="#444444")
    axB.set_xlim(0, 1.2)
    axB.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axB.set_xlabel("fração das receções (apertado → livre)")
    axB.set_title("Distribuição do espaço à receção\n(o Bernardo, mais fundo, recebe com mais espaço)",
                  fontsize=11, weight="bold")

    fig.suptitle("O benefício do rest-attack, testado: o Leão fica alto, mas recebe apertado",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig, (axP, axB)


def plot_leao_offense(match_id: int = 3942349, top_n: int = 8):
    """O payoff do rest-attack: o contributo ofensivo do Leão face à equipa.

    Dois painéis. À esquerda, a ameaça gerada (o mesmo proxy de ameaça) repartida por tipo de ação, para
    mostrar que a do Leão é sobretudo a conduzir. À direita, as conduções progressivas (definição
    FBref), onde o Leão lidera. O benefício do rest-attack materializa-se na criação, não no remate.
    """
    frag = threat_by_player(match_id).head(top_n).iloc[::-1]   # maior no topo do barh
    prog = progressive_carries(match_id).head(top_n).iloc[::-1]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))

    labels_a = [DISPLAY_NAMES.get(p, p) for p in frag.index]
    segments = [("Condução", "Carry", POR), ("Passe", "Pass", "#c9ccd1"),
                ("Remate", "Shot", "#6b6f76")]
    left = np.zeros(len(frag))
    for lab, col, color in segments:
        axA.barh(labels_a, frag[col], left=left, color=color, edgecolor="white", label=lab)
        left = left + frag[col].to_numpy()
    axA.set_xlabel("ameaça gerada (xT-lite, proxy)")
    axA.set_title("Ameaça gerada por tipo de ação\n(a do Leão é a mais assente em condução)",
                  fontsize=11, weight="bold")
    axA.legend(fontsize=8, loc="lower right")

    labels_b = [DISPLAY_NAMES.get(p, p) for p in prog.index]
    colors_b = [POR if p == LEAO else "#c9ccd1" for p in prog.index]
    axB.barh(labels_b, prog.to_numpy(), color=colors_b, edgecolor="white")
    for y, v in enumerate(prog.to_numpy()):
        axB.text(v + 0.15, y, str(int(v)), va="center", fontsize=8, color="#444444")
    axB.set_xlabel("conduções progressivas (definição FBref)")
    axB.set_title("Conduções progressivas\n(o Leão lidera a equipa)", fontsize=11, weight="bold")
    axB.margins(x=0.12)

    for ax in (axA, axB):                       # destacar o Leão nos dois eixos
        for tick in ax.get_yticklabels():
            if tick.get_text() == "Leão":
                tick.set_color(POR)
                tick.set_fontweight("bold")

    fig.suptitle("O payoff do rest-attack: o Leão poupa na pressão e rende na criação",
                 fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig, (axA, axB)


PALHINHA = "João Maria Lobo Alves Palhinha Gonçalves"
DEF_ACTIONS = ["Pressure", "Duel", "Interception", "Block", "Ball Recovery", "Clearance",
               "Foul Committed"]


def plot_palhinha(match_id: int = 3942349, ax=None):
    """Ações defensivas do Palhinha, para ler o ecrã que varre toda a largura.

    O pivot que segura a estrutura enquanto os avançados poupam. Ações defensivas (pressões,
    duelos, interceções, cortes, recuperações) plotadas para mostrar a cobertura em largura.
    """
    ev = build_all(match_id)["events"]
    p = ev[(ev["player"] == PALHINHA) & ev["type"].isin(DEF_ACTIONS) & ev["location_x"].notna()]

    pitch = Pitch(pitch_type="statsbomb", line_color="#9aa0a6", pitch_color="white", linewidth=1.1)
    if ax is None:
        fig, ax = pitch.draw(figsize=(9, 6.2))
    else:
        fig = ax.figure
        pitch.draw(ax=ax)

    pitch.scatter(p["location_x"], p["location_y"], ax=ax, s=90, color=POR, alpha=0.7,
                  edgecolors="white", linewidth=0.6, zorder=3)
    ax.set_title(f"Palhinha: {len(p)} ações defensivas a varrer a largura\n(Portugal ataca para a direita)",
                 fontsize=12, weight="bold")
    return fig, ax


# Papéis por flanco (nomes StatsBomb), para a viz de compensação defensiva.
LEFT_WINGER = "Rafael Alexandre Conceição Leão"
LEFT_BACK = "Nuno Mendes"
RIGHT_WINGER = "Bernardo Mota Veiga de Carvalho e Silva"
RIGHT_BACKS = ["João Pedro Cavaco Cancelo", "Nélson Cabral Semedo"]


def plot_press_thirds(match_id: int = 3942349, ax=None):
    """Onde cada avançado contesta, por terço do campo (a dimensão de pressão ofensiva).

    O terço de ataque é a pressão alta, o mais ligado à premissa original.
    Mostra que o Leão quase não pressiona alto, enquanto o Ronaldo o faz (lidera a linha).
    """
    ev = build_all(match_id)["events"]
    por = ev[ev["team"] == "Portugal"].copy()
    c = por[contest_mask(por) & por["location_x"].notna()]

    def counts(name):
        x = c.loc[c["player"] == name, "location_x"]
        return [int((x > 80).sum()), int(((x > 40) & (x <= 80)).sum()), int((x <= 40).sum())]

    colors = {"Leão": POR, "Ronaldo": "#f4a900", "Bernardo": "#0055A4"}
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 5))
    else:
        fig = ax.figure
    xpos = np.arange(3)
    w = 0.26
    for i, (lab, name) in enumerate(FRONT_LINE.items()):
        vals = counts(name)
        bars = ax.bar(xpos + (i - 1) * w, vals, w, label=lab, color=colors[lab])
        ax.bar_label(bars, fontsize=9, padding=2)
    ax.set_xticks(xpos)
    ax.set_xticklabels(["terço de ataque\n(pressão alta)", "meio-campo", "terço defensivo"])
    ax.set_ylabel("contests")
    ax.set_title("Onde cada um contesta: o Leão quase não pressiona alto\n"
                 "(o Ronaldo pressiona alto porque lidera a linha; perfis diferentes)",
                 fontsize=12, weight="bold")
    ax.legend(title="", fontsize=9)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig, ax


def plot_flank_defense(match_id: int = 3942349, ax=None):
    """Quem defende cada flanco: contests do extremo vs do lateral, por corredor.

    Mostra a inversão entre os dois lados. Na esquerda, o extremo (Leão) contesta pouco e o
    lateral (Nuno Mendes) carrega o flanco; na direita, o extremo (Bernardo) é quem mais defende.
    Contest = Pressure, Duel(Tackle) ou Dribbled Past. Contagem em bruto (cada posição ~120').
    """
    ev = build_all(match_id)["events"]
    por = ev[ev["team"] == "Portugal"].copy()
    por = por[contest_mask(por) & por["location_y"].notna()]
    por["corr"] = np.where(por["location_y"] < 26.7, "esq",
                           np.where(por["location_y"] < 53.3, "centro", "dir"))

    def cnt(corr, players):
        players = players if isinstance(players, list) else [players]
        return int(((por["corr"] == corr) & por["player"].isin(players)).sum())

    extremo = [cnt("esq", LEFT_WINGER), cnt("dir", RIGHT_WINGER)]
    lateral = [cnt("esq", LEFT_BACK), cnt("dir", RIGHT_BACKS)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure
    x = np.arange(2)
    w = 0.38
    b1 = ax.bar(x - w / 2, extremo, w, label="extremo", color=POR)
    b2 = ax.bar(x + w / 2, lateral, w, label="lateral", color="#0055A4")
    ax.bar_label(b1, padding=2, fontsize=10)
    ax.bar_label(b2, padding=2, fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(["Esquerda\n(Leão / Nuno Mendes)", "Direita\n(Bernardo / laterais dir.)"])
    ax.set_ylabel("contests no corredor")
    ax.set_title("Quem defende cada flanco: extremo vs lateral\n"
                 "(à esquerda o lateral cobre o extremo; à direita inverte-se)",
                 fontsize=12, weight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.margins(y=0.15)
    fig.tight_layout()
    return fig, ax


def _player_actions(ev, name):
    return ev[(ev["player"] == name) & ev["location_x"].notna()]


def plot_action_maps(match_id: int = 3942349):
    """Cobertura de terreno de cada avançado, para ler a divisão de trabalho.

    Fundo cinza: todas as posições de evento do jogador (envolvimentos com bola e ações defensivas),
    o footprint posicional. O X preto é a posição média (de eventos, não posição média real, que
    exigiria tracking contínuo). O título traz os dois sinais robustos da divisão de trabalho: quantas
    ações defensivas fez e que fração ocorreu na sua metade. Portugal ataca para a direita.
    """
    ev = build_all(match_id)["events"]
    pitch = Pitch(pitch_type="statsbomb", line_color="#9aa0a6", pitch_color="white", linewidth=1.0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.2))

    for ax, (lab, name) in zip(axs, FRONT_LINE.items()):
        pitch.draw(ax=ax)
        p = _player_actions(ev, name)
        pitch.scatter(p["location_x"], p["location_y"], ax=ax, s=26, color="#c9ccd1",
                      alpha=0.65, edgecolors="white", linewidth=0.2, zorder=2)
        cx, cy = float(p["location_x"].mean()), float(p["location_y"].mean())
        pitch.scatter([cx], [cy], ax=ax, s=280, color="black", marker="X",
                      edgecolors="white", linewidth=0.8, zorder=5)

        n_def = int(p["type"].isin(DEF_ACTIONS).sum())
        def_share = float((p["location_x"] < 60).mean())
        ax.set_title(f"{lab}\n{n_def} ações defensivas · {def_share:.0%} na metade defensiva",
                     fontsize=11, weight="bold")

    fig.suptitle("Cobertura de terreno do trio (Portugal ataca para a direita; X = posição média de eventos)",
                 fontsize=13, weight="bold")
    fig.tight_layout()
    return fig, axs


if __name__ == "__main__":
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs = {
        "momentum.png": plot_momentum,
        "pressing_ranking.png": plot_pressing_ranking,
        "press_thirds.png": plot_press_thirds,
        "flank_defense.png": plot_flank_defense,
        "palhinha.png": plot_palhinha,
        "space_by_zone.png": plot_space_by_zone,
        "reception_map.png": plot_reception_map,
        "leao_offense.png": plot_leao_offense,
        "action_maps.png": plot_action_maps,
    }
    for name, fn in figs.items():
        fig = fn()[0]
        fig.savefig(FIG_DIR / name, dpi=140, bbox_inches="tight")
        print(f"gravado: {FIG_DIR / name}")
