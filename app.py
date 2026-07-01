"""App visual do exercício (FC Porto, StatsBomb Open Data).

Interface Streamlit por cima da mesma pipeline Python (ETL + métricas + viz). Nada de
lógica nova aqui: a app só lê o que os módulos em `src/` produzem e apresenta.

Executar:  .venv/bin/streamlit run app.py

Três separadores: visão geral (contexto + momentum), o rest-attack e o seu preço (o coração
da análise, que cruza eventos e 360) e metodologia.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import streamlit as st

from src.build import build_all
from src.metrics.momentum import momentum_series
from src.metrics.offense import progressive_carries, threat_by_player
from src.metrics.pressing import FRONT_LINE, contest_mask, pressing_layer_a
from src.metrics.reception import reception_space, RECEPTION_PLAYERS
from src.metrics.space import space_conceded
from src import viz

MATCH_ID = 3942349
st.set_page_config(page_title="Portugal 0-0 França · 360", layout="wide")


@st.cache_data(show_spinner="A carregar dados do jogo...")
def get_context():
    return build_all(MATCH_ID)["context"].set_index("team")


@st.cache_data(show_spinner=False)
def get_pressing():
    data = build_all(MATCH_ID)
    pos = data["players"][["player", "primary_position"]]
    t = pressing_layer_a().merge(pos, on="player", how="left")
    t = t[(t["team"] == "Portugal") & t["exposure"].notna()
          & (t["minutes"] >= 30) & (t["primary_position"] != "Goalkeeper")].copy()
    t["jogador"] = t["player"].map(lambda p: viz.DISPLAY_NAMES.get(p, p))
    t = t.sort_values("contests_per100_exp", ascending=False)
    cols = {"jogador": "jogador", "minutes": "minutos", "contests": "contests",
            "contests_p90": "contests/90", "exposure": "exposição",
            "contests_per100_exp": "contests/100 exp"}
    return t[list(cols)].rename(columns=cols)


@st.cache_data(show_spinner=False)
def get_space():
    summary, _, _ = space_conceded()
    return summary.reset_index().rename(columns={
        "zone": "corredor", "n": "eventos",
        "dist_media": "dist. média (m)", "dist_mediana": "dist. mediana (m)",
    })


@st.cache_data(show_spinner=False)
def get_reception():
    summary, _ = reception_space(players=RECEPTION_PLAYERS)
    return summary.reset_index().rename(columns={
        "player": "jogador", "n": "receções", "espaco_mediana": "espaço mediano (m)",
        "x_medio": "x médio", "pct_apertado": "% <=5m", "pct_ult_terco": "% últ. terço",
        "pct_cercado": "% 2+ perto",
    })


@st.cache_data(show_spinner=False)
def get_offense():
    frag = threat_by_player().head(8).copy()
    frag.index = [viz.DISPLAY_NAMES.get(p, p) for p in frag.index]
    frag = frag[["Carry", "Pass", "Shot", "total"]].round(2)
    frag = frag.rename(columns={"Carry": "condução", "Pass": "passe",
                                "Shot": "remate", "total": "ameaça total"})
    prog = progressive_carries()
    prog.index = [viz.DISPLAY_NAMES.get(p, p) for p in prog.index]
    prog = prog.rename_axis("jogador").reset_index(name="cond. progressivas")
    return frag.reset_index().rename(columns={"index": "jogador"}), prog


@st.cache_data(show_spinner=False)
def get_blocos():
    m = momentum_series().copy()
    m["bloco"] = (m["minute"] // 15) * 15
    b = m.groupby("bloco")[["Portugal", "France"]].sum().round(1)
    b.index = [f"{i}-{i + 15}'" for i in b.index]
    return b.rename(columns={"France": "França"})


@st.cache_data(show_spinner=False)
def get_corridor():
    ev = build_all(MATCH_ID)["events"]
    por = ev[ev["team"] == "Portugal"].copy()
    por = por[contest_mask(por) & por["location_y"].notna()]
    por["corr"] = np.where(por["location_y"] < 26.7, "esquerda",
                           np.where(por["location_y"] < 53.3, "centro", "direita"))
    por["jogador"] = por["player"].map(lambda p: viz.DISPLAY_NAMES.get(p, p))
    out = {}
    for c in ["esquerda", "centro", "direita"]:
        vc = por[por["corr"] == c]["jogador"].value_counts().head(6)
        out[c] = vc.rename_axis("jogador").reset_index(name="contests")
    return out


@st.cache_resource(show_spinner=False)
def fig_momentum():
    return viz.plot_momentum()[0]


@st.cache_resource(show_spinner=False)
def fig_ranking():
    return viz.plot_pressing_ranking()[0]


@st.cache_resource(show_spinner=False)
def fig_press_thirds():
    return viz.plot_press_thirds()[0]


@st.cache_resource(show_spinner=False)
def fig_flank():
    return viz.plot_flank_defense()[0]


@st.cache_resource(show_spinner=False)
def fig_palhinha():
    return viz.plot_palhinha()[0]


@st.cache_resource(show_spinner=False)
def fig_space():
    return viz.plot_space_by_zone()[0]


@st.cache_resource(show_spinner=False)
def fig_offense():
    return viz.plot_leao_offense()[0]


@st.cache_resource(show_spinner=False)
def fig_reception():
    return viz.plot_reception_map()[0]


@st.cache_resource(show_spinner=False)
def fig_action():
    return viz.plot_action_maps()[0]


# ---------------------------- Cabeçalho ----------------------------
st.title("Portugal 0-0 França")
st.caption(
    "Euro 2024, quartos de final · decidido nos penáltis após prolongamento (120') · "
    "StatsBomb Open Data (Event + 360), match_id 3942349 · Análise de Henrique Pinto."
)

ctx = get_context()
cpor, cfra = st.columns(2)
with cpor:
    st.markdown("#### Portugal")
    a, b, c = st.columns(3)
    a.metric("Posse", f"{ctx.loc['Portugal', 'possession_pct']:.0f}%")
    b.metric("xG", f"{ctx.loc['Portugal', 'xg']:.2f}")
    c.metric("Remates", int(ctx.loc['Portugal', 'shots']))
    st.caption(f"Remates à baliza {int(ctx.loc['Portugal', 'shots_on_target'])} · "
               f"Passes {int(ctx.loc['Portugal', 'passes'])}")
with cfra:
    st.markdown("#### França")
    a, b, c = st.columns(3)
    a.metric("Posse", f"{ctx.loc['France', 'possession_pct']:.0f}%")
    b.metric("xG", f"{ctx.loc['France', 'xg']:.2f}")
    c.metric("Remates", int(ctx.loc['France', 'shots']))
    st.caption(f"Remates à baliza {int(ctx.loc['France', 'shots_on_target'])} · "
               f"Passes {int(ctx.loc['France', 'passes'])}")

st.divider()
tab1, tab2, tab3 = st.tabs(["Visão geral", "O rest-attack e o seu preço", "Metodologia"])

# ---------------------------- Tab 1: visão geral ----------------------------
with tab1:
    st.subheader("Fluxo do jogo: momentum de ameaça")
    st.pyplot(fig_momentum(), clear_figure=False)
    st.markdown(
        "Proxy de ameaça transparente (não é um xT treinado): cada ação vale pontos conforme "
        "progride para zonas perigosas (remate 1.0, entrada na área 0.5, entrada no último terço "
        "0.2). A curva é a diferença suavizada entre Portugal e França, e descreve o fluxo do jogo "
        "por si só, independente da história do Leão.\n\n"
        "**Leitura.** Jogo equilibrado, mas Portugal sombreou-o nos números de base (posse 60%, "
        "xG 1.41 vs 1.07, ameaça total 59.8 vs 55.1). A França teve o seu melhor período a meio da "
        "2ª parte (60-73'); a partir das substituições dos **73'** o jogo virou e Portugal ficou por "
        "cima até ao fim, prolongamento incluído. Essa viragem tem uma leitura própria, na conclusão "
        "secundária da página seguinte."
    )
    with st.expander("Ameaça acumulada por bloco de 15 min"):
        st.dataframe(get_blocos(), width="stretch")
    st.info(
        "**O jogo num relance.** Um 0-0 equilibrado, decidido nos penáltis após 120 minutos. Seguem-se "
        "duas leituras: a **principal**, como Portugal sustentou o rest-attack do Leão à esquerda e o "
        "que ele rendeu; e uma **secundária**, como as pernas frescas dos suplentes mudaram o fluxo."
    )

# ---------------------------- Tab 2: o rest-attack e o seu preço ----------------------------
with tab2:
    st.subheader("O rest-attack e quem o paga (cruza eventos e 360)")
    st.markdown(
        "**A premissa.** Por analogia com o *rest defence* (os jogadores que ficam atrás durante o "
        "ataque a cobrir o contra-ataque), defino **rest-attack** como manter avançados altos e frescos "
        "durante a posse adversária: poupam energia a defender para serem ameaça no ataque e na "
        "transição. A hipótese deste jogo é que Portugal o faz com **dois avançados ao mesmo tempo, "
        "Leão e Ronaldo**. É uma escolha invulgar, dois jogadores que pressionam pouco podem abrir "
        "fendas num sistema que se quer coeso a recuperar bola, e por isso vale a pena pô-la à prova, "
        "não assumi-la.\n\n"
        "Três perguntas: estes dois **defendem e pressionam mesmo menos** que os pares? Se sim, "
        "**quem paga** a conta? E **resultou**, rendeu à frente sem custar espaço atrás? Os eventos "
        "respondem ao *quem*, o 360 ao *espaço*. (O contraste é o Bernardo: fica no flanco mas com "
        "funções de médio, trabalha. E, como se verá, os próprios Leão e Ronaldo não são o mesmo "
        "perfil.)"
    )

    st.markdown("##### Defendem mesmo menos? Sim, e o fosso aguenta o ajuste por oportunidade")
    st.markdown(
        "Medir \"defender\" aqui é contar *contests* (pressões, duelos e dribles sofridos) e ajustar "
        "pela **oportunidade**: quantas vezes a bola da França passou perto de cada um enquanto esteve "
        "em campo. Assim comparo a vontade de disputar, não apenas quem teve mais oportunidades de o fazer."
    )
    lcol, rcol = st.columns([3, 2])
    with lcol:
        st.pyplot(fig_ranking(), clear_figure=False)
    with rcol:
        st.caption(
            "Taxa por oportunidade para toda a equipa. Contest = Pressure, Duel(Tackle) ou "
            "Dribbled Past. Exposição = bola da França na faixa do jogador enquanto em campo; "
            "a taxa é contests por 100 dessas oportunidades."
        )
        st.dataframe(get_pressing(), width="stretch", hide_index=True)
        st.markdown(
            "Entre quem jogou o jogo todo, o **Palhinha** lidera a taxa (9.8) e o **Leão (2.1)** e o "
            "**Ronaldo (3.2)** ficam no fundo. O fosso sobrevive ao ajuste por oportunidade, não é só "
            "de terem menos bola por perto. (Os suplentes Chico e Semedo aparecem no topo da tabela; "
            "isso tem leitura própria, na conclusão secundária no fim.)"
        )
    with st.expander("Robustez: e se a intensidade tardia estiver a distorcer?"):
        st.markdown(
            "Só a 1ª parte: **Leão 1.4, Ronaldo 3.7, Bernardo 3.8** (contests por 100 exp). Leão "
            "é o mais baixo também na 1ª parte, portanto o achado não depende das fases tardias "
            "nem do prolongamento."
        )

    st.markdown("##### Onde pressionam: a pressão alta")
    st.markdown("O terço de ataque é a pressão alta, a mais exigente e a que melhor expõe um avançado "
                "que se poupa: é lá que ele aparece menos.")
    st.pyplot(fig_press_thirds(), clear_figure=False)
    st.markdown(
        "**Leão faz só 1 contest no terço de ataque** em todo o jogo: a assinatura pura do "
        "rest-attack. Nuance importante que os dados revelam: o **Ronaldo pressiona alto (8)** "
        "porque lidera a linha. Não são o mesmo perfil, o Leão é o 'rester', o Ronaldo pressiona "
        "alto mas não recua.\n\n"
        "Confirma-se com a métrica nativa da StatsBomb: Leão fez só **1 counterpress** (re-pressão "
        "imediata após perder a bola) em 105 min, dos avançados o que menos fez das duas equipas "
        "(Ronaldo 3, Mbappé 2, Kolo Muani 6). São poucos eventos, portanto leitura direcional e não "
        "ranking fino, mas o proxy e a definição oficial da StatsBomb apontam no mesmo sentido."
    )

    st.markdown("##### A divisão de trabalho num relance (cobertura de terreno)")
    st.pyplot(fig_action(), clear_figure=False)
    st.caption(
        "Footprint de cada jogador (todos os envolvimentos com bola e ações defensivas, posições "
        "de evento). O X é a posição média (de eventos, não a real, que exigiria tracking contínuo). "
        "Lê-se a divisão de trabalho: o **Leão** faz só **9 ações defensivas** e opera quase todo no "
        "ataque (X alto, 17% na metade defensiva); o **Ronaldo** faz 25 e fica alto nas duas fases "
        "(pressiona mas não recua, X ao centro); o **Bernardo** faz **36** e cobre muito mais terreno "
        "(X fundo, 42%). Dois caveats: o Bernardo é médio, logo cobre mais por função, e o mapa do "
        "jogo todo mistura dois papéis (mais na ala e à direita até aos 73', mais central e adiantado "
        "depois, com o Chico a segurar o flanco)."
    )
    st.markdown(
        "E não é só na defesa que Leão fica de fora: fez só **36 passes (3.9% da equipa)**, dos "
        "jogadores de campo menos envolvidos na construção, um **endpoint alto** que espera em vez de "
        "circular a bola. A esquerda é do **Nuno Mendes**, que a carrega nas três frentes: **cobre** "
        "(13 contests), **constrói** (107 passes) e **ataca** (3º da equipa em ameaça gerada, 7.2; 4º "
        "em progressões). É o lateral que torna o rest-attack sustentável, faz o trabalho do corredor "
        "enquanto Leão fica em cima."
    )

    st.markdown("##### Quem paga a conta: o lateral e o pivot")
    fcol, tcol = st.columns([3, 2])
    with fcol:
        st.pyplot(fig_flank(), clear_figure=False)
    with tcol:
        st.markdown(
            "Na **esquerda do Leão**, o extremo contesta 6 vezes e o lateral **Nuno Mendes 13**: "
            "o lateral cobre o extremo que descansa. Na **direita** inverte-se, o próprio "
            "**Bernardo (17)** é quem mais defende o seu flanco."
        )
        with st.expander("Contests por corredor, todos os jogadores"):
            corr = get_corridor()
            st.caption("Esquerda (lado do Leão)")
            st.dataframe(corr["esquerda"], width="stretch", hide_index=True)
            st.caption("Direita (lado do Bernardo)")
            st.dataframe(corr["direita"], width="stretch", hide_index=True)
    st.pyplot(fig_palhinha(), clear_figure=False)
    st.markdown(
        "E no centro o **Palhinha** varre toda a largura (7 contests à esquerda, 12 à direita, "
        "mais o centro), o mais contestador do jogo. É o ecrã que segura a estrutura enquanto os "
        "avançados poupam. Detalhe: as ações dele **do lado do Leão são mais avançadas** (x médio "
        "~47 vs ~32 no resto), o que sugere cobrir a profundidade e a transição que o Leão, alto, "
        "deixa atrás de si."
    )

    st.markdown("##### A cobertura aguentou? O espaço no corredor do Leão (360)")
    scol, dcol = st.columns([3, 2])
    with scol:
        st.pyplot(fig_space(), clear_figure=False)
    with dcol:
        st.caption(
            "Barras por corredor de avançado. Cada barra é a mediana da distância do portador da "
            "França ao português mais próximo, tirada do freeze frame 360, nas bolas da França "
            "(passe, condução, receção) nos dois terços defensivos de Portugal, separada entre construção (longe "
            "da baliza) e junto à baliza."
        )
        st.dataframe(get_space(), width="stretch", hide_index=True)
        st.markdown(
            "Lê-se em duas partes. **Junto à baliza aperta em todos (3.5-4.6m)**, o bloco organizado, com o "
            "do Leão até marginalmente o mais alto. **Na construção fica mais solto (6.5-7.3m)** nos três, "
            "o reflexo de pressionar pouco alto, e aí o mais solto é o do **Bernardo (7.3)**, não o do Leão. "
            "Ou seja, o custo do rest-attack **não apareceu como mais espaço claro no lado do Leão**, o que é "
            "consistente com a cobertura (Nuno Mendes, Palhinha) a fazer o trabalho por ele, não com "
            "mérito do próprio. É o 360 a confirmar, no espaço, o que o 'quem paga' mostrou por eventos "
            "(medido só onde a bola está no enquadramento da câmara, 96.5% dos casos)."
        )

    st.markdown("##### O retorno: onde o Leão rende com bola")
    ocol, tcol = st.columns([3, 2])
    with ocol:
        st.pyplot(fig_offense(), clear_figure=False)
    with tcol:
        st.caption(
            "Ameaça gerada = o mesmo proxy de ameaça do gráfico de momentum (não é xT treinado), "
            "repartido por tipo de ação. Conduções progressivas na linha da definição FBref: avança "
            "10+ jardas para a baliza ou entra na área, excluindo a saída de bola dos centrais."
        )
        frag_tbl, _ = get_offense()
        st.dataframe(frag_tbl, width="stretch", hide_index=True)
        st.markdown(
            "O custo do rest-attack é a pressão que não fez; o retorno é a criação. Leão é **2º em "
            "ameaça gerada (7.6)**, e a dele é **a mais assente em condução da equipa (2.2)**; é ainda "
            "o **1º em conduções progressivas (14)**. Cria a conduzir, não a rematar (0.13 xG, a lente "
            "errada para um extremo)."
        )

    st.markdown("##### Como rendeu, e como não: o espaço à receção (360)")
    rcol, tcol = st.columns([3, 2])
    with rcol:
        st.pyplot(fig_reception(), clear_figure=False)
    with tcol:
        st.caption(
            "Para cada receção do Leão com freeze frame, a distância ao francês mais próximo (o "
            "espaço ao receber) e a profundidade, medida só onde a bola está no enquadramento da "
            "câmara (83% das receções; fora disso o 'mais próximo' seria lixo). O rest-attack pode "
            "render de duas formas: receber livre nas costas (a saída em transição) ou, com pernas "
            "frescas, desmarcar-se e conduzir. Este mapa testa a primeira; a segunda está na secção "
            "anterior. O Bernardo entra como contraste: um médio que recebe mais fundo, para separar o "
            "aperto do Leão do simples efeito de estar alto (mesmo a igual profundidade o Leão recebe "
            "mais marcado)."
        )
        st.dataframe(get_reception(), width="stretch", hide_index=True)
        st.markdown(
            "Leão recebeu **alto** (x médio 81, 54% no último terço), a posição do rest-attack. Mas aí "
            "recebeu **apertado**: 64% das receções a menos de 5m de um francês, e **marcado 1v1, não "
            "cercado** (só 5% com 2+ franceses perto). A **primeira via** (receber livre nas costas) "
            "quase não disparou: **só 5% das receções foram livres e altas** (>=10m e no último terço), "
            "contra uma França que quase não concedeu transições. O valor veio da **segunda via**, "
            "conduzir a partir dessas receções apertadas (secção anterior), não de receber livre. "
            "Recebeu mais apertado que o Bernardo (mediana 3.8m vs 5.9m), em parte porque fica alto, "
            "onde é mais congestionado, e o Bernardo cai mais fundo para o espaço; mas mesmo a igual "
            "profundidade (último terço) foi mais marcado (4.2m vs 5.6m). Duas leituras do "
            "aperto, e nenhuma é 'Leão falhou': ou a saída livre não chegou a existir, ou a França "
            "marcou-o de perto; com event+360 não as distingo (tracking diria melhor)."
        )

    st.success(
        "**Conclusão principal (o insight acionável).** Jogo equilibrado (Portugal com mais posse e "
        "xG), 0-0 até aos penáltis. Portugal pôde correr o rest-attack do Leão à esquerda porque a "
        "**estrutura o pagou**: o Nuno Mendes cobre atrás e o Palhinha protege a profundidade (a França "
        "esteve ativa pelos dois flancos, não foi um lado morto). O **custo defensivo foi real** e "
        "sobrevive ao ajuste por oportunidade. O **retorno apareceu, pela via das pernas frescas**: "
        "Leão ficou alto (posição) e criou **a conduzir** (2º em ameaça gerada, 1º em conduções "
        "progressivas da equipa), a desmarcar-se e a bater o homem, não a receber livre nas costas, "
        "que recebeu apertado e marcado 1v1 (a saída em transição quase não existiu). Perfil viável "
        "com a estrutura certa à volta: um lateral como o Nuno Mendes, que cobriu, construiu e ainda "
        "atacou o corredor, e um pivot que protege a profundidade. E os dois avançados não são "
        "o mesmo caso: Ronaldo contribui moderadamente mais na pressão e no posicionamento defensivo, "
        "mas sem a mesma preponderância ofensiva, por isso a análise centrou-se no Leão, o rest-attack "
        "mais puro e ao mesmo tempo o mais produtivo à frente. Leitura descritiva de um jogo, não "
        "projetável."
    )
    st.info(
        "**Conclusão secundária: as pernas frescas mudaram o fluxo.** A França teve o seu melhor "
        "período aos 60-73' (momentum, página anterior). Aos **73'** entram o Chico e o Semedo, e o "
        "jogo vira: no bloco 75-90' Portugal tem a maior vantagem de ameaça do jogo (+6.1) e fica por "
        "cima até ao fim. **Chico** teve a **maior taxa de contest do jogo (17.3/100 exp)**, acima "
        "do Palhinha (9.8), com impacto ofensivo (8 conduções progressivas); o Semedo (12.7) logo "
        "atrás. São o contraste das pernas frescas com quem se começou a poupar. Coincidência "
        "temporal, não causa provada (pernas frescas, ajuste tático e cansaço da França misturam-se), "
        "e as taxas em 47' de prolongamento aberto inflam. Mas o padrão vê-se nas duas métricas."
    )


# ---------------------------- Tab 3: metodologia ----------------------------
with tab3:
    st.subheader("Metodologia e limites")
    st.markdown(
        """
**Dados.** StatsBomb Open Data: Event Data e 360 Freeze Frames do Portugal 0-0 França
(Euro 2024). ETL próprio: ler os JSON em bruto, estruturar em DataFrames e ligar cada evento
ao seu freeze frame. Cobertura 360 de 86.8% dos eventos.

**As duas análises (as métricas avançadas do exercício).** O exercício pede pelo menos duas
métricas avançadas, uma a usar o 360. Aqui são a análise principal (o rest-attack e o seu preço),
com eventos e 360, e a de contexto (o momentum de ameaça), só com eventos.

**A análise principal: o rest-attack e o seu preço (eventos + 360).** Os eventos respondem ao
*quem*: contests por oportunidade (contests por 100 eventos de bola da França no corredor do
jogador, a lógica do PPDA ao nível individual) e a distribuição desses contests por corredor
e por jogador. O 360 responde ao *espaço*: a distância do portador da França ao defesa mais
próximo, tirada dos freeze frames, agregada por corredor. Os freeze frames não identificam os
jogadores sem bola, por isso a leitura espacial é zonal e anónima de propósito. O 360 entra
ainda no *benefício*: o espaço à receção do próprio Leão (distância ao francês mais próximo
quando recebe), para testar se a "saída livre" que o rest-attack promete chegou a disparar.
Meço o espaço só onde a bola cai dentro da área visível da câmara (96.5% dos casos), um
controlo de contenção para não sobrestimar o espaço com defesas fora de imagem.

**A análise de contexto: momentum de ameaça ao longo do tempo.** Proxy de ameaça transparente por
localização (não é um xT treinado), independente da hipótese Leão/Ronaldo. Por não depender dela,
descreve o fluxo por si só e aguenta melhor a amostra pequena.

**O que o 360 é, e o que não é.** São snapshots posicionais no momento de cada evento, não
tracking contínuo, e sem identidade dos jogadores sem bola. Usei-o para o que faz bem (medir
espaço no instante do evento) e evitei forçá-lo além disso.

**Limites.** É um jogo, portanto tudo é descritivo (o que aconteceu aqui),
não projetivo. As contagens por corredor são pequenas (6, 13, 17) e as taxas de suplentes com
poucos minutos têm amostra pequena. O counterpress também são poucos eventos (32 de Portugal, 1 a 6
por jogador), lido em direção e não em ranking. A distância ao mais próximo é um proxy cru de marcação.
"Descansar" vs "posicionar-se" é intenção, que um snapshot não observa.

**Lente só de Portugal.** Analisei o jogo do lado de Portugal; não meço os espaços nem os
jogadores da França. Nem sequer consigo dizer com confiança por que lado a França atacou mais:
por toques o jogo foi equilibrado nos dois flancos (mais pelo lado do Leão no meio-campo, mais pelo
do Mbappé no último terço). Fico pela descrição, sem reduzir a um lado nem inferir intenção.

**Trabalho futuro.** Duas direções. Uma que estes dados suportam e deixei por âmbito: a
sobrecarga numérica local (contar franceses vs portugueses à volta da bola no freeze frame), uma
leitura bilateral do espaço que a métrica de corredor, unilateral, não capta. Outra que os dados
não dão: o rest-attack por jogador (quantas vezes o Leão em concreto fica alto como saída de
transição), que precisa de tracking com identidade, pois um jogador parado e sem bola é anónimo
no 360 e invisível nos eventos. É a fronteira destes dados.
"""
    )
    st.info(
        "Nota: análise assistida pelo Claude Code (par técnico, sobretudo no código). "
        "Escolhas, julgamento de futebol e revisão do autor."
    )
