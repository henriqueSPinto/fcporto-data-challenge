# Portugal 0-0 França (Euro 2024): o rest-attack e o seu preço

Exercício técnico de Data Scientist sobre [StatsBomb Open Data](https://github.com/statsbomb/open-data).
Um jogo, com Event Data e 360 Freeze Frames, transformado numa leitura tática acionável.

**Jogo:** Portugal 0-0 França, quartos de final do Euro 2024 (`match_id = 3942349`), decidido
nos penáltis após prolongamento. App interativa em Streamlit por cima da pipeline de análise.

**App online:** https://fcporto-data-challenge-henrique-pinto.streamlit.app/

## A premissa

Por analogia com o *rest defence* (os jogadores que ficam atrás durante o ataque a cobrir o
contra-ataque), defino **rest-attack** como manter avançados altos e frescos durante a posse
adversária: poupam energia a defender para serem ameaça no ataque e na transição. A hipótese deste
jogo é que Portugal o faz com dois avançados ao mesmo tempo, **Leão e Ronaldo**. É uma escolha
invulgar (dois que pressionam pouco podem abrir fendas num sistema que se quer coeso a recuperar
bola), por isso ponho-a à prova em vez de a assumir:

> Estes dois avançados defendem e pressionam mesmo menos que os pares, *por oportunidade*? Se sim,
> **quem paga** o descanso, e **resultou** (rendeu à frente sem custar espaço atrás)?

A pergunta é robusta a qualquer resultado, e obriga a distinguir o *quem* (eventos, com identidade)
do *espaço* (360, posições de todos os jogadores no momento do evento).

## Lente e princípios

- **Individual e descritiva.** Um jogo é amostra pequena, por isso descrevo o que aconteceu *neste*
  jogo, não projeto traços estáveis do jogador.
- **O 360 é load-bearing, não decorativo.** Medir a não-ação (não pressionar) e o espaço concedido
  exige as posições de quem não tem a bola, que só o 360 dá.
- **Clareza sobre os limites.** O 360 são snapshots no momento do evento, não tracking contínuo,
  e sem identidade dos jogadores sem bola. Não forço a ferramenta além disso.

## ETL

O brief avalia o ETL, por isso ele é feito à mão a partir dos JSON em bruto (em produção usar-se-ia a
API oficial ou o `statsbombpy` para um warehouse):

1. `src/ingest.py` descarrega e faz cache dos três ficheiros (events, lineups, three-sixty).
2. `src/build.py` normaliza tudo em DataFrames tidy e faz o **join 360**: cada freeze frame liga-se ao
   evento por `event_uuid == event.id` (left join, parcial). Marca `has_360` nos eventos e guarda
   os freeze frames em formato longo (um jogador por linha), o que torna trivial calcular distâncias e
   contagens de jogadores à volta de uma ação.

Cobertura 360: **86.8%** dos eventos. Convenção de coordenadas validada empiricamente (cada equipa
ataca `x=120` no seu frame, sem troca ao intervalo).

## As duas métricas avançadas

(O exercício pede pelo menos duas, uma a usar o 360. A análise principal usa o 360.)

### A análise principal: o rest-attack e o seu preço (eventos + 360)

Responde à pergunta em três ângulos:

- **Quem engaja (eventos).** *Contest defensivo* = `{Pressure, Duel(Tackle), Dribbled Past}`,
  outcome-neutral (mede a vontade de contestar, não o êxito). Ajustado por **oportunidade**: contests
  por 100 eventos de bola da França no corredor do jogador enquanto em campo, a lógica do PPDA ao nível
  individual (`src/metrics/pressing.py`).
- **Quem paga (eventos, por corredor).** Distribuição dos contests por corredor e por jogador, para ver
  quem cobre cada flanco.
- **Se resultou (360).** *Espaço concedido* = distância do portador da França ao português de campo mais
  próximo, tirada do freeze frame, agregada por corredor (`src/metrics/space.py`), medida só onde a bola cai
  dentro da área visível da câmara (96.5%, para não sobrestimar o espaço com defesas fora de imagem). Anónima
  de propósito, porque o 360 não identifica os jogadores sem bola.
- **O payoff (eventos).** O outro lado do trade-off: quem poupa na pressão pode render na criação. Ameaça
  gerada (o mesmo proxy de ameaça, repartido por tipo de ação) e conduções progressivas na linha da
  definição FBref (`src/metrics/offense.py`), para ver *como* o Leão contribui no ataque.
- **O benefício, testado (360).** O espaço à receção do próprio Leão (`src/metrics/reception.py`): distância
  ao francês mais próximo quando recebe, para ver se a "saída livre" que o rest-attack promete chegou a
  disparar. É o segundo uso do 360, agora do lado ofensivo.

### A análise de contexto: momentum de ameaça ao longo do tempo

Proxy de ameaça transparente (não é um xT treinado): cada ação vale pontos conforme progride para zonas
perigosas (remate 1.0, entrada na área 0.5, entrada no último terço 0.2), somado por minuto e suavizado,
`Portugal menos França` (`src/metrics/momentum.py`). É **independente** da hipótese Leão/Ronaldo: descreve
o fluxo do jogo por si só, o que a torna robusta à amostra pequena.

## Insights e conclusões

### Insight 1: o rest-attack e quem o paga

- Leão e Ronaldo contestam menos a bola, e o fosso **sobrevive ao ajuste por oportunidade** (contests por 100 exp:
  Bernardo 4.9, Ronaldo 3.2, Leão 2.1). Estável na 1ª parte (Leão 1.4), não é efeito das fases tardias.
- **Onde pressionam (a pressão alta):** por terço, o Leão faz só **1 contest no terço de ataque**
  em todo o jogo, a assinatura pura do rest-attack. Ronaldo pressiona alto (8), lidera a linha, logo não
  são o mesmo perfil, o Leão é o "rester". A métrica nativa da StatsBomb corrobora: **1 counterpress** do
  Leão em 105 min, o avançado que menos fez das duas equipas (Ronaldo 3, Mbappé 2).
- **Quem paga:** na esquerda, o extremo (Leão) contesta 6 vezes no seu corredor e o lateral **Nuno Mendes 13**;
  na direita inverte-se (o próprio Bernardo, 17). O **Palhinha** varre toda a largura (o mais contestador do
  jogo) e do lado do Leão defende mais avançado (protege a profundidade que o Leão, alto, deixa atrás). E o
  **Nuno Mendes** carrega o corredor nas três frentes: cobre (13 contests), constrói (**107 passes** vs os 36
  do Leão, dos menos envolvidos) e ataca (**3º da equipa em ameaça gerada, 7.2; 4º em progressões**). É o
  lateral que torna o rest-attack sustentável, faz o trabalho do corredor enquanto o Leão fica alto.
- **Resultou:** o espaço concedido (360) lê-se em duas partes, junto à baliza apertado em todos (3.5-4.6m, bloco
  organizado) e na construção mais solto (6.5-7.3m, o reflexo de pressionar pouco alto). O corredor do Leão **não é
  o mais aberto** (o mais solto na construção é o do Bernardo, 7.3). O rest-attack do Leão não se traduziu em
  mais espaço no seu corredor.
- **O benefício, testado (duas vias):** o rest-attack pode render de duas formas, receber livre nas costas
  (a saída em transição) ou, com pernas frescas, desmarcar-se e conduzir. A **primeira via quase não disparou**:
  o Leão recebeu **alto mas apertado** (mediana 3.8m, 64% a menos de 5m, marcado 1v1; só 5% das receções foram
  livres e altas, 360), contra uma França que quase não concedeu transições. O valor veio da **segunda via**:
  o Leão é **2º em ameaça gerada** (xT-lite 7.6, a mais assente em condução da equipa, 2.2) e **1º em conduções
  progressivas** (14, definição FBref), a conduzir e a bater o homem, não a rematar (0.13 xG, a lente errada
  para um extremo). Duas leituras do aperto, e nenhuma é "o Leão falhou": ou a saída livre não chegou
  a existir, ou a França marcou-o de perto; com event+360 não as distingo (tracking diria melhor).

**Leitura acionável:** o perfil rest-attack é viável numa estrutura com um lateral que cobre e um pivot que
protege a profundidade. E os dois avançados não são o mesmo caso: Ronaldo contribui moderadamente mais na
pressão e no posicionamento defensivo, mas sem a mesma preponderância ofensiva, por isso a análise se centrou
no Leão, o rest-attack mais puro e o mais produtivo à frente. Não é um traço projetável do jogador a partir de
um jogo, é uma condição estrutural que o torna sustentável.

### Insight 2: o fluxo do jogo

Jogo equilibrado, com Portugal a sombrear nos números de base (posse 60%, xG 1.41 vs 1.07, ameaça total
59.8 vs 55.1). A França teve o seu melhor momento aos 60-75', e Portugal empurrou no início do prolongamento.

## Como correr

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# App visual (descarrega os dados na primeira execução e faz cache em data/raw/)
streamlit run app.py

# Ou correr uma métrica isolada no terminal
python -m src.metrics.pressing
python -m src.metrics.momentum
python -m src.viz            # regenera as figuras em reports/figures/
```

## Estrutura do projeto

```
src/
  ingest.py            download + cache dos JSON (events, lineups, 360)
  build.py             normalização em DataFrames + join 360
  metrics/
    pressing.py        rest-attack: contests por oportunidade (eventos)
    space.py           rest-attack: espaço concedido por zona (360)
    reception.py       rest-attack: espaço à receção do Leão, o teste do benefício (360)
    offense.py         rest-attack: o payoff ofensivo (ameaça gerada, conduções progressivas)
    momentum.py        contexto: momentum de ameaça ao longo do tempo
  viz.py               figuras (momentum, ranking, flanco, Palhinha, espaço, receções, payoff, ações)
app.py                 app Streamlit
reports/figures/       figuras geradas
```

## Limitações

- **Um jogo.** Tudo é descritivo, não projetável. As contagens por corredor são pequenas (6, 13, 17).
- **Lente só de Portugal.** Não meço os espaços nem os jogadores da França. Nem sequer consigo dizer
  com confiança por que lado a França atacou mais: por toques o jogo foi equilibrado nos dois flancos (mais
  pelo lado do Leão no meio-campo, mais pelo do Mbappé no último terço). Fico pela descrição, sem reduzir
  a um lado nem inferir intenção.
- **O 360 são snapshots, não tracking.** Sem identidade dos jogadores sem bola, sem distância percorrida.
  Duas direções futuras: a sobrecarga numérica local (contar franceses vs portugueses à volta da bola no
  freeze frame), que estes dados suportam e deixei por âmbito; e o rest-attack por jogador (quantas vezes
  o Leão em concreto fica alto como saída de transição), que precisaria de tracking com identidade, que
  estes dados não dão.
- **Proxies.** "Distância ao mais próximo" é um proxy cru de marcação; "descansar" vs "posicionar-se" é
  intenção, que um snapshot não observa.

## Nota sobre uso de IA

Usei o Claude Code como par técnico ao longo do trabalho, sobretudo para acelerar o código e para pressionar
o raciocínio (contra-argumentar e verificar afirmações contra os dados). A escolha do jogo, a definição das
métricas, o julgamento de futebol e a revisão são meus.
