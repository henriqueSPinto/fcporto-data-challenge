"""Ingestão dos dados em bruto do StatsBomb Open Data para um jogo.

Baixa e faz cache local dos três ficheiros JSON de um jogo (events, lineups e
360) a partir do repositório público da StatsBomb. O cache evita voltar a bater
na rede a cada execução e torna o pipeline reproduzível offline.

Nota de produção: num clube com subscrição, a ingestão seria feita contra a API
oficial da StatsBomb (ou via `statsbombpy`) para um data warehouse. Aqui usa-se
a via pública open-data, que é a fonte pedida pelo exercício.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# Mapeia um "tipo" lógico para o caminho do ficheiro no repositório open-data.
_ENDPOINTS = {
    "events": "events/{match_id}.json",
    "lineups": "lineups/{match_id}.json",
    "three_sixty": "three-sixty/{match_id}.json",
}

# data/raw/ na raiz do projeto (este ficheiro está em src/).
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def _cache_path(match_id, kind, data_dir) -> Path:
    return Path(data_dir) / str(match_id) / f"{kind}.json"


def _download(kind: str, match_id) -> object:
    url = f"{BASE_URL}/{_ENDPOINTS[kind].format(match_id=match_id)}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_match(match_id, data_dir=DEFAULT_DATA_DIR, force: bool = False) -> dict:
    """Baixa (ou lê do cache) os 3 JSON do jogo.

    Devolve um dict {kind: dados_json}. Se `force=True`, ignora o cache e
    volta a baixar.
    """
    out = {}
    for kind in _ENDPOINTS:
        path = _cache_path(match_id, kind, data_dir)
        if path.exists() and not force:
            out[kind] = json.loads(path.read_text(encoding="utf-8"))
        else:
            data = _download(kind, match_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            out[kind] = data
    return out


# Alias legível para quem só quer carregar (o cache trata do download).
load_match = fetch_match


if __name__ == "__main__":
    import sys

    mid = int(sys.argv[1]) if len(sys.argv) > 1 else 3942349
    data = fetch_match(mid)
    print(
        f"match {mid}: events={len(data['events'])} | "
        f"lineups={len(data['lineups'])} equipas | "
        f"360={len(data['three_sixty'])} frames"
    )
