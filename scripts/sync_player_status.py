"""
PitchMaster AI — Player Availability Sync
=========================================
Pulls current injuries/doubts from API-Football, maps each player onto the
Transfermarkt `player_id` used everywhere else in PitchMaster, and stores the
result in Redis (served by GET /api/v1/players/...) and on the Pinecone
profile vectors (seen by the RAG agent).

Needs API_FOOTBALL_KEY in .env (free key: https://dashboard.api-football.com)
and a running Redis (docker-compose up -d redis). One request per league per
run; raw responses are cached for API_FOOTBALL_CACHE_TTL, so re-runs are free.

Confirmed id matches are saved to data/player_id_map.json. To fix a wrong or
missing match, edit that file: {"<api_football_id>": "<transfermarkt_id>"}.

Usage:
    python scripts/sync_player_status.py                       # leagues/season from .env
    python scripts/sync_player_status.py --leagues 39,140 --season 2025
    python scripts/sync_player_status.py --dry-run             # fetch + match, store nothing
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from app.core.config import get_settings  # noqa: E402
from app.services import player_status  # noqa: E402
from app.services.api_football import ApiFootballError, api_football_client  # noqa: E402
from app.services.cache import redis_service  # noqa: E402
from app.services.player_matcher import Candidate, reconcile  # noqa: E402

settings = get_settings()

PLAYERS_CSV = PROJECT_ROOT / "kaggle_football" / "players.csv"
ID_MAP_PATH = PROJECT_ROOT / "data" / "player_id_map.json"


def load_candidates(season: int) -> dict[str, list[Candidate]]:
    """Dataset players still active around `season`, grouped by current club."""
    if not PLAYERS_CSV.exists():
        raise SystemExit(f"{PLAYERS_CSV} not found — download the Kaggle dataset first (see README).")
    cols = ["player_id", "name", "first_name", "last_name", "current_club_name", "last_season"]
    df = pd.read_csv(PLAYERS_CSV, usecols=cols, low_memory=False)
    df = df[df["last_season"] >= season - 1].fillna("")

    by_club: dict[str, list[Candidate]] = {}
    for row in df.itertuples(index=False):
        by_club.setdefault(row.current_club_name, []).append(Candidate(
            player_id=str(row.player_id),
            name=str(row.name),
            first_name=str(row.first_name),
            last_name=str(row.last_name),
            club=str(row.current_club_name),
        ))
    return by_club


def load_id_map() -> dict[str, str]:
    if ID_MAP_PATH.exists():
        return json.loads(ID_MAP_PATH.read_text(encoding="utf-8"))
    return {}


def save_id_map(id_map: dict[str, str]) -> None:
    ID_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    ID_MAP_PATH.write_text(json.dumps(id_map, indent=2, sort_keys=True), encoding="utf-8")


async def main(leagues: list[int], season: int, dry_run: bool) -> None:
    print(f"\n⚽  Syncing player availability — leagues {leagues}, season {season}\n")

    await redis_service.connect()
    if redis_service._use_memory and not dry_run:
        raise SystemExit("Redis is not reachable, so statuses would be lost on exit. "
                         "Start it with: docker-compose up -d redis")

    rows: list[dict] = []
    for league in leagues:
        try:
            league_rows = await api_football_client.get_injuries(league, season)
        except ApiFootballError as exc:
            raise SystemExit(f"❌  League {league}: {exc}")
        print(f"  📥  League {league}: {len(league_rows)} injury rows")
        rows.extend(league_rows)

    statuses = player_status.build_statuses(rows)
    print(f"  🩹  {len(statuses)} players currently out or doubtful")

    candidates = load_candidates(season)
    id_map = load_id_map()
    unmatched = []
    for s in statuses:
        key = str(s.api_football_id)
        if key in id_map:
            s.player_id = id_map[key]
            continue
        match = reconcile(s.name, s.team, candidates)
        if match:
            s.player_id = match.player_id
            id_map[key] = match.player_id
        else:
            unmatched.append(s)

    matched = len(statuses) - len(unmatched)
    print(f"  🔗  Matched {matched}/{len(statuses)} to dataset players")
    for s in unmatched:
        print(f"       ✗ {s.name} ({s.team}) — api_football_id {s.api_football_id}")

    if dry_run:
        print("\n  Dry run — nothing stored.\n")
        return

    save_id_map(id_map)
    cleared = await player_status.store_statuses(statuses)
    print(f"  💾  Stored {matched} statuses in Redis; {len(cleared)} players cleared as available")

    if settings.PINECONE_API_KEY:
        updated = await player_status.update_vector_metadata(statuses, cleared)
        print(f"  🌲  Updated status metadata on {updated} Pinecone vectors")
    else:
        print("  ⚠️  PINECONE_API_KEY not set — skipped vector metadata update")

    await redis_service.close()
    print("\n✅  Sync complete.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync player injuries from API-Football")
    parser.add_argument("--leagues", type=str, default=settings.API_FOOTBALL_LEAGUES,
                        help="Comma-separated API-Football league ids (39=EPL, 140=La Liga, "
                             "135=Serie A, 78=Bundesliga, 61=Ligue 1)")
    parser.add_argument("--season", type=int, default=settings.API_FOOTBALL_SEASON,
                        help="Season start year, e.g. 2025 for 2025/26")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and match, but store nothing")
    args = parser.parse_args()

    league_ids = [int(x) for x in args.leagues.split(",") if x.strip()]
    asyncio.run(main(league_ids, args.season, args.dry_run))
