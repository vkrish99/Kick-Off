"""
PitchMaster AI — Real Dataset Loader
======================================
Loads actual football data from the Kaggle and Transfermarkt datasets,
transforms it into rich player profiles, and ingests everything into the
PitchMaster system (Pinecone vector store + telemetry queue).

Data sources:
  kaggle_football/
    players.csv          — player bio, position, club, market value
    appearances.csv      — match appearances (goals, assists, minutes, cards)
    clubs.csv            — club names and metadata
    player_valuations.csv — market value history
  football-datasets-main/.../transfermarkt/
    player_profiles/     — detailed player profiles (height, foot, citizenship)
    player_injuries/     — injury history
    player_latest_market_value/ — latest valuations

Usage:
    python scripts/load_datasets.py                # load all active players
    python scripts/load_datasets.py --team "Manchester City"   # one team
    python scripts/load_datasets.py --limit 100    # limit player count
    python scripts/load_datasets.py --no-ai        # telemetry only
"""

import argparse
import asyncio
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

# Fix Windows console encoding for emoji/unicode
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx
import pandas as pd


BASE_URL = "http://localhost:8000"

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = PROJECT_ROOT / "kaggle_football"
TM_DIR = PROJECT_ROOT / "football-datasets-main" / "football-datasets-main" / "datalake" / "transfermarkt"


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------
def load_kaggle_players() -> pd.DataFrame:
    """Load player data from kaggle_football/players.csv."""
    path = KAGGLE_DIR / "players.csv"
    print(f"  📂  Loading {path.name} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"       → {len(df):,} players loaded")
    return df


def load_kaggle_appearances() -> pd.DataFrame:
    """Load appearance data from kaggle_football/appearances.csv."""
    path = KAGGLE_DIR / "appearances.csv"
    print(f"  📂  Loading {path.name} (this may take a moment)...")
    df = pd.read_csv(path, low_memory=False)
    print(f"       → {len(df):,} appearance records loaded")
    return df


def load_kaggle_clubs() -> pd.DataFrame:
    """Load club data."""
    path = KAGGLE_DIR / "clubs.csv"
    df = pd.read_csv(path, low_memory=False)
    return df


def load_valuations() -> pd.DataFrame:
    """Load player valuations."""
    path = KAGGLE_DIR / "player_valuations.csv"
    print(f"  📂  Loading {path.name} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"       → {len(df):,} valuation records loaded")
    return df


def load_tm_profiles() -> pd.DataFrame:
    """Load Transfermarkt player profiles."""
    path = TM_DIR / "player_profiles" / "player_profiles.csv"
    if not path.exists():
        print(f"  ⚠️  Transfermarkt profiles not found at {path}")
        return pd.DataFrame()
    print(f"  📂  Loading transfermarkt profiles ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"       → {len(df):,} TM profiles loaded")
    return df


def load_injuries() -> pd.DataFrame:
    """Load injury data."""
    path = TM_DIR / "player_injuries" / "player_injuries.csv"
    if not path.exists():
        return pd.DataFrame()
    print(f"  📂  Loading injury records ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"       → {len(df):,} injury records loaded")
    return df


# ---------------------------------------------------------------------------
# Data Transformation
# ---------------------------------------------------------------------------
def aggregate_appearances(appearances_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate appearance stats per player."""
    agg = appearances_df.groupby("player_id").agg(
        total_appearances=("appearance_id", "count"),
        total_goals=("goals", "sum"),
        total_assists=("assists", "sum"),
        total_minutes=("minutes_played", "sum"),
        total_yellow_cards=("yellow_cards", "sum"),
        total_red_cards=("red_cards", "sum"),
        last_appearance=("date", "max"),
    ).reset_index()

    # Calculate per-90 stats
    agg["minutes_safe"] = agg["total_minutes"].clip(lower=1)
    agg["goals_per_90"] = (agg["total_goals"] / agg["minutes_safe"] * 90).round(2)
    agg["assists_per_90"] = (agg["total_assists"] / agg["minutes_safe"] * 90).round(2)
    agg["goal_contributions_per_90"] = (agg["goals_per_90"] + agg["assists_per_90"]).round(2)

    return agg


def get_latest_valuation(valuations_df: pd.DataFrame) -> pd.DataFrame:
    """Get the most recent valuation for each player."""
    if valuations_df.empty:
        return pd.DataFrame(columns=["player_id", "market_value_in_eur"])
    valuations_df["date"] = pd.to_datetime(valuations_df["date"], errors="coerce")
    latest = valuations_df.sort_values("date").groupby("player_id").last().reset_index()
    return latest[["player_id", "market_value_in_eur"]]


def aggregate_injuries(injuries_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize injury history per player."""
    if injuries_df.empty:
        return pd.DataFrame(columns=["player_id", "total_injuries", "total_days_missed"])
    agg = injuries_df.groupby("player_id").agg(
        total_injuries=("injury_reason", "count"),
        total_days_missed=("days_missed", "sum"),
        total_games_missed=("games_missed", lambda x: pd.to_numeric(x, errors="coerce").sum()),
        last_injury=("injury_reason", "last"),
    ).reset_index()
    return agg


def format_market_value(value) -> str:
    """Format market value as human-readable string."""
    try:
        v = float(value)
        if v >= 1_000_000:
            return f"€{v / 1_000_000:.1f}M"
        elif v >= 1_000:
            return f"€{v / 1_000:.0f}K"
        else:
            return f"€{v:.0f}"
    except (ValueError, TypeError):
        return "N/A"


def position_to_skills(position: str, sub_position: str) -> list:
    """Map position to realistic skill keywords."""
    skills_map = {
        "Goalkeeper": ["shot-stopping", "distribution", "command of area", "reflexes", "positioning"],
        "Defender": ["tackling", "positioning", "aerial duels", "interceptions", "ball playing"],
        "Midfield": ["passing", "pressing", "positioning", "ball retention", "vision"],
        "Attack": ["finishing", "dribbling", "pace", "movement", "creativity"],
    }
    sub_skills = {
        "Centre-Forward": ["finishing", "heading", "hold-up play", "penalty area movement"],
        "Left Winger": ["pace", "dribbling", "crossing", "cutting inside"],
        "Right Winger": ["pace", "dribbling", "crossing", "1v1"],
        "Attacking Midfield": ["creativity", "through balls", "shooting", "vision"],
        "Central Midfield": ["passing", "box-to-box", "tackling", "ball carrying"],
        "Defensive Midfield": ["interceptions", "tackling", "positioning", "passing"],
        "Left-Back": ["overlapping runs", "crossing", "pace", "defending"],
        "Right-Back": ["overlapping runs", "crossing", "pace", "defending"],
        "Centre-Back": ["aerial duels", "tackling", "leadership", "reading the game"],
        "Second Striker": ["link-up play", "finishing", "movement", "pressing"],
    }

    base = skills_map.get(str(position), ["versatility", "work rate"])
    extra = sub_skills.get(str(sub_position), [])
    # Combine unique skills
    all_skills = list(dict.fromkeys(base + extra))
    return all_skills[:8]


def build_performance_summary(row: pd.Series) -> str:
    """Build a rich text summary from aggregated player stats."""
    parts = []

    name = row.get("name", "Unknown Player")
    position = row.get("position", "Unknown")
    sub_position = row.get("sub_position", "")
    club = row.get("current_club_name", "Unknown Club")
    nationality = row.get("country_of_citizenship", "")
    age = ""
    dob = row.get("date_of_birth", "")
    if pd.notna(dob) and dob:
        try:
            born = pd.to_datetime(str(dob).split(" ")[0])
            age = str((pd.Timestamp.now() - born).days // 365)
        except Exception:
            pass

    height = row.get("height_in_cm", "")
    foot = row.get("foot", "")

    parts.append(f"{name} is a {sub_position or position} currently at {club}.")
    if nationality:
        parts.append(f"Nationality: {nationality}.")
    if age:
        parts.append(f"Age: {age}.")
    if pd.notna(height) and height:
        parts.append(f"Height: {int(float(height))}cm.")
    if pd.notna(foot) and foot:
        parts.append(f"Preferred foot: {foot}.")

    # Performance stats
    apps = row.get("total_appearances", 0)
    goals = row.get("total_goals", 0)
    assists = row.get("total_assists", 0)
    minutes = row.get("total_minutes", 0)
    yellows = row.get("total_yellow_cards", 0)
    reds = row.get("total_red_cards", 0)
    g90 = row.get("goals_per_90", 0)
    a90 = row.get("assists_per_90", 0)

    if apps and apps > 0:
        parts.append(
            f"Career stats: {int(apps)} appearances, {int(goals)} goals, "
            f"{int(assists)} assists, {int(minutes):,} minutes played."
        )
        if g90 and g90 > 0:
            parts.append(f"Goals per 90 minutes: {g90:.2f}. Assists per 90: {a90:.2f}.")
        if yellows > 0 or reds > 0:
            parts.append(f"Discipline: {int(yellows)} yellow cards, {int(reds)} red cards.")

    # Market value
    mv = row.get("market_value_in_eur", None)
    hmv = row.get("highest_market_value_in_eur", None)
    if pd.notna(mv) and mv:
        parts.append(f"Current market value: {format_market_value(mv)}.")
    if pd.notna(hmv) and hmv:
        parts.append(f"Peak market value: {format_market_value(hmv)}.")

    # Injuries
    total_inj = row.get("total_injuries", 0)
    if pd.notna(total_inj) and total_inj > 0:
        days = row.get("total_days_missed", 0)
        parts.append(f"Injury history: {int(total_inj)} injuries, {int(days)} days missed.")

    return " ".join(parts)


def build_profiles(
    players_df: pd.DataFrame,
    appearances_agg: pd.DataFrame,
    injuries_agg: pd.DataFrame,
    team_filter: str = None,
    limit: int = None,
) -> list[dict]:
    """Build rich player profile objects ready for API ingestion."""

    # Filter to active players (have a current club and recent season)
    df = players_df.copy()
    df = df[df["current_club_name"].notna() & (df["current_club_name"] != "")]
    df = df[df["last_season"].notna()]

    # Optional team filter
    if team_filter:
        search = team_filter.lower()
        df = df[df["current_club_name"].str.lower().str.contains(search, na=False)]
        print(f"  🎯  Filtered to team '{team_filter}': {len(df)} players")

    # Sort by market value (descending) to get most prominent players first
    df["market_value_in_eur"] = pd.to_numeric(df["market_value_in_eur"], errors="coerce")
    df = df.sort_values("market_value_in_eur", ascending=False)

    if limit:
        df = df.head(limit)

    # Merge with aggregated stats
    df = df.merge(appearances_agg, on="player_id", how="left")
    df = df.merge(injuries_agg, on="player_id", how="left")

    # Fill NaN stats with 0
    stat_cols = [
        "total_appearances", "total_goals", "total_assists", "total_minutes",
        "total_yellow_cards", "total_red_cards", "goals_per_90", "assists_per_90",
        "total_injuries", "total_days_missed",
    ]
    for col in stat_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    profiles = []
    for _, row in df.iterrows():
        position = str(row.get("position", "Unknown"))
        sub_position = str(row.get("sub_position", ""))
        skills = position_to_skills(position, sub_position)
        summary = build_performance_summary(row)

        profile = {
            "player_id": str(row["player_id"]),
            "name": str(row.get("name", "Unknown")),
            "position": sub_position if sub_position and sub_position != "nan" else position,
            "team": str(row.get("current_club_name", "Unknown")),
            "skills": skills,
            "recent_performance_summary": summary,
        }
        profiles.append(profile)

    return profiles


# ---------------------------------------------------------------------------
# Telemetry generation from real appearance data
# ---------------------------------------------------------------------------
def generate_telemetry_from_appearances(
    players_df: pd.DataFrame,
    appearances_df: pd.DataFrame,
    team_filter: str = None,
    limit: int = 50,
) -> list[dict]:
    """Generate realistic telemetry snapshots from actual player data."""

    df = players_df.copy()
    df = df[df["current_club_name"].notna()]

    if team_filter:
        search = team_filter.lower()
        df = df[df["current_club_name"].str.lower().str.contains(search, na=False)]

    df = df.head(limit)
    player_ids = set(df["player_id"].values)

    # Get recent appearances for these players
    recent = appearances_df[appearances_df["player_id"].isin(player_ids)]

    pitch_zones = {
        "Goalkeeper": (5, 34),
        "Defender": (25, 34),
        "Midfield": (52, 34),
        "Attack": (78, 34),
    }

    telemetry = []
    for _, player in df.iterrows():
        pos = str(player.get("position", "Midfield"))
        base_x, base_y = pitch_zones.get(pos, (52, 34))

        # Simulate minute 65 telemetry
        speed_base = {"Goalkeeper": 5, "Defender": 16, "Midfield": 20, "Attack": 25}.get(pos, 18)
        hr_base = {"Goalkeeper": 120, "Defender": 148, "Midfield": 155, "Attack": 160}.get(pos, 150)
        fatigue_base = {"Goalkeeper": 0.1, "Defender": 0.25, "Midfield": 0.35, "Attack": 0.3}.get(pos, 0.25)

        t = {
            "player_id": str(player["player_id"]),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "speed": round(speed_base + random.uniform(-4, 4), 1),
            "distance_covered": round(random.uniform(0.6, 1.3), 2),
            "heart_rate": hr_base + random.randint(-10, 10),
            "position_x": round(base_x + random.uniform(-10, 10), 1),
            "position_y": round(base_y + random.uniform(-15, 15), 1),
            "fatigue_index": round(min(1.0, fatigue_base + 0.65 / 90 * 0.4 + random.uniform(-0.05, 0.05)), 2),
        }
        telemetry.append(t)

    return telemetry


# ---------------------------------------------------------------------------
# API interaction
# ---------------------------------------------------------------------------
async def ingest_profiles_batch(
    client: httpx.AsyncClient, profiles: list[dict], batch_size: int = 20
) -> int:
    """Send profiles to the API in batches."""
    total_ingested = 0
    total = len(profiles)

    for start in range(0, total, batch_size):
        batch = profiles[start : start + batch_size]
        batch_num = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        try:
            r = await client.post(
                "/api/v1/ingest/profiles", json=batch, timeout=120
            )
            if r.status_code == 201:
                count = r.json()["count"]
                total_ingested += count
                print(
                    f"    ✅  Batch {batch_num}/{total_batches}: "
                    f"ingested {count} profiles ({total_ingested}/{total} total)"
                )
            elif r.status_code == 503:
                print(f"    ⚠️   API keys not configured — skipping (503)")
                return total_ingested
            else:
                print(f"    ❌  Batch {batch_num} failed [{r.status_code}]: {r.text[:200]}")
        except Exception as e:
            print(f"    ❌  Batch {batch_num} error: {e}")

    return total_ingested


async def ingest_telemetry(client: httpx.AsyncClient, telemetry: list[dict]) -> None:
    """Send telemetry data to the API."""
    try:
        r = await client.post(
            "/api/v1/ingest/telemetry",
            json={"data": telemetry},
            timeout=10,
        )
        if r.status_code == 202:
            print(f"    ✅  Queued {r.json()['count']} telemetry records")
        else:
            print(f"    ❌  [{r.status_code}]: {r.text[:200]}")
    except Exception as e:
        print(f"    ❌  Error: {e}")


async def run_analysis(client: httpx.AsyncClient, query: str, telemetry: list[dict]) -> None:
    """Run an analysis query."""
    print(f"\n  🤖  Query: \"{query}\"")
    payload = {"query": query, "telemetry_context": telemetry[:15]}   # limit context size
    try:
        r = await client.post("/api/v1/analyze", json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            print(f"  📊  Agent: [{data.get('route_used', 'N/A')}]")
            print(f"\n{'-'*60}")
            insight = data["insight"]
            # Wrap long lines
            for line in insight.split("\n"):
                print(f"  {line}")
            print(f"{'-'*60}")
        elif r.status_code == 503:
            print(f"  ⚠️   LLM API key not configured")
        else:
            print(f"  ❌  [{r.status_code}]: {r.text[:300]}")
    except Exception as e:
        print(f"  ❌  Error: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main(no_ai: bool, team: str, limit: int) -> None:
    print(f"\n{'='*60}")
    print(f"  ⚽  PitchMaster AI — Real Dataset Loader")
    if team:
        print(f"  Team filter: {team}")
    print(f"  Player limit: {limit or 'ALL'}")
    print(f"{'='*60}")

    # --- Load datasets ---
    print(f"\n{'='*60}")
    print(f"  📊  Loading datasets ...")
    print(f"{'='*60}\n")

    players_df = load_kaggle_players()
    appearances_df = load_kaggle_appearances()
    valuations_df = load_valuations()
    injuries_df = load_injuries()

    # --- Aggregate stats ---
    print(f"\n  ⚙️   Aggregating player statistics ...")
    appearances_agg = aggregate_appearances(appearances_df)
    injuries_agg = aggregate_injuries(injuries_df)
    print(f"       → Stats computed for {len(appearances_agg):,} players")

    # --- Merge latest valuations into players ---
    # The players.csv already has market_value_in_eur, so we use that directly

    # --- Build profiles ---
    print(f"\n{'='*60}")
    print(f"  🏗️   Building player profiles ...")
    print(f"{'='*60}\n")

    profiles = build_profiles(
        players_df, appearances_agg, injuries_agg,
        team_filter=team, limit=limit,
    )
    print(f"\n  📋  Built {len(profiles)} player profiles")

    # Show sample
    print(f"\n  Sample profiles:")
    for p in profiles[:5]:
        print(f"    ▸ {p['name']:<30} ({p['position']}) — {p['team']}")

    # --- Generate telemetry ---
    telemetry = generate_telemetry_from_appearances(
        players_df, appearances_df, team_filter=team, limit=min(limit or 50, 50)
    )
    print(f"\n  📡  Generated {len(telemetry)} telemetry records\n")

    # --- Connect to server ---
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # Health check
        try:
            r = await client.get("/health", timeout=5)
            print(f"  🏥  Server: {r.json()['status'].upper()} (v{r.json()['version']})\n")
        except Exception:
            print(f"\n  ❌  Cannot reach server at {BASE_URL}")
            print(f"      Start with: python -m uvicorn app.main:app --reload")
            sys.exit(1)

        # --- Ingest profiles ---
        if not no_ai:
            print(f"{'='*60}")
            print(f"  📋  Ingesting {len(profiles)} player profiles into Pinecone ...")
            print(f"{'='*60}\n")
            ingested = await ingest_profiles_batch(client, profiles, batch_size=15)
            print(f"\n  ✅  Total ingested: {ingested} profiles\n")

        # --- Ingest telemetry ---
        print(f"{'='*60}")
        print(f"  📡  Pushing telemetry for {len(telemetry)} players ...")
        print(f"{'='*60}\n")
        await ingest_telemetry(client, telemetry)

        # Small wait for queue processing
        await asyncio.sleep(1)
        try:
            r = await client.get("/api/v1/queue/status", timeout=5)
            if r.status_code == 200:
                d = r.json()
                print(f"  📬  Queue: depth={d['depth']}  processed={d['processed_total']}")
        except Exception:
            pass

        # --- Demo analysis queries ---
        if not no_ai:
            print(f"\n{'='*60}")
            print(f"  🧠  Running analysis queries ...")
            print(f"{'='*60}")

            queries = [
                "Which players have the highest goal contribution per 90 minutes?",
                "Based on current fatigue levels, which players should be substituted?",
                "Recommend a formation based on the available player data.",
            ]
            for q in queries:
                await run_analysis(client, q, telemetry)
                print()

    print(f"\n{'='*60}")
    print(f"  ✅  Done! Open http://localhost:8000/docs to explore the API.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PitchMaster AI — Real Dataset Loader")
    parser.add_argument("--no-ai", action="store_true", help="Skip AI endpoints (telemetry only)")
    parser.add_argument("--team", type=str, default=None, help="Filter by team name (e.g. 'Manchester City')")
    parser.add_argument("--limit", type=int, default=200, help="Max players to ingest (default: 200)")
    args = parser.parse_args()

    asyncio.run(main(no_ai=args.no_ai, team=args.team, limit=args.limit))
