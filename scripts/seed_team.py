"""
PitchMaster AI — Manchester City Team Data Seeder
===================================================
Seeds a full 25-player squad with rich, real-world player profiles:
  1. Player profiles  → POST /api/v1/ingest/profiles   (requires GROQ + PINECONE keys)
  2. Live telemetry   → POST /api/v1/ingest/telemetry  (no keys needed)
  3. Demo analysis    → POST /api/v1/analyze            (requires GROQ key)

Usage:
    python scripts/seed_team.py              # full run (profiles + telemetry + analyze)
    python scripts/seed_team.py --no-ai      # telemetry only, skips AI endpoints
    python scripts/seed_team.py --team "Arsenal"  # swap team name in profiles

python -m pip install httpx is already in requirements.txt
"""

import argparse
import asyncio
import json
import random
import sys
from datetime import datetime, timezone

# Fix Windows console encoding for emoji/unicode
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import httpx

BASE_URL = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Manchester City 2024/25 Squad — Rich Player Profiles
# ---------------------------------------------------------------------------
SQUAD = [
    # ======================== GOALKEEPERS ========================
    {
        "player_id": "mci_gk1",
        "name": "Ederson Moraes",
        "position": "GK",
        "number": 31,
        "age": 30,
        "nationality": "Brazil",
        "height_cm": 188,
        "weight_kg": 86,
        "preferred_foot": "Left",
        "skills": [
            "shot-stopping", "distribution", "sweeper-keeper",
            "long passing", "penalty saving", "1v1 situations",
            "command of area", "playing out from back",
        ],
        "recent_performance_summary": (
            "Ederson has been outstanding this season with a 76% save percentage and "
            "7 clean sheets in 22 Premier League appearances. His distribution remains "
            "elite — averaging 42 accurate long balls per 90 minutes (89% accuracy). "
            "He has made 3 penalty saves this season. His sweeper-keeper actions "
            "(2.1 per 90) are the highest among Premier League goalkeepers. "
            "Comfortable playing as an extra midfielder when City build from the back."
        ),
    },
    {
        "player_id": "mci_gk2",
        "name": "Stefan Ortega",
        "position": "GK",
        "number": 18,
        "age": 31,
        "nationality": "Germany",
        "height_cm": 185,
        "weight_kg": 85,
        "preferred_foot": "Right",
        "skills": [
            "shot-stopping", "reflexes", "command of area",
            "distribution", "set piece organization",
        ],
        "recent_performance_summary": (
            "Stefan Ortega has been a reliable backup, starting 8 matches this season. "
            "Kept 4 clean sheets with a 79% save percentage. Strong reflexes and "
            "positioning have earned him trust in cup competitions. His distribution "
            "is slightly less adventurous than Ederson's but remains above average."
        ),
    },

    # ======================== DEFENDERS ========================
    {
        "player_id": "mci_rb1",
        "name": "Kyle Walker",
        "position": "RB",
        "number": 2,
        "age": 34,
        "nationality": "England",
        "height_cm": 178,
        "weight_kg": 80,
        "preferred_foot": "Right",
        "skills": [
            "pace", "recovery runs", "defending", "overlapping runs",
            "1v1 defending", "crossing", "leadership",
        ],
        "recent_performance_summary": (
            "Kyle Walker remains one of the fastest players in the Premier League "
            "with a top speed of 36.2 km/h recorded this season. Despite his age, "
            "he averages 2.4 tackles and 1.8 interceptions per 90 minutes. "
            "His overlapping and underlapping runs create width — 3.2 crosses per game. "
            "Occasionally inverts into midfield under Pep's system. 1 goal, 3 assists "
            "in 24 league appearances. Leadership on and off the pitch is invaluable."
        ),
    },
    {
        "player_id": "mci_cb1",
        "name": "Ruben Dias",
        "position": "CB",
        "number": 3,
        "age": 27,
        "nationality": "Portugal",
        "height_cm": 187,
        "weight_kg": 82,
        "preferred_foot": "Right",
        "skills": [
            "aerial duels", "tackling", "leadership", "ball playing",
            "positioning", "reading the game", "line-breaking passes",
        ],
        "recent_performance_summary": (
            "Ruben Dias has been Manchester City's defensive marshal. 93% tackle "
            "success rate, 4.2 aerial duels won per 90 minutes (68% win rate). "
            "His ball-playing ability is key to City's build-up: 91% pass accuracy "
            "and 3.8 progressive passes per 90. Scored 2 headed goals from set pieces. "
            "Has played every minute of City's last 14 league matches — true iron man."
        ),
    },
    {
        "player_id": "mci_cb2",
        "name": "Manuel Akanji",
        "position": "CB",
        "number": 25,
        "age": 29,
        "nationality": "Switzerland",
        "height_cm": 188,
        "weight_kg": 85,
        "preferred_foot": "Right",
        "skills": [
            "positioning", "tackling", "pace", "ball playing",
            "press resistance", "versatility", "recovery speed",
        ],
        "recent_performance_summary": (
            "Manuel Akanji has been remarkably consistent — the most minutes played "
            "by any City outfield player. Pass accuracy of 93%, with excellent "
            "press-resistant ability (0.3 dispossessions per 90). His pace allows "
            "City to play a high line — recovery speed clocked at 33.1 km/h. "
            "2.8 clearances and 1.3 interceptions per 90. 0 goals, 1 assist."
        ),
    },
    {
        "player_id": "mci_cb3",
        "name": "John Stones",
        "position": "CB",
        "number": 5,
        "age": 30,
        "nationality": "England",
        "height_cm": 188,
        "weight_kg": 80,
        "preferred_foot": "Right",
        "skills": [
            "ball playing", "positioning", "inverted fullback",
            "set pieces", "versatility", "composure on the ball",
        ],
        "recent_performance_summary": (
            "John Stones has been deployed as both CB and an inverted right-back, "
            "stepping into midfield during build-up. His pass accuracy (94%) is the "
            "highest among City's defenders. 5.2 progressive passes per 90 when "
            "deployed in the hybrid role. Injury has limited him to 16 appearances, "
            "but when fit he's crucial to Pep's positional play system. 1 goal, 2 assists."
        ),
    },
    {
        "player_id": "mci_lb1",
        "name": "Josko Gvardiol",
        "position": "LB",
        "number": 24,
        "age": 22,
        "nationality": "Croatia",
        "height_cm": 185,
        "weight_kg": 83,
        "preferred_foot": "Left",
        "skills": [
            "pace", "dribbling", "defending", "overlapping runs",
            "aerial duels", "ball carrying", "progressive passing",
        ],
        "recent_performance_summary": (
            "Josko Gvardiol has adapted superbly to the left-back role. His unique "
            "inside runs from LB into the box have yielded 5 goals and 4 assists — "
            "extraordinary for a defender. Averages 3.1 progressive carries per 90. "
            "Defensively solid: 2.0 tackles and 1.4 interceptions per 90. His top speed "
            "of 34.5 km/h makes him one of the fastest defenders in Europe."
        ),
    },
    {
        "player_id": "mci_rb2",
        "name": "Rico Lewis",
        "position": "RB",
        "number": 82,
        "age": 20,
        "nationality": "England",
        "height_cm": 169,
        "weight_kg": 65,
        "preferred_foot": "Right",
        "skills": [
            "inverted fullback", "passing", "positioning",
            "pressing", "tactical intelligence", "composure",
        ],
        "recent_performance_summary": (
            "Rico Lewis has become Pep's go-to inverted full-back option. Despite his "
            "small stature, his tactical intelligence is remarkable for a 20-year-old. "
            "91% pass accuracy, 2.8 key passes per 90 when he drifts centrally. "
            "Leads all full-backs in the squad for touches in the opposition half. "
            "3 assists in 20 league appearances. Needs to improve aerial defending."
        ),
    },
    {
        "player_id": "mci_cb4",
        "name": "Nathan Ake",
        "position": "CB",
        "number": 6,
        "age": 29,
        "nationality": "Netherlands",
        "height_cm": 180,
        "weight_kg": 75,
        "preferred_foot": "Left",
        "skills": [
            "tackling", "aerial duels", "versatility", "left-footed passing",
            "pace", "positioning", "covering left-back",
        ],
        "recent_performance_summary": (
            "Nathan Ake provides versatility — capable of playing CB or LB. "
            "His left foot is valuable in build-up from the left side. Averages "
            "3.5 aerial duels won per 90. Strong tackling (2.1 per 90) and "
            "clean in possession (90% pass accuracy). 1 goal, 1 assist in 18 "
            "appearances. A reliable rotation option."
        ),
    },

    # ======================== MIDFIELDERS ========================
    {
        "player_id": "mci_cdm1",
        "name": "Rodri",
        "position": "CDM",
        "number": 16,
        "age": 28,
        "nationality": "Spain",
        "height_cm": 191,
        "weight_kg": 82,
        "preferred_foot": "Right",
        "skills": [
            "passing", "pressing", "positioning", "set pieces",
            "long shots", "ball retention", "tactical awareness",
            "aerial duels", "leadership",
        ],
        "recent_performance_summary": (
            "Rodri is the heartbeat of Manchester City and the reigning Ballon d'Or "
            "winner. His pass accuracy of 94.2% is the best in the Premier League "
            "among midfielders. Averages 7.8 progressive passes per 90 and completes "
            "2.4 tackles per 90. City's win rate drops from 78% to 52% when he doesn't "
            "play — an incredible influence metric. 4 goals and 6 assists from CDM. "
            "Currently recovering from ACL injury (expected return date TBD)."
        ),
    },
    {
        "player_id": "mci_cm1",
        "name": "Kevin De Bruyne",
        "position": "CM",
        "number": 17,
        "age": 33,
        "nationality": "Belgium",
        "height_cm": 181,
        "weight_kg": 76,
        "preferred_foot": "Right",
        "skills": [
            "vision", "passing", "long shots", "creativity",
            "set pieces", "crossing", "through balls",
            "shooting power", "free kicks",
        ],
        "recent_performance_summary": (
            "Kevin De Bruyne has been managing fitness after hamstring issues but "
            "remains the most creative player in City's squad. In 14 league appearances, "
            "he has 3 goals and 7 assists. His expected assists (xA) of 0.52 per 90 is "
            "the highest in the league. Key pass rate: 3.8 per game. His long-range "
            "passing (avg. 4.2 successful long balls per 90) unlocks defences. "
            "When fit, he is the system's key orchestrator from CM or the right channel."
        ),
    },
    {
        "player_id": "mci_cm2",
        "name": "Bernardo Silva",
        "position": "CM",
        "number": 20,
        "age": 30,
        "nationality": "Portugal",
        "height_cm": 173,
        "weight_kg": 64,
        "preferred_foot": "Left",
        "skills": [
            "dribbling", "pressing", "passing", "work rate",
            "versatility", "ball retention", "counter-pressing",
            "close control", "playing between the lines",
        ],
        "recent_performance_summary": (
            "Bernardo Silva is City's engine — highest distance covered per 90 (11.8 km) "
            "and most pressures applied (24.3 per 90). Deployed across midfield and on "
            "the right wing. Pass accuracy 92% with 2.1 key passes per 90. His dribble "
            "success rate is 68% (2.4 per 90). 6 goals and 5 assists in 26 appearances. "
            "Vital to City's possession game with an incredible 0.4 dispossessions per 90."
        ),
    },
    {
        "player_id": "mci_cm3",
        "name": "Mateo Kovacic",
        "position": "CM",
        "number": 8,
        "age": 30,
        "nationality": "Croatia",
        "height_cm": 177,
        "weight_kg": 78,
        "preferred_foot": "Right",
        "skills": [
            "dribbling", "passing", "pressing", "box-to-box",
            "ball carrying", "progressive runs", "tackling",
        ],
        "recent_performance_summary": (
            "Mateo Kovacic has stepped up significantly with Rodri's absence. His "
            "ball-carrying ability is elite — 4.8 progressive carries per 90 (squad highest). "
            "Pass accuracy 91% and 1.9 tackles per 90. He's scored 5 goals with powerful "
            "late runs into the box. Covers an average of 11.3 km per game. His weakness "
            "is aerial duels (41% win rate) and occasional positional lapses in the single pivot."
        ),
    },
    {
        "player_id": "mci_cm4",
        "name": "Phil Foden",
        "position": "CM",
        "number": 47,
        "age": 24,
        "nationality": "England",
        "height_cm": 171,
        "weight_kg": 69,
        "preferred_foot": "Left",
        "skills": [
            "dribbling", "finishing", "creativity", "positioning",
            "passing", "shooting", "pressing",
            "half-space movement", "off-the-ball runs",
        ],
        "recent_performance_summary": (
            "Phil Foden — last season's PFA Player of the Year — has been used in "
            "CM, LW, and the false 9 this season. 8 goals and 4 assists in 22 appearances. "
            "Expected goals contribution (xGC) of 0.62 per 90. His ability to drift "
            "into half-spaces and finish is exceptional: 2.3 shots per 90 with 48% "
            "on target. Dribble success rate: 61%. A generational talent."
        ),
    },
    {
        "player_id": "mci_cm5",
        "name": "Matheus Nunes",
        "position": "CM",
        "number": 27,
        "age": 26,
        "nationality": "Portugal",
        "height_cm": 183,
        "weight_kg": 78,
        "preferred_foot": "Right",
        "skills": [
            "ball carrying", "pressing", "dribbling",
            "physicality", "progressive passing", "tackling",
        ],
        "recent_performance_summary": (
            "Matheus Nunes has been a useful rotational option, providing energy and "
            "ball-carrying from midfield. 3.2 progressive carries per 90, and his "
            "pressing stats (18.2 pressures per 90) are among the squad's highest. "
            "1 goal and 3 assists in 19 appearances. Needs to improve final-third "
            "decision-making and passing accuracy (87%) to become a regular starter."
        ),
    },

    # ======================== FORWARDS / WINGERS ========================
    {
        "player_id": "mci_rw1",
        "name": "Jeremy Doku",
        "position": "RW",
        "number": 11,
        "age": 22,
        "nationality": "Belgium",
        "height_cm": 173,
        "weight_kg": 72,
        "preferred_foot": "Both",
        "skills": [
            "pace", "dribbling", "1v1", "crossing",
            "directness", "ball carrying", "explosive acceleration",
        ],
        "recent_performance_summary": (
            "Jeremy Doku is the most prolific dribbler in the Premier League — "
            "averaging 6.8 successful dribbles per 90 (highest in Europe's top 5 leagues). "
            "Top speed clocked at 35.3 km/h. Creates 2.4 chances per 90 but needs to "
            "improve end-product: only 3 goals and 5 assists in 24 appearances. "
            "Draws an average of 3.5 fouls per 90, disrupting opposition defences."
        ),
    },
    {
        "player_id": "mci_lw1",
        "name": "Jack Grealish",
        "position": "LW",
        "number": 10,
        "age": 29,
        "nationality": "England",
        "height_cm": 180,
        "weight_kg": 76,
        "preferred_foot": "Right",
        "skills": [
            "dribbling", "creativity", "pressing", "link-up play",
            "drawing fouls", "ball retention", "final-third passing",
        ],
        "recent_performance_summary": (
            "Jack Grealish has had an inconsistent season. 2 goals and 3 assists in "
            "18 appearances. His dribbling remains effective (3.1 per 90, 64% success) "
            "and he draws 2.8 fouls per game. However, his attacking output has declined "
            "from previous seasons. Pressing contribution is strong (19.1 pressures per 90). "
            "Works best in tight spaces on the left channel, combining with the LB."
        ),
    },
    {
        "player_id": "mci_lw2",
        "name": "Savinho",
        "position": "LW",
        "number": 26,
        "age": 20,
        "nationality": "Brazil",
        "height_cm": 176,
        "weight_kg": 66,
        "preferred_foot": "Right",
        "skills": [
            "pace", "dribbling", "shooting", "directness",
            "1v1", "crossing", "pressing",
        ],
        "recent_performance_summary": (
            "Savinho has been a revelation since arriving — adapting quickly to the "
            "Premier League. 4 goals and 6 assists in 22 appearances, mostly from "
            "the right wing despite being naturally left-footed. His willingness to "
            "cut inside and shoot (2.8 shots per 90) adds goal threat. Dribble success "
            "rate: 55% (4.2 per 90). Top speed: 34.8 km/h. A future star."
        ),
    },

    # ======================== STRIKERS ========================
    {
        "player_id": "mci_st1",
        "name": "Erling Haaland",
        "position": "ST",
        "number": 9,
        "age": 24,
        "nationality": "Norway",
        "height_cm": 194,
        "weight_kg": 88,
        "preferred_foot": "Left",
        "skills": [
            "finishing", "heading", "pace", "strength",
            "penalty area movement", "hold-up play",
            "aerial duels", "counter-attacking runs",
        ],
        "recent_performance_summary": (
            "Erling Haaland continues to be the most lethal striker in world football. "
            "27 goals in 26 league appearances, 0.82 xG per 90 (actual: 1.04 goals per 90). "
            "6 headed goals this season — the most in the Premier League. "
            "His penalty-area touches (7.2 per 90) and shot conversion rate (28%) are elite. "
            "Link-up play has improved: 1.2 key passes per 90 and 78% pass accuracy. "
            "Top speed: 34.8 km/h. Physical stats: 88 kg of raw power. "
            "45 goals across all competitions — on track for the Golden Boot."
        ),
    },
    {
        "player_id": "mci_st2",
        "name": "Oscar Bobb",
        "position": "RW",
        "number": 52,
        "age": 21,
        "nationality": "Norway",
        "height_cm": 177,
        "weight_kg": 70,
        "preferred_foot": "Right",
        "skills": [
            "dribbling", "pace", "finishing", "creativity",
            "pressing", "off-the-ball movement",
        ],
        "recent_performance_summary": (
            "Oscar Bobb has been promoted from the academy and shown flashes of "
            "brilliance. 2 goals and 2 assists in 12 appearances. His pace (34.1 km/h) "
            "and dribbling (3.2 per 90, 58% success) add dynamism off the bench. "
            "Needs more minutes to develop consistency. Strong work ethic in pressing."
        ),
    },
    {
        "player_id": "mci_sq1",
        "name": "James McAtee",
        "position": "CM",
        "number": 87,
        "age": 22,
        "nationality": "England",
        "height_cm": 174,
        "weight_kg": 68,
        "preferred_foot": "Right",
        "skills": [
            "passing", "positioning", "technical skill",
            "work rate", "creativity", "shooting",
        ],
        "recent_performance_summary": (
            "James McAtee returned from loan and has made 8 appearances, mostly as "
            "a substitute. Shows excellent technical quality and creativity (1.8 key "
            "passes per 90 in limited minutes). 1 goal. Needs consistent game time "
            "to prove he belongs at City's level."
        ),
    },
]

TEAM_NAME = "Manchester City"

# ---------------------------------------------------------------------------
# Position-specific performance summaries used as fallback
# ---------------------------------------------------------------------------
PERFORMANCE_SUMMARIES = {
    "GK":  "Consistently solid with world-class distribution. Commanding in 1v1 situations.",
    "RB":  "Explosive attacking fullback. Recorded 3 assists and 2 goals in last 10 matches.",
    "LB":  "Dynamic overlapping runs and improved defensive positioning. Average 4.2 crosses/game.",
    "CB":  "Aerially dominant. 92% tackle success rate over last month. Leader on set pieces.",
    "CDM": "Highest pass accuracy in squad (94%). Breaks up play and recycles possession efficiently.",
    "CM":  "Creative output rated 8/10 over last 5 matches. Key pass rate: 3.2/game.",
    "RW":  "Averaging 5.4 successful dribbles per 90min. Pace measured at 34.2 km/h top speed.",
    "LW":  "High press contribution (8.5 pressures/90). Occasional decision-making in final third.",
    "ST":  "Elite movement, clinical finishing. 0.78 xG per 90min. 23 goals in 27 appearances.",
}


def make_telemetry(player: dict, minute: int) -> dict:
    """Generate realistic telemetry for a player at a given match minute."""
    pos = player["position"]

    # Position-specific base stats
    base = {
        "GK":  dict(speed=5,  distance=0.4, hr=120, fatigue=0.05),
        "RB":  dict(speed=24, distance=1.1, hr=158, fatigue=0.30),
        "LB":  dict(speed=23, distance=1.1, hr=155, fatigue=0.28),
        "CB":  dict(speed=14, distance=0.8, hr=145, fatigue=0.20),
        "CDM": dict(speed=15, distance=0.9, hr=148, fatigue=0.22),
        "CM":  dict(speed=18, distance=1.0, hr=152, fatigue=0.25),
        "RW":  dict(speed=30, distance=1.2, hr=165, fatigue=0.35),
        "LW":  dict(speed=28, distance=1.1, hr=162, fatigue=0.33),
        "ST":  dict(speed=26, distance=1.0, hr=160, fatigue=0.30),
    }.get(pos, dict(speed=18, distance=1.0, hr=150, fatigue=0.25))

    # Fatigue builds with match minutes
    fatigue_factor = minute / 90
    fatigue = min(1.0, base["fatigue"] + fatigue_factor * 0.4 + random.uniform(-0.05, 0.05))

    # Position-based x/y coordinates on pitch (0-105 x, 0-68 y)
    pitch_zones = {
        "GK":  (5,  34), "CB": (25, 34), "RB": (28, 60),
        "LB":  (28, 10), "CDM": (45, 34), "CM": (55, 34),
        "RW":  (70, 58), "LW": (70, 10), "ST": (85, 34),
    }
    base_x, base_y = pitch_zones.get(pos, (52, 34))

    return {
        "player_id":        player["player_id"],
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "speed":            round(base["speed"] + random.uniform(-4, 4), 1),
        "distance_covered": round(base["distance"] + random.uniform(-0.1, 0.1), 2),
        "heart_rate":       base["hr"] + random.randint(-8, 8),
        "position_x":       round(base_x + random.uniform(-8, 8), 1),
        "position_y":       round(base_y + random.uniform(-6, 6), 1),
        "fatigue_index":    round(fatigue, 2),
    }


async def seed_profiles(client: httpx.AsyncClient, team: str) -> bool:
    print(f"\n{'='*60}")
    print(f"  📋  Ingesting {len(SQUAD)} player profiles for {team}...")
    print(f"{'='*60}")

    profiles = []
    for p in SQUAD:
        pos = p["position"]
        profiles.append({
            "player_id": p["player_id"],
            "name": p["name"],
            "position": pos,
            "team": team,
            "skills": p["skills"],
            "recent_performance_summary": p.get(
                "recent_performance_summary",
                PERFORMANCE_SUMMARIES.get(pos, "Solid contributor to the squad."),
            ),
        })
        # Print each player being ingested
        print(f"    ▸ {p['name']:<22} ({pos})  — {len(p['skills'])} skills")

    print()

    try:
        r = await client.post("/api/v1/ingest/profiles", json=profiles, timeout=120)
        if r.status_code == 201:
            data = r.json()
            print(f"  ✅  Ingested {data['count']} profiles → Pinecone vector store")
            return True
        elif r.status_code == 503:
            print(f"  ⚠️   Pinecone/Groq keys not configured — skipping profile ingestion")
            print(f"       Set GROQ_API_KEY + PINECONE_API_KEY in .env to enable RAG.")
            return False
        else:
            print(f"  ❌  Failed [{r.status_code}]: {r.text[:300]}")
            return False
    except Exception as e:
        print(f"  ❌  Error: {e}")
        return False


async def seed_telemetry(client: httpx.AsyncClient, minute: int = 65) -> None:
    print(f"\n{'='*60}")
    print(f"  📡  Pushing live telemetry (match minute {minute}) for all {len(SQUAD)} players...")
    print(f"{'='*60}")

    telemetry = [make_telemetry(p, minute) for p in SQUAD]

    # Show a preview
    high_fatigue = [t for t in telemetry if t["fatigue_index"] > 0.6]
    print(f"\n  Live snapshot at minute {minute}:")
    for t in telemetry[:8]:
        player_name = next(p["name"] for p in SQUAD if p["player_id"] == t["player_id"])
        print(f"    {player_name:<22} speed={t['speed']:5.1f} km/h  HR={t['heart_rate']}bpm  fatigue={t['fatigue_index']:.2f}")
    print(f"    ... ({len(SQUAD)-8} more players)")

    if high_fatigue:
        print(f"\n  ⚠️   Players with high fatigue (>0.6):")
        for t in high_fatigue:
            player_name = next(p["name"] for p in SQUAD if p["player_id"] == t["player_id"])
            print(f"    🔴 {player_name:<22} fatigue={t['fatigue_index']:.2f}")

    try:
        r = await client.post(
            "/api/v1/ingest/telemetry",
            json={"data": telemetry},
            timeout=10,
        )
        if r.status_code == 202:
            print(f"\n  ✅  Queued {r.json()['count']} telemetry records → Redis worker")
        else:
            print(f"  ❌  [{r.status_code}]: {r.text[:200]}")
    except Exception as e:
        print(f"  ❌  Error: {e}")


async def run_analysis(client: httpx.AsyncClient, query: str, minute: int = 65) -> None:
    print(f"\n  🤖  Query: \"{query}\"")
    telemetry = [make_telemetry(p, minute) for p in SQUAD]
    payload = {
        "query": query,
        "telemetry_context": telemetry,
    }
    try:
        r = await client.post("/api/v1/analyze", json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            print(f"  📊  Route: [{data.get('route_used', 'N/A')}]")
            print(f"\n{'-'*60}")
            print(f"  {data['insight']}")
            print(f"{'-'*60}")
        elif r.status_code == 503:
            print(f"  ⚠️   Groq API key not configured — set GROQ_API_KEY in .env")
        else:
            print(f"  ❌  [{r.status_code}]: {r.text[:300]}")
    except Exception as e:
        print(f"  ❌  Error: {e}")


async def demo_analysis(client: httpx.AsyncClient) -> None:
    print(f"\n{'='*60}")
    print(f"  🧠  Running Demo Analysis Queries...")
    print(f"{'='*60}")

    queries = [
        ("What formation should we use at minute 65 given current fatigue levels?", 65),
        ("Which players are showing tactical issues — spacing or positioning problems?",  65),
        ("Who should we substitute right now and why?",                                  65),
        ("Tell me about Erling Haaland's recent performance and goal-scoring record.",   65),
        ("Compare Kevin De Bruyne and Bernardo Silva's creative output this season.",    65),
    ]

    for query, minute in queries:
        await run_analysis(client, query, minute)
        print()


async def check_queue(client: httpx.AsyncClient) -> None:
    r = await client.get("/api/v1/queue/status", timeout=5)
    if r.status_code == 200:
        d = r.json()
        print(f"  📬  Queue status — depth: {d['depth']}  |  processed total: {d['processed_total']}")


async def main(no_ai: bool, team: str) -> None:
    print(f"\n{'='*60}")
    print(f"  ⚽  PitchMaster AI — Manchester City Team Seeder")
    print(f"  Team: {team}")
    print(f"  Players: {len(SQUAD)}")
    print(f"{'='*60}")

    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        # 1. Health check
        try:
            r = await client.get("/health", timeout=5)
            print(f"\n  Server: {r.json()['status'].upper()} ({r.json()['version']})")
        except Exception:
            print("\n  ❌  Cannot reach server at http://localhost:8000")
            print("      Run: python -m uvicorn app.main:app --reload")
            sys.exit(1)

        # 2. Ingest profiles (needs API keys)
        if not no_ai:
            await seed_profiles(client, team)

        # 3. Push live telemetry (no API keys needed)
        await seed_telemetry(client, minute=65)

        # Small wait for worker to process
        await asyncio.sleep(1.5)
        await check_queue(client)

        # 4. Demo analyses (needs OpenAI key)
        if not no_ai:
            await demo_analysis(client)

    print(f"\n{'='*60}")
    print("  ✅  Done! Open http://localhost:8000/docs to explore the API.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PitchMaster AI Team Seeder")
    parser.add_argument("--no-ai",  action="store_true",  help="Skip AI endpoints (telemetry only)")
    parser.add_argument("--minute", type=int, default=65, help="Match minute to simulate (default: 65)")
    parser.add_argument("--team",   type=str, default=TEAM_NAME, help="Team name (default: Manchester City)")
    args = parser.parse_args()

    asyncio.run(main(no_ai=args.no_ai, team=args.team))
