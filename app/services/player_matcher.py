"""
Player ID Reconciliation
========================
API-Football and the Transfermarkt/Kaggle dataset PitchMaster is built on use
different player ids, and API-Football's injury feed usually abbreviates first
names ("E. Haaland"). This module maps an API-Football (name, team) pair onto
a Transfermarkt `player_id` using only the names:

1. Resolve the club: exact match after normalisation, then substring
   containment, then difflib fuzzy match.
2. Within that club: exact full-name match, else surname match (plus first
   initial when one is given). Only a *unique* candidate counts as a match —
   ambiguity is reported as unmatched rather than guessed.

Pure functions only (no I/O) so they are cheap to unit test. Confirmed
matches are persisted by the sync script so this only runs for new players.
"""
import difflib
import re
import unicodedata
from dataclasses import dataclass

# Tokens that differ between data sources without changing which club is meant
# ("Manchester City" vs "Manchester City Football Club").
_CLUB_NOISE = {
    "fc", "cf", "afc", "sc", "ac", "as", "ssc", "club", "football", "futbol", "calcio",
    "de", "the", "sad", "spa", "s.p.a.", "e.v.", "ev", "1.",
}

# Short names API-Football uses that share no tokens with the dataset's legal
# club names. Keys and values are both in normalize_club() form.
_CLUB_ALIASES = {
    "wolves": "wolverhampton wanderers",
    "rb leipzig": "rasenballsport leipzig",
    "athletic": "athletic bilbao",
    "psg": "paris saint germain",
    "vfb stuttgart": "stuttgart",
    "rennes": "stade rennais",
}


@dataclass(frozen=True)
class Candidate:
    player_id: str
    name: str
    first_name: str
    last_name: str
    club: str


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"[^a-z0-9. ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def normalize_club(name: str) -> str:
    tokens = [t for t in normalize(name).replace(".", " ").split() if t not in _CLUB_NOISE]
    return " ".join(tokens)


def resolve_club(team: str, clubs: list[str]) -> str | None:
    """Return the dataset club name that best matches an API-Football team name."""
    target = normalize_club(team)
    target = _CLUB_ALIASES.get(target, target)
    if not target:
        return None
    by_norm: dict[str, str] = {}
    for club in clubs:
        by_norm.setdefault(normalize_club(club), club)

    if target in by_norm:
        return by_norm[target]

    containing = [orig for norm, orig in by_norm.items() if norm and (target in norm or norm in target)]
    if len(containing) == 1:
        return containing[0]

    # Every word of the short name appears in the long one ("bayer leverkusen" ⊂ "bayer 04 leverkusen ...")
    target_tokens = set(target.split())
    subset = [orig for norm, orig in by_norm.items() if target_tokens <= set(norm.split())]
    if len(subset) == 1:
        return subset[0]

    close = difflib.get_close_matches(target, list(by_norm), n=2, cutoff=0.75)
    # Reject near-ties ("manchester city" vs "manchester united") rather than guess.
    if len(close) == 1 or (
        len(close) == 2
        and difflib.SequenceMatcher(None, target, close[0]).ratio()
        - difflib.SequenceMatcher(None, target, close[1]).ratio() > 0.1
    ):
        return by_norm[close[0]]
    return None


def _split_name(name: str) -> tuple[str | None, str]:
    """'E. Haaland' -> ('e', 'haaland'); 'Kevin De Bruyne' -> (None, 'kevin de bruyne')."""
    norm = normalize(name)
    parts = norm.split(" ")
    if len(parts) > 1 and re.fullmatch(r"[a-z]{1,2}\.", parts[0]):
        return parts[0][0], " ".join(parts[1:]).replace(".", "")
    return None, norm.replace(".", "")


def match_player(name: str, candidates: list[Candidate]) -> Candidate | None:
    """Pick the single candidate that `name` refers to, or None if absent/ambiguous."""
    initial, rest = _split_name(name)

    if initial is None:
        exact = [c for c in candidates if normalize(c.name).replace(".", "") == rest]
        if len(exact) == 1:
            return exact[0]
        # Treat the full name's last token as a surname fallback ("Rodrigo Hernandez" -> "hernandez")
        surname = rest.split(" ")[-1]
    else:
        surname = rest

    def surname_matches(c: Candidate) -> bool:
        full = normalize(c.name).replace(".", "")
        last = normalize(c.last_name).replace(".", "")
        return last == surname or full == surname or full.endswith(" " + surname)

    hits = [c for c in candidates if surname_matches(c)]
    if initial is not None:
        hits = [c for c in hits if not c.first_name or normalize(c.first_name).startswith(initial)]
    return hits[0] if len(hits) == 1 else None


def reconcile(name: str, team: str, candidates_by_club: dict[str, list[Candidate]]) -> Candidate | None:
    """Match an API-Football (name, team) pair against the dataset, club first."""
    club = resolve_club(team, list(candidates_by_club))
    if club is None:
        return None
    return match_player(name, candidates_by_club[club])
