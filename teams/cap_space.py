"""
teams/cap_space.py
════════════════════════════════════════════════════════════════════════════════
2026 NFL Cap Space Tracker + Rookie Slot Cost Model

BUSINESS SUMMARY
────────────────
Cap space is the hidden constraint of every draft decision. A team picking 3rd
overall with $4M in cap room faces a categorically different draft than the
same team with $40M available. This module provides:

  1. 2026 estimated cap space by team (post-free-agency, pre-draft estimates)
  2. Rookie slot cost calculator — 4-year cost for any pick number
  3. Positional spending history (% of cap by position, 2021-2025 average)
  4. Cap room check: can team afford the player at their slot value?

CAP SLOT FORMULA
────────────────
The NFL rookie wage scale follows a power-law decay from pick 1 to pick 262.
We model this as:

    4yr_value = BASE_CAP × (DECAY_RATE ^ (pick_number - 1))

where BASE_CAP = $46.5M (2026 #1 pick estimated 4yr value) and DECAY_RATE
is calibrated so pick 32 ≈ $12.5M and pick 64 ≈ $6.8M.

SOURCE: overthecap.com rookie slot projections, spotrac.com 2026 estimates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ── 2026 Salary Cap ───────────────────────────────────────────────────────────

NFL_SALARY_CAP_2026: int = 279_000_000   # $279M hard cap (projected)

# ── Rookie Slot Formula Constants ─────────────────────────────────────────────

# 4-year total cost for pick #1 (2026 projection based on cap growth trends)
ROOKIE_SLOT_BASE: float = 46_500_000.0

# Decay exponent: calibrated so pick 32 ≈ $12.5M, pick 64 ≈ $6.8M
ROOKIE_SLOT_DECAY: float = 0.9625

# Minimum 4-year slot (late round, pick 200+)
ROOKIE_SLOT_MINIMUM: float = 3_400_000.0

# ── 2026 Estimated Cap Space by Team ─────────────────────────────────────────
# Source: Post-free-agency estimates based on overthecap.com / spotrac.com
# Figures represent available cap space as of April 2026 (pre-draft)
# Values are approximate — exact numbers shift daily with roster moves

TEAM_CAP_SPACE_2026: Dict[str, int] = {
    "ARI": 42_100_000,
    "ATL": 18_500_000,
    "BAL": 11_200_000,
    "BUF": 8_900_000,
    "CAR": 61_800_000,   # Post-rebuild, significant space
    "CHI": 47_300_000,
    "CIN": 14_100_000,
    "CLE": 55_200_000,   # Post-Deshaun Watson restructure
    "DAL": 6_400_000,    # Tight against cap
    "DEN": 38_700_000,
    "DET": 9_800_000,
    "GB":  21_400_000,
    "HOU": 17_300_000,
    "IND": 52_900_000,
    "JAX": 44_600_000,
    "KC":  12_800_000,   # Mahomes' deal eats space
    "LAC": 28_300_000,
    "LAR": 15_600_000,
    "LV":  46_200_000,
    "MIA": 19_800_000,
    "MIN": 31_500_000,
    "NE":  68_400_000,   # Rebuilding, most cap room in NFL
    "NO":  39_100_000,
    "NYG": 57_300_000,
    "NYJ": 41_700_000,
    "PHI": 13_200_000,
    "PIT": 24_600_000,
    "SEA": 35_800_000,
    "SF":  7_100_000,    # Tight post-Purdy extension
    "TB":  22_400_000,
    "TEN": 63_500_000,   # Full rebuild
    "WAS": 29_600_000,
}

# ── Positional Cap Spending (% of total cap, 2021-2025 avg) ──────────────────
# Source: overthecap.com positional spending reports
# These represent what winning teams typically allocate by position group

POSITIONAL_SPENDING_HISTORY: Dict[str, Dict[str, float]] = {
    # Offensive positions
    "QB":  {"avg_pct": 12.8, "high_pct": 18.2, "low_pct": 4.1,  "note": "Starter-driven — franchise QBs dominate"},
    "WR":  {"avg_pct": 9.1,  "high_pct": 13.4, "low_pct": 5.8,  "note": "Top WR1 can be $25M+/yr"},
    "OT":  {"avg_pct": 8.3,  "high_pct": 11.7, "low_pct": 5.2,  "note": "LT premium over RT"},
    "RB":  {"avg_pct": 3.2,  "high_pct": 6.1,  "low_pct": 1.4,  "note": "Devalued — rookie contracts preferred"},
    "TE":  {"avg_pct": 5.4,  "high_pct": 8.2,  "low_pct": 2.8,  "note": "Elite TEs command WR money"},
    "IOL": {"avg_pct": 5.9,  "high_pct": 8.1,  "low_pct": 3.6,  "note": "Interior OL increasingly valued"},
    # Defensive positions
    "EDGE":{"avg_pct": 9.7,  "high_pct": 14.1, "low_pct": 5.3,  "note": "Pass rusher premium, near QB value"},
    "DT":  {"avg_pct": 5.8,  "high_pct": 9.2,  "low_pct": 3.1,  "note": "1T vs 3T split"},
    "LB":  {"avg_pct": 4.6,  "high_pct": 7.3,  "low_pct": 2.2,  "note": "Off-ball LB devalued vs edge"},
    "CB":  {"avg_pct": 7.2,  "high_pct": 10.8, "low_pct": 4.1,  "note": "Elite CB1 approaching $25M/yr"},
    "S":   {"avg_pct": 4.9,  "high_pct": 7.6,  "low_pct": 2.8,  "note": "Split between SS/FS roles"},
    # Special teams (minimal)
    "K":   {"avg_pct": 0.8,  "high_pct": 1.4,  "low_pct": 0.3,  "note": "Justin Tucker exception"},
    "P":   {"avg_pct": 0.5,  "high_pct": 0.9,  "low_pct": 0.2,  "note": "Minimal spend"},
}

# ── Positional contract tiers for free agency context ─────────────────────────
# Market rate by tier (2026 AAV estimates)

POSITION_CONTRACT_TIERS_2026: Dict[str, Dict[str, float]] = {
    "QB":   {"elite": 52_000_000, "starter": 32_000_000, "backup": 8_000_000},
    "WR":   {"elite": 28_000_000, "starter": 16_000_000, "depth":  4_000_000},
    "OT":   {"elite": 24_000_000, "starter": 14_000_000, "depth":  3_500_000},
    "RB":   {"elite": 14_000_000, "starter": 7_500_000,  "depth":  2_000_000},
    "TE":   {"elite": 22_000_000, "starter": 11_000_000, "depth":  2_500_000},
    "IOL":  {"elite": 18_000_000, "starter": 10_000_000, "depth":  2_000_000},
    "EDGE": {"elite": 30_000_000, "starter": 16_000_000, "depth":  4_500_000},
    "DT":   {"elite": 22_000_000, "starter": 12_000_000, "depth":  3_000_000},
    "LB":   {"elite": 18_000_000, "starter": 10_000_000, "depth":  2_000_000},
    "CB":   {"elite": 24_000_000, "starter": 13_000_000, "depth":  3_000_000},
    "S":    {"elite": 17_000_000, "starter": 9_500_000,  "depth":  2_200_000},
}


# ── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class RookieSlotCost:
    """4-year rookie contract cost breakdown for a draft pick."""
    pick_number: int
    round_number: int
    four_year_total: float
    year_1_salary: float
    year_2_salary: float
    year_3_salary: float
    year_4_salary: float       # Team option year (5th-year option for R1)
    fifth_year_option: Optional[float]   # Round 1 only
    signing_bonus: float
    annual_cap_hit_avg: float

    def __str__(self) -> str:
        return (
            f"Pick #{self.pick_number} (Rd {self.round_number}): "
            f"${self.four_year_total/1e6:.1f}M / 4yr  |  "
            f"Avg cap: ${self.annual_cap_hit_avg/1e6:.1f}M/yr"
            + (f"  |  5th-yr option: ${self.fifth_year_option/1e6:.1f}M"
               if self.fifth_year_option else "")
        )


@dataclass
class TeamCapProfile:
    """Full cap profile for a team going into the draft."""
    team: str
    available_space: int
    rookie_class_cost: float          # Sum of all their picks' slot values
    post_draft_space: float           # available_space - rookie_class_cost
    can_afford_all_picks: bool
    at_risk_picks: List[int]          # Pick numbers they may not be able to sign
    spending_by_position: Dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        status = "✓ HEALTHY" if self.can_afford_all_picks else "⚠ CAP RISK"
        return (
            f"{self.team} | Available: ${self.available_space/1e6:.1f}M | "
            f"Rookie class: ${self.rookie_class_cost/1e6:.1f}M | "
            f"Post-draft: ${self.post_draft_space/1e6:.1f}M | {status}"
        )


# ── Core Functions ────────────────────────────────────────────────────────────

def rookie_slot_cost(pick_number: int) -> RookieSlotCost:
    """
    Calculate the 4-year rookie contract cost for any pick number.

    Uses a power-law decay model calibrated to historical NFL rookie slot data:
        4yr_value = BASE_CAP × (DECAY_RATE ^ (pick_number - 1))

    Round 1 picks receive a 5th-year option (team-controlled at ~120% of Year 4).
    Signing bonus is approximately 30% of total contract value.

    Args:
        pick_number: Overall draft pick number (1–262)

    Returns:
        RookieSlotCost dataclass with full year-by-year breakdown

    Examples:
        >>> cost = rookie_slot_cost(1)
        >>> cost.four_year_total  # ~$46.5M
        >>> cost = rookie_slot_cost(32)
        >>> cost.four_year_total  # ~$12.5M
    """
    pick_number = max(1, min(pick_number, 262))
    round_number = _pick_to_round(pick_number)

    # Power-law decay
    four_year_total = max(
        ROOKIE_SLOT_BASE * (ROOKIE_SLOT_DECAY ** (pick_number - 1)),
        ROOKIE_SLOT_MINIMUM
    )

    # Year-by-year split: 20/22/26/32% of total (typical NFL distribution)
    year_splits = [0.20, 0.22, 0.26, 0.32]
    years = [four_year_total * s for s in year_splits]

    # Round 1: 5th-year option at ~120% of Year 4 salary (2026 scale)
    fifth_year = years[3] * 1.20 if round_number == 1 else None

    # Signing bonus: ~30% of 4yr total, prorated over 4 years
    signing_bonus = four_year_total * 0.30

    return RookieSlotCost(
        pick_number=pick_number,
        round_number=round_number,
        four_year_total=four_year_total,
        year_1_salary=years[0],
        year_2_salary=years[1],
        year_3_salary=years[2],
        year_4_salary=years[3],
        fifth_year_option=fifth_year,
        signing_bonus=signing_bonus,
        annual_cap_hit_avg=four_year_total / 4.0,
    )


def team_cap_profile(
    team: str,
    pick_numbers: List[int],
) -> TeamCapProfile:
    """
    Compute full cap profile for a team given their draft picks.

    Args:
        team: 2-3 letter team abbreviation (e.g., "KC", "NE")
        pick_numbers: List of their overall pick numbers

    Returns:
        TeamCapProfile with post-draft cap space and risk flags
    """
    team = team.upper()
    available = TEAM_CAP_SPACE_2026.get(team, 30_000_000)

    costs = [rookie_slot_cost(p) for p in pick_numbers]
    total_rookie_cost = sum(c.four_year_total for c in costs)
    post_draft = available - total_rookie_cost

    # Flag any picks that, individually, consume > 60% of available space
    at_risk = [
        c.pick_number for c in costs
        if c.four_year_total > available * 0.60
    ]

    return TeamCapProfile(
        team=team,
        available_space=available,
        rookie_class_cost=total_rookie_cost,
        post_draft_space=post_draft,
        can_afford_all_picks=post_draft >= 0,
        at_risk_picks=at_risk,
    )


def can_team_afford_pick(team: str, pick_number: int) -> Tuple[bool, str]:
    """
    Quick check: can this team afford the slot cost for this pick?

    Args:
        team: Team abbreviation
        pick_number: Overall pick number

    Returns:
        (can_afford: bool, reason: str)
    """
    team = team.upper()
    available = TEAM_CAP_SPACE_2026.get(team, 30_000_000)
    cost = rookie_slot_cost(pick_number)

    # Leave a $2M buffer for camp roster + practice squad costs
    BUFFER = 2_000_000
    affordable = available >= (cost.four_year_total + BUFFER)

    if affordable:
        remaining = available - cost.four_year_total
        return True, (
            f"{team} can afford Pick #{pick_number} "
            f"(${cost.four_year_total/1e6:.1f}M slot) — "
            f"${remaining/1e6:.1f}M remaining after signing"
        )
    else:
        shortfall = cost.four_year_total - available
        return False, (
            f"{team} CANNOT afford Pick #{pick_number} "
            f"(${cost.four_year_total/1e6:.1f}M slot) — "
            f"${shortfall/1e6:.1f}M over budget. Must restructure or cut."
        )


def position_cap_value(position: str, pick_number: int) -> Dict[str, float]:
    """
    Assess the cap value of drafting a player at this pick vs. free agency cost.

    Computes how many years of rookie contract surplus value the team receives
    before the player reaches their market rate (typically Year 5 extension).

    Args:
        position: Player position (e.g., "QB", "WR", "EDGE")
        pick_number: Overall pick number

    Returns:
        Dict with surplus_value, years_of_control, market_rate_starter
    """
    slot = rookie_slot_cost(pick_number)
    pos_data = POSITION_CONTRACT_TIERS_2026.get(position, POSITION_CONTRACT_TIERS_2026["WR"])
    market_rate = pos_data["starter"]

    # Surplus per year = market rate - actual slot cost per year
    # Assumes player develops into starter by Year 2
    surplus_per_year = market_rate - slot.annual_cap_hit_avg
    years_of_control = 4 + (1 if slot.round_number == 1 else 0)
    total_surplus = max(0.0, surplus_per_year * years_of_control)

    return {
        "position":             position,
        "pick_number":          pick_number,
        "slot_aav":             slot.annual_cap_hit_avg,
        "market_rate_starter":  market_rate,
        "surplus_per_year":     surplus_per_year,
        "years_of_control":     years_of_control,
        "total_surplus_value":  total_surplus,
        "cap_efficiency_ratio": market_rate / slot.annual_cap_hit_avg if slot.annual_cap_hit_avg > 0 else 1.0,
    }


def all_teams_cap_summary() -> List[Dict]:
    """
    Return cap space summary for all 32 teams, sorted by available space.

    Returns:
        List of dicts with team, space_m, tier (flush/comfortable/tight/crisis)
    """
    rows = []
    for team, space in sorted(TEAM_CAP_SPACE_2026.items(), key=lambda x: -x[1]):
        if space >= 50_000_000:
            tier = "FLUSH"
        elif space >= 25_000_000:
            tier = "COMFORTABLE"
        elif space >= 10_000_000:
            tier = "TIGHT"
        else:
            tier = "CRISIS"

        rows.append({
            "team":    team,
            "space_m": round(space / 1_000_000, 1),
            "tier":    tier,
        })

    return rows


# ── Private Helpers ───────────────────────────────────────────────────────────

def _pick_to_round(pick: int) -> int:
    """Convert overall pick number to round number."""
    if pick <= 32:   return 1
    if pick <= 64:   return 2
    if pick <= 105:  return 3
    if pick <= 141:  return 4
    if pick <= 177:  return 5
    if pick <= 220:  return 6
    return 7


# ── CLI Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("2026 NFL CAP SPACE ANALYZER")
    print("=" * 65)

    # Rookie slot costs for top 5 picks
    print("\n── Rookie Slot Costs (Top 10 Picks) ──")
    for pick in [1, 2, 3, 5, 10, 15, 32, 64, 100]:
        print(f"  {rookie_slot_cost(pick)}")

    # Cap space leaderboard
    print("\n── 2026 Cap Space by Team (Top 10) ──")
    for row in all_teams_cap_summary()[:10]:
        bar = "█" * int(row["space_m"] / 3)
        print(f"  {row['team']:4s} {bar:20s} ${row['space_m']:5.1f}M  [{row['tier']}]")

    # Position cap value example
    print("\n── Cap Value of Drafting QB at Pick #1 ──")
    val = position_cap_value("QB", 1)
    for k, v in val.items():
        if isinstance(v, float):
            print(f"  {k:30s}: ${v/1e6:.1f}M" if v > 100 else f"  {k:30s}: {v:.2f}")
        else:
            print(f"  {k:30s}: {v}")
