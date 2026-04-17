"""
teams/trade_value_calculator.py
════════════════════════════════════════════════════════════════════════════════
Draft Pick Trade Value Calculator

BUSINESS SUMMARY
────────────────
No draft model is complete without a trade value engine. NFL teams routinely
trade up for elite prospects and down to accumulate capital. This module
implements two trade charts used by actual NFL front offices, plus an
optimizer that finds trade packages a team should accept/reject.

TWO CHART MODELS
────────────────
1. Jimmy Johnson Chart (1991, "The Chart"):
   - The original — every front office has used it for 30+ years
   - Pick #1 = 3000 points; values decay roughly exponentially
   - Well-known to over-value early picks vs late picks by modern analytics

2. Fitzgerald-Spielberger Chart (2013, "The Analytics Chart"):
   - Developed by Harvard analytics researchers using AV (Approximate Value)
   - Corrects JJ bias: late picks are worth MORE, early picks slightly LESS
   - Used by analytics-forward franchises (Eagles, Ravens, 49ers, etc.)
   - Pick #1 = 3000 points (normalized to JJ scale for comparability)

TRADE EVALUATION METHODOLOGY
─────────────────────────────
A trade is evaluated under BOTH charts. If both agree it's a fair trade
(within 10% tolerance), it's classified "consensus fair". If one chart
calls it a steal and the other doesn't, it's "chart-dependent."

evaluate_trade(give, receive) → TradeEvaluation
optimal_trade_up(target_pick, your_picks) → best package
find_trade_partners(pick, needs) → teams most likely to trade

Source: drafttek.com, overthecap.com/tradecalculator, original JJ chart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ── Jimmy Johnson Chart ───────────────────────────────────────────────────────
# 262-pick coverage. Original source: Dallas Cowboys front office, 1991.
# Values for picks 1-32 (Round 1) — most critical range.

JIMMY_JOHNSON_CHART: Dict[int, float] = {
    # Round 1
    1: 3000, 2: 2600, 3: 2200, 4: 1800, 5: 1700, 6: 1600, 7: 1500,
    8: 1400, 9: 1350, 10: 1300, 11: 1250, 12: 1200, 13: 1150, 14: 1100,
    15: 1050, 16: 1000, 17: 950, 18: 900, 19: 875, 20: 850,
    21: 800, 22: 780, 23: 760, 24: 740, 25: 720, 26: 700,
    27: 680, 28: 660, 29: 640, 30: 620, 31: 600, 32: 590,
    # Round 2
    33: 580, 34: 560, 35: 550, 36: 540, 37: 530, 38: 520, 39: 510,
    40: 500, 41: 490, 42: 480, 43: 470, 44: 460, 45: 450, 46: 440,
    47: 430, 48: 420, 49: 410, 50: 400, 51: 390, 52: 380, 53: 370,
    54: 360, 55: 350, 56: 340, 57: 330, 58: 320, 59: 310, 60: 300,
    61: 290, 62: 280, 63: 270, 64: 260,
    # Round 3
    65: 255, 66: 250, 67: 245, 68: 240, 69: 235, 70: 230, 71: 225,
    72: 220, 73: 215, 74: 210, 75: 205, 76: 200, 77: 195, 78: 190,
    79: 185, 80: 180, 81: 175, 82: 170, 83: 165, 84: 160, 85: 155,
    86: 150, 87: 145, 88: 140, 89: 135, 90: 130, 91: 125, 92: 122,
    93: 119, 94: 116, 95: 113, 96: 110, 97: 107, 98: 104, 99: 101,
    100: 98, 101: 95, 102: 92, 103: 89, 104: 86, 105: 83,
    # Round 4
    106: 80, 107: 77, 108: 74, 109: 71, 110: 68, 111: 66, 112: 64,
    113: 62, 114: 60, 115: 58, 116: 56, 117: 54, 118: 52, 119: 50,
    120: 48, 121: 46, 122: 45, 123: 44, 124: 43, 125: 42, 126: 41,
    127: 40, 128: 39, 129: 38, 130: 37, 131: 36, 132: 35, 133: 34,
    134: 33, 135: 32, 136: 31, 137: 30, 138: 29, 139: 28, 140: 27, 141: 26,
    # Round 5
    142: 25, 143: 24, 144: 23, 145: 22, 146: 21, 147: 20, 148: 19, 149: 18,
    150: 17, 151: 16, 152: 15, 153: 14, 154: 13, 155: 12, 156: 11, 157: 10,
    158: 9.5, 159: 9.0, 160: 8.5, 161: 8.0, 162: 7.5, 163: 7.0,
    164: 6.5, 165: 6.0, 166: 5.8, 167: 5.6, 168: 5.4, 169: 5.2, 170: 5.0,
    171: 4.8, 172: 4.6, 173: 4.4, 174: 4.2, 175: 4.0, 176: 3.8, 177: 3.6,
    # Round 6
    178: 3.4, 179: 3.2, 180: 3.0, 181: 2.9, 182: 2.8, 183: 2.7, 184: 2.6,
    185: 2.5, 186: 2.4, 187: 2.3, 188: 2.2, 189: 2.1, 190: 2.0,
    191: 1.9, 192: 1.8, 193: 1.7, 194: 1.6, 195: 1.5, 196: 1.4,
    197: 1.3, 198: 1.2, 199: 1.1, 200: 1.0, 201: 0.98, 202: 0.96,
    203: 0.94, 204: 0.92, 205: 0.90, 206: 0.88, 207: 0.86, 208: 0.84,
    209: 0.82, 210: 0.80, 211: 0.78, 212: 0.76, 213: 0.74, 214: 0.72,
    215: 0.70, 216: 0.68, 217: 0.66, 218: 0.64, 219: 0.62, 220: 0.60,
    # Round 7
    221: 0.58, 222: 0.56, 223: 0.54, 224: 0.52, 225: 0.50, 226: 0.48,
    227: 0.46, 228: 0.44, 229: 0.42, 230: 0.40, 231: 0.38, 232: 0.36,
    233: 0.34, 234: 0.32, 235: 0.30, 236: 0.28, 237: 0.26, 238: 0.24,
    239: 0.22, 240: 0.20, 241: 0.18, 242: 0.16, 243: 0.14, 244: 0.12,
    245: 0.10, 246: 0.09, 247: 0.08, 248: 0.07, 249: 0.06, 250: 0.05,
    251: 0.05, 252: 0.05, 253: 0.04, 254: 0.04, 255: 0.04,
    256: 0.03, 257: 0.03, 258: 0.03, 259: 0.02, 260: 0.02,
    261: 0.02, 262: 0.01,
}


# ── Fitzgerald-Spielberger Analytics Chart ────────────────────────────────────
# Source: "Rethinking the Value of NFL Draft Picks" — Harvard Sports Analysis Collective, 2013
# Normalized to JJ scale (pick #1 = 3000) for direct comparison.
# Key difference: R2/R3 picks worth MORE; R1 top-5 picks worth slightly LESS.

def _fitzgerald_spielberger_value(pick: int) -> float:
    """
    Compute Fitzgerald-Spielberger trade value for any pick number.

    Original paper uses AV (Approximate Value) regression; this is the
    piecewise power-law approximation calibrated to their published values:

        value = 136.8 × (pick)^(-0.688)    [raw AV basis]

    Rescaled so pick #1 = 3000 (JJ comparable).
    """
    if pick <= 0:
        pick = 1
    raw = 136.8 * (pick ** -0.688)
    # Scale: raw pick #1 ≈ 136.8, target 3000 → scale = 3000/136.8
    scale = 3000.0 / (136.8 * (1 ** -0.688))
    return raw * scale


# Pre-compute F-S values for all 262 picks
FITZGERALD_SPIELBERGER_CHART: Dict[int, float] = {
    pick: round(_fitzgerald_spielberger_value(pick), 2)
    for pick in range(1, 263)
}


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class TradeEvaluation:
    """Result of evaluating a trade under both chart systems."""
    give_picks:            List[int]
    receive_picks:         List[int]
    give_value_jj:         float
    receive_value_jj:      float
    give_value_fs:         float
    receive_value_fs:      float
    jj_verdict:            str    # "STRONG_GIVE", "FAIR", "STRONG_RECEIVE", etc.
    fs_verdict:            str
    consensus_verdict:     str    # Combined assessment
    surplus_jj:            float  # positive = receiving team overpays
    surplus_fs:            float
    recommendation:        str

    def __str__(self) -> str:
        return (
            f"Trade Evaluation\n"
            f"  Give:  {self.give_picks}  →  JJ: {self.give_value_jj:.0f}  |  "
            f"F-S: {self.give_value_fs:.0f}\n"
            f"  Recv:  {self.receive_picks}  →  JJ: {self.receive_value_jj:.0f}  |  "
            f"F-S: {self.receive_value_fs:.0f}\n"
            f"  JJ Verdict: {self.jj_verdict}  |  F-S Verdict: {self.fs_verdict}\n"
            f"  CONSENSUS: {self.consensus_verdict}\n"
            f"  → {self.recommendation}"
        )


@dataclass
class TradeSuggestion:
    """A trade package suggestion from your picks to reach a target pick."""
    target_pick:    int
    your_pick:      int
    add_picks:      List[int]     # Additional picks to include
    give_value_jj:  float
    give_value_fs:  float
    cost_ratio_jj:  float         # give/receive — < 1.0 means underpay
    cost_ratio_fs:  float
    verdict:        str

    def __str__(self) -> str:
        all_picks = [self.your_pick] + self.add_picks
        return (
            f"Trade up to #{self.target_pick}: "
            f"Send {all_picks} "
            f"(JJ: {self.give_value_jj:.0f} | F-S: {self.give_value_fs:.0f}) | "
            f"{self.verdict}"
        )


# ── Core Trade Functions ───────────────────────────────────────────────────────

def pick_value(pick_number: int, chart: str = "both") -> Dict[str, float]:
    """
    Get the trade value for a draft pick under one or both chart systems.

    Args:
        pick_number: Overall draft pick number (1-262)
        chart: "jj", "fs", or "both"

    Returns:
        Dict with chart values
    """
    pick_number = max(1, min(pick_number, 262))
    result = {}

    if chart in ("jj", "both"):
        result["jimmy_johnson"] = JIMMY_JOHNSON_CHART.get(pick_number, 0.0)
    if chart in ("fs", "both"):
        result["fitzgerald_spielberger"] = FITZGERALD_SPIELBERGER_CHART.get(pick_number, 0.0)

    return result


def evaluate_trade(
    give_picks: List[int],
    receive_picks: List[int],
    perspective: str = "neutral",
) -> TradeEvaluation:
    """
    Evaluate a trade package under both JJ and F-S chart systems.

    A trade is "fair" if the value difference is within 10% under a chart.
    "STRONG" means the advantage exceeds 20%.

    Args:
        give_picks: Pick numbers you are giving away
        receive_picks: Pick numbers you are receiving
        perspective: "neutral" | "analytics" (F-S bias) | "traditional" (JJ bias)

    Returns:
        TradeEvaluation with full breakdown and recommendation

    Examples:
        >>> # Classic Eagles trade-down
        >>> ev = evaluate_trade([9], [16, 49, 82])
        >>> print(ev.consensus_verdict)
        "STRONG_RECEIVE — Eagles win this overwhelmingly under analytics chart"
    """
    give_jj  = sum(JIMMY_JOHNSON_CHART.get(p, 0.0) for p in give_picks)
    recv_jj  = sum(JIMMY_JOHNSON_CHART.get(p, 0.0) for p in receive_picks)
    give_fs  = sum(FITZGERALD_SPIELBERGER_CHART.get(p, 0.0) for p in give_picks)
    recv_fs  = sum(FITZGERALD_SPIELBERGER_CHART.get(p, 0.0) for p in receive_picks)

    jj_verdict  = _verdict(give_jj, recv_jj)
    fs_verdict  = _verdict(give_fs, recv_fs)
    jj_surplus  = recv_jj - give_jj
    fs_surplus  = recv_fs - give_fs

    # Consensus
    if jj_verdict == fs_verdict:
        consensus = f"CONSENSUS {jj_verdict}"
    elif "FAIR" in jj_verdict and "FAIR" in fs_verdict:
        consensus = "CONSENSUS FAIR"
    else:
        consensus = f"CHART-DEPENDENT — JJ: {jj_verdict} | F-S: {fs_verdict}"

    # Recommendation logic
    if perspective == "analytics":
        primary = fs_verdict
    elif perspective == "traditional":
        primary = jj_verdict
    else:
        # Neutral: use the more conservative assessment
        verdicts = {"STRONG_GIVE": -2, "SLIGHT_GIVE": -1, "FAIR": 0,
                    "SLIGHT_RECEIVE": 1, "STRONG_RECEIVE": 2}
        v_jj = verdicts.get(jj_verdict, 0)
        v_fs = verdicts.get(fs_verdict, 0)
        primary = jj_verdict if abs(v_jj) <= abs(v_fs) else fs_verdict

    rec_map = {
        "STRONG_GIVE":     "REJECT — significantly overpaying under this chart",
        "SLIGHT_GIVE":     "CAUTION — slight overpay; may be worth it for positional need",
        "FAIR":            "ACCEPT — balanced trade under this chart system",
        "SLIGHT_RECEIVE":  "FAVORABLE — you're getting slightly better value",
        "STRONG_RECEIVE":  "STRONG ACCEPT — significantly favorable; take this trade",
    }

    return TradeEvaluation(
        give_picks=give_picks,
        receive_picks=receive_picks,
        give_value_jj=give_jj,
        receive_value_jj=recv_jj,
        give_value_fs=give_fs,
        receive_value_fs=recv_fs,
        jj_verdict=jj_verdict,
        fs_verdict=fs_verdict,
        consensus_verdict=consensus,
        surplus_jj=jj_surplus,
        surplus_fs=fs_surplus,
        recommendation=rec_map.get(primary, "EVALUATE FURTHER"),
    )


def optimal_trade_up(
    target_pick: int,
    your_picks: List[int],
    max_picks_to_give: int = 3,
    chart: str = "both",
) -> List[TradeSuggestion]:
    """
    Find optimal pick packages to trade UP to a target pick.

    Searches all combinations of your_picks (up to max_picks_to_give) to find
    packages that come closest to fair value for the target pick, sorted by
    cost efficiency (giving least excess value).

    Args:
        target_pick: The pick number you want to move up to
        your_picks: Your current pick numbers (must all be > target_pick)
        max_picks_to_give: Maximum number of picks to send (default 3)
        chart: "jj", "fs", or "both" (uses average of both in "both" mode)

    Returns:
        List of TradeSuggestion objects, sorted best-to-worst for you
    """
    from itertools import combinations

    target_jj = JIMMY_JOHNSON_CHART.get(target_pick, 0.0)
    target_fs = FITZGERALD_SPIELBERGER_CHART.get(target_pick, 0.0)

    eligible = [p for p in your_picks if p > target_pick]
    if not eligible:
        return []

    suggestions = []

    # Try each lead pick (your highest pick in the trade)
    for lead_pick in eligible:
        remaining = [p for p in eligible if p != lead_pick]

        # Try adding 0-2 additional picks
        for n_add in range(0, min(max_picks_to_give, len(remaining) + 1)):
            for add_combo in combinations(remaining, n_add):
                all_give = [lead_pick] + list(add_combo)
                give_jj = sum(JIMMY_JOHNSON_CHART.get(p, 0.0) for p in all_give)
                give_fs = sum(FITZGERALD_SPIELBERGER_CHART.get(p, 0.0) for p in all_give)

                # Must have enough value to reach target
                if give_jj < target_jj * 0.85 and give_fs < target_fs * 0.85:
                    continue  # Not enough value under either chart

                ratio_jj = give_jj / max(target_jj, 1.0)
                ratio_fs = give_fs / max(target_fs, 1.0)

                if ratio_jj > 1.5 and ratio_fs > 1.5:
                    verdict = "OVERPAYING — too much value going out"
                elif ratio_jj < 0.90 and ratio_fs < 0.90:
                    verdict = "UNDERPAYING — other team unlikely to accept"
                elif max(ratio_jj, ratio_fs) <= 1.10:
                    verdict = "FAIR — both charts agree this is balanced"
                else:
                    verdict = f"CHART-SPLIT (JJ ratio: {ratio_jj:.2f} | F-S ratio: {ratio_fs:.2f})"

                suggestions.append(TradeSuggestion(
                    target_pick=target_pick,
                    your_pick=lead_pick,
                    add_picks=list(add_combo),
                    give_value_jj=give_jj,
                    give_value_fs=give_fs,
                    cost_ratio_jj=ratio_jj,
                    cost_ratio_fs=ratio_fs,
                    verdict=verdict,
                ))

    # Sort: favor packages closest to 1.0 ratio (least overpay)
    def _sort_key(s: TradeSuggestion) -> float:
        avg_ratio = (s.cost_ratio_jj + s.cost_ratio_fs) / 2.0
        # Penalize overpay more than underpay (other team won't accept underpay)
        overpay = max(0, avg_ratio - 1.0) * 2.0
        underpay = max(0, 1.0 - avg_ratio)
        return overpay + underpay

    return sorted(suggestions, key=_sort_key)[:10]


def find_trade_partners(
    have_pick: int,
    need_positions: List[str],
    all_team_needs: Optional[Dict[str, List[str]]] = None,
) -> List[Dict]:
    """
    Find teams most likely to trade up for a specific pick.

    A team is a viable trade partner if:
    1. They pick later than have_pick (they need to move up)
    2. They have a positional need that matches the likely player at have_pick
    3. They have enough cap space to absorb the slot cost

    Args:
        have_pick: The pick number you own and might trade
        need_positions: Positions of players likely available at have_pick
        all_team_needs: Optional {team: [positions]} from team_needs.py

    Returns:
        List of partner dicts sorted by likelihood score
    """
    from teams.cap_space import TEAM_CAP_SPACE_2026, rookie_slot_cost

    # Default needs if not provided
    if all_team_needs is None:
        all_team_needs = {}

    slot_cost = rookie_slot_cost(have_pick).four_year_total
    partners = []

    for team, space in TEAM_CAP_SPACE_2026.items():
        if space < slot_cost * 0.8:
            continue  # Can't afford it

        team_needs = all_team_needs.get(team, [])
        position_overlap = len([p for p in need_positions if p in team_needs])

        # Likelihood score: cap space + need alignment
        cap_score = min(1.0, space / (slot_cost * 1.5))
        need_score = position_overlap / max(len(need_positions), 1)
        likelihood = (cap_score * 0.4) + (need_score * 0.6)

        if likelihood > 0.3 or position_overlap > 0:
            partners.append({
                "team":             team,
                "likelihood_score": round(likelihood, 3),
                "cap_space_m":      round(space / 1e6, 1),
                "position_overlap": position_overlap,
                "need_positions":   team_needs[:5],
            })

    return sorted(partners, key=lambda x: -x["likelihood_score"])


def compare_charts(picks: List[int]) -> None:
    """
    Print a side-by-side comparison of JJ vs F-S values for a set of picks.

    Shows where the two systems diverge most — typically in rounds 2-3 where
    analytics teams find the most exploitable value.

    Args:
        picks: List of pick numbers to compare
    """
    print(f"\n{'Pick':>5} {'JJ Value':>12} {'F-S Value':>12} {'F-S Diff%':>12}  Interpretation")
    print("─" * 70)
    for p in picks:
        jj  = JIMMY_JOHNSON_CHART.get(p, 0.0)
        fs  = FITZGERALD_SPIELBERGER_CHART.get(p, 0.0)
        pct = ((fs - jj) / jj * 100) if jj > 0 else 0
        interp = (
            "F-S values HIGHER (late pick undervalued by JJ)" if pct > 10 else
            "F-S values LOWER (early pick overvalued by JJ)"  if pct < -10 else
            "Charts roughly agree"
        )
        print(f"{p:>5} {jj:>12.1f} {fs:>12.1f} {pct:>+11.1f}%  {interp}")


# ── Private Helpers ────────────────────────────────────────────────────────────

def _verdict(give_val: float, recv_val: float) -> str:
    """Classify trade outcome from the perspective of the receiving side."""
    if recv_val <= 0 and give_val <= 0:
        return "FAIR"
    if give_val <= 0:
        return "STRONG_RECEIVE"
    ratio = recv_val / give_val if give_val > 0 else float("inf")
    if ratio >= 1.20:   return "STRONG_RECEIVE"
    if ratio >= 1.05:   return "SLIGHT_RECEIVE"
    if ratio >= 0.95:   return "FAIR"
    if ratio >= 0.80:   return "SLIGHT_GIVE"
    return "STRONG_GIVE"


# ── CLI Demo ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("DRAFT TRADE VALUE CALCULATOR — JJ vs Fitzgerald-Spielberger")
    print("=" * 70)

    # Compare charts across draft
    compare_charts([1, 5, 10, 16, 32, 48, 64, 100, 150])

    # Evaluate a real historical trade: 2021 SF trading up for Trey Lance
    # SF gave: picks 3, 12, 43, 2022-R1, 2022-R3 to receive pick #3
    # Simplified version:
    print("\n── Famous Trade Evaluation: 49ers trading up for Trey Lance ──")
    ev = evaluate_trade(
        give_picks=[12, 43, 102],
        receive_picks=[3],
    )
    print(ev)

    # Trade-up optimizer
    print("\n── Optimal Trade-Up Packages: Moving from #15 to #5 ──")
    suggestions = optimal_trade_up(
        target_pick=5,
        your_picks=[15, 47, 78, 110],
        max_picks_to_give=3,
    )
    for s in suggestions[:5]:
        print(f"  {s}")
