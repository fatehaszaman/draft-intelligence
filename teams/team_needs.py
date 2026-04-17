"""
teams/team_needs.py
════════════════════════════════════════════════════════════════════════════════
Team Need Scoring + Event-Driven Draft State Machine

BUSINESS SUMMARY
────────────────
A player's draft grade means nothing in isolation. The 12th-best player in
the draft who fills a desperate need is more valuable than the 8th-best who
plays a position already stacked. This module encodes 2026 team needs for all
32 NFL teams, computes player-team fit scores, and runs an event-driven mock
draft state machine where each pick updates every team's need profile in real
time.

EVENT-DRIVEN STATE MACHINE
──────────────────────────
Each draft pick fires a DraftEvent that updates the live state:
  PickMade        → removes player from board, updates drafting team's needs
  PlayerRemoved   → signals board thinning at a position
  RoundCompleted  → resets pick counter, logs round summary
  TeamPassedOnPlayer → logged when team skips a value player (future trade signal)
  TradeOccurred   → swaps pick assignments between teams

Events are dispatched through DraftStateMachine.process_event(), which
maintains a DraftState dataclass containing the full current snapshot.
The event log is preserved for post-draft analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple


# ── Event Types ───────────────────────────────────────────────────────────────

class EventType(Enum):
    PICK_MADE           = auto()
    PLAYER_REMOVED      = auto()
    ROUND_COMPLETED     = auto()
    TEAM_PASSED         = auto()
    TRADE_OCCURRED      = auto()
    BOARD_UPDATED       = auto()


@dataclass
class DraftEvent:
    """Base event dataclass. All draft events share these core fields."""
    event_type:   EventType
    timestamp:    float = field(default_factory=time.time)
    pick_number:  Optional[int]   = None
    round_number: Optional[int]   = None
    team:         Optional[str]   = None
    player:       Optional[str]   = None
    position:     Optional[str]   = None
    metadata:     Dict            = field(default_factory=dict)

    def __str__(self) -> str:
        parts = [f"[{self.event_type.name}]"]
        if self.pick_number:
            parts.append(f"Pick #{self.pick_number}")
        if self.team:
            parts.append(f"Team: {self.team}")
        if self.player:
            parts.append(f"Player: {self.player}")
        if self.position:
            parts.append(f"({self.position})")
        if self.metadata:
            parts.append(str(self.metadata))
        return " | ".join(parts)


@dataclass
class PickMadeEvent(DraftEvent):
    """Fired when a team makes a selection. Updates board + team need."""
    event_type: EventType = EventType.PICK_MADE
    overall_grade: float = 0.0
    fit_score:     float = 0.0


@dataclass
class TradeOccurredEvent(DraftEvent):
    """Records a pick swap between two teams."""
    event_type:      EventType = EventType.TRADE_OCCURRED
    trade_partner:   Optional[str] = None
    picks_given:     List[int] = field(default_factory=list)
    picks_received:  List[int] = field(default_factory=list)
    value_surplus:   float = 0.0   # positive = receiving team wins the trade


@dataclass
class RoundCompletedEvent(DraftEvent):
    """Fired at the end of each round."""
    event_type:        EventType = EventType.ROUND_COMPLETED
    picks_in_round:    int = 32
    top_pick:          Optional[str] = None
    biggest_steal:     Optional[str] = None


@dataclass
class TeamPassedEvent(DraftEvent):
    """Fired when a team passes on an expected value player."""
    event_type:       EventType = EventType.TEAM_PASSED
    player_grade:     float = 0.0
    team_need_score:  float = 0.0
    reason:           str = ""


# ── Draft State ────────────────────────────────────────────────────────────────

@dataclass
class DraftState:
    """
    Full snapshot of the draft at any moment in time.

    Mutated in-place by DraftStateMachine.process_event().
    Immutable fields are set at initialization.
    """
    # Immutable setup
    total_rounds:     int
    teams_in_order:   List[str]   # Pick order for round 1 (snake in later rounds)
    # Live state
    current_pick:     int         = 1
    current_round:    int         = 1
    remaining_board:  List[str]   = field(default_factory=list)  # player names, ordered
    picks_made:       List[DraftEvent] = field(default_factory=list)
    # Team needs: team → position → need score [0-1] (mutable)
    team_needs:       Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Pick assignments: pick_number → team (may be swapped by trades)
    pick_assignments: Dict[int, str]  = field(default_factory=dict)
    event_log:        List[str]       = field(default_factory=list)


# ── Draft State Machine ───────────────────────────────────────────────────────

class DraftStateMachine:
    """
    Event-driven draft simulation engine.

    Each call to process_event() consumes one DraftEvent, updates the
    DraftState, notifies subscribers, and appends to the event log.

    Subscribers are callables registered via subscribe(). They receive
    (event, state) and can be used for logging, UI updates, or analytics.

    PICK RESOLUTION ORDER
    ─────────────────────
    1. Receive PickMadeEvent
    2. Remove player from remaining_board
    3. Reduce drafting team's need at that position by 0.30 (diminishing need)
    4. Emit BOARD_UPDATED event
    5. Check if round is complete → emit RoundCompletedEvent
    6. Advance current_pick and current_round
    """

    def __init__(self, state: DraftState) -> None:
        self.state = state
        self._subscribers: List[Callable] = []

    def subscribe(self, callback: Callable) -> None:
        """Register a callback (event, state) → None for event notifications."""
        self._subscribers.append(callback)

    def process_event(self, event: DraftEvent) -> DraftState:
        """
        Consume an event, mutate DraftState, notify subscribers.

        Returns the updated DraftState after processing.
        """
        state = self.state

        if event.event_type == EventType.PICK_MADE:
            self._handle_pick_made(event, state)

        elif event.event_type == EventType.TRADE_OCCURRED:
            self._handle_trade(event, state)

        elif event.event_type == EventType.TEAM_PASSED:
            state.event_log.append(
                f"Pick #{state.current_pick}: {event.team} passed on "
                f"{event.player} ({event.position}) — {event.reason}"
            )

        # Notify all subscribers
        log_entry = str(event)
        state.event_log.append(log_entry)
        for cb in self._subscribers:
            try:
                cb(event, state)
            except Exception:
                pass

        return state

    def _handle_pick_made(self, event: DraftEvent, state: DraftState) -> None:
        # 1. Remove player from board
        if event.player and event.player in state.remaining_board:
            state.remaining_board.remove(event.player)

        # 2. Record the pick
        state.picks_made.append(event)

        # 3. Diminish team's need at this position
        if event.team and event.position:
            needs = state.team_needs.get(event.team, {})
            pos = event.position.upper()
            current_need = needs.get(pos, 0.0)
            # First pick at a position reduces need by 0.35; subsequent by 0.20
            picks_at_pos = sum(
                1 for p in state.picks_made
                if p.team == event.team and p.position == event.position
            )
            reduction = 0.35 if picks_at_pos <= 1 else 0.20
            needs[pos] = max(0.0, current_need - reduction)
            state.team_needs[event.team] = needs

        # 4. Emit board updated event internally
        board_event = DraftEvent(
            event_type=EventType.BOARD_UPDATED,
            pick_number=state.current_pick,
            metadata={"remaining": len(state.remaining_board)},
        )
        state.event_log.append(str(board_event))

        # 5. Check round completion
        picks_per_round = len(state.teams_in_order)
        if state.current_pick % picks_per_round == 0:
            picks_this_round = [
                p for p in state.picks_made
                if p.round_number == state.current_round
            ]
            round_event = RoundCompletedEvent(
                pick_number=state.current_pick,
                round_number=state.current_round,
                picks_in_round=len(picks_this_round),
                top_pick=picks_this_round[0].player if picks_this_round else None,
            )
            state.event_log.append(str(round_event))
            state.current_round += 1

        # 6. Advance pick counter
        state.current_pick += 1

    def _handle_trade(self, event: TradeOccurredEvent, state: DraftState) -> None:
        """Swap pick assignments between two teams."""
        if not isinstance(event, TradeOccurredEvent):
            return
        partner = event.trade_partner
        if not partner or not event.team:
            return
        for pick in event.picks_given:
            if pick in state.pick_assignments:
                state.pick_assignments[pick] = partner
        for pick in event.picks_received:
            if pick in state.pick_assignments:
                state.pick_assignments[pick] = event.team
        state.event_log.append(
            f"TRADE: {event.team} gives picks {event.picks_given} "
            f"to {partner}, receives picks {event.picks_received}. "
            f"Value surplus: {event.value_surplus:+.1f}"
        )


# ── 2026 Team Needs Database ──────────────────────────────────────────────────
# Need scores: 0.0 = no need (set at position), 1.0 = desperate need
# Based on realistic 2026 depth chart analysis post-2025 FA/cuts

TEAM_NEEDS_2026: Dict[str, Dict[str, float]] = {
    "TEN": {"QB": 1.00, "WR": 0.70, "OT": 0.55, "LB": 0.50, "CB": 0.40, "DT": 0.35, "TE": 0.30},
    "CLE": {"QB": 0.80, "CB": 0.85, "WR": 0.75, "OT": 0.60, "EDGE": 0.55, "S": 0.40},
    "NYG": {"EDGE": 0.90, "QB": 0.85, "OT": 0.65, "CB": 0.55, "WR": 0.50, "LB": 0.45},
    "NE":  {"OT": 0.90, "QB": 0.85, "CB": 0.65, "WR": 0.60, "EDGE": 0.50, "LB": 0.45},
    "JAX": {"DT": 0.85, "QB": 0.75, "OT": 0.70, "CB": 0.60, "WR": 0.55, "LB": 0.50},
    "LV":  {"WR": 0.88, "EDGE": 0.75, "OT": 0.65, "CB": 0.55, "DT": 0.50, "S": 0.45},
    "NYJ": {"S": 0.85, "WR": 0.70, "EDGE": 0.65, "OT": 0.60, "DT": 0.50, "CB": 0.45},
    "CAR": {"EDGE": 0.88, "OT": 0.75, "DT": 0.70, "WR": 0.55, "CB": 0.50, "LB": 0.45},
    "NO":  {"DT": 0.82, "CB": 0.75, "EDGE": 0.70, "WR": 0.55, "OT": 0.50, "S": 0.45},
    "CHI": {"EDGE": 0.80, "CB": 0.75, "OT": 0.65, "WR": 0.55, "DT": 0.50, "LB": 0.45},
    "ATL": {"EDGE": 0.75, "DT": 0.70, "OT": 0.65, "CB": 0.55, "LB": 0.50, "S": 0.40},
    "DEN": {"CB": 0.78, "EDGE": 0.72, "WR": 0.65, "OT": 0.55, "DT": 0.50, "LB": 0.45},
    "ARI": {"OT": 0.82, "CB": 0.75, "LB": 0.68, "EDGE": 0.55, "DT": 0.50, "WR": 0.45},
    "IND": {"EDGE": 0.80, "CB": 0.72, "WR": 0.65, "OT": 0.55, "DT": 0.50, "S": 0.40},
    "LAR": {"DT": 0.75, "CB": 0.70, "OT": 0.65, "LB": 0.55, "EDGE": 0.50, "WR": 0.40},
    "SEA": {"QB": 0.78, "EDGE": 0.72, "OT": 0.65, "CB": 0.55, "WR": 0.50, "DT": 0.45},
    "MIA": {"OT": 0.82, "EDGE": 0.75, "DT": 0.65, "CB": 0.55, "LB": 0.50, "WR": 0.40},
    "MIN": {"CB": 0.78, "EDGE": 0.70, "OT": 0.65, "DT": 0.55, "WR": 0.45, "S": 0.40},
    "WAS": {"DT": 0.75, "CB": 0.70, "WR": 0.65, "OT": 0.55, "LB": 0.50, "TE": 0.40},
    "GB":  {"CB": 0.78, "WR": 0.72, "EDGE": 0.65, "OT": 0.50, "DT": 0.45, "LB": 0.40},
    "DAL": {"EDGE": 0.80, "CB": 0.72, "OT": 0.65, "WR": 0.55, "DT": 0.50, "S": 0.40},
    "DET": {"CB": 0.75, "S": 0.70, "LB": 0.65, "EDGE": 0.50, "WR": 0.45, "DT": 0.40},
    "PHI": {"CB": 0.72, "EDGE": 0.68, "WR": 0.65, "DT": 0.55, "OT": 0.45, "LB": 0.40},
    "SF":  {"CB": 0.78, "EDGE": 0.70, "DT": 0.62, "WR": 0.55, "OT": 0.45, "LB": 0.40},
    "BUF": {"EDGE": 0.75, "DT": 0.68, "CB": 0.62, "OT": 0.55, "WR": 0.45, "LB": 0.35},
    "HOU": {"CB": 0.72, "OT": 0.68, "LB": 0.62, "EDGE": 0.50, "DT": 0.45, "WR": 0.40},
    "PIT": {"QB": 0.75, "OT": 0.70, "CB": 0.65, "EDGE": 0.55, "WR": 0.45, "DT": 0.40},
    "CIN": {"OT": 0.80, "CB": 0.70, "EDGE": 0.65, "DT": 0.55, "LB": 0.45, "S": 0.40},
    "BAL": {"WR": 0.75, "EDGE": 0.68, "CB": 0.62, "DT": 0.55, "OT": 0.45, "LB": 0.40},
    "KC":  {"WR": 0.72, "CB": 0.68, "EDGE": 0.62, "DT": 0.55, "OT": 0.40, "LB": 0.35},
    "LAC": {"OT": 0.75, "CB": 0.68, "EDGE": 0.62, "WR": 0.55, "DT": 0.45, "LB": 0.40},
    "TB":  {"QB": 0.70, "EDGE": 0.68, "CB": 0.62, "OT": 0.55, "DT": 0.45, "WR": 0.40},
}

# Scheme archetypes per team (relevant for fit scoring)
TEAM_SCHEMES: Dict[str, str] = {
    "TEN": "pro_style", "CLE": "pro_style",  "NYG": "4-3_defense",
    "NE":  "pro_style", "JAX": "4-3_defense","LV":  "west_coast",
    "NYJ": "cover_2",   "CAR": "4-3_defense","NO":  "west_coast",
    "CHI": "4-3_defense","ATL": "spread_option","DEN": "cover_2",
    "ARI": "spread_option","IND": "west_coast","LAR": "spread_option",
    "SEA": "spread_option","MIA": "spread_option","MIN": "west_coast",
    "WAS": "4-3_defense","GB":  "west_coast", "DAL": "4-3_defense",
    "DET": "west_coast", "PHI": "spread_option","SF":  "pro_style",
    "BUF": "pro_style",  "HOU": "spread_option","PIT": "3-4_defense",
    "CIN": "west_coast", "BAL": "pro_style",  "KC":  "spread_option",
    "LAC": "west_coast", "TB":  "west_coast",
}


class TeamNeedScorer:
    """
    Computes team-specific need scores, fit scores, and mock draft
    slot predictions. Integrates with DraftStateMachine for live
    need updates as picks are made.
    """

    def __init__(self) -> None:
        self._needs = {t: dict(n) for t, n in TEAM_NEEDS_2026.items()}
        self._schemes = TEAM_SCHEMES
        # Draft order for 2026 (simplified: worst-to-best 2025 records)
        self._draft_order: List[str] = [
            "TEN", "CLE", "NYG", "NE",  "JAX", "LV",  "NYJ", "CAR",
            "NO",  "CHI", "ATL", "DEN", "ARI", "IND", "LAR", "SEA",
            "MIA", "MIN", "WAS", "GB",  "DAL", "DET", "PHI", "SF",
            "BUF", "HOU", "PIT", "CIN", "BAL", "KC",  "LAC", "TB",
        ]

    # ── Public API ────────────────────────────────────────────────────────────

    def get_team_need(self, team: str, position: str) -> float:
        """
        Return the 0-1 need score for a team at a position.
        0.0 = no need (set at that position)
        1.0 = desperate need (no starter, no backup)
        """
        team = team.upper()
        pos  = position.upper()
        return self._needs.get(team, {}).get(pos, 0.30)  # default: some need

    def fit_score(self, player: str, team: str,
                  player_position: Optional[str] = None,
                  player_overall: float = 80.0) -> float:
        """
        Compute a 0-100 player-team fit score combining:
          - Need score at player's position (50% weight)
          - Scheme fit (30% weight)
          - Player overall grade scaled to 0-100 (20% weight)

        Args:
            player:          Player name (display)
            team:            NFL team abbreviation
            player_position: Position code; required for need lookup
            player_overall:  Overall draft grade 0-100

        Returns:
            float in [0, 100]
        """
        need = self.get_team_need(team, player_position or "QB")
        scheme_fit = self._scheme_fit(player_position or "QB",
                                      self._schemes.get(team.upper(), "pro_style"))
        grade_component = player_overall

        score = (
            0.50 * need * 100
            + 0.30 * scheme_fit * 100
            + 0.20 * grade_component
        )
        return round(min(100.0, score), 1)

    def mock_draft_slot(self, player: str,
                        consensus_rank: int,
                        position: Optional[str] = None) -> Dict:
        """
        Predict pick range for a player based on consensus rank + need analysis.

        Scans the draft order and finds the first team with:
          (a) A high need at the player's position, AND
          (b) The pick lands near the consensus rank

        Returns:
            dict: low, high, consensus pick range, most likely team
        """
        pick_ranges = [
            (1, 5),   (6, 10),  (11, 15), (16, 20),
            (21, 32), (33, 64), (65, 100),(101, 256),
        ]

        # Find which range the consensus rank falls in
        for lo, hi in pick_ranges:
            if lo <= consensus_rank <= hi:
                target_range = (lo, hi)
                break
        else:
            target_range = (consensus_rank - 3, consensus_rank + 5)

        # Find most likely team in that range
        best_team = "TBD"
        best_need = 0.0
        for i, team in enumerate(self._draft_order):
            pick_num = i + 1
            if target_range[0] <= pick_num <= target_range[1]:
                need = self.get_team_need(team, position or "QB")
                if need > best_need:
                    best_need = need
                    best_team = team

        return {
            "player":           player,
            "consensus_rank":   consensus_rank,
            "predicted_low":    max(1, target_range[0]),
            "predicted_high":   target_range[1],
            "most_likely_team": best_team,
            "team_need_score":  round(best_need, 2),
        }

    def value_pick_analysis(self, player: str, pick_number: int,
                            player_rank: int) -> str:
        """
        Classify whether this pick is value/fair/reach based on the
        difference between where the player is taken vs. their rank.

        Args:
            player:      Player name
            pick_number: Actual draft pick number
            player_rank: Model's consensus board rank for this player

        Returns:
            str: "STEAL" | "FAIR" | "REACH" | "SIGNIFICANT_REACH"
        """
        diff = pick_number - player_rank  # positive = taken later than expected
        if diff >= 8:
            return "STEAL"
        elif diff >= -3:
            return "FAIR"
        elif diff >= -8:
            return "REACH"
        else:
            return "SIGNIFICANT_REACH"

    def top_fits_for_player(self, player: str,
                             position: str,
                             player_overall: float,
                             top_n: int = 5) -> List[Dict]:
        """
        Return the top N team fits for a player, sorted by fit score desc.
        """
        results = []
        for team in self._draft_order:
            score = self.fit_score(player, team, position, player_overall)
            results.append({"team": team, "fit_score": score,
                             "need": self.get_team_need(team, position)})
        return sorted(results, key=lambda x: x["fit_score"], reverse=True)[:top_n]

    def build_draft_state(self, big_board: List[Tuple[str, str, float]],
                           total_rounds: int = 3) -> DraftState:
        """
        Build an initial DraftState for the mock draft simulation.

        Args:
            big_board: List of (player_name, position, overall_grade) sorted best-first
            total_rounds: Rounds to simulate

        Returns:
            DraftState ready for DraftStateMachine
        """
        pick_assignments = {}
        for rnd in range(1, total_rounds + 1):
            order = self._draft_order if rnd % 2 == 1 else list(reversed(self._draft_order))
            for i, team in enumerate(order):
                pick_num = (rnd - 1) * len(self._draft_order) + i + 1
                pick_assignments[pick_num] = team

        return DraftState(
            total_rounds=total_rounds,
            teams_in_order=self._draft_order,
            current_pick=1,
            current_round=1,
            remaining_board=[p[0] for p in big_board],
            team_needs={t: dict(n) for t, n in self._needs.items()},
            pick_assignments=pick_assignments,
        )

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _scheme_fit(self, position: str, scheme: str) -> float:
        """
        Return a 0-1 scheme fit score based on position × scheme compatibility.
        Simple lookup table reflecting which positions thrive in each scheme.
        """
        scheme_position_fits = {
            "west_coast":     {"QB": 0.8, "WR": 0.9, "TE": 0.9, "RB": 0.7,
                               "OT": 0.8, "EDGE": 0.7, "CB": 0.8, "S": 0.8,
                               "DT": 0.7, "LB": 0.7},
            "spread_option":  {"QB": 0.9, "WR": 0.8, "TE": 0.7, "RB": 0.8,
                               "OT": 0.7, "EDGE": 0.8, "CB": 0.8, "S": 0.7,
                               "DT": 0.8, "LB": 0.7},
            "pro_style":      {"QB": 0.8, "WR": 0.8, "TE": 0.8, "RB": 0.8,
                               "OT": 0.9, "EDGE": 0.8, "CB": 0.8, "S": 0.8,
                               "DT": 0.8, "LB": 0.8},
            "4-3_defense":    {"QB": 0.8, "WR": 0.7, "TE": 0.7, "RB": 0.7,
                               "OT": 0.8, "EDGE": 0.9, "CB": 0.9, "S": 0.8,
                               "DT": 0.9, "LB": 0.8},
            "3-4_defense":    {"QB": 0.8, "WR": 0.7, "TE": 0.7, "RB": 0.7,
                               "OT": 0.8, "EDGE": 0.8, "CB": 0.8, "S": 0.8,
                               "DT": 0.8, "LB": 0.9},
            "cover_2":        {"QB": 0.8, "WR": 0.8, "TE": 0.7, "RB": 0.7,
                               "OT": 0.8, "EDGE": 0.8, "CB": 0.8, "S": 0.9,
                               "DT": 0.8, "LB": 0.8},
            "man_coverage":   {"QB": 0.8, "WR": 0.8, "TE": 0.7, "RB": 0.7,
                               "OT": 0.8, "EDGE": 0.8, "CB": 0.9, "S": 0.8,
                               "DT": 0.8, "LB": 0.7},
            "air_raid":       {"QB": 0.8, "WR": 0.9, "TE": 0.8, "RB": 0.6,
                               "OT": 0.7, "EDGE": 0.7, "CB": 0.8, "S": 0.7,
                               "DT": 0.7, "LB": 0.7},
        }
        fits = scheme_position_fits.get(scheme, {})
        return fits.get(position.upper(), 0.75)
