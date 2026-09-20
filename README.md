# draft-intelligence

[Algorithm guide: pseudocode, time complexity, and memory](docs/ALGORITHM_GUIDE.md).

![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)
![NFL Data](https://img.shields.io/badge/data-nfl--data--py-orange)

**NFL Draft outcome prediction and player evaluation engine.** Predicts draft order, team-pick fits, and career success probability using combine biometrics, college production, positional scarcity, team need scoring, historical comp analysis, and social/commercial value signals.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                             │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  NFL Combine │  │ College Stats│  │  Team Needs  │  │ Historical │  │
│  │  Biometrics  │  │ PFF / Yards  │  │  Depth Charts│  │   Comps    │  │
│  │  & Athleticsm│  │ TDs / Grades │  │  Cap Space   │  │  2010-2024 │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
└─────────┼─────────────────┼─────────────────┼────────────────┼─────────┘
          │                 │                 │                │
          ▼                 ▼                 │                ▼
┌─────────────────────────────────────┐      │    ┌────────────────────────┐
│       FEATURE ENGINEERING           │      │    │  COMP FINDER (KNN)     │
│                                     │      │    │                        │
│  • Position-adjusted athletic score │      │    │  Euclidean distance on │
│  • Conference/system adjustments    │      │    │  normalized 12-dim     │
│  • Dominator rating                 │      │    │  feature vector        │
│  • Breakout age                     │      │    │  → bust/avg/good/elite │
│  • Career trajectory slope          │      │    │    probability         │
└────────────────┬────────────────────┘      │    └──────────┬─────────────┘
                 │                           │               │
                 ▼                           ▼               ▼
┌────────────────────────────────────────────────────────────────────────┐
│                         PREDICTION MODELS                              │
│                                                                        │
│   ┌─────────────────────────────┐    ┌──────────────────────────────┐  │
│   │   Draft Position Predictor  │    │   Career Success Model       │  │
│   │                             │    │                              │  │
│   │  combine   × 0.25           │    │  comp outcomes × 0.40        │  │
│   │  college   × 0.30           │    │  athleticism  × 0.25         │  │
│   │  comps     × 0.25           │    │  production   × 0.25         │  │
│   │  intangibles × 0.20         │    │  intangibles  × 0.10         │  │
│   └──────────────┬──────────────┘    └──────────────┬───────────────┘  │
└──────────────────┼────────────────────────────────┬─┘                  │
                   │                                │                    │
                   ▼                                ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          OUTPUT LAYER                                │
│                                                                      │
│   ┌──────────────────────┐       ┌────────────────────────────────┐  │
│   │   Ranked Big Board   │       │   Team Fit Scores + Mock Draft  │ │
│   │                      │       │                                │  │
│   │  #1  Cam Ward  QB  97│       │  NE @ #4: Cam Ward (QB) 94.2  │  │
│   │  #2  Travis Hunter 95│       │  NYG @ #3: Travis Hunter 91.7 │  │
│   │  #3  Abdul Carter 93 │       │  TEN @ #1: Cam Ward (QB) 96.8 │  │
│   │  ...                 │       │  ...                          │  │
│   └──────────────────────┘       └────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Module | Description |
|--------|-------------|
| `scouting/combine_analyzer.py` | Position-weighted athletic scoring with historical percentiles |
| `scouting/college_production.py` | Conference/system-adjusted production, dominator rating, breakout age |
| `scouting/historical_comps.py` | KNN comp finder over 60+ historical prospects, career outcome probabilities |
| `teams/team_needs.py` | 2026 team need scores, scheme fit, mock draft slot predictions |
| `teams/cap_space.py` | 2026 cap space data, rookie slot costs, positional spending history |
| `teams/trade_value_calculator.py` | Pick-value lookup and trade-package comparison |

The integrated draft-board runner and NFL data client are not implemented in
this checkout. The architecture and big-board mockup describe the intended
integration, not a working end-to-end command.

---

## Quickstart

```bash
# Clone and create an isolated environment (macOS/Linux)
git clone https://github.com/fatehaszaman/draft-intelligence.git
cd draft-intelligence
python3 -m venv .venv
source .venv/bin/activate

# Offline example: uses the standard library only
python -c "from teams.trade_value_calculator import evaluate_trade; print(evaluate_trade([9], [16, 49, 82]))"

# Optional dependencies for the other analysis modules
python -m pip install -r requirements.txt
```

### Sample Output

The following big-board table is an illustrative mockup, not output from the
offline trade example above. No integrated big-board runner is shipped.

```
╔══════════════════════════════════════════════════════════════════╗
║               DRAFT INTELLIGENCE — 2026 BIG BOARD               ║
╠════╦══════════════════╦═════╦════════╦═══════╦═══════╦══════════╣
║ RK ║ PLAYER           ║ POS ║ SCHOOL ║ COMB  ║ PROD  ║ OVERALL  ║
╠════╬══════════════════╬═════╬════════╬═══════╬═══════╬══════════╣
║  1 ║ Cam Ward         ║ QB  ║ Miami  ║ 78.4  ║ 91.2  ║  94.8    ║
║  2 ║ Travis Hunter    ║ CB  ║ Col St ║ 95.1  ║ 96.3  ║  93.6    ║
║  3 ║ Abdul Carter     ║ EDGE║ Penn St║ 96.7  ║ 89.4  ║  92.1    ║
║  4 ║ Will Campbell    ║ OT  ║ LSU    ║ 82.3  ║ 88.6  ║  90.4    ║
║  5 ║ Mason Graham     ║ DT  ║ Mich   ║ 88.9  ║ 87.1  ║  89.7    ║
║  6 ║ Tetairoa McMillan║ WR  ║ Arizona║ 90.2  ║ 92.4  ║  89.2    ║
║  7 ║ Malaki Starks    ║ S   ║ Georgia║ 87.3  ║ 85.9  ║  87.8    ║
║  8 ║ Jalon Walker     ║ EDGE║ Georgia║ 91.4  ║ 84.7  ║  86.3    ║
║  9 ║ Darius Alexander ║ DT  ║ Toledo ║ 93.1  ║ 83.2  ║  85.9    ║
║ 10 ║ Mykel Williams   ║ EDGE║ Georgia║ 89.6  ║ 82.8  ║  85.1    ║
╚════╩══════════════════╩═════╩════════╩═══════╩═══════╩══════════╝

2026 MOCK DRAFT — ROUND 1 (First 10 Picks)
─────────────────────────────────────────────────────────────────
Pick  1 │ TEN  │ Cam Ward         QB   │ Fit: 97.1 │ Value: FAIR
Pick  2 │ CLE  │ Travis Hunter    CB   │ Fit: 88.4 │ Value: STEAL
Pick  3 │ NYG  │ Abdul Carter     EDGE │ Fit: 91.2 │ Value: FAIR
Pick  4 │ NE   │ Will Campbell    OT   │ Fit: 86.7 │ Value: FAIR
Pick  5 │ JAX  │ Mason Graham     DT   │ Fit: 84.3 │ Value: FAIR
Pick  6 │ LV   │ Tetairoa McMillan WR  │ Fit: 89.1 │ Value: STEAL
Pick  7 │ NYJ  │ Malaki Starks    S    │ Fit: 83.8 │ Value: FAIR
Pick  8 │ CAR  │ Jalon Walker     EDGE │ Fit: 81.6 │ Value: STEAL
Pick  9 │ NO   │ Darius Alexander DT   │ Fit: 79.4 │ Value: FAIR
Pick 10 │ CHI  │ Mykel Williams   EDGE │ Fit: 82.1 │ Value: FAIR
```

---

## API Reference

### CombineAnalyzer

```python
from scouting.combine_analyzer import CombineAnalyzer

ca = CombineAnalyzer()

# Get athletic score for a player
score = ca.score_combine("Cam Ward", "QB", {
    "forty_yard": 4.71,
    "hand_size": 9.75,
    "arm_length": 32.25,
    "completion_pct": 67.2
})
# → 78.4

# Percentile vs historical players at that position
pct = ca.position_percentile("Abdul Carter", "EDGE", "forty_yard")
# → 94.2  (94th percentile among all EDGE rushers)
```

### CollegeProductionScorer

```python
from scouting.college_production import CollegeProductionScorer

cps = CollegeProductionScorer()

# Adjusted stat (accounts for conference + offensive system)
adj = cps.adjusted_stat(1847, "SEC", "pro_style")
# → 1739.2  (slight downward adjustment for SEC competition)

# Career trajectory
trajectory = cps.career_trajectory([
    {"year": 2022, "yards": 650, "tds": 5, "catches": 48},
    {"year": 2023, "yards": 1100, "tds": 9, "catches": 78},
    {"year": 2024, "yards": 1847, "tds": 17, "catches": 96},
])
# → {"slope": 598.5, "acceleration": 1.14, "trending": "up"}

# Breakout age
age = cps.breakout_age([...])
# → 19  (broke out as true freshman — excellent signal)
```

### TeamNeedScorer

```python
from teams.team_needs import TeamNeedScorer

tns = TeamNeedScorer()

# Get team need at a position
need = tns.get_team_need("NE", "OT")
# → 0.88  (high need)

# Full fit score
fit = tns.fit_score("Will Campbell", "NE")
# → 86.7

# Mock draft slot prediction
slot = tns.mock_draft_slot("Cam Ward")
# → {"low": 1, "high": 2, "consensus": 1}
```

### DraftBoardEngine

```python
from valuation.draft_board import DraftBoardEngine

engine = DraftBoardEngine()

# Overall composite grade
grade = engine.overall_grade("Travis Hunter")
# → 93.6

# Full big board as DataFrame
board = engine.generate_big_board(players)
print(board.to_string())

# Full 3-round mock draft
mock = engine.generate_mock_draft(rounds=3)
print(mock.to_string())
```

### Data Client

```python
from data.nfl_data_client import NFLDataClient

client = NFLDataClient(use_live=True)  # or use_live=False for mock data

# Fetch combine results for a draft year
combine_df = client.get_combine_data(year=2025)

# Get draft picks
draft_df = client.get_draft_picks(year=2025)
```

---

## Configuration

Edit `config.py` to tune model weights, enable live data, and adjust output verbosity.

```python
# config.py
USE_LIVE_DATA = False          # Set True to use nfl-data-py live feeds
COMBINE_WEIGHT = 0.25
COLLEGE_WEIGHT = 0.30
COMPS_WEIGHT   = 0.25
INTANGIBLES_WEIGHT = 0.20
ROUNDS_TO_SIMULATE = 3
```

---

## Testing

The `tests/` directory currently contains scaffolding, not executable test
cases. No passing-build or coverage percentage is claimed; the offline
quickstart is a smoke check only, not a validated prediction benchmark.

---

## License

MIT © 2026 draft-intelligence contributors
