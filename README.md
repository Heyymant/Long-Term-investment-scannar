# Indian Equity Long-Term Alpha Framework

A multi-sleeve, long-only systematic equity research and decision-support system for Indian
markets, with a fast Rust web dashboard as the cockpit.

> **This system never places trades.** It produces rankings, target weights, event signals and a
> rebalance checklist that **you execute manually**. Every broker integration is read-only.

---

## What it does

Three complementary long-only return sleeves share one data layer, cost/tax model, backtester and
statistical validation suite:

| Sleeve | Strategy | Horizon | Evidence base |
|---|---|---|---|
| **A** | Quantamental multi-factor (Quality, Value, Momentum, Low-Vol) | Long | IIMA four-factor & quality research |
| **B** | Statistical mean reversion (residual reversion; pairs variant) | Short | NSE cointegration / stat-arb studies |
| **C** | Event-driven earnings drift (PEAD via SUE) | Tactical | Nifty 500 PEAD evidence |

Everything is measured net of Indian frictions (STT, stamp duty, GST, DP charges, ADV-scaled market
impact) and net of tax (lot-level FIFO, STCG/LTCG), then run through a statistical battery designed
to **reject** results that are merely lucky or overfit.

### The research it is built on

- **Quality is the strongest Indian factor** — QMJ earns ~0.92%/month four-factor alpha, with low
  turnover and shallow drawdowns, and works long-only (Jacob, Pradeep & Varma, IIMA).
- **Momentum earns the most but crashes** — roughly 17% annualized, with a ~-70% drawdown in 2008
  and a 65-month recovery. Hence the vol-adjusted momentum leg and the trend overlay.
- **Low volatility has the best risk-adjusted profile** in India and is a distinct effect.
- **Value works but is cyclical** — used *with* quality, because cheap is not the same as undervalued.
- **Factor premia are structural, not reliably timeable**, so the design diversifies rather than times.
- **425 active Indian funds (2013-2024) showed no significant net alpha** — which is exactly why this
  reports alpha *net of factor exposure*, not just excess return.

---

## Quick start (no broker account needed)

```bash
cd python
pip install -r requirements.txt

# Fetch real NSE prices, TRI benchmarks and overnight rates
python scripts/fetch_nse.py --all --start 2022-01-01

# Run a backtest on that archive
python scripts/run_backtest.py --preset price_only_core --source nse --start 2022-01-01
```

That writes a full set of artifacts to `artifacts/runs/<run_id>/`. Then start the dashboard:

```bash
cd ../dashboard
cargo run --release
# open http://127.0.0.1:8080
```

The dashboard and the test suite both read **exchange data only** — NSE bhavcopy prices, official
index TRI reconstructions, the Nifty 1D Rate overnight series, and recorded XBRL filings. There is
no simulated market.

---

## Deploy on Netlify

The hosted site is a **static** snapshot of the terminal (Launchpad, Screener, Book, 1-year NSE
charts, horse-race GO filters). Push this repo and connect it to Netlify — `netlify.toml` already
sets the build command and publish directory.

```bash
# refresh the JSON snapshot after a backtest / nightly run, then commit
python python/scripts/build_netlify.py
git add dashboard/assets dashboard/assets/data netlify.toml python/scripts/build_netlify.py
git commit -m "Publish latest NSE snapshot for Netlify"
git push
```

On [app.netlify.com](https://app.netlify.com): **Add new site → Import an existing project → GitHub**,
pick this repo. Leave build settings as detected from `netlify.toml`.

| Works on Netlify | Stays on your machine |
|---|---|
| Screener, income ratios, 1-year NSE charts | `cargo run` dashboard (Kite session, live ticks) |
| Upload a portfolio CSV → BUY **and** SELL vs those positions | Python jobs (backtest, horse race, rebalance) |
| Horse-race GO filters, Launchpad | Secrets in `.env` (never commit) |

Portfolio upload on the hosted site is stored in **your browser** (localStorage), not on Netlify's
servers. Refreshing the snapshot is `python python/scripts/build_netlify.py` then push
`dashboard/assets/data/`.

---

## Using real data

### Option A (recommended): NSE public data — free, and better than most paid feeds

NSE publishes everything the framework needs except live ticks:

```bash
cd python
python scripts/fetch_nse.py --all --start 2015-01-01
```

That single command gives you:

| Step | What you get |
|---|---|
| Security master | **Real ISINs**, listing dates and sector classification for ~2,600 NSE equities |
| Bhavcopy archive | Daily OHLCV for the **entire market**, ISIN-keyed, one file per trading day |
| Corporate actions | Splits, bonuses and dividends with ex-dates |
| Results calendar | Forthcoming board meetings (the forward-looking earnings calendar) |
| Fundamentals | Revenue, PAT and **EPS** parsed from each company's XBRL filing |

Two things make this genuinely valuable rather than just free:

**It solves survivorship bias.** Each historical bhavcopy lists exactly what was trading that
day, so replaying the archive reconstructs a database that still contains companies which later
delisted. On a 2022-2024 pull the framework found 2,565 ISINs, flagged 487 as delisted or
suspended, and reconstructed ~150 symbol changes — all of which a "current constituents" snapshot
would have quietly erased.

**It gives real announcement timestamps.** The results feed carries `broadCastDate` — the moment
the market actually learned the numbers, down to the second. That is precisely what point-in-time
discipline requires, and it is the one thing free fundamentals sources normally lack.

Then run against it:

```bash
python scripts/run_backtest.py --preset price_only_core --source nse
```

Honest limitations:
- The results API returns a **rolling recent window**, so deep fundamentals history has to
  accumulate over repeated runs (the cache is append-only for this reason). Until it does,
  Quality and Value will be sparse and the universe filter will warn you rather than silently
  emptying the universe.
- Quarterly filings carry P&L detail; balance-sheet items appear mainly in half-yearly and annual
  filings, so leverage/cash-flow Quality legs are thinner than the profitability ones.
- NSE publishes no machine-readable index inclusion/exclusion log, so point-in-time **membership**
  builds up from snapshots over time.
- Be considerate: it is a free public service. The client is rate-limited and caches to disk.

### 1. Zerodha Kite (prices, holdings, live quotes)

```bash
cp .env.example .env        # then fill in KITE_API_KEY / KITE_API_SECRET

cd python
python scripts/fetch_data.py --login              # prints the login URL
python scripts/fetch_data.py --request-token <token_from_redirect>
python scripts/fetch_data.py --build-master       # ISIN-keyed security master
python scripts/fetch_data.py --prices --start 2010-01-01
```

Notes that matter in practice:
- Kite **access tokens expire daily** — you re-authenticate each trading day.
- Day candles reach back to roughly 2005/06 and each request is capped at 2000 days; the client
  paginates and caches to Parquet automatically.
- Kite's instrument dump has **no ISIN**. Prefer `scripts/fetch_nse.py --master` (EQUITY_L.csv) so
  every row has a real ISIN and survives symbol changes.
- Kite provides **no fundamentals** — see below.

### 2. Fundamentals (required for Quality and Value)

Set `data_sources.fundamentals.provider` in `python/config.yaml`:

| Provider | Cost | Announcement dates | Notes |
|---|---|---|---|
| `nse` | Free | **Yes** (`broadCastDate`) | **Recommended.** Exchange-sourced XBRL. History accumulates over time. |
| `prowess` | Paid | Yes | CMIE Prowess — the academic standard, deepest history. Reads local exports. |
| `eodhd` | Paid | Yes | Convenient API with long history. Set `EODHD_API_KEY`. |
| `free` | Free | **No** | yfinance `.NS`. Prototyping only: patchy, no filing dates. |
| `csv` | — | Depends | Your own export at `python/data/fundamentals.csv`. |

Start with `nse`; move to Prowess or EODHD if you need deep history immediately rather than
accumulating it.

**Point-in-time discipline:** a fundamental is only usable from its *announcement date*. When a
provider omits filing dates we approximate with `period_end + reporting_lag_days` and log a warning.
That approximation is fine for prototyping and **not** fine for a final backtest.

### 3. IIMA factor library (for true-alpha attribution)

Download the Indian Fama-French-Momentum factors from
[faculty.iima.ac.in/iffm](https://faculty.iima.ac.in/iffm/Indian-Fama-French-Momentum/) and save as
`python/data/iima_factors.csv`. Without it you only get CAPM alpha versus the benchmark, which is
**not** the same thing. If the library has no QMJ series, the framework builds one in-repo from your
fundamentals.

---

## Daily workflow

```bash
cd python

# 1. Refresh data and regenerate signals (schedule after market close)
python scripts/update_daily.py

# 2. Produce the manual execution checklist
python scripts/generate_rebalance.py --preset full_composite
```

`artifacts/rebalance_orders.csv` is the deliverable: symbol, action, current qty, target qty, delta,
price, estimated cost, and a warning when an order is large relative to ADV. **You place these orders
yourself.**

---

## Research workflow

```bash
# Compare against the literature before optimizing anything
python scripts/run_backtest.py --preset quality_only
python scripts/run_backtest.py --preset momentum_only     # expect deep drawdowns
python scripts/run_backtest.py --preset low_vol_only      # expect the best Sharpe

# Walk-forward optimization (out-of-sample selection, turnover-penalized)
python scripts/run_optimization.py --preset full_composite --method random --n 40

# Full statistical battery, including data-snooping tests across presets
python scripts/run_validation.py --preset full_composite --compare-presets

# Highest-fidelity check of a chosen design
python scripts/run_backtest.py --preset full_composite --engine both
```

### Why the optimizer reports two answers

It reports the **in-sample best** and the **plateau choice** — and recommends the plateau. A
configuration surrounded by other good configurations is far more likely to survive out-of-sample
than an isolated peak, which is usually noise. It also reports **PBO** (probability of backtest
overfitting) so you can see how much to distrust the search itself.

---

## How it avoids fooling itself

This is the part that matters most, because a backtest will happily tell you whatever you want.

**Data integrity**
- Point-in-time index membership (survivorship bias)
- Point-in-time fundamentals keyed on announcement date (look-ahead bias)
- ISIN-keyed security master tracking symbol changes, mergers and delistings
- Corporate-action adjustment and total-return series; benchmarks are **Total Return** indices
- A data-health report surfaced in the dashboard

**Execution realism**
- `t+1` execution: signals computed at the close of *t* trade on *t+1*
- Integer share quantities, an explicit cash balance, and per-order costs in the event engine
- Market impact scaling with order size versus ADV
- Capacity analysis bound by the *least liquid* position, not the average

**Statistical honesty**
- Sharpe standard errors corrected for autocorrelation, skew and kurtosis (Lo 2002)
- Block bootstrap confidence intervals that preserve time dependence
- **Deflated Sharpe** and **haircut Sharpe**, using the *real* trial count from the run registry
- **PBO** via combinatorially symmetric cross-validation
- **White's Reality Check** and **Hansen's SPA** for data snooping across candidates
- **MinTRL**: how long a track record must be before the Sharpe is believable
- Breakeven-cost analysis: the cost level at which the edge disappears
- Purged, embargoed walk-forward validation
- Leak detectors that must *fire* on deliberately leaked signals (tested)

**Two engines**
A fast vectorized engine does the searching; a realistic event-driven engine validates the final
design. A built-in consistency test fails loudly if they disagree.

---

## Dashboard

```bash
cd dashboard
cargo run --release
```

| View | Shows |
|---|---|
| Overview | Equity curve, drawdown, full metrics catalog, monthly returns |
| Backtest Lab | Compose a design, run it, watch progress live |
| Optimization | Sweeps, sensitivity, plateau recommendation, PBO |
| Validation | Every statistical test with a plain-language verdict |
| Attribution | True alpha with t-stats, factor betas, IC/ICIR |
| Holdings & Rebalance | Live holdings vs targets, the manual checklist |
| Rankings | Sortable factor scores |
| Stat-Arb / Events | Sleeve B pairs and signals; Sleeve C calendar and SUE |
| Risk | Exposure, drawdown, sector weights, cost sensitivity, regimes |
| Journal | Executed fills vs checklist, realized slippage, live-vs-backtest drift |
| Data Health | Data-quality issues and the run registry |
| Live | Streaming quotes |

**Security.** The process holds Kite credentials, so it binds to `127.0.0.1` by default and never
sends secrets to the browser. If you bind to a non-loopback address, set `DASHBOARD_AUTH_TOKEN` —
the server warns you at startup if you do not.

---

## Layout

```
artifacts/          # shared contract: Python writes, Rust reads (gitignored)
schema/             # strategy_config JSON Schema, validated by both sides
python/
  config.yaml       # environment config (paths, costs, taxes, universe)
  src/
    data/           # Kite, security master, corporate actions, universe, fundamentals, earnings
    factors/        # quality, value, momentum, low-vol, composite, sector-neutral, IC/ICIR
    portfolio/      # construction, rank buffer, regime overlay, vol scaling
    statarb/        # Sleeve B: residual reversion, cointegration, spread/Kalman, signals
    events/         # Sleeve C: calendar, SUE, PEAD signals
    risk/           # portfolio-level risk limits
    backtest/       # engines, costs, taxes, metrics, attribution, statistics, optimization, runner
    journal/        # tradebook import, reconciliation, live tracking
  scripts/          # fetch_data, update_daily, run_backtest, run_optimization, run_validation, generate_rebalance
  tests/
dashboard/          # Rust (Axum) cockpit + static frontend
```

Strategy *designs* live in `strategy_config.json` (validated by pydantic and the shared JSON Schema);
`config.yaml` holds *environment* settings. Keeping them separate is what lets the dashboard, the CLI
and the test suite all run the identical design.

---

## Testing

```bash
cd python
python -m pytest tests/ -v          # 80 tests
python -m pytest tests/ -m "not slow"
```

Coverage includes factor and engine tests on a recorded NSE slice (60 liquid names, 2022–2024),
parser tests against real UDiFF/legacy bhavcopy rows and a real XBRL filing, golden-number cost/tax
tests, leak detectors, engine consistency, and Python↔Rust artifact contract tests.

---

## Limitations, stated plainly

- **Backtests are not forecasts.** Indian factor research shows real out-of-sample decay.
- **Fundamentals coverage is the binding constraint** for Quality and Value. Free sources are not
  point-in-time; use EODHD or Prowess for anything you intend to act on.
- **Sleeve B is long-only, so it is not market-neutral.** Dropping the short leg leaves residual beta.
- **PEAD is weak in NSE large caps.** The edge concentrates in less liquid mid/small caps, where
  costs bite hardest.
- **Synthetic mode is for development**, not for investment decisions.
- The default cost and tax rates are configurable and **must be verified** against current rate cards
  and tax law before you rely on after-tax numbers.

---

## License

Private project. Educational and research use. Not investment advice.
