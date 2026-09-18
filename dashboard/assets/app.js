/* Indian Equity Alpha Framework - dashboard frontend.
 *
 * Deliberately dependency-light: vanilla JS plus uPlot for charts. The
 * backend hands us columnar arrays, so rendering is cheap even with long
 * daily histories.
 */

const state = {
  runId: null,
  view: 'overview',
  loaded: new Set(),
  charts: {},
  rankings: null,
  ticks: new Map(),
};

// ---------------------------------------------------------------- utilities
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

function pct(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return `${(v * 100).toFixed(digits)}%`;
}
function num(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  return Number(v).toFixed(digits);
}
function inr(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const a = Math.abs(v);
  if (a >= 1e7) return `₹${(v / 1e7).toFixed(2)} Cr`;
  if (a >= 1e5) return `₹${(v / 1e5).toFixed(2)} L`;
  return `₹${v.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
}
function signClass(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return '';
  return v > 0 ? 'pos' : v < 0 ? 'neg' : '';
}

async function api(path, options) {
  const url = state.runId && !path.includes('?')
    ? `/api${path}?run_id=${encodeURIComponent(state.runId)}`
    : `/api${path}`;
  const res = await fetch(url, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || body.hint || `HTTP ${res.status}`);
  return body;
}

async function post(path, payload) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
}

// ------------------------------------------------------------ table rendering
function renderTable(container, table, opts = {}) {
  const node = typeof container === 'string' ? $(container) : container;
  node.innerHTML = '';

  if (!table || !table.columns || table.rows.length === 0) {
    node.appendChild(el('div', 'empty', table?.note || 'No data available.'));
    return;
  }

  const wrap = el('div', 'table-wrap');
  const tbl = el('table');
  const thead = el('thead');
  const hrow = el('tr');

  const cols = opts.columns
    ? opts.columns.filter((c) => table.columns.includes(c))
    : table.columns;
  const idx = cols.map((c) => table.columns.indexOf(c));

  cols.forEach((c, i) => {
    const th = el('th', null, c.replace(/_/g, ' '));
    th.onclick = () => sortTable(node, table, opts, idx[i]);
    hrow.appendChild(th);
  });
  thead.appendChild(hrow);
  tbl.appendChild(thead);

  const tbody = el('tbody');
  const rows = opts.limit ? table.rows.slice(0, opts.limit) : table.rows;

  for (const row of rows) {
    const tr = el('tr');
    cols.forEach((c, i) => {
      const raw = row[idx[i]];
      const td = el('td');
      const fmt = opts.format && opts.format[c];
      if (fmt) {
        const out = fmt(raw, row, table.columns);
        if (out instanceof Node) td.appendChild(out);
        else td.innerHTML = out;
      } else if (typeof raw === 'number') {
        td.textContent = Number.isInteger(raw) ? raw : raw.toFixed(4);
        td.className = opts.colorize?.includes(c) ? signClass(raw) : '';
      } else {
        td.textContent = raw === null || raw === undefined ? '—' : String(raw);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
  tbl.appendChild(tbody);
  wrap.appendChild(tbl);
  node.appendChild(wrap);

  if (opts.limit && table.rows.length > opts.limit) {
    node.appendChild(el('div', 'hint', `Showing ${opts.limit} of ${table.rows.length} rows.`));
  }
}

function sortTable(node, table, opts, colIndex) {
  const dir = node._sortDir === colIndex ? -1 : 1;
  node._sortDir = dir === 1 ? colIndex : null;
  const sorted = {
    ...table,
    rows: [...table.rows].sort((a, b) => {
      const x = a[colIndex], y = b[colIndex];
      if (x === null) return 1;
      if (y === null) return -1;
      if (typeof x === 'number' && typeof y === 'number') return (y - x) * dir;
      return String(y).localeCompare(String(x)) * dir;
    }),
  };
  renderTable(node, sorted, opts);
}

function metricCard(label, value, cls) {
  const card = el('div', 'metric');
  card.appendChild(el('div', 'label', label));
  card.appendChild(el('div', `value ${cls || ''}`, value));
  return card;
}

// ------------------------------------------------------------------- charts
function drawChart(id, data, seriesNames, opts = {}) {
  const node = $(id);
  if (!node) return;
  node.innerHTML = '';

  if (!data || !data.t || data.t.length === 0) {
    node.appendChild(el('div', 'empty', 'No chart data.'));
    return;
  }

  const available = seriesNames.filter((n) => data.series[n]);
  if (available.length === 0) {
    node.appendChild(el('div', 'empty', 'No matching series.'));
    return;
  }

  const palette = ['#4c8dff', '#3ecf8e', '#f5a623', '#ff5f6d', '#b78bff'];
  const uData = [data.t, ...available.map((n) => data.series[n])];

  const chart = new uPlot({
    width: node.clientWidth || 900,
    height: opts.height || 320,
    padding: [12, 16, 0, 0],
    series: [
      { label: 'Date' },
      ...available.map((n, i) => ({
        label: n.replace(/_/g, ' '),
        stroke: palette[i % palette.length],
        width: 1.6,
        fill: opts.fill && i === 0 ? 'rgba(76,141,255,.12)' : undefined,
        value: (_, v) => (v === null ? '—' : opts.percent ? pct(v) : num(v, 2)),
      })),
    ],
    axes: [
      { stroke: '#9aa3b2', grid: { stroke: '#2a2f3a', width: 0.5 } },
      {
        stroke: '#9aa3b2',
        grid: { stroke: '#2a2f3a', width: 0.5 },
        values: (_, ticks) => ticks.map((v) => (opts.percent ? pct(v, 0) : compact(v))),
      },
    ],
    legend: { live: true },
  }, uData, node);

  state.charts[id] = chart;
}

function compact(v) {
  if (v === null || v === undefined) return '';
  const a = Math.abs(v);
  if (a >= 1e7) return `${(v / 1e7).toFixed(1)}Cr`;
  if (a >= 1e5) return `${(v / 1e5).toFixed(1)}L`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(0)}k`;
  return v.toFixed(0);
}

// -------------------------------------------------------------------- views
const views = {
  async overview() {
    const data = await api('/backtest');
    const m = data.metrics?.sleeves?.combined || {};

    const grid = $('#overview-metrics');
    grid.innerHTML = '';
    [
      ['CAGR', pct(m.cagr), signClass(m.cagr)],
      ['Sharpe', num(m.sharpe), signClass(m.sharpe)],
      ['Sortino', num(m.sortino), ''],
      ['Max drawdown', pct(m.max_drawdown), 'neg'],
      ['Volatility', pct(m.volatility), ''],
      ['Calmar', num(m.calmar), ''],
      ['Benchmark CAGR', pct(m.benchmark_cagr), ''],
      ['Excess return', pct(m.excess_return), signClass(m.excess_return)],
      ['Info ratio', num(m.information_ratio), signClass(m.information_ratio)],
      ['Turnover', num(m.turnover), ''],
      ['Recovery (days)', m.recovery_days ? num(m.recovery_days, 0) : '—', ''],
      ['After-tax CAGR', pct(m.after_tax_cagr), signClass(m.after_tax_cagr)],
    ].forEach(([l, v, c]) => grid.appendChild(metricCard(l, v, c)));

    const curve = data.equity_curve?.data;
    drawChart('#equity-chart', curve, ['equity', 'after_tax_equity', 'benchmark'], { fill: true });
    drawChart('#drawdown-chart', curve, ['drawdown'], { percent: true, height: 200 });

    try {
      renderTable('#monthly-table', await api('/monthly'));
    } catch {
      $('#monthly-table').innerHTML = '<div class="empty">No monthly table.</div>';
    }
  },

  async optimization() {
    const data = await api('/optimize');
    const s = data.summary || {};
    const box = $('#optimization-summary');
    box.innerHTML = '<h2>Summary</h2>';

    if (!s || Object.keys(s).length === 0) {
      box.appendChild(el('div', 'empty', 'No optimization has been run yet.'));
    } else {
      const grid = el('div', 'metric-grid');
      grid.appendChild(metricCard('Trials', s.n_trials ?? '—'));
      grid.appendChild(metricCard('Best score', num(s.best_score)));
      grid.appendChild(metricCard('Plateau score', num(s.plateau_score)));
      grid.appendChild(metricCard('Free params', s.degrees_of_freedom ?? '—'));
      box.appendChild(grid);
      box.appendChild(el('div', 'hint',
        `Plateau config (recommended): ${JSON.stringify(s.plateau_params || {})}`));
      box.appendChild(el('div', 'hint',
        `In-sample best: ${JSON.stringify(s.best_params || {})}`));
    }
    renderTable('#optimization-table', data.results, { limit: 100, colorize: ['score', 'sharpe'] });
  },

  async validation() {
    const v = await api('/validation');
    const box = $('#validation-verdict');
    box.innerHTML = '<h2>Verdict</h2>';

    const verdict = v.verdict || 'No validation available.';
    const cls = verdict.startsWith('PASS') ? 'pass'
      : verdict.startsWith('CAUTION') ? 'caution'
      : verdict.startsWith('FAIL') ? 'fail' : '';
    const vb = el('div', `verdict ${cls}`, verdict);
    if (v.flags?.length) {
      const ul = el('ul', 'flags');
      v.flags.forEach((f) => ul.appendChild(el('li', null, f)));
      vb.appendChild(ul);
    }
    box.appendChild(vb);

    const rows = [];
    const push = (test, metric, value, note) => rows.push([test, metric, value, note || '']);
    const st = v.sharpe_test || {};
    if (st.sharpe != null) {
      push('Sharpe t-test', 'Sharpe', num(st.sharpe),
        `95% CI [${num(st.ci_lower)}, ${num(st.ci_upper)}], p=${num(st.p_value, 4)}`);
    }
    const bs = v.bootstrap || {};
    if (bs.ci_lower != null) {
      push('Block bootstrap', 'CI', `[${num(bs.ci_lower)}, ${num(bs.ci_upper)}]`,
        `P(Sharpe>0) = ${pct(bs.prob_positive)}`);
    }
    const dsr = v.deflated_sharpe || {};
    if (dsr.dsr != null) {
      push('Deflated Sharpe', 'DSR', num(dsr.dsr, 3),
        `${dsr.n_trials} trials; >0.95 desired`);
    }
    const hc = v.haircut || {};
    if (hc.haircut_sharpe != null) {
      push('Multiple-testing haircut', 'Adj. Sharpe', num(hc.haircut_sharpe),
        `cut ${num(hc.haircut_pct, 0)}%`);
    }
    const trl = v.min_trl || {};
    if (trl.min_track_record_years != null) {
      push('MinTRL', 'Years needed', num(trl.min_track_record_years, 1), '');
    }
    if (v.pbo?.pbo != null) push('PBO (CSCV)', 'PBO', num(v.pbo.pbo, 3), v.pbo.interpretation);
    if (v.spa?.p_value != null) push('Hansen SPA', 'p-value', num(v.spa.p_value, 4), '');
    if (v.reality_check?.p_value != null) {
      push('White Reality Check', 'p-value', num(v.reality_check.p_value, 4), '');
    }
    if (v.leaks?.note) push('Leak detection', '', '', v.leaks.note);

    renderTable('#validation-tests', {
      columns: ['test', 'metric', 'value', 'note'],
      rows, n_rows: rows.length,
    });
  },

  async attribution() {
    const data = await api('/attribution');
    const a = data.attribution || {};
    const fm = a.factor_model;
    const box = $('#attribution-summary');
    box.innerHTML = '<h2>True alpha</h2>';

    if (!fm) {
      box.appendChild(el('div', 'empty',
        a.note || 'Factor attribution unavailable (IIMA factor library not loaded).'));
      if (a.capm) {
        const g = el('div', 'metric-grid');
        g.appendChild(metricCard('CAPM alpha (ann.)', pct(a.capm.alpha_annualized),
          signClass(a.capm.alpha_annualized)));
        g.appendChild(metricCard('Beta', num(a.capm.beta)));
        g.appendChild(metricCard('R²', num(a.capm.r_squared)));
        box.appendChild(g);
        box.appendChild(el('div', 'hint',
          'This is excess return vs the benchmark, NOT alpha net of factor exposure.'));
      }
    } else {
      const g = el('div', 'metric-grid');
      g.appendChild(metricCard('Alpha (annualized)', pct(fm.alpha_annualized),
        signClass(fm.alpha_annualized)));
      g.appendChild(metricCard('t-statistic', num(fm.alpha_tstat)));
      g.appendChild(metricCard('p-value', num(fm.alpha_pvalue, 4)));
      g.appendChild(metricCard('Significant (5%)', fm.alpha_significant_5pct ? 'Yes' : 'No',
        fm.alpha_significant_5pct ? 'pos' : 'neg'));
      g.appendChild(metricCard('R²', num(fm.adj_r_squared)));
      g.appendChild(metricCard('Observations', fm.n_obs));
      box.appendChild(g);
      box.appendChild(el('div', 'hint',
        'Alpha is the regression intercept after paying for market, size, value, momentum and quality exposure.'));

      const rows = Object.entries(fm.betas || {}).map(([k, v]) => [
        k, num(v), num(fm.beta_tstats?.[k]), num(fm.beta_pvalues?.[k], 4),
      ]);
      renderTable('#attribution-betas', {
        columns: ['factor', 'beta', 't_stat', 'p_value'], rows, n_rows: rows.length,
      });
    }

    renderTable('#ic-table', data.ic_icir || { columns: [], rows: [], note: 'No IC data.' });
  },

  async holdings() {
    const reb = await api('/rebalance');
    const s = reb.summary || {};
    const box = $('#rebalance-summary');
    box.innerHTML = '<h2>Rebalance summary</h2>';

    if (!s || Object.keys(s).length === 0) {
      box.appendChild(el('div', 'empty', 'No rebalance generated yet.'));
    } else {
      const g = el('div', 'metric-grid');
      g.appendChild(metricCard('Orders', s.n_orders ?? '—'));
      g.appendChild(metricCard('Buys', s.n_buys ?? '—', 'pos'));
      g.appendChild(metricCard('Sells', s.n_sells ?? '—', 'neg'));
      g.appendChild(metricCard('Portfolio value', inr(s.portfolio_value)));
      g.appendChild(metricCard('Est. cost', inr(s.est_total_cost)));
      g.appendChild(metricCard('Turnover', `${num(s.turnover_pct, 1)}%`));
      box.appendChild(g);
    }

    renderTable('#orders-table', reb.orders, {
      columns: ['symbol', 'action', 'current_qty', 'target_qty', 'delta_qty',
                'price', 'target_weight', 'delta_value', 'est_cost', 'notes'],
      format: {
        action: (v) => {
          const t = el('span', `tag ${String(v).toLowerCase()}`, v);
          return t;
        },
        delta_value: (v) => `<span class="${signClass(v)}">${inr(v)}</span>`,
        target_weight: (v) => pct(v),
      },
    });

    try {
      const h = await api('/holdings');
      const node = $('#holdings-table');
      if (h.note) {
        node.innerHTML = '';
        node.appendChild(el('div', 'empty', h.note));
      } else {
        const rows = h.holdings.map((x) => [
          x.tradingsymbol, x.quantity, x.average_price, x.last_price, x.pnl,
          h.targets[x.tradingsymbol] ?? null,
        ]);
        renderTable(node, {
          columns: ['symbol', 'qty', 'avg_price', 'last_price', 'pnl', 'target_weight'],
          rows, n_rows: rows.length,
        }, { colorize: ['pnl'], format: { target_weight: (v) => (v == null ? '—' : pct(v)) } });
      }
    } catch (e) {
      $('#holdings-table').innerHTML = `<div class="empty">${e.message}</div>`;
    }
  },

  async rankings() {
    const data = await api('/rankings');
    state.rankings = data;
    renderTable('#rankings-table', data, {
      limit: 200,
      columns: ['rank', 'symbol', 'sector', 'composite', 'quality', 'value', 'momentum', 'low_vol'],
      colorize: ['composite', 'quality', 'value', 'momentum', 'low_vol'],
    });
  },

  async statarb() {
    const data = await api('/statarb');
    renderTable('#pairs-table', data.pairs, { limit: 100 });
    renderTable('#statarb-signals-table', data.signals, { limit: 200 });
  },

  async events() {
    const data = await api('/events');
    renderTable('#calendar-table', data.calendar, { limit: 200 });
    renderTable('#event-signals-table', data.signals, { limit: 200, colorize: ['sue'] });
  },

  async risk() {
    const data = await api('/risk');
    const r = data.risk?.latest || {};
    const grid = $('#risk-metrics');
    grid.innerHTML = '';
    grid.appendChild(metricCard('Gross exposure', pct(r.gross_exposure)));
    grid.appendChild(metricCard('Cash', pct(r.cash_weight)));
    grid.appendChild(metricCard('Drawdown', pct(r.drawdown), 'neg'));
    grid.appendChild(metricCard('Exposure multiplier', num(r.exposure_multiplier)));
    grid.appendChild(metricCard('Max position', pct(r.concentration?.max_position)));
    grid.appendChild(metricCard('Top-5 weight', pct(r.concentration?.top5)));
    grid.appendChild(metricCard('Effective N', num(r.concentration?.effective_n, 1)));
    const be = data.risk?.breakeven_cost_bps;
    grid.appendChild(metricCard('Breakeven cost', be ? `${num(be, 0)} bps` : '—'));

    const sectors = Object.entries(r.sector_weights || {})
      .sort((a, b) => b[1] - a[1])
      .map(([k, v]) => [k, v]);
    renderTable('#sector-table', {
      columns: ['sector', 'weight'], rows: sectors, n_rows: sectors.length,
    }, { format: { weight: (v) => pct(v) } });

    renderTable('#cost-table', data.cost_sensitivity || { columns: [], rows: [] });
    renderTable('#regime-table', data.regime || { columns: [], rows: [] },
      { colorize: ['annualized', 'sharpe'] });
  },

  async journal() {
    const data = await api('/journal');
    const box = $('#drift-summary');
    box.innerHTML = '<h2>Live vs backtest</h2>';
    const d = data.drift;
    if (!d || d === null) {
      box.appendChild(el('div', 'empty',
        'No live tracking yet. Import your tradebook to compare execution against the model.'));
    } else {
      const g = el('div', 'metric-grid');
      g.appendChild(metricCard('Live return', pct(d.live_return), signClass(d.live_return)));
      g.appendChild(metricCard('Expected', pct(d.expected_return)));
      g.appendChild(metricCard('Shortfall', pct(d.implementation_shortfall),
        signClass(d.implementation_shortfall)));
      g.appendChild(metricCard('Tracking error', pct(d.tracking_error)));
      g.appendChild(metricCard('Slippage', d.realized_slippage_bps ? `${num(d.realized_slippage_bps, 1)} bps` : '—'));
      box.appendChild(g);
      (d.alerts || []).forEach((a) => box.appendChild(el('div', 'hint warn', a)));
    }
    renderTable('#journal-table', data.entries, { limit: 200, colorize: ['slippage_bps'] });
  },

  async health() {
    const h = await api('/health');
    const dh = h.data_health || {};
    const box = $('#health-summary');
    box.innerHTML = '<h2>Status</h2>';
    const g = el('div', 'metric-grid');
    g.appendChild(metricCard('Errors', dh.n_errors ?? 0, dh.n_errors ? 'neg' : 'pos'));
    g.appendChild(metricCard('Warnings', dh.n_warnings ?? 0, dh.n_warnings ? '' : 'pos'));
    g.appendChild(metricCard('Coverage', dh.coverage_pct ? `${dh.coverage_pct}%` : '—'));
    g.appendChild(metricCard('Instruments', dh.n_instruments ?? '—'));
    g.appendChild(metricCard('Kite', h.kite ?? '—'));
    g.appendChild(metricCard('Latest run', h.latest_run ? h.latest_run.slice(0, 15) : '—'));
    box.appendChild(g);

    const issues = (dh.issues || []).map((i) => [i.severity, i.check, i.message, i.count]);
    renderTable('#health-issues', {
      columns: ['severity', 'check', 'message', 'count'], rows: issues, n_rows: issues.length,
    });

    const runs = await api('/runs');
    renderTable('#runs-table', runs.registry, { limit: 50 });
  },

  async live() {
    $('#live-status').textContent = state.ws?.readyState === 1
      ? 'Streaming live ticks from Zerodha.'
      : 'Not connected. Live quotes require a Kite session and market hours.';
    renderTicks();
  },

  async lab() { /* static form; nothing to load */ },
};

function renderTicks() {
  const rows = [...state.ticks.values()].map((t) => [
    t.instrument_token, t.last_price, t.change ?? null, t.volume ?? null,
  ]);
  renderTable('#ticks-table', {
    columns: ['instrument_token', 'last_price', 'change_pct', 'volume'],
    rows, n_rows: rows.length,
    note: 'No ticks received yet.',
  }, { colorize: ['change_pct'] });
}

// -------------------------------------------------------------- app wiring
async function switchView(name) {
  state.view = name;
  document.querySelectorAll('.tabs button').forEach((b) =>
    b.classList.toggle('active', b.dataset.view === name));
  document.querySelectorAll('.view').forEach((v) =>
    v.classList.toggle('active', v.id === `view-${name}`));

  if (views[name]) {
    try {
      await views[name]();
    } catch (e) {
      const node = $(`#view-${name}`);
      const existing = node.querySelector('.load-error');
      if (existing) existing.remove();
      const err = el('div', 'card load-error');
      err.appendChild(el('div', 'empty', e.message));
      node.prepend(err);
    }
  }
}

async function loadRuns() {
  try {
    const runs = await api('/runs');
    state.runId = state.runId || runs.latest;
    const sel = $('#run-select');
    sel.innerHTML = '';
    (runs.run_ids || []).forEach((id) => {
      const o = el('option', null, id === runs.latest ? `${id} (latest)` : id);
      o.value = id;
      if (id === state.runId) o.selected = true;
      sel.appendChild(o);
    });
    $('#run-label').textContent = state.runId ? `run ${state.runId}` : 'no runs yet';
  } catch {
    $('#run-label').textContent = 'artifacts unavailable';
  }
}

async function loadStatus() {
  try {
    const h = await api('/health');
    const badge = $('#kite-status');
    badge.textContent = `Kite: ${h.kite}`;
    badge.className = `badge ${h.kite === 'authenticated' ? 'ok' : h.kite === 'needs login' ? 'err' : ''}`;
  } catch { /* status is best-effort */ }
}

function connectSockets() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';

  const jobs = new WebSocket(`${proto}://${location.host}/ws/jobs`);
  const log = $('#job-log');
  jobs.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === 'snapshot') return;
    if (log.textContent.startsWith('Waiting')) log.textContent = '';
    log.textContent += `[${msg.status}] ${msg.message}\n`;
    log.scrollTop = log.scrollHeight;
    if (msg.status === 'completed') {
      if (msg.run_id) state.runId = msg.run_id;
      loadRuns().then(() => switchView(state.view));
    }
  };

  state.ws = new WebSocket(`${proto}://${location.host}/ws/ticks`);
  state.ws.onmessage = (ev) => {
    const tick = JSON.parse(ev.data);
    if (tick.type === 'connected') return;
    state.ticks.set(tick.instrument_token, tick);
    if (state.view === 'live') renderTicks();
  };
  state.ws.onopen = () => {
    if (state.view === 'live') $('#live-status').textContent = 'Streaming live ticks from Zerodha.';
  };
}

function wireControls() {
  document.querySelectorAll('.tabs button').forEach((b) => {
    b.onclick = () => switchView(b.dataset.view);
  });

  $('#refresh-btn').onclick = () => switchView(state.view);
  $('#run-select').onchange = (e) => {
    state.runId = e.target.value;
    $('#run-label').textContent = `run ${state.runId}`;
    switchView(state.view);
  };

  const body = () => ({
    preset: $('#lab-preset').value,
    engine: $('#lab-engine').value,
    source: $('#lab-source').value || null,
    validate: $('#lab-validate').checked,
  });

  const launch = async (path, btn) => {
    btn.disabled = true;
    $('#job-log').textContent = 'Starting…\n';
    try {
      await post(path, body());
    } catch (e) {
      $('#job-log').textContent += `ERROR: ${e.message}\n`;
    } finally {
      setTimeout(() => { btn.disabled = false; }, 1500);
    }
  };

  $('#run-backtest').onclick = (e) => launch('/backtest/run', e.target);
  $('#run-optimize').onclick = (e) => launch('/optimize/run', e.target);
  $('#run-validate').onclick = (e) => launch('/validation/run', e.target);
  $('#run-rebalance').onclick = (e) => launch('/rebalance/run', e.target);

  $('#rankings-filter').oninput = (e) => {
    if (!state.rankings) return;
    const q = e.target.value.toLowerCase();
    const filtered = {
      ...state.rankings,
      rows: state.rankings.rows.filter((r) =>
        r.some((c) => String(c).toLowerCase().includes(q))),
    };
    renderTable('#rankings-table', filtered, {
      limit: 200,
      columns: ['rank', 'symbol', 'sector', 'composite', 'quality', 'value', 'momentum', 'low_vol'],
      colorize: ['composite', 'quality', 'value', 'momentum', 'low_vol'],
    });
  };

  // Redraw charts on resize (uPlot needs an explicit size).
  let t;
  window.addEventListener('resize', () => {
    clearTimeout(t);
    t = setTimeout(() => {
      if (state.view === 'overview') views.overview().catch(() => {});
    }, 250);
  });
}

(async function init() {
  wireControls();
  await loadRuns();
  await loadStatus();
  connectSockets();
  await switchView('overview');
})();
