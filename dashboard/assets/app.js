/* Indian Equity Alpha Framework - dashboard frontend.
 *
 * Vanilla JS plus D3 for charts. The backend hands us columnar arrays
 * (unix seconds + named series); D3 turns those into interactive SVG.
 */

const state = {
  runId: null,
  view: 'overview',
  loaded: new Set(),
  charts: {},
  resizeObs: null,
  rankings: null,
  screener: null,
  screenerMeta: { top_n: 40, buffer_n: 80 },
  screenerPage: 0,
  screenerPageSize: 80,
  screenerPreset: 'universe',
  activePresets: new Set(),
  sparksFile: null,
  screenerSet: 'overview',
  screenerFiltered: null,
  backtestMeta: null,
  paperSignals: null,
  historyCache: new Map(),
  msSelected: { exchange: new Set(), country: new Set(), sector: new Set(), industry: new Set() },
  ticks: new Map(),
  scanner: {
    threshold: 0.03,
    lit: new Set(),
    alerts: [],
    lastClose: new Map(),
    tokenToSymbol: new Map(),
    overlay: new Map(),
    ticks: 0,
    lastAt: null,
  },
};

function screenerPageSize() {
  return parseInt(document.getElementById('screener-page-size')?.value, 10) || state.screenerPageSize || 80;
}

const SCREENER_RANGE_IDS = [
  'screener-rank-min', 'screener-rank-max',
  'screener-comp-min', 'screener-comp-max',
  'screener-mom-min', 'screener-mom-max',
  'screener-q-min', 'screener-q-max',
  'screener-v-min', 'screener-v-max',
  'screener-lv-min', 'screener-lv-max',
  'screener-prof-min', 'screener-prof-max',
  'screener-growth-min', 'screener-growth-max',
  'screener-safety-min', 'screener-safety-max',
  'screener-payoutz-min', 'screener-payoutz-max',
];

const SCREENER_SELECT_IDS = [
  'screener-mcap-min', 'screener-mcap-max',
  'screener-pe-min', 'screener-pe-max',
  'screener-div-min', 'screener-div-max',
  'screener-price-min', 'screener-price-max',
  'screener-pb-min', 'screener-pb-max',
  'screener-de-min', 'screener-de-max',
  'screener-roe-min', 'screener-roe-max',
  'screener-ey-min', 'screener-ey-max',
  'screener-eps-min', 'screener-eps-max',
  'screener-rev-min', 'screener-rev-max',
  'screener-ni-min', 'screener-ni-max',
  'screener-payout-min', 'screener-payout-max',
  'screener-nm-min', 'screener-nm-max',
  'screener-r1d-min', 'screener-r1d-max',
  'screener-r1w-min', 'screener-r1w-max',
  'screener-r1m-min', 'screener-r1m-max',
  'screener-r3m-min', 'screener-r3m-max',
  'screener-r6m-min', 'screener-r6m-max',
  'screener-r1y-min', 'screener-r1y-max',
  'screener-r3y-min', 'screener-r3y-max',
  'screener-tick-price-min', 'screener-tick-price-max',
  'screener-tick-chg-min', 'screener-tick-chg-max',
  'screener-vol-min', 'screener-vol-max',
];

const MACRO_PAD = [
  'name', 'exchange', 'country', 'industry',
  'close', 'pe', 'pb', 'market_cap', 'roe', 'debt_equity', 'div_yield', 'payout',
  'ret_1w', 'ret_1m', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y',
  'revenue', 'eps', 'net_income', 'earnings_yield',
  'profitability', 'growth', 'safety', 'payout_z', 'net_margin',
  'pretax_margin', 'gross_profitability',
];

const CR = 1e7;
const RANGE_PRESETS = {
  mcap: [[50*CR,'₹50 Cr'],[100*CR,'₹100 Cr'],[300*CR,'₹300 Cr'],[500*CR,'₹500 Cr'],[1000*CR,'₹1,000 Cr'],[2000*CR,'₹2,000 Cr'],[5000*CR,'₹5,000 Cr'],[10000*CR,'₹10,000 Cr'],[25000*CR,'₹25,000 Cr'],[50000*CR,'₹50,000 Cr'],[1e12,'₹1 Lakh Cr']],
  pe: [5,10,15,20,25,30,40,50,75,100].map((n) => [n, String(n)]),
  div: [0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.10].map((n) => [n, `${n * 100}%`]),
  price: [5,10,20,50,100,200,500,1000,2000,5000].map((n) => [n, `₹${n}`]),
  pb: [0.5,1,1.5,2,3,4,5,10].map((n) => [n, String(n)]),
  de: [0.1,0.2,0.5,1,1.5,2,3,5].map((n) => [n, String(n)]),
  roe: [0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50].map((n) => [n, `${n * 100}%`]),
  ey: [0.02,0.04,0.06,0.08,0.10,0.15,0.20].map((n) => [n, `${n * 100}%`]),
  eps: [1,5,10,20,50,100,200].map((n) => [n, `₹${n}`]),
  rev: [[100*CR,'₹100 Cr'],[500*CR,'₹500 Cr'],[1000*CR,'₹1,000 Cr'],[5000*CR,'₹5,000 Cr'],[10000*CR,'₹10,000 Cr'],[50000*CR,'₹50,000 Cr']],
  payout: [0.1,0.2,0.3,0.4,0.5,0.7,1].map((n) => [n, `${n * 100}%`]),
  r1d: [-10,-5,-3,0,3,5,10,20].map((n) => [n, `${n}%`]),
  ret: [-0.50,-0.25,-0.10,-0.05,0,0.05,0.10,0.25,0.50,1].map((n) => [n, `${(n * 100).toFixed(0)}%`]),
  vol: [[50000,'50k'],[1e5,'1 L'],[5e5,'5 L'],[1e6,'10 L'],[2e6,'20 L'],[5e6,'50 L']],
};

const COLUMN_SETS = {
  overview: { label: 'Overview', cols: ['name','symbol','last_price','ret_1y','pe','earnings_yield','net_margin','revenue','net_income','action'] },
  income: { label: 'Income Ratios', cols: ['name','symbol','last_price','pe','earnings_yield','roe','eps','revenue','net_income','net_margin','pretax_margin','gross_profitability'] },
  descriptive: { label: 'Descriptive', cols: ['name','symbol','exchange','country','sector','industry','asset_class','isin'] },
  dividends: { label: 'Dividends', cols: ['name','symbol','div_yield','payout'] },
  perf_st: { label: 'Performance (Short-Term)', cols: ['name','symbol','last_price','day_chg','ret_1w','ret_1m','ret_3m'] },
  perf_lt: { label: 'Performance (Long-Term)', cols: ['name','symbol','last_price','ret_6m','ret_1y','ret_3y'] },
  debt: { label: 'Debt Ratios', cols: ['name','symbol','debt_equity','pb'] },
  revenue: { label: 'Revenue & Earnings', cols: ['name','symbol','last_price','revenue','eps','net_income','net_margin'] },
  factors: { label: 'Factors', cols: ['action','name','symbol','rank','composite','momentum','quality','value','low_vol','held','reason'] },
  ticker: { label: 'Ticker', cols: ['name','symbol','last_price','prev_close','day_chg','volume','action','rank'] },
  iima: { label: 'IIMA Quality', cols: ['name','symbol','last_price','pe','market_cap','profitability','growth','safety','payout_z','net_margin'] },
};

const COL_HEADERS = {
  name: 'Stock Name',
  symbol: 'Ticker',
  industry: 'Industry',
  market_cap: 'Market Cap (₹ Cr)',
  last_price: 'Closing Price',
  ret_1y: '1 Year % Change',
  pe: 'P/E Ratio',
  div_yield: 'Dividend Yield',
  exchange: 'Exchange',
  country: 'Country',
  sector: 'Sector',
  asset_class: 'Asset',
  isin: 'ISIN',
  payout: 'Payout',
  day_chg: '1 Day %',
  ret_1w: '1 Week %',
  ret_1m: '1 Month %',
  ret_3m: '3 Month %',
  ret_6m: '6 Month %',
  ret_3y: '3 Year %',
  roe: 'ROE',
  earnings_yield: 'Earnings Yield',
  eps: 'EPS',
  net_income: 'Net Income (₹ Cr)',
  revenue: 'Revenue (₹ Cr)',
  debt_equity: 'Debt / Equity',
  pb: 'Price / Book',
  prev_close: 'Prev Close',
  volume: 'Volume',
  profitability: 'Profitability z',
  growth: 'Growth z',
  safety: 'Safety z',
  payout_z: 'Payout z',
  net_margin: 'Net Margin',
  pretax_margin: 'Pretax Margin',
  gross_profitability: 'Gross Profitability',
};

// ---------------------------------------------------------------- utilities
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

function finiteNum(v) {
  if (v === null || v === undefined || v === '') return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function positiveNum(v) {
  const n = finiteNum(v);
  return n != null && n > 0 ? n : null;
}
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
  if (window.StaticAPI) {
    await window.StaticAPI.ready;
    if (window.StaticAPI.mode === 'static') {
      return window.StaticAPI.handle(path, options);
    }
  }
  let url = `/api${path}`;
  if (state.runId && !path.includes('run_id=')) {
    url += (path.includes('?') ? '&' : '?') + `run_id=${encodeURIComponent(state.runId)}`;
  }
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
    const th = el('th', null, (opts.headers && opts.headers[c]) || c.replace(/_/g, ' '));
    th.onclick = () => sortTable(node, table, opts, idx[i]);
    hrow.appendChild(th);
  });
  thead.appendChild(hrow);
  tbl.appendChild(thead);

  const tbody = el('tbody');
  const start = opts.offset || 0;
  const rows = opts.limit ? table.rows.slice(start, start + opts.limit) : table.rows;
  const keyIdx = opts.rowKey ? table.columns.indexOf(opts.rowKey) : -1;

  for (const row of rows) {
    const tr = el('tr');
    if (keyIdx >= 0) tr.dataset.symbol = String(row[keyIdx] || '');
    if (typeof opts.rowClass === 'function') {
      const extra = opts.rowClass(row, table.columns);
      if (extra) tr.classList.add(extra);
    }
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

  if (opts.limit && table.rows.length > opts.limit && !opts.hideCount) {
    const shown = Math.min(start + opts.limit, table.rows.length);
    node.appendChild(el('div', 'hint', `Showing ${start + 1}–${shown} of ${table.rows.length} rows.`));
  }
}

function sortTable(node, table, opts, colIndex) {
  const dir = node._sortDir === colIndex ? -1 : 1;
  node._sortDir = dir === 1 ? colIndex : null;
  if (opts && 'offset' in opts) {
    opts.offset = 0;
    if (node && node.id === 'screener-table') {
      state.screenerPage = 0;
      const pages = Math.max(1, Math.ceil(table.rows.length / screenerPageSize()));
      const label = $('#screener-page-label');
      if (label) label.textContent = `Page 1 of ${pages}`;
      const prev = $('#screener-prev');
      const next = $('#screener-next');
      if (prev) prev.disabled = true;
      if (next) next.disabled = pages <= 1;
    }
  }
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

function mergeTables(a, b) {
  const cols = (a && a.columns && a.columns.length) ? a.columns
    : (b && b.columns) ? b.columns : [];
  const rows = [...(a && a.rows ? a.rows : []), ...(b && b.rows ? b.rows : [])];
  return {
    columns: cols,
    rows,
    n_rows: rows.length,
    note: rows.length ? undefined : (a && a.note) || (b && b.note) || 'No names in this list.',
  };
}

function parseBound(id) {
  const n = parseFloat(document.getElementById(id)?.value);
  return Number.isFinite(n) ? n : null;
}

function inFactorRange(raw, minId, maxId) {
  const lo = parseBound(minId);
  const hi = parseBound(maxId);
  if (lo === null && hi === null) return true;
  if (raw === null || raw === undefined || raw === '') return false;
  const x = Number(raw);
  if (!Number.isFinite(x)) return false;
  if (lo !== null && x < lo) return false;
  if (hi !== null && x > hi) return false;
  return true;
}

function factorWeights() {
  const f = state.backtestMeta?.factors || {};
  return {
    mom: Number(f.weight_momentum) || 0,
    lv: Number(f.weight_low_vol) || 0,
    q: Number(f.weight_quality) || 0,
    v: Number(f.weight_value) || 0,
  };
}

function winningPresets() {
  const top = Number(state.screenerMeta.top_n) || 40;
  const buf = Number(state.screenerMeta.buffer_n) || top * 2;
  const w = factorWeights();
  const m = state.backtestMeta || {};
  const minPx = Number(m.min_price) || 0;
  const parts = [];
  const set = { 'rank-max': String(top) };
  if (w.mom > 0) { parts.push(`momentum ${Math.round(w.mom * 100)}%`); set['mom-min'] = '0'; }
  if (w.lv > 0) { parts.push(`low-vol ${Math.round(w.lv * 100)}%`); set['lv-min'] = '0'; }
  if (w.q > 0) { parts.push(`quality ${Math.round(w.q * 100)}%`); set['q-min'] = '0'; }
  if (w.v > 0) { parts.push(`value ${Math.round(w.v * 100)}%`); set['v-min'] = '0'; }
  const sharpe = Number(m.sharpe);
  const list = [
    {
      id: 'winner',
      label: 'Winning recipe',
      hint: `${m.name || 'this run'}: ${parts.join(' + ') || 'rank buffer'} · rank ≤ ${top}`
        + (Number.isFinite(sharpe) ? ` · Sharpe ${sharpe.toFixed(2)}` : ''),
      set, colset: 'factors', rail: 'backtest', winner: true,
    },
    { id: 'buy', label: 'Rank-buffer BUY', hint: `Enter the top ${top} when a slot is free`,
      set: { action: 'BUY' }, colset: 'factors', rail: 'backtest', winner: true },
    { id: 'hold', label: 'HOLD in buffer', hint: `Keep ranks ${top + 1}–${buf} to cut turnover`,
      set: { action: 'HOLD' }, colset: 'factors', rail: 'backtest', winner: true },
    { id: 'sell', label: 'Rank-buffer SELL', hint: `Held names past rank ${buf}`,
      set: { action: 'SELL' }, colset: 'factors', rail: 'backtest', winner: true },
    { id: 'entry', label: `Entry zone (top ${top})`, hint: 'Jegadeesh/Titman skip-month rank window',
      set: { 'rank-max': String(top) }, colset: 'factors', rail: 'backtest', winner: true },
    { id: 'buffer', label: `Buffer band (${top + 1}–${buf})`, hint: 'Oscillation around top-N is ignored',
      set: { 'rank-min': String(top + 1), 'rank-max': String(buf) }, colset: 'factors', rail: 'backtest', winner: true },
  ];
  if (w.mom > 0) {
    list.push({
      id: 'momentum', label: `Momentum ${Math.round(w.mom * 100)}%`,
      hint: 'Skip-month 12–1 / 6–1 — the primary sleeve on this passing run',
      set: { 'mom-min': '0' }, colset: 'factors', rail: 'backtest', winner: true,
    });
  }
  if (w.lv > 0) {
    list.push({
      id: 'lowvol', label: `Low-vol ${Math.round(w.lv * 100)}%`,
      hint: 'Low-risk overlay — the second sleeve on this passing run',
      set: { 'lv-min': '0' }, colset: 'factors', rail: 'backtest', winner: true,
    });
  }
  if (minPx > 0) {
    list.push({
      id: 'minprice', label: `Price ≥ ₹${minPx}`,
      hint: 'Universe floor from the backtest — uses live last price',
      set: { 'tick-price-min': String(minPx) }, quoted: true, colset: 'ticker', rail: 'ticker', winner: true,
    });
  }
  return list;
}

function paperHint(s) {
  if (s.status === 'pass') {
    const bits = [];
    if (Number.isFinite(s.sharpe)) bits.push(`Sharpe ${s.sharpe.toFixed(2)}`);
    if (Number.isFinite(s.excess)) bits.push(`excess ${pct(s.excess)}`);
    if (Number.isFinite(s.cagr)) bits.push(`CAGR ${pct(s.cagr)}`);
    return `${s.paper}${bits.length ? ` · ${bits.join(' · ')}` : ''}`;
  }
  return s.note || s.paper || s.status;
}

function paperPresets(status) {
  const rows = state.paperSignals?.signals || [];
  return rows.filter((s) => s.status === status).map((s) => ({
    id: `paper-${s.id}`,
    label: status === 'pass' ? s.label
      : status === 'fail' ? `${s.label} (did not beat Nifty)`
      : `${s.label} (untested)`,
    hint: paperHint(s),
    set: s.filter || {},
    junk: !!s.junk,
    colset: s.colset || 'factors',
    rail: status === 'pass' ? 'backtest' : 'rules',
    winner: status === 'pass',
  }));
}

const LAUNCH_BOOKS = [
  { id: 'paper-profitability_only', kicker: 'CORE COMPOUNDER', fallback: 'IIMA profitability · hold ~12 months' },
  { id: 'paper-value_quality', kicker: 'CHEAP AND GOOD', fallback: 'Value × quality · junk gate on' },
  { id: 'paper-momentum_only', kicker: 'MOMENTUM SATELLITE', fallback: '12-1 / 6-1 skip-month · hold ~5 months' },
];

async function loadPaperSignals() {
  try {
    const r = await fetch('/paper_signals.json?v=bb5');
    state.paperSignals = r.ok ? await r.json() : null;
  } catch {
    state.paperSignals = null;
  }
  return state.paperSignals;
}

async function openProvenBook(id) {
  state.activePresets = new Set([id]);
  await switchView('signals');
  applyActivePresets();
}

function fillLaunchpadBooks() {
  const box = $('#bb-books');
  if (!box) return;
  const byId = Object.fromEntries((state.paperSignals?.signals || []).map((s) => [s.id, s]));
  box.innerHTML = '';
  LAUNCH_BOOKS.forEach((book) => {
    const sid = book.id.replace(/^paper-/, '');
    const s = byId[sid];
    const b = el('button', 'bb-go');
    b.type = 'button';
    const pass = s?.status === 'pass';
    const stats = pass && Number.isFinite(s.sharpe)
      ? `Sharpe ${s.sharpe.toFixed(2)} · excess ${pct(s.excess)} · CAGR ${pct(s.cagr)}`
      : (s?.note || book.fallback);
    b.innerHTML = `<span class="bb-k">${book.kicker}${pass ? ' · PASS' : ''}</span>`
      + `<b>${s?.label || sid}</b>`
      + `<span>${stats}</span>`
      + `<span class="bb-cta">GO → SCREEN PROFITABLE NAMES</span>`;
    b.onclick = () => openProvenBook(book.id);
    box.appendChild(b);
  });
}

function fillBbRace() {
  const rows = (state.paperSignals?.signals || [])
    .filter((s) => Number.isFinite(s.excess))
    .sort((a, b) => (Number(b.excess) || 0) - (Number(a.excess) || 0))
    .map((s) => ({ label: s.label, value: s.excess, status: s.status }));
  Charts.drawBars('#bb-race', rows, {
    diverging: true,
    format: (v) => pct(v, 1),
    color: (d) => (d.status === 'pass' ? '#5dff6b' : d.status === 'fail' ? '#ff4d4d' : '#8a7d5e'),
    empty: 'Horse race not loaded.',
  });
}

function fillBbTreemap(table) {
  if (!table?.columns) {
    Charts.drawTreemap('#bb-treemap', [], { empty: 'Load a run to map sectors.' });
    return;
  }
  const rows = Charts.tableToObjects(table);
  const hasProf = table.columns.includes('profitability');
  const picked = hasProf
    ? rows.filter((r) => Number(r.profitability) > 0)
    : rows.filter((r) => Number(r.rank) > 0 && Number(r.rank) <= 50);
  const counts = {};
  picked.forEach((r) => {
    const sec = r.sector || 'UNKNOWN';
    counts[sec] = (counts[sec] || 0) + 1;
  });
  const items = Object.entries(counts)
    .sort((a, b) => b[1] - a[1])
    .map(([label, value]) => ({ label, value }));
  Charts.drawTreemap('#bb-treemap', items, {
    height: 360,
    empty: 'No names with profitability z > 0 on this run.',
  });
}

function fillBbTape(table) {
  const box = $('#bb-tape');
  if (!box) return;
  if (!table?.columns) {
    box.innerHTML = '<div class="tape-track">WAITING FOR RANKINGS…</div>';
    return;
  }
  const rows = Charts.tableToObjects(table);
  const bits = rows
    .filter((r) => r.action && r.action !== 'PASS')
    .slice(0, 48)
    .map((r) => {
      const cls = `tape-${String(r.action).toLowerCase()}`;
      const px = Number.isFinite(Number(r.last_price)) ? ` ₹${Number(r.last_price).toFixed(0)}` : '';
      return `<span class="${cls}">${r.action} ${r.symbol || ''}${px}</span>`;
    });
  if (!bits.length) {
    box.innerHTML = '<div class="tape-track">NO ACTIONABLE NAMES · CONNECT KITE TO SIZE THE BOOK</div>';
    return;
  }
  const line = bits.join('   ·   ');
  box.innerHTML = `<div class="tape-track">${line}   ·   ${line}</div>`;
}

async function fillBbStamp() {
  const box = $('#bb-stamp');
  if (!box) return;
  let nightly = null;
  try {
    const r = await fetch(`/nightly.json?v=${Date.now()}`);
    nightly = r.ok ? await r.json() : null;
  } catch {
    nightly = null;
  }
  const nPass = (state.paperSignals?.passed || []).length;
  const nFail = (state.paperSignals?.failed || []).length;
  const when = nightly?.ran_at
    ? `${new Date(nightly.ran_at).toLocaleString('en-IN', { timeZone: 'Asia/Kolkata' })} IST`
    : 'not yet this session';
  const pytest = nightly?.pytest_ok === false ? 'FAIL' : nightly?.pytest_ok ? 'PASS' : '—';
  box.innerHTML = `NIGHTLY RESEARCH <b>${nightly?.mode || 'pending'}</b> · last ${when} · pytest ${pytest} · horse-race PASS ${nPass} / FAIL ${nFail} · next full run 00:00 IST`;
}

function fillProvenStrip() {
  const box = $('#proven-strip');
  if (!box) return;
  box.innerHTML = '';
  paperPresets('pass').forEach((p) => {
    const b = el('button', `proven-chip${state.activePresets.has(p.id) ? ' active' : ''}`);
    b.type = 'button';
    b.innerHTML = `<b>GO</b>${p.label}`;
    b.onclick = () => toggleScreenerPreset(p.id);
    box.appendChild(b);
  });
}

function tickerPresets() {
  return [
    { id: 'quoted', label: 'Has NSE close', hint: 'Names with a rupee close from the NSE bhavcopy',
      quoted: true, colset: 'ticker', rail: 'ticker' },
    { id: 'movers3', label: 'Live movers ≥3%', hint: 'Databento latch: |last / prev_close − 1| ≥ 3%',
      set: { 'move-min': '3' }, live: true, colset: 'ticker', rail: 'ticker' },
    { id: 'movers2', label: 'Live movers ≥2%', hint: 'Same scanner, lower threshold',
      set: { 'move-min': '2' }, live: true, colset: 'ticker', rail: 'ticker' },
    { id: 'liveup', label: 'Live up ≥3%', hint: 'Only names up vs yesterday’s close',
      set: { 'move-min': '3', 'move-dir': 'up' }, live: true, colset: 'ticker', rail: 'ticker' },
    { id: 'livedn', label: 'Live down ≥3%', hint: 'Only names down vs yesterday’s close',
      set: { 'move-min': '3', 'move-dir': 'down' }, live: true, colset: 'ticker', rail: 'ticker' },
    { id: 'lit', label: 'Latched tape', hint: 'Already tripped the % latch this session',
      lit: true, colset: 'ticker', rail: 'ticker' },
    { id: 'vol1l', label: 'Volume ≥ 1 L', hint: 'Today’s traded shares from the ticker',
      set: { 'vol-min': String(1e5) }, quoted: true, colset: 'ticker', rail: 'ticker' },
  ];
}

function screenerPresets() {
  const top = Number(state.screenerMeta.top_n) || 40;
  const buf = Number(state.screenerMeta.buffer_n) || top * 2;
  const w = factorWeights();
  const extra = [];
  if (w.q > 0) {
    extra.push({ id: 'quality', label: 'Quality / QMJ', hint: 'Asness–Frazzini–Pedersen quality minus junk',
      set: { 'q-min': '0' } });
  } else {
    extra.push({ id: 'quality', label: 'Quality / QMJ (untested)', hint: 'Weight was 0 on this run — no quality scores to screen',
      set: { 'q-min': '0' } });
  }
  if (w.v > 0 || w.q > 0) {
    extra.push({ id: 'vxq', label: 'Value × Quality', hint: 'Drop cheap-and-junk (positive value, negative quality)',
      set: { 'v-min': '0', 'q-min': '0' }, junk: true });
  } else {
    extra.push({ id: 'vxq', label: 'Value × Quality (untested)', hint: 'Needs a fundamentals backtest — empty on price_only_core',
      set: { 'v-min': '0', 'q-min': '0' }, junk: true });
  }
  extra.push({ id: 'etf', label: 'ETF dual momentum', hint: 'Antonacci: relative rank plus positive 12-month return',
    set: { asset: 'etf', 'mom-min': '0' } });
  return [
    { id: 'universe', label: 'Full universe', hint: 'Every scored name in the latest run' },
    { id: 'watch', label: 'WATCH / queued', hint: 'Blocked by a paper rule or waiting for a slot',
      set: { action: 'WATCH' } },
    { id: 'held', label: 'Held in Kite', hint: 'Names already on the live book',
      set: { held: 'yes' } },
    { id: 'candidates', label: 'Not held', hint: 'Research candidates sized if you connect Kite',
      set: { held: 'no' } },
    ...extra,
  ];
}

function allScreenerPresets() {
  return [
    ...winningPresets(),
    ...paperPresets('pass'),
    ...tickerPresets(),
    ...screenerPresets(),
    ...paperPresets('fail'),
    ...paperPresets('no_data'),
    ...paperPresets('error'),
  ];
}

function paintPresetList(box, presets) {
  if (!box) return;
  box.innerHTML = '';
  presets.forEach((p) => {
    const on = state.activePresets.has(p.id);
    const b = el('button', `preset${on ? ' active' : ''}${p.winner ? ' winner' : ''}`);
    b.type = 'button';
    b.dataset.preset = p.id;
    b.appendChild(el('b', null, p.label));
    b.appendChild(el('span', null, p.hint));
    b.onclick = () => toggleScreenerPreset(p.id);
    box.appendChild(b);
  });
}

function fillScreenerPresets() {
  paintPresetList($('#screener-presets'), [
    ...screenerPresets(),
    ...paperPresets('fail'),
    ...paperPresets('no_data'),
    ...paperPresets('error'),
  ]);
  paintPresetList($('#screener-winners'), [...winningPresets(), ...paperPresets('pass')]);
  paintPresetList($('#screener-ticker-presets'), tickerPresets());
  fillProvenStrip();
}

function fillBacktestCard() {
  const box = $('#backtest-card');
  if (!box) return;
  const m = state.backtestMeta;
  if (!m?.name) {
    box.innerHTML = '<div class="w-title">No backtest metrics on this run.</div>';
    fillPaperRace();
    return;
  }
  const w = factorWeights();
  const mix = [
    w.mom && `Mom ${Math.round(w.mom * 100)}%`,
    w.lv && `Low-vol ${Math.round(w.lv * 100)}%`,
    w.q && `Quality ${Math.round(w.q * 100)}%`,
    w.v && `Value ${Math.round(w.v * 100)}%`,
  ].filter(Boolean).join(' · ') || 'rank buffer only';
  let regimeNote = '';
  const table = m.regime;
  if (table?.columns && table.rows?.length) {
    const i = Object.fromEntries(table.columns.map((c, n) => [c, n]));
    const best = [...table.rows].sort((a, b) => (Number(b[i.sharpe]) || -99) - (Number(a[i.sharpe]) || -99))[0];
    if (best && Number.isFinite(Number(best[i.sharpe]))) {
      regimeNote = `Best regime: ${best[i.regime]} Sharpe ${Number(best[i.sharpe]).toFixed(2)}`;
    }
  }
  const pass = String(m.verdict || '').toUpperCase().includes('PASS');
  box.innerHTML = `
    <div class="w-title">${m.name}${pass ? ' · passed validation' : ''}</div>
    <div class="w-grid">
      <div>Sharpe <b>${Number.isFinite(m.sharpe) ? m.sharpe.toFixed(2) : '—'}</b></div>
      <div>CAGR <b>${pct(m.cagr)}</b></div>
      <div>Excess <b>${pct(m.excess)}</b></div>
      <div>Max DD <b>${pct(m.max_dd)}</b></div>
    </div>
    <div class="hint" style="margin:6px 0 0">${mix}${regimeNote ? ` · ${regimeNote}` : ''}</div>
    <div class="hint">${m.description || ''}</div>`;
  fillPaperRace();
}

function fillPaperRace() {
  const box = $('#paper-race');
  if (!box) return;
  const payload = state.paperSignals;
  if (!payload?.signals?.length) {
    box.innerHTML = '<div class="hint">Paper-signal horse race has not finished yet.</div>';
    return;
  }
  const w = payload.window || {};
  const rows = payload.signals.map((s) => {
    const mark = s.status === 'pass' ? 'PASS' : s.status === 'fail' ? 'FAIL' : '—';
    return `<tr>
      <td>${s.label}</td>
      <td class="${s.status}">${mark}</td>
      <td>${Number.isFinite(s.sharpe) ? s.sharpe.toFixed(2) : '—'}</td>
      <td>${pct(s.cagr)}</td>
      <td>${pct(s.excess)}</td>
    </tr>`;
  }).join('');
  box.innerHTML = `
    <div class="w-title">Horse race ${w.start || ''} → ${w.end || ''} vs Nifty 50 TRI</div>
    <table class="paper-race">
      <thead><tr><th>Signal</th><th></th><th>Sharpe</th><th>CAGR</th><th>Excess</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

function ensureScreenerColumns(table) {
  if (!table?.columns || table._macro) return table;
  MACRO_PAD.forEach((c) => {
    if (table.columns.includes(c)) return;
    table.columns.push(c);
    const si = table.columns.indexOf('symbol');
    const sec = table.columns.indexOf('sector');
    table.rows.forEach((r) => {
      if (c === 'name') r.push(r[si] || null);
      else if (c === 'exchange') r.push('NSE');
      else if (c === 'country') r.push('India');
      else if (c === 'industry') r.push(sec >= 0 ? (r[sec] || 'UNKNOWN') : 'UNKNOWN');
      else r.push(null);
    });
  });
  table._macro = true;
  return table;
}

function fillSelectPair(minId, maxId, pairs) {
  const min = document.getElementById(minId);
  const max = document.getElementById(maxId);
  if (!min || !max || min.options.length) return;
  min.appendChild(new Option('No Min', ''));
  max.appendChild(new Option('No Max', ''));
  pairs.forEach(([val, label]) => {
    min.appendChild(new Option(label, String(val)));
    max.appendChild(new Option(label, String(val)));
  });
}

function fillScreenerRanges() {
  fillSelectPair('screener-mcap-min', 'screener-mcap-max', RANGE_PRESETS.mcap);
  fillSelectPair('screener-pe-min', 'screener-pe-max', RANGE_PRESETS.pe);
  fillSelectPair('screener-div-min', 'screener-div-max', RANGE_PRESETS.div);
  fillSelectPair('screener-price-min', 'screener-price-max', RANGE_PRESETS.price);
  fillSelectPair('screener-pb-min', 'screener-pb-max', RANGE_PRESETS.pb);
  fillSelectPair('screener-de-min', 'screener-de-max', RANGE_PRESETS.de);
  fillSelectPair('screener-roe-min', 'screener-roe-max', RANGE_PRESETS.roe);
  fillSelectPair('screener-ey-min', 'screener-ey-max', RANGE_PRESETS.ey);
  fillSelectPair('screener-eps-min', 'screener-eps-max', RANGE_PRESETS.eps);
  fillSelectPair('screener-rev-min', 'screener-rev-max', RANGE_PRESETS.rev);
  fillSelectPair('screener-ni-min', 'screener-ni-max', RANGE_PRESETS.rev);
  fillSelectPair('screener-payout-min', 'screener-payout-max', RANGE_PRESETS.payout);
  fillSelectPair('screener-nm-min', 'screener-nm-max', RANGE_PRESETS.roe);
  fillSelectPair('screener-r1d-min', 'screener-r1d-max', RANGE_PRESETS.r1d);
  ['r1w', 'r1m', 'r3m', 'r6m', 'r1y', 'r3y'].forEach((k) => {
    fillSelectPair(`screener-${k}-min`, `screener-${k}-max`, RANGE_PRESETS.ret);
  });
  fillSelectPair('screener-tick-price-min', 'screener-tick-price-max', RANGE_PRESETS.price);
  fillSelectPair('screener-tick-chg-min', 'screener-tick-chg-max', RANGE_PRESETS.r1d);
  fillSelectPair('screener-vol-min', 'screener-vol-max', RANGE_PRESETS.vol);
}

function uniqueCol(table, col) {
  const i = table.columns.indexOf(col);
  if (i < 0) return [];
  const seen = new Set();
  (table.rows || []).forEach((r) => { if (r[i]) seen.add(String(r[i])); });
  return [...seen].sort();
}

function updateMsToggle(box, selected) {
  const btn = box.querySelector('.ms-toggle');
  if (btn) btn.textContent = selected.size ? `${selected.size} selected` : 'All';
}

function fillScreenerMultis() {
  if (!state.screener?.columns) return;
  document.querySelectorAll('.ms').forEach((box) => {
    const name = box.dataset.ms;
    const col = box.dataset.col;
    const selected = state.msSelected[name] || new Set();
    const wrap = box.querySelector('.ms-opts');
    if (!wrap) return;
    wrap.innerHTML = '';
    uniqueCol(state.screener, col).forEach((v) => {
      const lab = el('label', 'checkbox');
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = v;
      cb.checked = selected.has(v);
      cb.onchange = () => {
        if (cb.checked) selected.add(v);
        else selected.delete(v);
        state.msSelected[name] = selected;
        updateMsToggle(box, selected);
        renderScreener();
      };
      lab.appendChild(cb);
      lab.appendChild(document.createTextNode(` ${v}`));
      wrap.appendChild(lab);
    });
    updateMsToggle(box, selected);
    const search = box.querySelector('.ms-search');
    if (search && !search._wired) {
      search._wired = true;
      search.oninput = () => {
        const q = search.value.toLowerCase();
        wrap.querySelectorAll('label').forEach((lab) => {
          lab.style.display = lab.textContent.toLowerCase().includes(q) ? '' : 'none';
        });
      };
    }
  });
}

function fillColSets() {
  const bar = $('#screener-colsets');
  if (!bar) return;
  bar.innerHTML = '';
  Object.entries(COLUMN_SETS).forEach(([id, spec]) => {
    const b = el('button', state.screenerSet === id ? 'active' : '');
    b.type = 'button';
    b.textContent = spec.label;
    b.onclick = () => {
      state.screenerSet = id;
      fillColSets();
      renderScreener({ keepPage: true });
    };
    bar.appendChild(b);
  });
}

function msMatch(name, value) {
  const sel = state.msSelected[name];
  if (!sel || sel.size === 0) return true;
  return sel.has(String(value || ''));
}

function clearScreenerFilters() {
  ['screener-q', 'screener-action', 'screener-asset', 'screener-held',
   'screener-move-min', 'screener-move-dir', 'screener-tick-dir']
    .forEach((id) => { const n = document.getElementById(id); if (n) n.value = ''; });
  SCREENER_RANGE_IDS.forEach((id) => { const n = document.getElementById(id); if (n) n.value = ''; });
  SCREENER_SELECT_IDS.forEach((id) => { const n = document.getElementById(id); if (n) n.value = ''; });
  Object.keys(state.msSelected).forEach((k) => { state.msSelected[k] = new Set(); });
  fillScreenerMultis();
  ['screener-no-junk', 'screener-quoted', 'screener-lit'].forEach((id) => {
    const n = document.getElementById(id);
    if (n) n.checked = false;
  });
}

function stricterMin(a, b) {
  const na = parseFloat(a);
  const nb = parseFloat(b);
  if (!Number.isFinite(na)) return b;
  if (!Number.isFinite(nb)) return a;
  return String(Math.max(na, nb));
}

function stricterMax(a, b) {
  const na = parseFloat(a);
  const nb = parseFloat(b);
  if (!Number.isFinite(na)) return b;
  if (!Number.isFinite(nb)) return a;
  return String(Math.min(na, nb));
}

function mergePresets(presets) {
  const set = {};
  let junk = false;
  let quoted = false;
  let lit = false;
  let colset = null;
  let rail = null;
  presets.forEach((p) => {
    junk = junk || !!p.junk;
    quoted = quoted || !!p.quoted;
    lit = lit || !!p.lit;
    if (p.colset) colset = p.colset === 'iima' || !colset ? (p.colset || colset) : colset;
    if (p.rail) rail = p.rail === 'backtest' || !rail ? (p.rail || rail) : rail;
    Object.entries(p.set || {}).forEach(([key, val]) => {
      if (key.endsWith('-min')) set[key] = stricterMin(set[key], val);
      else if (key.endsWith('-max')) set[key] = stricterMax(set[key], val);
      else set[key] = val;
    });
  });
  return { set, junk, quoted, lit, colset, rail };
}

function applyMergedPreset(merged) {
  Object.entries(merged.set || {}).forEach(([key, val]) => {
    const n = document.getElementById(`screener-${key}`);
    if (n) n.value = val;
  });
  if (merged.junk) {
    const junk = $('#screener-no-junk');
    if (junk) junk.checked = true;
  }
  if (merged.quoted) {
    const q = $('#screener-quoted');
    if (q) q.checked = true;
  }
  if (merged.lit) {
    const lit = $('#screener-lit');
    if (lit) lit.checked = true;
  }
  if (merged.colset && COLUMN_SETS[merged.colset]) {
    state.screenerSet = merged.colset;
    fillColSets();
  }
  if (merged.live) {
    const th = $('#screener-threshold');
    if (th && merged.set && merged.set['move-min']) th.value = merged.set['move-min'];
    state.scanner.threshold = (parseFloat(merged.set?.['move-min']) || 3) / 100;
  }
  if (merged.rail) switchRail(merged.rail);
}

function applyActivePresets() {
  const presets = [...state.activePresets]
    .map((id) => allScreenerPresets().find((x) => x.id === id))
    .filter(Boolean);
  state.screenerPreset = presets.map((p) => p.id).join('+') || 'universe';
  clearScreenerFilters();
  if (presets.length) applyMergedPreset(mergePresets(presets));
  fillScreenerPresets();
  fillActiveFilters();
  renderScreener();
}

function toggleScreenerPreset(id) {
  if (state.activePresets.has(id)) state.activePresets.delete(id);
  else state.activePresets.add(id);
  applyActivePresets();
}

function applyScreenerPreset(id) {
  toggleScreenerPreset(id);
}

async function openProvenBook(id) {
  state.activePresets = new Set([id]);
  await switchView('signals');
  applyActivePresets();
}

function fillActiveFilters() {
  const box = $('#active-filters');
  if (!box) return;
  box.innerHTML = '';
  if (!state.activePresets.size) return;
  [...state.activePresets].forEach((id) => {
    const p = allScreenerPresets().find((x) => x.id === id);
    const chip = el('button', 'active-chip');
    chip.type = 'button';
    chip.textContent = `${p?.label || id} ×`;
    chip.onclick = () => toggleScreenerPreset(id);
    box.appendChild(chip);
  });
}

function fillProvenStrip() {
  const box = $('#proven-strip');
  if (!box) return;
  box.innerHTML = '';
  paperPresets('pass').forEach((p) => {
    const b = el('button', `proven-chip${state.activePresets.has(p.id) ? ' active' : ''}`);
    b.type = 'button';
    b.innerHTML = `<b>GO</b>${p.label}`;
    b.onclick = () => toggleScreenerPreset(p.id);
    box.appendChild(b);
  });
}

function switchRail(name) {
  document.querySelectorAll('.screener-rail-tabs button').forEach((x) => {
    x.classList.toggle('active', x.dataset.rail === name);
  });
  document.querySelectorAll('.rail-panel').forEach((p) => {
    p.classList.toggle('active', p.id === `rail-${name}`);
  });
}

function renderScreener(opts = {}) {
  const table = state.screener;
  const node = $('#screener-table');
  if (!node) return;
  if (!table || !table.columns) {
    node.innerHTML = '<div class="empty">No rankings to screen.</div>';
    const count = $('#screener-count');
    if (count) count.textContent = 'Matching names: 0';
    return;
  }
  if (!opts.keepPage) state.screenerPage = 0;

  const idx = Object.fromEntries(table.columns.map((c, i) => [c, i]));
  const q = ($('#screener-q')?.value || '').toLowerCase();
  const action = $('#screener-action')?.value || '';
  const asset = $('#screener-asset')?.value || '';
  const held = $('#screener-held')?.value || '';
  const noJunk = $('#screener-no-junk')?.checked;

  const rows = table.rows.filter((r) => {
    if (action && r[idx.action] !== action) return false;
    if (asset && r[idx.asset_class] !== asset) return false;
    if (!msMatch('exchange', r[idx.exchange])) return false;
    if (!msMatch('country', r[idx.country])) return false;
    if (!msMatch('sector', r[idx.sector])) return false;
    if (!msMatch('industry', r[idx.industry] || r[idx.sector])) return false;
    if (held === 'yes' && !r[idx.held]) return false;
    if (held === 'no' && r[idx.held]) return false;
    if (q) {
      const blob = `${r[idx.name] || ''} ${r[idx.symbol] || ''} ${r[idx.isin] || ''} ${r[idx.sector] || ''} ${r[idx.industry] || ''}`.toLowerCase();
      if (!blob.includes(q)) return false;
    }
    if (!inFactorRange(r[idx.rank], 'screener-rank-min', 'screener-rank-max')) return false;
    if (!inFactorRange(r[idx.composite], 'screener-comp-min', 'screener-comp-max')) return false;
    if (!inFactorRange(r[idx.momentum], 'screener-mom-min', 'screener-mom-max')) return false;
    if (!inFactorRange(r[idx.quality], 'screener-q-min', 'screener-q-max')) return false;
    if (!inFactorRange(r[idx.value], 'screener-v-min', 'screener-v-max')) return false;
    if (!inFactorRange(r[idx.low_vol], 'screener-lv-min', 'screener-lv-max')) return false;
    if (!inFactorRange(r[idx.profitability], 'screener-prof-min', 'screener-prof-max')) return false;
    if (!inFactorRange(r[idx.growth], 'screener-growth-min', 'screener-growth-max')) return false;
    if (!inFactorRange(r[idx.safety], 'screener-safety-min', 'screener-safety-max')) return false;
    if (!inFactorRange(r[idx.payout_z], 'screener-payoutz-min', 'screener-payoutz-max')) return false;
    if (!inFactorRange(r[idx.net_margin], 'screener-nm-min', 'screener-nm-max')) return false;
    if (!inFactorRange(r[idx.pe], 'screener-pe-min', 'screener-pe-max')) return false;
    if (!inFactorRange(r[idx.market_cap], 'screener-mcap-min', 'screener-mcap-max')) return false;
    if (!inFactorRange(r[idx.div_yield], 'screener-div-min', 'screener-div-max')) return false;
    if (!inFactorRange(livePrice(r, idx), 'screener-price-min', 'screener-price-max')) return false;
    if (!inFactorRange(r[idx.pb], 'screener-pb-min', 'screener-pb-max')) return false;
    if (!inFactorRange(r[idx.debt_equity], 'screener-de-min', 'screener-de-max')) return false;
    if (!inFactorRange(r[idx.roe], 'screener-roe-min', 'screener-roe-max')) return false;
    if (!inFactorRange(r[idx.earnings_yield], 'screener-ey-min', 'screener-ey-max')) return false;
    if (!inFactorRange(r[idx.eps], 'screener-eps-min', 'screener-eps-max')) return false;
    if (!inFactorRange(r[idx.revenue], 'screener-rev-min', 'screener-rev-max')) return false;
    if (!inFactorRange(r[idx.net_income], 'screener-ni-min', 'screener-ni-max')) return false;
    if (!inFactorRange(r[idx.payout], 'screener-payout-min', 'screener-payout-max')) return false;
    if (!inFactorRange(r[idx.ret_1w], 'screener-r1w-min', 'screener-r1w-max')) return false;
    if (!inFactorRange(r[idx.ret_1m], 'screener-r1m-min', 'screener-r1m-max')) return false;
    if (!inFactorRange(r[idx.ret_3m], 'screener-r3m-min', 'screener-r3m-max')) return false;
    if (!inFactorRange(r[idx.ret_6m], 'screener-r6m-min', 'screener-r6m-max')) return false;
    if (!inFactorRange(r[idx.ret_1y], 'screener-r1y-min', 'screener-r1y-max')) return false;
    if (!inFactorRange(r[idx.ret_3y], 'screener-r3y-min', 'screener-r3y-max')) return false;
    if (!inFactorRange(liveDayChg(r, idx), 'screener-r1d-min', 'screener-r1d-max')) return false;
    if (!inFactorRange(livePrice(r, idx), 'screener-tick-price-min', 'screener-tick-price-max')) return false;
    if (!inFactorRange(liveDayChg(r, idx), 'screener-tick-chg-min', 'screener-tick-chg-max')) return false;
    if (!inFactorRange(liveVol(r, idx), 'screener-vol-min', 'screener-vol-max')) return false;
    const quotedOnly = $('#screener-quoted')?.checked;
    const litOnly = $('#screener-lit')?.checked;
    const sym = String(r[idx.symbol] || '').toUpperCase();
    if (quotedOnly && !Number.isFinite(livePrice(r, idx))) return false;
    if (litOnly && !state.scanner.lit.has(sym)) return false;
    const tickDir = $('#screener-tick-dir')?.value || '';
    if (tickDir) {
      const chg = liveDayChg(r, idx);
      if (!Number.isFinite(chg)) return false;
      if (tickDir === 'up' && !(chg > 0)) return false;
      if (tickDir === 'down' && !(chg < 0)) return false;
    }
    if (noJunk) {
      const val = Number(r[idx.value]);
      const qlt = Number(r[idx.quality]);
      if (Number.isFinite(val) && Number.isFinite(qlt) && val > 0 && qlt < 0) return false;
    }
    const moveMin = parseBound('screener-move-min');
    const moveDir = $('#screener-move-dir')?.value || '';
    if (moveMin !== null || moveDir) {
      const chg = liveDayChg(r, idx);
      if (!Number.isFinite(chg)) return false;
      if (moveMin !== null && Math.abs(chg) < moveMin) return false;
      if (moveDir === 'up' && !(chg > 0)) return false;
      if (moveDir === 'down' && !(chg < 0)) return false;
    }
    return true;
  });

  const pageSize = screenerPageSize();
  const pages = Math.max(1, Math.ceil(rows.length / pageSize));
  if (state.screenerPage >= pages) state.screenerPage = pages - 1;
  if (state.screenerPage < 0) state.screenerPage = 0;

  const missingHint = (() => {
    const has = (col) => table.rows.some((r) => {
      const raw = r[idx[col]];
      if (raw === null || raw === undefined || raw === '') return false;
      return Number.isFinite(Number(raw));
    });
    if ((parseBound('screener-q-min') !== null || parseBound('screener-q-max') !== null) && !has('quality')) {
      return 'This run has no quality scores (price-only core). Re-run full_composite to use QMJ.';
    }
    if ((parseBound('screener-v-min') !== null || parseBound('screener-v-max') !== null) && !has('value')) {
      return 'This run has no value scores. Re-run a fundamentals preset to use Value × Quality.';
    }
    if (asset === 'etf' && !table.rows.some((r) => r[idx.asset_class] === 'etf')) {
      return 'No listed ETFs in this universe yet. The next backtest writes etf_rankings when INF names clear the ADV floor.';
    }
    if ((parseBound('screener-move-min') !== null || $('#screener-quoted')?.checked
        || parseBound('screener-vol-min') !== null || parseBound('screener-tick-chg-min') !== null)
        && !table.rows.some((r) => Number.isFinite(liveDayChg(r, idx)) || Number.isFinite(livePrice(r, idx)))) {
      return 'No NSE EOD last / prev / volume on this run.';
    }
    const emptyFund = [
      ['screener-pe-min', 'screener-pe-max', 'pe', 'P/E is empty on this run. Re-run a fundamentals preset.'],
      ['screener-mcap-min', 'screener-mcap-max', 'market_cap', 'Market cap is empty until fundamentals are loaded.'],
      ['screener-div-min', 'screener-div-max', 'div_yield', 'Dividend yield is not in this run.'],
      ['screener-pb-min', 'screener-pb-max', 'pb', 'Price / Book is empty until fundamentals are loaded.'],
      ['screener-de-min', 'screener-de-max', 'debt_equity', 'Debt / Equity is empty until fundamentals are loaded.'],
      ['screener-roe-min', 'screener-roe-max', 'roe', 'ROE is empty until fundamentals are loaded.'],
      ['screener-prof-min', 'screener-prof-max', 'profitability', 'IIMA profitability is empty until NSE filings print margins/ROIC.'],
      ['screener-growth-min', 'screener-growth-max', 'growth', 'IIMA growth needs two years of profitability prints.'],
      ['screener-safety-min', 'screener-safety-max', 'safety', 'IIMA safety is empty until leverage or low-vol scores exist.'],
      ['screener-payoutz-min', 'screener-payoutz-max', 'payout_z', 'IIMA payout is empty — NSE quarterlies rarely print payout.'],
      ['screener-nm-min', 'screener-nm-max', 'net_margin', 'Net margin is empty until filings are loaded.'],
      ['screener-r1y-min', 'screener-r1y-max', 'ret_1y', '1-year returns are written on the next backtest (need a price panel).'],
    ];
    for (const [a, b, col, msg] of emptyFund) {
      if ((parseBound(a) !== null || parseBound(b) !== null) && !has(col)) return msg;
    }
    return '';
  })();

  const count = $('#screener-count');
  if (count) count.textContent = `Matching names: ${rows.length.toLocaleString('en-IN')}`;
  const hint = $('#screener-count')?.nextElementSibling;
  if (hint && hint.classList.contains('hint') && missingHint) {
    hint.dataset.base = hint.dataset.base || hint.textContent;
    hint.textContent = missingHint;
  } else if (hint && hint.dataset.base) {
    hint.textContent = hint.dataset.base;
  }
  const label = $('#screener-page-label');
  if (label) label.textContent = `Page ${state.screenerPage + 1} of ${pages}`;
  const prev = $('#screener-prev');
  const next = $('#screener-next');
  const first = $('#screener-first');
  const last = $('#screener-last');
  if (prev) prev.disabled = state.screenerPage <= 0;
  if (first) first.disabled = state.screenerPage <= 0;
  if (next) next.disabled = state.screenerPage >= pages - 1;
  if (last) last.disabled = state.screenerPage >= pages - 1;

  state.screenerFiltered = { columns: table.columns, rows, n_rows: rows.length };
  const set = COLUMN_SETS[state.screenerSet] || COLUMN_SETS.overview;

  renderTable(node, {
    ...table,
    rows,
    n_rows: rows.length,
    note: rows.length ? undefined : (missingHint || 'No names match these filters.'),
  }, {
    columns: set.cols,
    headers: COL_HEADERS,
    limit: pageSize,
    offset: state.screenerPage * pageSize,
    hideCount: true,
    rowKey: 'symbol',
    rowClass: (row, cols) => {
      const i = cols.indexOf('symbol');
      const sym = String(row[i] || '').toUpperCase();
      return state.scanner.lit.has(sym) ? 'lit' : '';
    },
    colorize: ['composite', 'momentum', 'quality', 'value', 'low_vol', 'day_chg',
               'ret_1w', 'ret_1m', 'ret_3m', 'ret_6m', 'ret_1y', 'ret_3y'],
    format: {
      name: (v, row, cols) => nameCell(v || row[cols.indexOf('symbol')], row, cols),
      action: (v) => el('span', `tag ${String(v).toLowerCase()}`, v),
      held: (v) => (v ? 'yes' : 'no'),
      last_price: (v, row, cols) => {
        const px = positiveNum(overlayField(row, cols, 'last'))
          ?? positiveNum(v)
          ?? positiveNum(row[cols.indexOf('close')]);
        return px != null ? inr(px) : '—';
      },
      prev_close: (v, row, cols) => {
        const px = positiveNum(overlayField(row, cols, 'prev')) ?? positiveNum(v);
        return px != null ? inr(px) : '—';
      },
      day_chg: (v, row, cols) => {
        const chg = overlayField(row, cols, 'chg') ?? v;
        return finiteNum(chg) != null
          ? `<span class="${signClass(chg)}">${chg >= 0 ? '+' : ''}${Number(chg).toFixed(2)}%</span>` : '—';
      },
      ret_1w: fmtRet, ret_1m: fmtRet, ret_3m: fmtRet, ret_6m: fmtRet, ret_1y: fmtRet, ret_3y: fmtRet,
      pe: (v) => (finiteNum(v) != null ? num(v, 1) : '—'),
      pb: (v) => (finiteNum(v) != null ? num(v, 2) : '—'),
      market_cap: (v) => (positiveNum(v) != null ? `₹${(Number(v) / CR).toFixed(0)} Cr` : '—'),
      revenue: (v) => (finiteNum(v) != null ? `₹${(Number(v) / CR).toFixed(0)} Cr` : '—'),
      net_income: (v) => (finiteNum(v) != null ? `₹${(Number(v) / CR).toFixed(0)} Cr` : '—'),
      div_yield: (v) => (finiteNum(v) != null ? pct(v) : '—'),
      payout: (v) => (finiteNum(v) != null ? pct(v) : '—'),
      roe: (v) => (finiteNum(v) != null ? pct(v) : '—'),
      earnings_yield: (v) => (finiteNum(v) != null ? pct(v) : '—'),
      debt_equity: (v) => (finiteNum(v) != null ? num(v, 2) : '—'),
      eps: (v) => (finiteNum(v) != null ? num(v, 2) : '—'),
      volume: (v, row, cols) => {
        const vol = overlayField(row, cols, 'vol') ?? v;
        return typeof vol === 'number' ? Number(vol).toLocaleString('en-IN') : '—';
      },
      current_qty: (v) => (typeof v === 'number' ? num(v, 0) : '—'),
      target_qty: (v) => (typeof v === 'number' ? num(v, 0) : '—'),
      composite: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      momentum: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      quality: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      profitability: (v) => (finiteNum(v) != null ? num(v, 2) : '—'),
      growth: (v) => (finiteNum(v) != null ? num(v, 2) : '—'),
      safety: (v) => (finiteNum(v) != null ? num(v, 2) : '—'),
      payout_z: (v) => (finiteNum(v) != null ? num(v, 2) : '—'),
      net_margin: (v) => (finiteNum(v) != null ? pct(v) : '—'),
      pretax_margin: (v) => (finiteNum(v) != null ? pct(v) : '—'),
      gross_profitability: (v) => (finiteNum(v) != null ? pct(v) : '—'),
      value: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      low_vol: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      rank: (v) => (typeof v === 'number' ? num(v, 0) : '—'),
      delta_value: (v) => (typeof v === 'number'
        ? `<span class="${signClass(v)}">${v >= 0 ? '+' : ''}${inr(v)}</span>` : '—'),
      delta_qty: (v) => (typeof v === 'number'
        ? `<span class="${signClass(v)}">${v > 0 ? '+' : ''}${num(v, 0)}</span>` : '—'),
    },
  });
}

function fmtRet(v) {
  return finiteNum(v) != null ? `<span class="${signClass(v)}">${pct(v)}</span>` : '—';
}

function livePrice(row, idx) {
  const ov = state.scanner.overlay.get(String(row[idx.symbol] || '').toUpperCase());
  if (ov && ov.last != null && Number.isFinite(ov.last) && ov.last > 0) return ov.last;
  const n = positiveNum(row[idx.last_price]);
  if (n != null) return n;
  return positiveNum(idx.close >= 0 ? row[idx.close] : null);
}

function liveVol(row, idx) {
  const ov = state.scanner.overlay.get(String(row[idx.symbol] || '').toUpperCase());
  if (ov && ov.vol != null && Number.isFinite(ov.vol)) return ov.vol;
  const n = Number(row[idx.volume]);
  return Number.isFinite(n) && n > 0 ? n : null;
}

function nameCell(label, row, cols) {
  const a = el('a', 'name-link', label || '—');
  const sym = String(row[cols.indexOf('symbol')] || '');
  a.href = '#';
  a.dataset.symbol = sym;
  a.onmouseenter = (e) => showSpark(e, sym);
  a.onmouseleave = hideSpark;
  a.onclick = (e) => { e.preventDefault(); openProfile(sym); };
  return a;
}

async function loadSparksFile() {
  if (state.sparksFile) return state.sparksFile;
  try {
      const r = await fetch('/sparks.json?v=bb5');
    state.sparksFile = r.ok ? await r.json() : { closes: {} };
  } catch {
    state.sparksFile = { closes: {} };
  }
  return state.sparksFile;
}

async function loadHistory(symbol) {
  const key = String(symbol || '').toUpperCase();
  if (!key) return { ok: false, closes: [], note: 'No ticker' };
  const hit = state.historyCache.get(key);
  if (hit) return hit;
  const file = await loadSparksFile();
  let closes = (file.closes?.[key] || []).map(Number).filter(Number.isFinite);
  let note = closes.length > 1
    ? `${closes.length} NSE EOD closes`
    : 'No NSE EOD series';
  if (closes.length < 2) {
    try {
      const body = await api(`/history?symbol=${encodeURIComponent(key)}`);
      closes = (body.data?.series?.close || []).filter((v) => v != null).map(Number);
      if (closes.length > 1) note = body.note || `${closes.length} NSE EOD closes`;
    } catch (e) {
      note = e.message || note;
    }
  }
  const out = { ok: closes.length > 1, closes, note };
  state.historyCache.set(key, out);
  return out;
}

function showSpark(ev, symbol) {
  const tip = $('#spark-tip');
  if (!tip) return;
  tip.hidden = false;
  tip.style.left = `${Math.min(ev.clientX + 14, window.innerWidth - 300)}px`;
  tip.style.top = `${Math.min(ev.clientY + 14, window.innerHeight - 150)}px`;
  $('#spark-tip-note').textContent = 'Loading…';
  $('#spark-tip-chart').innerHTML = '';
  loadHistory(symbol).then((h) => {
    if (tip.hidden) return;
    $('#spark-tip-note').textContent = h.note;
    if (h.ok) Charts.drawSparkline('#spark-tip-chart', h.closes, { height: 72 });
    else $('#spark-tip-chart').innerHTML = '';
  });
}

function hideSpark() {
  const tip = $('#spark-tip');
  if (tip) tip.hidden = true;
}

async function openProfile(symbol) {
  const table = state.screener;
  if (!table) return;
  const idx = Object.fromEntries(table.columns.map((c, i) => [c, i]));
  const row = table.rows.find((r) => String(r[idx.symbol] || '').toUpperCase() === String(symbol).toUpperCase());
  const drawer = $('#screener-drawer');
  if (!drawer || !row) return;
  drawer.hidden = false;
  const name = row[idx.name] || symbol;
  $('#drawer-title').textContent = name;
  $('#drawer-sub').textContent = [
    row[idx.symbol], row[idx.exchange] || 'NSE', row[idx.country] || 'India',
    row[idx.industry] || row[idx.sector], row[idx.action],
  ].filter(Boolean).join(' · ');
  const stats = $('#drawer-stats');
  stats.className = 'drawer-stats';
  stats.innerHTML = '';
  const add = (k, v) => {
    const cell = el('div');
    cell.appendChild(el('div', 'k', k));
    const val = el('div', 'v');
    val.innerHTML = v;
    cell.appendChild(val);
    stats.appendChild(cell);
  };
  add('Ticker', row[idx.symbol] || '—');
  add('Action', row[idx.action] || '—');
  add('Rank', typeof row[idx.rank] === 'number' ? num(row[idx.rank], 0) : '—');
  add('Composite', typeof row[idx.composite] === 'number' ? num(row[idx.composite], 2) : '—');
  add('Closing price', inr(livePrice(row, idx)));
  add('1 day', Number.isFinite(liveDayChg(row, idx))
    ? `<span class="${signClass(liveDayChg(row, idx))}">${liveDayChg(row, idx).toFixed(2)}%</span>` : '—');
  add('1 year', finiteNum(row[idx.ret_1y]) != null ? pct(row[idx.ret_1y]) : '—');
  add('P/E', finiteNum(row[idx.pe]) != null ? num(row[idx.pe], 1) : '—');
  add('Earnings yield', finiteNum(row[idx.earnings_yield]) != null ? pct(row[idx.earnings_yield]) : '—');
  add('EPS', finiteNum(row[idx.eps]) != null ? num(row[idx.eps], 2) : '—');
  add('Revenue', finiteNum(row[idx.revenue]) != null ? `₹${(Number(row[idx.revenue]) / CR).toFixed(0)} Cr` : '—');
  add('Net income', finiteNum(row[idx.net_income]) != null ? `₹${(Number(row[idx.net_income]) / CR).toFixed(0)} Cr` : '—');
  add('Net margin', finiteNum(row[idx.net_margin]) != null ? pct(row[idx.net_margin]) : '—');
  add('Pretax margin', finiteNum(row[idx.pretax_margin]) != null ? pct(row[idx.pretax_margin]) : '—');
  add('ROE', finiteNum(row[idx.roe]) != null ? pct(row[idx.roe]) : '—');
  add('Debt / Equity', typeof row[idx.debt_equity] === 'number' ? num(row[idx.debt_equity], 2) : '—');
  add('Momentum', typeof row[idx.momentum] === 'number' ? num(row[idx.momentum], 2) : '—');
  add('Quality', typeof row[idx.quality] === 'number' ? num(row[idx.quality], 2) : '—');
  add('Reason', row[idx.reason] || '—');
  $('#drawer-chart').innerHTML = '<div class="empty">Loading 1-year chart…</div>';
  const hist = await loadHistory(symbol);
  if (hist.ok) Charts.drawLineChart('#drawer-chart', {
    t: hist.closes.map((_, i) => i),
    series: { close: hist.closes },
  }, ['close'], { height: 220, rebase: true });
  else $('#drawer-chart').innerHTML = `<div class="empty">${hist.note}</div>`;
}

function closeProfile() {
  const drawer = $('#screener-drawer');
  if (drawer) drawer.hidden = true;
}

function exportScreenerCsv() {
  const t = state.screenerFiltered;
  if (!t?.rows?.length) return;
  const set = COLUMN_SETS[state.screenerSet] || COLUMN_SETS.overview;
  const cols = set.cols.filter((c) => t.columns.includes(c));
  const ix = cols.map((c) => t.columns.indexOf(c));
  const lines = [cols.map((c) => COL_HEADERS[c] || c).join(',')];
  t.rows.forEach((r) => {
    lines.push(ix.map((i) => {
      const v = r[i];
      if (v === null || v === undefined) return '';
      const s = String(v).replaceAll('"', '""');
      return s.includes(',') ? `"${s}"` : s;
    }).join(','));
  });
  const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `screener-${state.screenerSet}.csv`;
  a.click();
  URL.revokeObjectURL(a.href);
}

function overlayField(row, cols, key) {
  const i = cols.indexOf('symbol');
  const ov = state.scanner.overlay.get(String(row[i] || '').toUpperCase());
  return ov ? ov[key] : undefined;
}

function liveDayChg(row, idx) {
  const ov = state.scanner.overlay.get(String(row[idx.symbol] || '').toUpperCase());
  if (ov && ov.chg != null && Number.isFinite(ov.chg)) return ov.chg;
  const raw = row[idx.day_chg];
  if (raw === null || raw === undefined || raw === '') return NaN;
  const n = Number(raw);
  return Number.isFinite(n) ? n : NaN;
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

    await loadPaperSignals();
    let screenerTable = null;
    try {
      const sig = await api('/signals');
      screenerTable = ensureScreenerColumns(sig.screener);
      fillBbTape(screenerTable);
      fillBbTreemap(screenerTable);
      const topN = Number(sig.summary?.top_n) || 40;
      const used = Number(sig.summary?.n_keep) || 0;
      Charts.drawGauge('#overview-slots', topN ? Math.min(1, used / topN) : 0, {
        label: `${used} / ${topN}`,
        color: used >= topN ? '#ffb000' : '#ff9900',
      });
    } catch {
      fillBbTape(null);
      fillBbTreemap(null);
      Charts.drawGauge('#overview-slots', 0, { label: '—' });
    }
    fillLaunchpadBooks();
    fillBbRace();
    await fillBbStamp();

    const curve = data.equity_curve?.data;
    Charts.drawLineChart('#equity-chart', curve, ['equity', 'after_tax_equity', 'benchmark'], {
      fill: true, height: 400, rebase: true,
    });
    Charts.drawLineChart('#drawdown-chart', curve, ['drawdown'], {
      percent: true, fill: true, zeroBaseline: true, height: 240,
    });

    const eq = (curve?.series?.equity || []).filter((v) => v != null);
    const bm = (curve?.series?.benchmark || []).filter((v) => v != null);
    const eqRet = eq.length > 1 ? eq[eq.length - 1] / eq[0] - 1 : 0;
    const bmRet = bm.length > 1 ? bm[bm.length - 1] / bm[0] - 1 : 0;
    Charts.drawDonut('#overview-mix', [
      { label: 'Strategy', value: Math.max(0.01, eqRet), color: '#ff9900' },
      { label: 'Nifty 50 TRI', value: Math.max(0.01, bmRet), color: '#5dff6b' },
    ], { center: `${((eqRet - bmRet) * 100).toFixed(1)} pp` });
    const ddNow = (curve?.series?.drawdown || []).filter((v) => v != null).at(-1);
    const ddWorst = m.max_drawdown || -0.11;
    Charts.drawGauge('#overview-dd-gauge',
      (ddWorst && ddNow != null) ? Math.min(1, Math.abs(ddNow) / Math.abs(ddWorst)) : 0,
      { label: ddNow != null ? `${(ddNow * 100).toFixed(1)}%` : '—', color: '#ff4d4d' });

    try {
      const monthly = await api('/monthly');
      Charts.drawHeatmap('#monthly-heatmap', monthly);
      renderTable('#monthly-table', monthly, {
        format: Object.fromEntries(
          [...['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Year']]
            .map((c) => [c, (v) => (typeof v === 'number'
              ? `<span class="${signClass(v)}">${pct(v, 1)}</span>` : '—')]),
        ),
      });
    } catch {
      $('#monthly-heatmap').innerHTML = '';
      $('#monthly-table').innerHTML = '<div class="empty">No monthly table.</div>';
    }
  },

  async signals() {
    const [data, bt, cfg, val, risk] = await Promise.all([
      api('/signals'),
      api('/backtest').catch(() => null),
      api('/config').catch(() => null),
      api('/validation').catch(() => null),
      api('/risk').catch(() => null),
    ]);
    const s = data.summary || {};
    const combined = bt?.metrics?.sleeves?.combined || {};
    const factors = cfg?.sleeve_a?.factors || {};
    state.backtestMeta = {
      name: cfg?.name,
      description: cfg?.description,
      sharpe: combined.sharpe,
      cagr: combined.cagr,
      excess: combined.excess_return,
      max_dd: combined.max_drawdown,
      hit_rate: combined.hit_rate,
      verdict: val?.verdict,
      factors,
      min_price: cfg?.universe?.min_price_inr,
      min_adv: cfg?.universe?.min_adv_inr,
      regime: risk?.regime || risk,
    };
    state.screener = ensureScreenerColumns(data.screener);
    state.screenerMeta = { top_n: s.top_n || 40, buffer_n: s.buffer_n || 80 };
    seedScanner(s, data.screener);
    fillScreenerRanges();
    fillScreenerMultis();
    fillColSets();
    try {
      await loadPaperSignals();
    } catch {
      state.paperSignals = null;
    }
    fillBacktestCard();
    fillScreenerPresets();

    const banner = $('#kite-book-banner');
    const src = String(s.book_source || '');
    const nHeld = s.n_held ?? 0;
    if (src === 'upload') {
      banner.className = 'card hint';
      banner.innerHTML = `Sized against your <b>uploaded portfolio</b> (${nHeld} names, NAV ${inr(s.nav)}). Prices from NSE EOD. <a href="#book">Open Book tab</a> to replace the file.`;
    } else if (src === 'kite') {
      banner.className = 'card hint';
      banner.innerHTML = `Sized against your live Kite book (${nHeld} holdings, NAV ${inr(s.nav)}). NSE EOD fills last/PE.`;
    } else {
      banner.className = 'card hint warn';
      banner.innerHTML = `${s.kite_status || 'No portfolio loaded.'} Upload a CSV on the <a href="#book">Book tab</a> (or <a class="kite-login" href="/kite/login">connect Kite</a>) so the screen can emit <b>SELL</b> as well as BUY. Until then this is a research screener — no sized orders.`;
    }
    banner.querySelectorAll('a[href="#book"]').forEach((a) => {
      a.onclick = (e) => { e.preventDefault(); switchView('holdings'); };
    });

    const grid = $('#signals-metrics');
    grid.innerHTML = '';
    grid.appendChild(metricCard('Book NAV', inr(s.nav)));
    grid.appendChild(metricCard('Cash', inr(s.cash)));
    grid.appendChild(metricCard('Buys', s.n_buys ?? 0, 'pos'));
    grid.appendChild(metricCard('Buy value', inr(s.buy_notional), 'pos'));
    grid.appendChild(metricCard('Sells', s.n_sells ?? 0, 'neg'));
    grid.appendChild(metricCard('Sell value', inr(s.sell_notional), 'neg'));
    grid.appendChild(metricCard('Queued', s.n_queued ?? 0));
    grid.appendChild(metricCard('Holds', s.n_holds ?? 0));
    grid.appendChild(metricCard('Entry zone', `top ${s.top_n ?? '—'}`));
    grid.appendChild(metricCard('Sell after rank', s.buffer_n ?? '—'));
    grid.appendChild(metricCard('Slots free', s.n_room ?? '—'));
    grid.appendChild(metricCard('Holdings in buffer', s.n_keep ?? '—'));
    grid.appendChild(metricCard('Equities held', s.n_equity_held ?? 0));
    grid.appendChild(metricCard('ETFs held', s.n_etf_held ?? 0));
    grid.appendChild(metricCard('Mutual funds', s.n_mf_held ?? 0));
    grid.appendChild(metricCard('Live quotes', s.n_quoted ?? 0));
    grid.appendChild(metricCard('Lit movers', state.scanner.lit.size));

    Charts.drawDonut('#action-mix', [
      { label: 'BUY', value: s.n_buys || 0, color: '#5dff6b' },
      { label: 'SELL', value: s.n_sells || 0, color: '#ff4d4d' },
      { label: 'HOLD', value: s.n_holds || 0, color: '#ff9900' },
      { label: 'WATCH', value: s.n_watch || 0, color: '#7ec8ff' },
    ], { center: (s.n_buys || 0) + (s.n_sells || 0) });
    Charts.drawDonut('#asset-mix', [
      { label: 'Equity', value: s.n_equity_held || 0, color: '#ff9900' },
      { label: 'ETF', value: s.n_etf_held || 0, color: '#5dff6b' },
      { label: 'MF', value: s.n_mf_held || 0, color: '#7ec8ff' },
    ], { center: s.n_held ?? 0, empty: 'Upload a portfolio on the Book tab to see the mix.' });
    const topN = Number(s.top_n) || 40;
    const used = Number(s.n_keep) || 0;
    Charts.drawGauge('#slot-gauge', topN ? used / topN : 0, {
      label: `${used} / ${topN}`,
      color: used >= topN ? '#ffb000' : '#ff9900',
    });

    const rules = $('#signals-rules');
    rules.innerHTML = '';
    (s.rules || []).forEach((r) => {
      const li = document.createElement('li');
      li.innerHTML = `<b>${r.action}</b> — ${r.text}`;
      rules.appendChild(li);
    });
    if (!s.rules?.length) {
      rules.innerHTML = '<li>No rule text on this run. Rank buffer still applies.</li>';
    }

    const signalFmt = {
      action: (v) => el('span', `tag ${String(v).toLowerCase()}`, v),
      composite: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      momentum: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      quality: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      value: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      low_vol: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      conviction: (v) => (typeof v === 'number' ? num(v, 2) : '—'),
      current_weight: (v) => (typeof v === 'number' ? pct(v, 1) : '—'),
      target_weight: (v) => (typeof v === 'number' ? pct(v, 1) : '—'),
      last_price: (v) => (typeof v === 'number' ? inr(v) : '—'),
      current_value: (v) => (typeof v === 'number' ? inr(v) : '—'),
      delta_value: (v) => (typeof v === 'number'
        ? `<span class="${signClass(v)}">${v >= 0 ? '+' : ''}${inr(v)}</span>` : '—'),
      pnl: (v) => (typeof v === 'number'
        ? `<span class="${signClass(v)}">${inr(v)}</span>` : '—'),
      held: (v) => (v ? 'yes' : 'no'),
      current_qty: (v) => (typeof v === 'number' ? num(v, 0) : '—'),
      target_qty: (v) => (typeof v === 'number' ? num(v, 0) : '—'),
      delta_qty: (v) => (typeof v === 'number'
        ? `<span class="${signClass(v)}">${v > 0 ? '+' : ''}${num(v, 0)}</span>` : '—'),
    };
    const cols = ['action', 'symbol', 'asset_class', 'sector', 'rank', 'composite', 'momentum',
                  'current_qty', 'target_qty', 'delta_qty', 'last_price',
                  'delta_value', 'target_weight', 'reason'];

    const bars = (table, color) => (table.rows || []).slice(0, 18).map((row) => {
      const obj = {};
      table.columns.forEach((c, i) => { obj[c] = row[i]; });
      const v = Number(obj.delta_value);
      return {
        label: String(obj.symbol || obj.isin || ''),
        value: Number.isFinite(v) && v !== 0 ? Math.abs(v) : (Number(obj.conviction) || 0),
        color,
      };
    });

    const buyBars = [...bars(data.buys || {}, '#3ecf8e'), ...bars(data.queued || {}, '#f5a623')];
    Charts.drawBars('#buy-chart', buyBars, {
      format: (v) => (v >= 1000 ? inr(v) : num(v, 2)),
      color: (d) => (d.color || '#3ecf8e'),
      empty: 'No buy candidates in the entry zone.',
    });
    Charts.drawBars('#sell-chart', bars(data.sells || {}, '#ff5f6d'), {
      format: (v) => (v >= 1000 ? inr(v) : num(v, 2)),
      color: () => '#ff5f6d',
      empty: src === 'empty'
        ? 'No sells until a portfolio is loaded. Upload a CSV on the Book tab.'
        : 'No sells. Every holding is still inside the rank buffer.',
    });

    renderTable('#buy-table', mergeTables(data.buys, data.queued), {
      columns: cols, format: signalFmt, colorize: ['composite', 'momentum'],
    });
    renderTable('#sell-table', {
      ...(data.sells || {}),
      note: (data.sells && data.sells.rows && data.sells.rows.length)
        ? undefined
        : 'No sells versus your book. Upload holdings on the Book tab, or every name is still inside the buffer.',
    }, { columns: cols, format: signalFmt, colorize: ['composite', 'momentum'] });
    renderTable('#watch-table', data.watch, { columns: cols, limit: 80, format: signalFmt });
    renderTable('#hold-table', data.holds, { columns: cols, limit: 80, format: signalFmt });
    renderScreener();
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
      $('#beta-chart').innerHTML = '';
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
      Charts.drawBars('#beta-chart',
        Object.entries(fm.betas || {}).map(([label, value]) => ({ label, value: Number(value) })),
        { diverging: true, format: (v) => num(v, 2), empty: 'No factor betas in this run.' });
      renderTable('#attribution-betas', {
        columns: ['factor', 'beta', 't_stat', 'p_value'], rows, n_rows: rows.length,
      });
    }

    renderTable('#ic-table', data.ic_icir || { columns: [], rows: [], note: 'No IC data.' });
  },

  async holdings() {
    const reb = await api('/rebalance').catch(() => ({ summary: {}, orders: null }));
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

    let book = null;
    let sig = null;
    try { book = await api('/book'); } catch (e) {
      $('#positions-table').innerHTML = `<div class="empty">${e.message}</div>`;
    }
    try { sig = await api('/signals'); } catch { /* optional */ }

    const metrics = $('#book-metrics');
    if (metrics) {
      metrics.innerHTML = '';
      const pos = book?.positions || {};
      metrics.appendChild(metricCard('Book', book?.source || 'empty'));
      metrics.appendChild(metricCard('Names', pos.n ?? 0));
      metrics.appendChild(metricCard('NSE matched', pos.n_matched ?? 0));
      metrics.appendChild(metricCard('Equity value', inr(book?.equity_value)));
      metrics.appendChild(metricCard('Cash', inr(book?.cash)));
      metrics.appendChild(metricCard('NAV', inr(book?.nav)));
      metrics.appendChild(metricCard('Buys', sig?.summary?.n_buys ?? 0, 'pos'));
      metrics.appendChild(metricCard('Sells', sig?.summary?.n_sells ?? 0, 'neg'));
    }
    const status = $('#book-upload-status');
    if (status) {
      if (book?.note) status.textContent = book.note;
      else if (book?.source === 'upload') status.textContent = `Loaded ${book.positions?.n ?? 0} names from your file. NSE filled last price / PE / 1-year %.`;
      else if (book?.source === 'kite') status.textContent = 'Using Kite holdings + positions. Upload a CSV to override.';
      else status.textContent = '';
    }
    if ($('#book-cash') && book && Number(book.cash) > 0 && !$('#book-cash').value) {
      $('#book-cash').value = book.cash;
    }

    const actionBySym = new Map();
    const table = sig?.signals || sig?.screener;
    if (table?.columns && table.rows) {
      const ai = table.columns.indexOf('action');
      const si = table.columns.indexOf('symbol');
      table.rows.forEach((r) => {
        const sym = String(r[si] || '').toUpperCase();
        if (sym) actionBySym.set(sym, r[ai]);
      });
    }

    const posRows = (book?.positions?.rows || []).map((p) => [
      actionBySym.get(String(p.symbol || '').toUpperCase()) || (p.matched_nse ? 'HOLD' : 'SELL'),
      p.symbol,
      p.name || p.symbol,
      p.qty,
      p.avg_price,
      p.last_price,
      p.pnl,
      p.value,
      p.pe,
      p.ret_1y,
      p.rank,
      p.sector,
      p.matched_nse ? 'NSE' : 'unmatched',
    ]);
    renderTable('#positions-table', {
      columns: ['action', 'symbol', 'name', 'qty', 'avg_price', 'last_price', 'pnl',
                'value', 'pe', 'ret_1y', 'rank', 'sector', 'nse'],
      rows: posRows,
      n_rows: posRows.length,
      note: posRows.length ? undefined : (book?.note || 'Upload a CSV to load positions.'),
    }, {
      colorize: ['pnl', 'ret_1y'],
      format: {
        action: (v) => el('span', `tag ${String(v).toLowerCase()}`, v),
        avg_price: (v) => (typeof v === 'number' ? inr(v) : '—'),
        last_price: (v) => (typeof v === 'number' ? inr(v) : '—'),
        pnl: (v) => (typeof v === 'number' ? `<span class="${signClass(v)}">${inr(v)}</span>` : '—'),
        value: (v) => (typeof v === 'number' ? inr(v) : '—'),
        pe: (v) => (typeof v === 'number' ? num(v, 1) : '—'),
        ret_1y: (v) => (typeof v === 'number' ? `<span class="${signClass(v)}">${pct(v, 1)}</span>` : '—'),
        qty: (v) => (typeof v === 'number' ? num(v, 0) : '—'),
      },
    });

    try {
      const h = await api('/holdings');
      const node = $('#holdings-table');
      if (h.note && !(h.holdings && h.holdings.length)) {
        node.innerHTML = '';
        node.appendChild(el('div', 'empty', h.note));
      } else if (h.holdings && h.holdings.length) {
        const rows = h.holdings.map((x) => [
          x.tradingsymbol, x.quantity, x.average_price, x.last_price, x.pnl,
          h.targets[x.tradingsymbol] ?? null,
        ]);
        renderTable(node, {
          columns: ['symbol', 'qty', 'avg_price', 'last_price', 'pnl', 'target_weight'],
          rows, n_rows: rows.length,
        }, { colorize: ['pnl'], format: { target_weight: (v) => (v == null ? '—' : pct(v)) } });
      } else {
        node.innerHTML = '<div class="empty">No broker holdings. The table above is your uploaded book.</div>';
      }
    } catch (e) {
      $('#holdings-table').innerHTML = `<div class="empty">${e.message}</div>`;
    }
  },

  async rankings() {
    const data = await api('/rankings');
    state.rankings = data;
    const symbolIdx = (data.columns || []).indexOf('symbol');
    const scoreIdx = (data.columns || []).indexOf('composite');
    if (symbolIdx >= 0 && scoreIdx >= 0) {
      const top = data.rows.slice(0, 18).map((r) => ({
        label: String(r[symbolIdx]),
        value: Number(r[scoreIdx]),
      }));
      Charts.drawBars('#rankings-chart', top, {
        diverging: true,
        format: (v) => num(v, 2),
        empty: 'No rankings in this run.',
      });
    }
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
    Charts.drawBars('#sector-chart',
      sectors.map(([label, value]) => ({ label, value: Number(value) })),
      { format: (v) => pct(v, 1), empty: 'No sector weights in this run.' });
    renderTable('#sector-table', {
      columns: ['sector', 'weight'], rows: sectors, n_rows: sectors.length,
    }, { format: { weight: (v) => pct(v) } });

    const cost = data.cost_sensitivity;
    Charts.drawXYLines('#cost-chart', Charts.tableToObjects(cost), 'cost_bps', ['sharpe', 'cagr'], {
      xLabel: 'Assumed cost (bps)',
      height: 280,
      dualY: true,
      yFormats: [(v) => num(v, 2), (v) => pct(v, 1)],
      empty: 'No cost-sensitivity table in this run.',
    });
    renderTable('#cost-table', cost || { columns: [], rows: [] });
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
    if (h.hosted || h.kite === 'hosted') {
      badge.textContent = 'Hosted';
      badge.className = 'badge ok';
      badge.style.cursor = 'default';
      badge.title = 'Static Netlify snapshot. Upload a portfolio on the Book tab. Kite stays on the local dashboard.';
      badge.onclick = null;
      return;
    }
    badge.textContent = `Kite: ${h.kite}`;
    badge.className = `badge ${h.kite === 'authenticated' ? 'ok' : h.kite === 'needs login' ? 'err' : ''}`;
    badge.style.cursor = h.kite === 'authenticated' ? 'default' : 'pointer';
    badge.onclick = () => {
      if (h.kite !== 'authenticated') location.href = '/kite/login';
    };
  } catch { /* status is best-effort */ }
}

function connectSockets() {
  if (window.StaticAPI?.mode === 'static') {
    const node = document.getElementById('live-status');
    if (node) node.textContent = 'Hosted site — live ticks run only on the local dashboard.';
    return;
  }
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
    ingestTick(tick);
  };
  state.ws.onopen = () => {
    if (state.view === 'live') $('#live-status').textContent = 'Streaming live ticks from Zerodha.';
    updateLiveHud();
  };
}

function seedScanner(summary, table) {
  const sc = state.scanner;
  (summary.directory || []).forEach((d) => {
    const sym = String(d.symbol || '').toUpperCase();
    if (d.token && sym) sc.tokenToSymbol.set(Number(d.token), sym);
    if (sym && d.prev_close > 0) sc.lastClose.set(sym, d.prev_close);
  });
  if (table?.columns) {
    const idx = Object.fromEntries(table.columns.map((c, i) => [c, i]));
    (table.rows || []).forEach((r) => {
      const sym = String(r[idx.symbol] || '').toUpperCase();
      if (!sym) return;
      const token = Number(r[idx.kite_token]);
      const prev = r[idx.prev_close];
      const last = r[idx.last_price];
      const close = idx.close >= 0 ? r[idx.close] : null;
      const chg = r[idx.day_chg];
      const vol = r[idx.volume];
      if (Number.isFinite(token) && token > 0) sc.tokenToSymbol.set(token, sym);
      if (typeof prev === 'number' && prev > 0) sc.lastClose.set(sym, prev);
      const lastPx = positiveNum(last) ?? positiveNum(close);
      if (lastPx != null || typeof chg === 'number') {
        sc.overlay.set(sym, {
          last: lastPx,
          prev: positiveNum(prev),
          chg: finiteNum(chg),
          vol: positiveNum(vol),
        });
      }
    });
  }
  updateLiveHud();
}

function ingestTick(tick) {
  if (!tick || tick.type === 'connected') return;
  state.ticks.set(tick.instrument_token, tick);
  const sc = state.scanner;
  const sym = tick.symbol || sc.tokenToSymbol.get(tick.instrument_token);
  if (sym) {
    const prev = (tick.ohlc_close > 0 ? tick.ohlc_close : null) || sc.lastClose.get(sym);
    const last = tick.last_price;
    const chg = tick.change != null
      ? tick.change
      : (prev > 0 ? ((last - prev) / prev) * 100 : null);
    sc.overlay.set(sym, { last, prev: prev || null, chg, vol: tick.volume ?? null });
    if (prev > 0) sc.lastClose.set(sym, prev);
    sc.ticks += 1;
    sc.lastAt = Date.now();
    const absR = prev > 0 ? Math.abs(last - prev) / prev : 0;
    if (absR > sc.threshold && !sc.lit.has(sym)) {
      sc.lit.add(sym);
      sc.alerts.unshift({
        ts: new Date(), symbol: sym, last, previous: prev, abs_return: absR,
      });
      if (sc.alerts.length > 50) sc.alerts.pop();
      renderTape();
    }
    patchScreenerRow(sym);
  }
  if (state.view === 'live') renderTicks();
  updateLiveHud();
}

function patchScreenerRow(sym) {
  const tr = document.querySelector(`#screener-table tr[data-symbol="${CSS.escape(sym)}"]`);
  if (!tr) return;
  const live = state.scanner.overlay.get(sym);
  if (!live) return;
  const table = state.screener;
  if (!table?.columns) return;
  const shown = ['action', 'symbol', 'asset_class', 'sector', 'rank', 'composite', 'momentum',
    'day_chg', 'last_price', 'prev_close', 'volume', 'quality', 'value', 'low_vol',
    'held', 'current_qty', 'target_qty', 'delta_qty', 'delta_value', 'reason']
    .filter((c) => table.columns.includes(c));
  const cells = tr.querySelectorAll('td');
  const set = (name, html) => {
    const i = shown.indexOf(name);
    if (i < 0 || !cells[i]) return;
    cells[i].innerHTML = html;
  };
  if (live.last != null) set('last_price', inr(live.last));
  if (live.prev != null) set('prev_close', inr(live.prev));
  if (live.chg != null) {
    set('day_chg', `<span class="${signClass(live.chg)}">${live.chg >= 0 ? '+' : ''}${live.chg.toFixed(2)}%</span>`);
  }
  if (live.vol != null) set('volume', Number(live.vol).toLocaleString('en-IN'));
  if (state.scanner.lit.has(sym)) tr.classList.add('lit');
}

function renderTape() {
  const box = $('#live-tape');
  if (!box) return;
  const alerts = state.scanner.alerts;
  box.hidden = alerts.length === 0;
  box.innerHTML = '';
  alerts.slice(0, 12).forEach((a) => {
    const line = el('div', 'alert');
    const ts = a.ts.toLocaleTimeString('en-IN', { hour12: false });
    line.innerHTML = `[${ts}] <b>${a.symbol}</b> moved by ${(a.abs_return * 100).toFixed(2)}% (current: ${a.last.toFixed(2)}, previous: ${Number(a.previous).toFixed(2)})`;
    box.appendChild(line);
  });
}

function updateLiveHud() {
  const hud = $('#live-hud');
  if (!hud) return;
  const sc = state.scanner;
  const wsOk = state.ws?.readyState === 1;
  const age = sc.lastAt ? `${Math.round((Date.now() - sc.lastAt) / 1000)}s ago` : 'none';
  hud.textContent = wsOk
    ? `${sc.ticks} ticks · ${sc.lit.size} lit · last ${age} · latch ${(sc.threshold * 100).toFixed(1)}%`
    : 'WebSocket idle. Connect Kite and keep this tab open during market hours.';
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
  const researchQuick = $('#run-research-quick');
  const researchFull = $('#run-research-full');
  if (researchQuick) {
    researchQuick.onclick = (e) => {
      e.target.disabled = true;
      $('#job-log').textContent = 'Starting nightly research (quick)…\n';
      post('/research/run', { full: false }).catch((err) => {
        $('#job-log').textContent += `ERROR: ${err.message}\n`;
      }).finally(() => setTimeout(() => { e.target.disabled = false; }, 1500));
    };
  }
  if (researchFull) {
    researchFull.onclick = (e) => {
      e.target.disabled = true;
      $('#job-log').textContent = 'Starting horse race (full)…\n';
      post('/research/run', { full: true }).catch((err) => {
        $('#job-log').textContent += `ERROR: ${err.message}\n`;
      }).finally(() => setTimeout(() => { e.target.disabled = false; }, 1500));
    };
  }

  const BOOK_TEMPLATE = 'Instrument,Qty.,Avg. cost\nINFY,10,1500\nRELIANCE,2,2400\nHDFCBANK,5,1600\n';

  async function uploadPortfolio() {
    const status = $('#book-upload-status');
    const cash = Number($('#book-cash')?.value || 0);
    const pasted = ($('#book-paste')?.value || '').trim();
    const file = $('#book-file')?.files?.[0];
    let csv = pasted;
    if (file) {
      csv = await file.text();
    }
    if (!csv && !(cash > 0)) {
      if (status) status.textContent = 'Paste a CSV or pick a file (symbol, qty, avg cost).';
      return;
    }
    status.textContent = 'Loading… fetching NSE EOD for your positions.';
    try {
      await post('/book', { csv: csv || undefined, cash: cash > 0 ? cash : undefined });
      status.textContent = 'Portfolio loaded. Sizing BUY and SELL against these positions.';
      await switchView('holdings');
    } catch (e) {
      status.textContent = e.message || String(e);
    }
  }

  const uploadBtn = $('#book-upload-btn');
  if (uploadBtn) uploadBtn.onclick = () => uploadPortfolio();
  const tmplBtn = $('#book-template-btn');
  if (tmplBtn) {
    tmplBtn.onclick = () => {
      const blob = new Blob([BOOK_TEMPLATE], { type: 'text/csv' });
      const a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'portfolio-template.csv';
      a.click();
      URL.revokeObjectURL(a.href);
    };
  }
  const clearBtn = $('#book-clear-btn');
  if (clearBtn) {
    clearBtn.onclick = async () => {
      try {
        await api('/book', { method: 'DELETE' });
        if ($('#book-paste')) $('#book-paste').value = '';
        if ($('#book-file')) $('#book-file').value = '';
        if ($('#book-cash')) $('#book-cash').value = '';
        await switchView('holdings');
      } catch (e) {
        const status = $('#book-upload-status');
        if (status) status.textContent = e.message;
      }
    };
  }

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

  [
    'screener-q', 'screener-action', 'screener-asset', 'screener-held',
    'screener-move-min', 'screener-move-dir', 'screener-tick-dir',
    ...SCREENER_RANGE_IDS,
    ...SCREENER_SELECT_IDS,
  ].forEach((id) => {
    const n = document.getElementById(id);
    if (n) n.addEventListener('input', () => renderScreener());
    if (n && n.tagName === 'SELECT') n.addEventListener('change', () => renderScreener());
  });
  ['screener-no-junk', 'screener-quoted', 'screener-lit'].forEach((id) => {
    const n = document.getElementById(id);
    if (n) n.addEventListener('change', () => renderScreener());
  });
  const clear = $('#screener-clear');
  if (clear) {
    clear.onclick = () => {
      state.screenerPreset = 'universe';
      state.activePresets = new Set();
      clearScreenerFilters();
      fillScreenerPresets();
      fillActiveFilters();
      renderScreener();
    };
  }
  const pageTo = (fn) => {
    fn();
    renderScreener({ keepPage: true });
  };
  const prev = $('#screener-prev');
  const next = $('#screener-next');
  const first = $('#screener-first');
  const last = $('#screener-last');
  if (prev) prev.onclick = () => pageTo(() => { state.screenerPage -= 1; });
  if (next) next.onclick = () => pageTo(() => { state.screenerPage += 1; });
  if (first) first.onclick = () => pageTo(() => { state.screenerPage = 0; });
  if (last) last.onclick = () => pageTo(() => { state.screenerPage = 1e9; });
  const psz = $('#screener-page-size');
  if (psz) {
    psz.onchange = () => {
      state.screenerPageSize = screenerPageSize();
      state.screenerPage = 0;
      renderScreener();
    };
  }
  const csv = $('#screener-csv');
  if (csv) csv.onclick = exportScreenerCsv;
  const dclose = $('#drawer-close');
  const dback = $('#drawer-back');
  if (dclose) dclose.onclick = closeProfile;
  if (dback) dback.onclick = closeProfile;
  document.addEventListener('click', (e) => {
    const toggle = e.target.closest('.ms-toggle');
    if (toggle) {
      const panel = toggle.parentElement.querySelector('.ms-panel');
      const willOpen = panel.hidden;
      document.querySelectorAll('.ms-panel').forEach((p) => { p.hidden = true; });
      document.querySelectorAll('.ms-toggle').forEach((b) => b.classList.remove('open'));
      panel.hidden = !willOpen;
      toggle.classList.toggle('open', willOpen);
      e.stopPropagation();
      return;
    }
    if (!e.target.closest('.ms-panel')) {
      document.querySelectorAll('.ms-panel').forEach((p) => { p.hidden = true; });
      document.querySelectorAll('.ms-toggle').forEach((b) => b.classList.remove('open'));
    }
  });
  document.querySelectorAll('.screener-rail-tabs button').forEach((b) => {
    b.onclick = () => switchRail(b.dataset.rail);
  });
  const th = $('#screener-threshold');
  if (th) {
    th.addEventListener('input', () => {
      state.scanner.threshold = (parseFloat(th.value) || 3) / 100;
      updateLiveHud();
    });
  }
  const reset = $('#screener-reset-latch');
  if (reset) {
    reset.onclick = () => {
      state.scanner.lit.clear();
      state.scanner.alerts = [];
      renderTape();
      renderScreener({ keepPage: true });
      updateLiveHud();
    };
  }

}

(async function init() {
  if (window.StaticAPI?.ready) await window.StaticAPI.ready;
  wireControls();
  await loadRuns();
  await loadStatus();
  connectSockets();
  await switchView('overview');
})();
