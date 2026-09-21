/* Static / Netlify backend for the Indian Equity Terminal.
 *
 * When /api/health is unreachable (the hosted site), every dashboard call is
 * served from /data/*.json plus a localStorage book. BUY and SELL both fire
 * against that book. This file never places orders.
 */
(function () {
  const BOOK_KEY = 'lt-terminal-book-v1';
  const ETF_TOP_N = 8;
  const ETF_TOKENS = ['ETF', 'BEES', 'IETF', 'SETFNIF', 'SETFNN50', 'MON100', 'MOSNIFTY',
    'NIFTYBEES', 'GOLDBEES', 'BANKBEES', 'LIQUIDBE', 'JUNIORBE'];
  const MIN_ORDER = 5000;

  const cache = new Map();

  const api = {
    mode: 'unknown',
    manifest: null,
    ready: null,
  };

  async function init() {
    try {
      const r = await fetch('/api/health', { cache: 'no-store' });
      if (r.ok) {
        const body = await r.json().catch(() => ({}));
        if (body && body.status === 'ok' && !body.hosted) {
          api.mode = 'live';
          return;
        }
      }
    } catch {
      /* hosted site, or local assets without the Rust server */
    }
    api.mode = 'static';
    api.manifest = await loadJson('/data/manifest.json', { run_id: null, hosted: true, files: [] });
  }

  api.ready = init();

  async function loadJson(url, fallback) {
    if (cache.has(url)) return cache.get(url);
    try {
      const r = await fetch(url, { cache: 'no-store' });
      if (!r.ok) throw new Error(`${url} ${r.status}`);
      const body = await r.json();
      cache.set(url, body);
      return body;
    } catch (e) {
      if (fallback !== undefined) return fallback;
      throw e;
    }
  }

  function data(name) {
    return loadJson(`/data/${name}`);
  }

  function emptyTable(note) {
    return { columns: [], rows: [], n_rows: 0, note: note || 'No data on this hosted snapshot.' };
  }

  function asTable(doc) {
    if (!doc || !Array.isArray(doc.columns)) return emptyTable();
    return {
      columns: doc.columns,
      rows: doc.rows || [],
      n_rows: (doc.rows || []).length,
      note: doc.note,
    };
  }

  function col(table, name) {
    return table.columns.indexOf(name);
  }

  function cell(row, idx) {
    if (idx < 0) return null;
    const v = row[idx];
    return v === undefined ? null : v;
  }

  function num(v) {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }

  function classify(isin, symbol, exchange) {
    const isinU = String(isin || '').toUpperCase();
    const sym = String(symbol || '').toUpperCase();
    const ex = String(exchange || '').toUpperCase();
    if (ex === 'MF' || ex === 'MFSS') return 'mf';
    if (isinU.startsWith('INF') || ETF_TOKENS.some((t) => sym.includes(t))) return 'etf';
    return 'equity';
  }

  function normalizeSymbol(raw) {
    let s = String(raw || '').trim().toUpperCase();
    const m = s.match(/^(NSE|BSE|NFO|MCX|CDS|MF):(.+)$/);
    if (m) s = m[2].trim();
    s = s.replace(/-(EQ|BE|BL|BZ|SM|ST|IL)$/, '').replace(/\.(NS|BO)$/, '');
    return s.trim();
  }

  function parseNumber(raw) {
    const t = String(raw || '').trim().replace(/[",₹%]/g, '').replace(/,/g, '');
    if (!t || t === '-' || t === '—' || t.toLowerCase() === 'na') return null;
    const n = Number(t);
    return Number.isFinite(n) ? n : null;
  }

  function canonHeader(raw) {
    const h = String(raw || '').trim().toLowerCase().replace(/[._-]/g, ' ').replace(/\s+/g, ' ').trim();
    if (['symbol', 'tradingsymbol', 'ticker', 'instrument', 'nse symbol', 'scrip', 'stock',
      'stock name', 'name', 'scrip name', 'trading symbol'].includes(h)) return 'symbol';
    if (['qty', 'quantity', 'qty available', 'shares', 'units', 'net qty', 'net quantity', 'filled qty'].includes(h)) return 'qty';
    if (['avg', 'avg price', 'average price', 'avg cost', 'average cost', 'buy price'].includes(h)) return 'avg_price';
    if (['ltp', 'last', 'last price', 'close', 'price', 'cmp'].includes(h)) return 'last_price';
    if (h === 'isin') return 'isin';
    if (['exchange', 'exch', 'ex'].includes(h)) return 'exchange';
    return h.replace(/ /g, '_');
  }

  function splitLine(line, delim) {
    const out = [];
    let cur = '';
    let q = false;
    for (const ch of line) {
      if (ch === '"') { q = !q; continue; }
      if (ch === delim && !q) { out.push(cur.trim()); cur = ''; }
      else cur += ch;
    }
    out.push(cur.trim());
    return out;
  }

  function parseCsv(text) {
    const raw = String(text || '').replace(/^\uFEFF/, '').trim();
    if (!raw) throw new Error('empty file');
    const lines = raw.split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith('#') && l !== '---');
    if (!lines.length) throw new Error('no rows');
    const commas = (lines[0].match(/,/g) || []).length;
    const tabs = (lines[0].match(/\t/g) || []).length;
    const delim = tabs > commas ? '\t' : ',';
    const header = splitLine(lines[0], delim).map(canonHeader);
    const hasHeader = header.some((h) => ['symbol', 'qty', 'avg_price', 'last_price', 'isin', 'exchange'].includes(h));
    const rows = [];
    let cash = 0;
    const body = hasHeader ? lines.slice(1) : lines;
    for (const line of body) {
      const cols = splitLine(line, delim);
      const get = (key) => {
        const i = header.indexOf(key);
        return i >= 0 ? (cols[i] || '') : '';
      };
      const symbol = normalizeSymbol(hasHeader ? get('symbol') : cols[0]);
      if (!symbol) continue;
      const qty = hasHeader ? (parseNumber(get('qty')) || 0) : (parseNumber(cols[1]) ?? 1);
      const avg = hasHeader ? (parseNumber(get('avg_price')) || 0) : (parseNumber(cols[2]) || 0);
      const last = hasHeader ? (parseNumber(get('last_price')) || 0) : (parseNumber(cols[3]) || 0);
      const isin = hasHeader ? get('isin').trim().toUpperCase() : '';
      const exchange = hasHeader ? (get('exchange').trim().toUpperCase() || 'NSE') : 'NSE';
      if (symbol === 'CASH' || symbol === 'LIQUID') {
        cash += qty > 0 ? qty : Math.max(last, avg);
        continue;
      }
      if (qty <= 0) continue;
      rows.push({ symbol, isin, qty, avg_price: avg, last_price: last, exchange });
    }
    if (!rows.length && cash <= 0) throw new Error('no holdings found — need symbol and qty columns');
    return { rows, cash };
  }

  function loadBook() {
    try {
      return JSON.parse(localStorage.getItem(BOOK_KEY) || 'null') || { present: false, cash: 0, rows: [] };
    } catch {
      return { present: false, cash: 0, rows: [] };
    }
  }

  function saveBook(book) {
    localStorage.setItem(BOOK_KEY, JSON.stringify(book));
  }

  function tableToSeries(table, timeCol) {
    const ti = col(table, timeCol || 'date');
    if (ti < 0) return null;
    const t = table.rows.map((r) => {
      const v = r[ti];
      if (typeof v === 'number') return v > 1e11 ? Math.floor(v / 1000) : v;
      const s = String(v || '').slice(0, 10);
      const ms = Date.parse(`${s}T00:00:00Z`);
      return Number.isFinite(ms) ? Math.floor(ms / 1000) : 0;
    });
    const series = {};
    table.columns.forEach((c, i) => {
      if (i === ti) return;
      const vals = table.rows.map((r) => {
        const n = Number(r[i]);
        return Number.isFinite(n) ? n : null;
      });
      if (vals.some((v) => v != null)) series[c] = vals;
    });
    return { t, series };
  }

  function lookup(book, isin, symbol) {
    const i = String(isin || '').toUpperCase();
    const s = String(symbol || '').toUpperCase();
    return book.find((n) => (i && n.isin === i) || (s && n.symbol === s));
  }

  function suggest(rankings, live, config) {
    const topN = Number(config?.sleeve_a?.top_n) || 40;
    const bufferMult = Number(config?.sleeve_a?.rank_buffer_multiple) || 2;
    const momW = Number(config?.sleeve_a?.factors?.weight_momentum) || 0;
    const qW = Number(config?.sleeve_a?.factors?.weight_quality) || 0;
    const lvW = Number(config?.sleeve_a?.factors?.weight_low_vol) || 0;
    const qGate = config?.sleeve_a?.factors?.value_quality_gate !== false;
    const maxW = Number(config?.sleeve_a?.max_weight_per_stock) || 0.05;
    const nScored = Math.max(1, rankings.n_rows);
    const bufferN = Math.min(nScored, Math.max(topN, Math.round(topN * Math.max(1, bufferMult))));
    const etfBuf = Math.round(ETF_TOP_N * Math.max(1, bufferMult));
    const bookLoaded = live.length > 0 || (loadBook().present === true);
    const idx = Object.fromEntries(rankings.columns.map((c, i) => [c, i]));
    const g = (row, name) => cell(row, idx[name] ?? -1);

    const momOn = momW > 0 && rankings.rows.some((r) => num(g(r, 'momentum')) != null);
    const hasQ = rankings.rows.some((r) => num(g(r, 'quality')) != null);
    const hasV = rankings.rows.some((r) => num(g(r, 'value')) != null);
    const hasLv = rankings.rows.some((r) => num(g(r, 'low_vol')) != null);
    const gateOn = qGate && hasQ && hasV;
    const qualityOn = qW > 0 && hasQ;
    const lvOn = lvW > 0 && hasLv;

    let keep = 0;
    let etfKeep = 0;
    const matched = new Set();
    rankings.rows.forEach((row) => {
      const isin = String(g(row, 'isin') || '');
      const symbol = String(g(row, 'symbol') || '');
      const rank = num(g(row, 'rank')) ?? 1e9;
      const asset = String(g(row, 'asset_class') || classify(isin, symbol, g(row, 'exchange')));
      if (lookup(live, isin, symbol)) {
        matched.add(isin || symbol);
        if (asset === 'etf' && rank <= etfBuf) etfKeep += 1;
        else if (asset === 'equity' && rank <= bufferN) keep += 1;
      }
    });
    const room = Math.max(0, topN - keep);
    const etfRoom = Math.max(0, ETF_TOP_N - etfKeep);
    let buyUsed = 0;
    let etfBuyUsed = 0;

    const equity = live.reduce((s, n) => s + (n.qty * (n.last_price || 0)), 0);
    const cash = Number(loadBook().cash) || 0;
    const nav = Math.max(0, equity + cash);

    const order = rankings.rows.map((_, i) => i)
      .sort((a, b) => (num(g(rankings.rows[a], 'rank')) ?? 1e9) - (num(g(rankings.rows[b], 'rank')) ?? 1e9));

    const objects = [];
    for (const i of order) {
      const row = rankings.rows[i];
      const isin = String(g(row, 'isin') || '');
      const symbol = String(g(row, 'symbol') || isin);
      if (!isin && !symbol) continue;
      const heldRow = lookup(live, isin, symbol);
      const isHeld = Boolean(heldRow);
      const rank = num(g(row, 'rank')) ?? 1e9;
      const asset = String(g(row, 'asset_class') || classify(isin, symbol, heldRow?.exchange || g(row, 'exchange')));
      const isEtf = asset === 'etf';
      const useTop = isEtf ? ETF_TOP_N : topN;
      const useBuf = isEtf ? etfBuf : bufferN;
      const useRoom = isEtf ? etfRoom : room;
      const slot = isEtf ? () => (++etfBuyUsed) : () => (++buyUsed);
      const used = isEtf ? etfBuyUsed : buyUsed;
      const { action, rules, reason } = decide({
        isHeld, rank, useTop, useBuf, useRoom, used, slot,
        momentum: num(g(row, 'momentum')),
        quality: num(g(row, 'quality')),
        value: num(g(row, 'value')),
        lowVol: num(g(row, 'low_vol')),
        absMom: num(g(row, 'abs_momentum')),
        profitability: num(g(row, 'profitability')),
        momOn, gateOn: gateOn && !isEtf, qualityOn: qualityOn && !isEtf, lvOn, dualOn: isEtf,
        asset, bookLoaded,
      });
      if (!action) continue;
      const last = (heldRow && heldRow.last_price) || num(g(row, 'last_price')) || num(g(row, 'close')) || 0;
      const qty = heldRow ? heldRow.qty : 0;
      const avg = heldRow ? heldRow.avg_price : 0;
      const value = qty * last;
      const pnl = heldRow && heldRow.pnl ? heldRow.pnl : (avg > 0 && last > 0 ? (last - avg) * qty : 0);
      const obj = {};
      rankings.columns.forEach((c, ci) => { obj[c] = row[ci]; });
      Object.assign(obj, {
        symbol, isin, action, sleeve: isEtf ? 'ETF' : 'A',
        held: isHeld, current_qty: qty, last_price: last || null,
        current_value: value || null, current_weight: nav > 0 ? value / nav : null,
        target_weight: null, target_qty: null, delta_qty: null, delta_value: null,
        conviction: conviction(action, num(g(row, 'composite')), rank, useTop, useBuf),
        rules, reason, pnl: pnl || null, asset_class: asset,
        kite_token: null,
      });
      objects.push(obj);
    }

    live.forEach((n) => {
      if (matched.has(n.isin) || matched.has(n.symbol)) return;
      const last = n.last_price || 0;
      const value = n.qty * last;
      const action = n.asset_class === 'mf' ? 'HOLD' : n.asset_class === 'etf' ? 'WATCH' : 'SELL';
      objects.push({
        symbol: n.symbol, isin: n.isin, action,
        sleeve: n.asset_class === 'mf' ? 'MF' : n.asset_class === 'etf' ? 'ETF' : 'A',
        rank: null, composite: null, quality: null, value: null, momentum: null, low_vol: null,
        sector: 'UNKNOWN', held: true, current_qty: n.qty, last_price: last || null,
        current_value: value || null, current_weight: nav > 0 ? value / nav : null,
        target_weight: action === 'SELL' ? 0 : null, target_qty: 0,
        delta_qty: action === 'SELL' ? -n.qty : 0,
        delta_value: action === 'SELL' ? -value : 0,
        conviction: 0.5,
        rules: n.asset_class === 'mf' ? 'mf_satellite' : n.asset_class === 'etf' ? 'etf_unscored' : 'outside_universe',
        reason: n.asset_class === 'mf'
          ? 'Mutual-fund folio: keep as a satellite. Prefer a listed ETF when adding.'
          : n.asset_class === 'etf'
            ? 'Listed ETF is not in the liquid momentum/low-vol sleeve yet. Review; do not auto-sell.'
            : 'Not in the current research universe. Review and sell if it no longer belongs.',
        pnl: n.pnl || null, asset_class: n.asset_class, name: n.symbol, exchange: n.exchange || 'NSE',
      });
    });

    applyTargets(objects, maxW, nav);

    const columns = [
      'symbol', 'isin', 'action', 'sleeve', 'rank', 'composite', 'quality', 'value',
      'momentum', 'low_vol', 'sector', 'held', 'current_qty', 'last_price',
      'current_value', 'current_weight', 'target_weight', 'target_qty',
      'delta_qty', 'delta_value', 'conviction', 'rules', 'reason', 'pnl', 'asset_class',
      'kite_token', 'prev_close', 'day_chg', 'volume',
      'name', 'exchange', 'country', 'industry', 'close', 'pe', 'pb', 'market_cap',
      'roe', 'debt_equity', 'div_yield', 'payout', 'ret_1w', 'ret_1m', 'ret_3m', 'ret_6m',
      'ret_1y', 'ret_3y', 'revenue', 'eps', 'net_income', 'earnings_yield', 'eps_ttm',
      'net_margin', 'pretax_margin', 'gross_profitability', 'profitability', 'growth',
      'safety', 'payout_z',
    ];
    const ord = { SELL: 0, BUY: 1, WATCH: 2, HOLD: 3 };
    objects.sort((a, b) => (ord[a.action] ?? 9) - (ord[b.action] ?? 9) || (b.conviction || 0) - (a.conviction || 0));
    const rows = objects.map((o) => columns.map((c) => (o[c] === undefined ? null : o[c])));
    const table = { columns, rows, n_rows: rows.length };
    const filter = (action) => {
      const rs = rows.filter((r) => r[2] === action);
      return { columns, rows: rs, n_rows: rs.length };
    };
    const queued = {
      columns,
      rows: rows.filter((r) => r[2] === 'WATCH' && Number(r[4]) <= topN),
      n_rows: 0,
    };
    queued.n_rows = queued.rows.length;
    const buys = filter('BUY');
    const sells = filter('SELL');
    const sumDelta = (t) => t.rows.reduce((s, r) => s + (Number(r[19]) || 0), 0);
    const book = loadBook();
    const source = book.present ? 'upload' : (live.length ? 'kite' : 'empty');
    return {
      summary: {
        as_of_ranks: nScored, top_n: topN, buffer_n: bufferN, n_keep: keep, n_room: room,
        n_held: live.length, n_buys: buys.n_rows, n_sells: sells.n_rows,
        n_holds: filter('HOLD').n_rows, n_watch: filter('WATCH').n_rows, n_queued: queued.n_rows,
        n_equity_held: live.filter((n) => n.asset_class === 'equity').length,
        n_etf_held: live.filter((n) => n.asset_class === 'etf').length,
        n_mf_held: live.filter((n) => n.asset_class === 'mf').length,
        equity_value: equity, cash, nav,
        buy_notional: sumDelta(buys), sell_notional: Math.abs(sumDelta(sells)),
        book_source: source,
        kite_status: source === 'upload'
          ? `uploaded portfolio (${live.length} names)`
          : 'Hosted site — upload a CSV on the Book tab. Kite login runs only on the local dashboard.',
        login_url: '/kite/login',
        source: 'nse_screener',
        n_quoted: 0,
        rules: [
          { id: 'rank_buffer_entry', action: 'BUY', text: `Buy names that enter the top ${topN} when a slot is free.` },
          { id: 'rank_buffer_exit', action: 'SELL', text: `Sell a holding only after its rank falls past ${bufferN} (top ${topN} × buffer).` },
          { id: 'no_book', action: 'WATCH', text: 'Without a loaded book the screen does not emit BUY. Upload your portfolio so sells can fire on what you hold.' },
          { id: 'outside_universe', action: 'SELL', text: 'A holding that is not in the current research universe is flagged to review and sell.' },
        ],
        note: 'Decision support only — no orders are placed. Execute manually.',
      },
      signals: table,
      buys, sells, queued, watch: filter('WATCH'), holds: filter('HOLD'),
      screener: table,
      note: 'Decision support only — no orders are placed. Execute manually.',
    };
  }

  function decide(p) {
    const { isHeld, rank, useTop, useBuf, useRoom, used, asset, bookLoaded } = p;
    if (asset === 'mf') {
      return { action: isHeld ? 'HOLD' : 'WATCH', rules: 'mf_satellite', reason: 'Mutual-fund folio: satellite holding.' };
    }
    if (isHeld && rank > useBuf) {
      return { action: 'SELL', rules: 'rank_buffer_exit', reason: `Rank ${rank.toFixed(0)} is past the sell line (${useBuf}). The buffer no longer protects this holding.` };
    }
    if (isHeld && rank > useTop && asset === 'equity' && p.profitability != null && p.profitability < 0) {
      return { action: 'SELL', rules: 'profitability_broke', reason: `Rank ${rank.toFixed(0)} left the top ${useTop} and IIMA profitability turned negative. Close.` };
    }
    if (isHeld && rank <= useBuf) {
      return {
        action: 'HOLD', rules: 'rank_buffer_keep',
        reason: rank <= useTop
          ? `Rank ${rank.toFixed(0)} is still inside the top ${useTop}. Keep.`
          : `Rank ${rank.toFixed(0)} is outside the top ${useTop} but inside the buffer (sell only after ${useBuf}). Keep.`,
      };
    }
    if (!isHeld && rank <= useTop) {
      if (!bookLoaded) {
        return { action: 'WATCH', rules: 'no_book', reason: 'In the entry zone, but no portfolio is loaded. Upload a CSV on the Book tab so BUY and SELL can both be sized against what you hold.' };
      }
      if (p.profitability != null && p.profitability < 0 && asset === 'equity') {
        return { action: 'WATCH', rules: 'profitability_failed', reason: 'IIMA profitability is negative. Do not open a new long.' };
      }
      if (p.momOn && p.momentum != null && p.momentum <= 0) {
        return { action: 'WATCH', rules: 'momentum_failed', reason: 'In the entry zone, but 12–1 / 6–1 momentum is non-positive.' };
      }
      if (p.qualityOn && p.quality != null && p.quality < 0) {
        return { action: 'WATCH', rules: 'quality_failed', reason: 'Quality (QMJ) is negative. Do not start a new long.' };
      }
      if (p.gateOn && p.quality != null && p.value != null && p.value > 0 && p.quality < 0) {
        return { action: 'WATCH', rules: 'value_quality_gate', reason: 'Cheap but poor quality — the value×quality gate blocks a new buy.' };
      }
      if (used < useRoom) {
        p.slot();
        const left = useRoom - used - 1;
        return { action: 'BUY', rules: 'rank_buffer_entry', reason: `Rank ${rank.toFixed(0)} entered the top ${useTop} and a slot is free${left > 0 ? ` (${left} more after this name).` : '.'}` };
      }
      return { action: 'WATCH', rules: 'book_full', reason: `Rank ${rank.toFixed(0)} is in the entry zone but the book is full. Wait for a sell.` };
    }
    if (!isHeld && rank <= useBuf) {
      return { action: 'WATCH', rules: 'buffer_zone', reason: `Rank ${rank.toFixed(0)} is inside the buffer band (≤ ${useBuf}) but not yet in the top ${useTop}.` };
    }
    return { action: null, rules: '', reason: '' };
  }

  function conviction(action, composite, rank, topN, bufferN) {
    if (action === 'SELL') {
      const span = Math.max(1, bufferN);
      return Math.min(1, Math.max(0, (rank - bufferN) / span));
    }
    if (composite != null) return Math.min(1, Math.max(0, 0.5 + composite / 4));
    return Math.max(0, 1 - (rank - 1) / Math.max(topN, 1));
  }

  function applyTargets(objects, maxW, nav) {
    const selected = objects.filter((o) => o.action === 'BUY' || o.action === 'HOLD');
    const n = selected.length;
    if (!n) return;
    const cap = n * maxW < 1 ? 1 / n : maxW;
    const total = selected.reduce((s, o) => s + Math.max(0.1, Number(o.composite) || 0.1), 0);
    selected.forEach((o) => {
      const raw = total > 0 ? Math.max(0.1, Number(o.composite) || 0.1) / total : 1 / n;
      o.target_weight = Math.min(cap, raw);
    });
    objects.forEach((o) => {
      const last = Number(o.last_price) || 0;
      const qty = Number(o.current_qty) || 0;
      if (o.action === 'SELL') {
        o.target_weight = 0;
        o.target_qty = 0;
        o.delta_qty = -qty;
        o.delta_value = -qty * last;
        return;
      }
      if (o.action === 'WATCH' || o.action === 'PASS') {
        o.target_weight = 0;
        o.target_qty = qty;
        o.delta_qty = 0;
        o.delta_value = 0;
        return;
      }
      const tw = o.target_weight || 0;
      const tgt = last > 0 && nav > 0 ? Math.floor(tw * nav / last) : 0;
      let dq = tgt - qty;
      let dv = dq * last;
      if (Math.abs(dv) < MIN_ORDER) { dq = 0; dv = 0; }
      o.target_qty = tgt;
      o.delta_qty = dq;
      o.delta_value = dv;
    });
  }

  function enrichLive(live, rankings) {
    const si = col(rankings, 'symbol');
    const ii = col(rankings, 'isin');
    const li = col(rankings, 'last_price');
    const ci = col(rankings, 'close');
    const bySym = new Map();
    rankings.rows.forEach((r) => {
      const sym = String(cell(r, si) || '').toUpperCase();
      const isin = String(cell(r, ii) || '').toUpperCase();
      const px = num(cell(r, li)) || num(cell(r, ci));
      if (sym) bySym.set(sym, { isin, px });
    });
    live.forEach((n) => {
      const hit = bySym.get(n.symbol);
      if (!hit) return;
      if (!n.isin && hit.isin) n.isin = hit.isin;
      if (!(n.last_price > 0) && hit.px > 0) n.last_price = hit.px;
      if (n.avg_price > 0 && n.last_price > 0) n.pnl = (n.last_price - n.avg_price) * n.qty;
    });
  }

  function liveFromBook(rankings) {
    const book = loadBook();
    if (!book.present) return [];
    const live = (book.rows || []).map((r) => ({
      symbol: normalizeSymbol(r.symbol),
      isin: String(r.isin || '').toUpperCase(),
      qty: Number(r.qty) || 0,
      last_price: Number(r.last_price) || 0,
      avg_price: Number(r.avg_price) || 0,
      pnl: 0,
      exchange: r.exchange || 'NSE',
      asset_class: classify(r.isin, r.symbol, r.exchange),
    })).filter((n) => n.qty > 0 && (n.symbol || n.isin));
    if (rankings) enrichLive(live, rankings);
    return live;
  }

  async function rankingsTable() {
    const doc = await data('rankings.json').catch(() => null);
    return asTable(doc);
  }

  async function configDoc() {
    return data('strategy_config.json').catch(() => ({ sleeve_a: { top_n: 40, rank_buffer_multiple: 2 } }));
  }

  async function signalsPayload() {
    const rankings = await rankingsTable();
    const config = await configDoc();
    const live = liveFromBook(rankings);
    return suggest(rankings, live, config);
  }

  async function bookPayload() {
    const rankings = await rankingsTable();
    const live = liveFromBook(rankings);
    const book = loadBook();
    const si = col(rankings, 'symbol');
    const extras = ['name', 'sector', 'rank', 'pe', 'ret_1y', 'prev_close', 'earnings_yield', 'net_margin', 'composite'];
    const idx = Object.fromEntries(rankings.columns.map((c, i) => [c, i]));
    const bySym = new Map(rankings.rows.map((r) => [String(cell(r, si) || '').toUpperCase(), r]));
    const rows = live.map((n) => {
      const src = bySym.get(n.symbol);
      const extra = (k) => (src && idx[k] != null ? src[idx[k]] : null);
      return {
        symbol: n.symbol, isin: n.isin, qty: n.qty, avg_price: n.avg_price,
        last_price: n.last_price, pnl: n.pnl, value: n.qty * (n.last_price || 0),
        exchange: n.exchange, asset_class: n.asset_class, matched_nse: Boolean(src),
        name: extra('name'), sector: extra('sector'), rank: extra('rank'),
        pe: extra('pe'), ret_1y: extra('ret_1y'), prev_close: extra('prev_close'),
        earnings_yield: extra('earnings_yield'), net_margin: extra('net_margin'),
        composite: extra('composite'),
      };
    });
    const missing = rows.filter((r) => !r.matched_nse).map((r) => r.symbol);
    const equity = live.reduce((s, n) => s + n.qty * (n.last_price || 0), 0);
    const source = book.present ? 'upload' : 'empty';
    return {
      source,
      status: book.present ? `uploaded portfolio (${live.length} names)` : 'No portfolio loaded.',
      cash: book.cash || 0,
      equity_value: equity,
      nav: equity + (book.cash || 0),
      holdings: live.map((n) => ({
        tradingsymbol: n.symbol, isin: n.isin, quantity: n.qty,
        average_price: n.avg_price, last_price: n.last_price, pnl: n.pnl,
        exchange: n.exchange,
      })),
      positions: { n: live.length, n_matched: live.length - missing.length, missing, rows },
      note: book.present ? null : 'No portfolio loaded. Upload a CSV on the Book tab.',
    };
  }

  async function handle(path, options) {
    const method = (options?.method || 'GET').toUpperCase();
    const [rawPath, qs] = String(path || '').split('?');
    const route = rawPath.replace(/^\//, '').split('/')[0];
    const params = new URLSearchParams(qs || '');

    if (method === 'POST' && /\/run$/.test(rawPath)) {
      throw new Error('Jobs (backtest, horse race, rebalance) run on the local dashboard, not on the hosted site.');
    }

    if (route === 'health') {
      const manifest = api.manifest || {};
      const dh = await data('data_health.json').catch(() => null);
      return {
        status: 'ok', hosted: true, kite: 'hosted',
        latest_run: manifest.run_id, data_health: dh,
        book: { present: loadBook().present, n: (loadBook().rows || []).length },
      };
    }
    if (route === 'runs') {
      const id = api.manifest?.run_id;
      return { latest: id, run_ids: id ? [id] : [], registry: emptyTable('Hosted snapshot of the latest run.') };
    }
    if (route === 'backtest') {
      const metrics = await data('backtest.json');
      const curve = asTable(await data('equity_curve.json').catch(() => null));
      const series = tableToSeries(curve, 'date');
      return {
        run_id: metrics.run_id || api.manifest?.run_id,
        metrics,
        equity_curve: series ? { n_points: series.t.length, run_id: metrics.run_id, data: series } : null,
      };
    }
    if (route === 'rankings') return rankingsTable();
    if (route === 'signals' || route === 'screener') return signalsPayload();
    if (route === 'config') return configDoc();
    if (route === 'validation') return data('validation_stats.json');
    if (route === 'attribution') {
      const attribution = await data('attribution.json').catch(() => ({ note: 'No attribution on this snapshot.' }));
      const ic = asTable(await data('ic_icir.json').catch(() => null));
      return { attribution, ic_icir: ic };
    }
    if (route === 'risk') {
      const risk = await data('risk.json');
      const costs = asTable(await data('cost_sensitivity.json').catch(() => null));
      const regime = asTable(await data('regime_breakdown.json').catch(() => null));
      return { risk, cost_sensitivity: costs.n_rows ? costs : null, regime: regime.n_rows ? regime : null };
    }
    if (route === 'monthly') return asTable(await data('monthly_returns.json'));
    if (route === 'rebalance') {
      const orders = asTable(await data('rebalance_orders.json').catch(() => null));
      const targets = asTable(await data('holdings_target.json').catch(() => null));
      const summary = await data('rebalance_summary.json').catch(() => ({}));
      if (!orders.n_rows) orders.note = 'No rebalance generated yet.';
      return { orders, targets, summary };
    }
    if (route === 'optimize') {
      return { results: emptyTable('Optimization is a local job. Run it from the desktop dashboard.'), summary: {} };
    }
    if (route === 'statarb') {
      return { pairs: emptyTable('Sleeve B is not in this snapshot.'), signals: emptyTable() };
    }
    if (route === 'events') {
      return { calendar: emptyTable('No earnings calendar in this snapshot.'), signals: emptyTable() };
    }
    if (route === 'journal') {
      return { entries: emptyTable('Import the tradebook on the local dashboard.'), drift: null };
    }
    if (route === 'history') {
      const symbol = (params.get('symbol') || '').trim().toUpperCase();
      if (!symbol) throw new Error('pass ?symbol=INFY');
      const sparks = await loadJson('/sparks.json', await data('sparks.json').catch(() => null));
      const closes = sparks?.closes?.[symbol];
      if (!Array.isArray(closes) || closes.length < 2) {
        throw new Error(`No NSE EOD series for ${symbol}.`);
      }
      return {
        symbol, source: 'nse_eod', n_points: closes.length,
        data: { t: closes.map((_, i) => i), series: { close: closes } },
        note: sparks.note || 'NSE EOD last ~1 year. Decision support only.',
      };
    }
    if (route === 'book') {
      if (method === 'DELETE') {
        saveBook({ present: false, cash: 0, rows: [] });
        return { ok: true, source: 'empty' };
      }
      if (method === 'POST') {
        const body = JSON.parse(options?.body || '{}');
        let rows = Array.isArray(body.rows) ? body.rows.slice() : [];
        let cash = Number(body.cash) || 0;
        if (body.csv && String(body.csv).trim()) {
          const trimmed = String(body.csv).replace(/^\uFEFF/, '').trim();
          if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
            const parsed = JSON.parse(trimmed);
            const list = Array.isArray(parsed) ? parsed : (parsed.rows || []);
            rows = rows.concat(list);
            cash += Number(parsed.cash) || 0;
          } else {
            const parsed = parseCsv(trimmed);
            rows = rows.concat(parsed.rows);
            cash += parsed.cash;
          }
        }
        const bySym = new Map();
        rows.forEach((r) => {
          const symbol = normalizeSymbol(r.symbol || r.tradingsymbol || r.ticker);
          if (!symbol) return;
          bySym.set(symbol, {
            symbol,
            isin: String(r.isin || '').toUpperCase(),
            qty: Number(r.qty ?? r.quantity) || 0,
            avg_price: Number(r.avg_price ?? r.average_price) || 0,
            last_price: Number(r.last_price) || 0,
            exchange: String(r.exchange || 'NSE').toUpperCase(),
          });
        });
        const merged = [...bySym.values()].filter((r) => r.qty > 0);
        if (!merged.length && cash <= 0) throw new Error('need at least one holding (symbol, qty) or a cash amount');
        saveBook({ present: true, cash, rows: merged, uploaded_at: new Date().toISOString() });
        return bookPayload();
      }
      return bookPayload();
    }
    if (route === 'holdings') {
      const b = await bookPayload();
      return {
        holdings: b.holdings, total_value: b.equity_value, total_pnl: (b.holdings || []).reduce((s, h) => s + (h.pnl || 0), 0),
        targets: {}, note: b.note,
      };
    }
    if (route === 'quotes' || route === 'positions' || route === 'trades') {
      throw new Error('Broker endpoints are not available on the hosted site. Upload a CSV on the Book tab.');
    }
    throw new Error(`Unknown endpoint ${rawPath} on the hosted snapshot.`);
  }

  api.handle = handle;
  window.StaticAPI = api;
})();
