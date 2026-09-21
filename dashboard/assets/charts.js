/* D3 charts for the Indian Equity Alpha dashboard. */

const Charts = (() => {
  const PALETTE = ['#4c8dff', '#3ecf8e', '#f5a623', '#ff5f6d', '#b78bff', '#5ad0e6'];
  const LABELS = {
    equity: 'Strategy',
    after_tax_equity: 'After tax',
    benchmark: 'Nifty 50 TRI',
    drawdown: 'Drawdown',
    sharpe: 'Sharpe',
    cagr: 'CAGR',
  };

  const pretty = (key) => LABELS[key] || String(key).replace(/_/g, ' ');

  function compact(v) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '';
    const n = Number(v);
    const a = Math.abs(n);
    if (a >= 1e7) return `${(n / 1e7).toFixed(1)} Cr`;
    if (a >= 1e5) return `${(n / 1e5).toFixed(1)} L`;
    if (a >= 1e3) return `${(n / 1e3).toFixed(0)}k`;
    return n.toFixed(Math.abs(n) < 10 ? 2 : 0);
  }

  function pct(v, digits = 2) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
    return `${(Number(v) * 100).toFixed(digits)}%`;
  }

  function num(v, digits = 2) {
    if (v === null || v === undefined || Number.isNaN(Number(v))) return '—';
    return Number(v).toFixed(digits);
  }

  function nodeOf(sel) {
    return typeof sel === 'string' ? document.querySelector(sel) : sel;
  }

  function ensureTooltip() {
    let tip = document.querySelector('.d3-tooltip');
    if (!tip) {
      tip = document.createElement('div');
      tip.className = 'd3-tooltip';
      tip.style.opacity = '0';
      document.body.appendChild(tip);
    }
    return tip;
  }

  function clearChart(node) {
    if (!node) return;
    node.innerHTML = '';
  }

  function emptyChart(node, msg) {
    clearChart(node);
    const d = document.createElement('div');
    d.className = 'empty';
    d.textContent = msg;
    node.appendChild(d);
  }

  function watchSize(node, draw) {
    if (typeof ResizeObserver === 'undefined') return;
    if (node._ro) node._ro.disconnect();
    let last = node.clientWidth;
    let t;
    node._ro = new ResizeObserver(() => {
      const w = node.clientWidth;
      if (Math.abs(w - last) < 2) return;
      last = w;
      clearTimeout(t);
      t = setTimeout(draw, 60);
    });
    node._ro.observe(node);
  }

  function chartBox(node, height, extra = {}) {
    const w = Math.max(node.clientWidth || 640, 280);
    const h = height || Math.max(node.clientHeight || 280, 180);
    const margin = {
      top: extra.top ?? 16,
      right: extra.right ?? 20,
      bottom: extra.bottom ?? 36,
      left: extra.left ?? 64,
    };
    return {
      w, h, margin,
      innerW: Math.max(w - margin.left - margin.right, 40),
      innerH: Math.max(h - margin.top - margin.bottom, 40),
    };
  }

  function firstPositive(values) {
    for (const v of values || []) {
      if (v != null && Number.isFinite(Number(v)) && Number(v) > 0) return Number(v);
    }
    return null;
  }

  function trimLeading(data, keys) {
    const n = data.t.length;
    let start = 0;
    while (start < n) {
      const any = keys.some((k) => {
        const v = data.series?.[k]?.[start];
        return v != null && Number.isFinite(Number(v));
      });
      if (any) break;
      start += 1;
    }
    if (start === 0) return data;
    const series = {};
    Object.keys(data.series || {}).forEach((k) => {
      series[k] = data.series[k].slice(start);
    });
    return { t: data.t.slice(start), series };
  }

  function rebaseData(data, keys, base = 100) {
    const series = { ...(data.series || {}) };
    keys.forEach((key) => {
      const vals = data.series?.[key] || [];
      const origin = firstPositive(vals);
      series[key] = origin
        ? vals.map((v) => (v != null && Number.isFinite(Number(v)) ? base * Number(v) / origin : null))
        : vals;
    });
    return { t: data.t, series };
  }

  function paddedDomain(values, opts = {}) {
    const nums = values.filter((v) => v != null && Number.isFinite(v));
    if (!nums.length) return [0, 1];
    const y0 = d3.min(nums);
    const y1 = d3.max(nums);
    const span = (y1 - y0) || Math.abs(y1) || 1;
    const pad = span * (opts.pad ?? (opts.percent ? 0.16 : 0.10));
    let lo = y0 - pad;
    let hi = y1 + pad;
    if (opts.zeroBaseline) {
      if (y0 >= 0) lo = 0;
      else if (y1 <= 0) hi = 0;
      else {
        lo = Math.min(0, lo);
        hi = Math.max(0, hi);
      }
    } else if (y0 > 0 && lo < y0 * 0.92) {
      // Keep strictly-positive series off a zero baseline so the path uses the full plot.
      lo = Math.max(y0 * 0.94, y0 - pad);
    } else if (y1 < 0 && hi > y1 * 0.92) {
      hi = Math.min(y1 * 0.94, y1 + pad);
    }
    if (lo === hi) {
      const bump = Math.abs(lo) * 0.05 || 1;
      lo -= bump;
      hi += bump;
    }
    return [lo, hi];
  }

  function seriesMaximaRatio(rows) {
    const maxima = rows.map((r) => d3.max(r.points, (p) => Math.abs(p.value)) || 0)
      .filter((v) => v > 0);
    if (maxima.length < 2) return 1;
    return d3.max(maxima) / d3.min(maxima);
  }

  function drawAxes(g, x, y, box, yFormat) {
    g.append('g')
      .attr('class', 'axis axis-x')
      .attr('transform', `translate(0,${box.innerH})`)
        .call(d3.axisBottom(x).ticks(7).tickSizeOuter(0));
    g.append('g')
      .attr('class', 'axis axis-y')
      .call(
        d3.axisLeft(y)
          .ticks(6)
          .tickSize(-box.innerW)
          .tickSizeOuter(0)
          .tickFormat(yFormat),
      );
  }

  function zipSeries(t, values) {
    return t.map((ts, i) => ({
      date: new Date(Number(ts) * 1000),
      value: values[i],
    }));
  }

  function drawLineChart(selector, data, seriesNames, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    if (typeof d3 === 'undefined') {
      emptyChart(node, 'D3 failed to load.');
      return;
    }
    if (!data || !data.t || data.t.length === 0) {
      emptyChart(node, 'No chart data.');
      return;
    }
    const hasValues = (arr) => Array.isArray(arr)
      && arr.some((v) => v != null && Number.isFinite(Number(v)));
    const keys = (seriesNames || []).filter((n) => hasValues(data.series && data.series[n]));
    if (!keys.length) {
      emptyChart(node, 'No matching series.');
      return;
    }

    const hidden = node._hidden instanceof Set ? node._hidden : new Set();
    node._hidden = hidden;

    const paint = () => {
      const prevHidden = node._hidden;
      let plot = trimLeading(data, keys);
      const previewRows = keys
        .filter((k) => !prevHidden.has(k))
        .map((key) => ({
          key,
          points: zipSeries(plot.t, plot.series[key]),
        }));
      const shouldRebase = !opts.percent && (opts.rebase
        || (opts.rebase !== false && seriesMaximaRatio(previewRows) >= 4));
      if (shouldRebase) plot = rebaseData(plot, keys, 100);

      const box = chartBox(node, opts.height || 360, {
        left: shouldRebase ? 74 : (opts.percent ? 56 : 72),
      });
      clearChart(node);
      node._hidden = prevHidden;

      const svg = d3.select(node).append('svg')
        .attr('width', box.w)
        .attr('height', box.h)
        .attr('viewBox', `0 0 ${box.w} ${box.h}`);
      const g = svg.append('g')
        .attr('transform', `translate(${box.margin.left},${box.margin.top})`);

      const visible = keys.filter((k) => !prevHidden.has(k));
      const rows = visible.map((key) => ({
        key,
        color: PALETTE[keys.indexOf(key) % PALETTE.length],
        points: zipSeries(plot.t, plot.series[key]),
      }));
      const allPts = rows.flatMap((r) =>
        r.points.filter((p) => p.value != null && Number.isFinite(p.value)));
      if (!allPts.length) {
        emptyChart(node, 'No finite values to plot.');
        return;
      }

      const x = d3.scaleTime()
        .domain(d3.extent(allPts, (d) => d.date))
        .range([0, box.innerW]);
      const y = d3.scaleLinear()
        .domain(paddedDomain(allPts.map((d) => d.value), opts))
        .range([box.innerH, 0]);

      const yFormat = opts.percent
        ? (v) => pct(v, 1)
        : shouldRebase
          ? (v) => num(v, 0)
          : compact;
      drawAxes(g, x, y, box, yFormat);
      if (shouldRebase) {
        g.append('text').attr('class', 'axis-title')
          .attr('x', -box.innerH / 2).attr('y', -44)
          .attr('transform', 'rotate(-90)')
          .attr('text-anchor', 'middle')
          .text('Indexed (start = 100)');
      }

      if (opts.zeroBaseline) {
        g.append('line').attr('class', 'zero-line')
          .attr('x1', 0).attr('x2', box.innerW)
          .attr('y1', y(0)).attr('y2', y(0));
      }

      const line = d3.line()
        .curve(d3.curveLinear)
        .defined((d) => d.value != null && Number.isFinite(d.value))
        .x((d) => x(d.date))
        .y((d) => y(d.value));

      rows.forEach((row, i) => {
        if (opts.fill && i === 0) {
          const area = d3.area()
            .curve(d3.curveLinear)
            .defined((d) => d.value != null && Number.isFinite(d.value))
            .x((d) => x(d.date))
            .y0(y(opts.percent || opts.zeroBaseline ? 0 : y.domain()[0]))
            .y1((d) => y(d.value));
          g.append('path').datum(row.points)
            .attr('class', 'area')
            .attr('fill', row.color)
            .attr('d', area);
        }
        g.append('path').datum(row.points)
          .attr('class', 'line')
          .attr('stroke', row.color)
          .attr('d', line);
      });

      const legend = d3.select(node).insert('div', 'svg').attr('class', 'chart-legend');
      keys.forEach((key, i) => {
        const item = legend.append('button')
          .attr('type', 'button')
          .attr('class', `legend-item${prevHidden.has(key) ? ' off' : ''}`);
        item.append('span').attr('class', 'swatch')
          .style('background', PALETTE[i % PALETTE.length]);
        item.append('span').text(pretty(key));
        item.on('click', () => {
          if (prevHidden.has(key)) prevHidden.delete(key);
          else if (prevHidden.size < keys.length - 1) prevHidden.add(key);
          paint();
        });
      });

      const tip = ensureTooltip();
      const bisect = d3.bisector((d) => d.date).center;
      const hover = g.append('g').attr('class', 'hover').style('display', 'none');
      hover.append('line').attr('class', 'crosshair').attr('y1', 0).attr('y2', box.innerH);
      rows.forEach((row) => {
        hover.append('circle')
          .attr('class', `dot-${keySafe(row.key)}`)
          .attr('r', 3.5)
          .attr('fill', row.color);
      });

      svg.append('rect')
        .attr('class', 'overlay')
        .attr('fill', 'transparent')
        .attr('x', box.margin.left)
        .attr('y', box.margin.top)
        .attr('width', box.innerW)
        .attr('height', box.innerH)
        .on('mouseenter', () => { hover.style('display', null); tip.style.opacity = '1'; })
        .on('mouseleave', () => { hover.style('display', 'none'); tip.style.opacity = '0'; })
        .on('mousemove', (event) => {
          const [mx] = d3.pointer(event, g.node());
          const at = x.invert(mx);
          hover.select('.crosshair').attr('x1', x(at)).attr('x2', x(at));
          const lines = [`<div class="tip-date">${d3.timeFormat('%d %b %Y')(at)}</div>`];
          rows.forEach((row) => {
            const pts = row.points.filter((p) => p.value != null && Number.isFinite(p.value));
            if (!pts.length) return;
            const i = Math.max(0, Math.min(pts.length - 1, bisect(pts, at)));
            const pt = pts[i];
            hover.select(`.dot-${keySafe(row.key)}`)
              .attr('cx', x(pt.date))
              .attr('cy', y(pt.value));
            let shown = opts.percent ? pct(pt.value) : (shouldRebase ? num(pt.value, 1) : compact(pt.value));
            if (shouldRebase && Number.isFinite(pt.value)) {
              shown += `  (${pt.value >= 100 ? '+' : ''}${num(pt.value - 100, 1)}%)`;
            }
            lines.push(
              `<div><span class="swatch" style="background:${row.color}"></span>`
              + `${pretty(row.key)} <b>${shown}</b></div>`,
            );
          });
          tip.innerHTML = lines.join('');
          tip.style.left = `${event.clientX + 14}px`;
          tip.style.top = `${event.clientY + 14}px`;
        });
    };

    paint();
    watchSize(node, paint);
  }

  function keySafe(s) {
    return String(s).replace(/[^a-zA-Z0-9_-]/g, '_');
  }

  function drawHeatmap(selector, table) {
    const node = nodeOf(selector);
    if (!node) return;
    if (!table || !table.columns || !table.rows?.length) {
      emptyChart(node, 'No monthly returns.');
      return;
    }
    if (typeof d3 === 'undefined') {
      emptyChart(node, 'D3 failed to load.');
      return;
    }

    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const cols = table.columns;
    const yearIdx = cols.indexOf('year');
    const monthIdx = months.map((m) => cols.indexOf(m));
    const yearTotalIdx = cols.indexOf('Year');

    const paint = () => {
      const cellW = 56;
      const cellH = 28;
      const labelW = 52;
      const totalW = 68;
      const w = Math.max(node.clientWidth || 720, labelW + months.length * cellW + totalW + 8);
      const h = 28 + table.rows.length * cellH + 10;
      clearChart(node);

      const values = table.rows.flatMap((row) => monthIdx
        .map((i) => (i >= 0 ? row[i] : null))
        .filter((v) => typeof v === 'number'));
      const maxAbs = Math.max(0.02, d3.max(values, (v) => Math.abs(v)) || 0.02);
      const color = d3.scaleDiverging(
        d3.interpolateRgbBasis(['#ff5f6d', '#2a2f3a', '#3ecf8e']),
      ).domain([-maxAbs, maxAbs]);

      const svg = d3.select(node).append('svg').attr('width', w).attr('height', h);
      months.forEach((m, i) => {
        svg.append('text').attr('class', 'heat-label')
          .attr('x', labelW + i * cellW + cellW / 2).attr('y', 16)
          .attr('text-anchor', 'middle').text(m);
      });
      svg.append('text').attr('class', 'heat-label')
        .attr('x', labelW + months.length * cellW + totalW / 2).attr('y', 16)
        .attr('text-anchor', 'middle').text('Year');

      const tip = ensureTooltip();
      table.rows.forEach((row, r) => {
        const yy = 24 + r * cellH;
        svg.append('text').attr('class', 'heat-label')
          .attr('x', labelW - 8).attr('y', yy + cellH / 2 + 4)
          .attr('text-anchor', 'end').text(row[yearIdx]);

        monthIdx.forEach((ci, i) => {
          const v = ci >= 0 ? row[ci] : null;
          const rect = svg.append('rect')
            .attr('x', labelW + i * cellW + 2)
            .attr('y', yy + 2)
            .attr('width', cellW - 4)
            .attr('height', cellH - 4)
            .attr('rx', 3)
            .attr('fill', typeof v === 'number' ? color(v) : '#1e222b');
          if (typeof v === 'number') {
            svg.append('text').attr('class', 'heat-cell')
              .attr('x', labelW + i * cellW + cellW / 2)
              .attr('y', yy + cellH / 2 + 4)
              .attr('text-anchor', 'middle')
              .attr('fill', '#e6e8ee')
              .text((v * 100).toFixed(1));
            rect.on('mousemove', (event) => {
              tip.style.opacity = '1';
              tip.innerHTML = `<div class="tip-date">${months[i]} ${row[yearIdx]}</div><div><b>${pct(v)}</b></div>`;
              tip.style.left = `${event.clientX + 12}px`;
              tip.style.top = `${event.clientY + 12}px`;
            }).on('mouseleave', () => { tip.style.opacity = '0'; });
          }
        });

        const yr = yearTotalIdx >= 0 ? row[yearTotalIdx] : null;
        if (typeof yr === 'number') {
          svg.append('text').attr('class', 'heat-year')
            .attr('x', labelW + months.length * cellW + totalW / 2)
            .attr('y', yy + cellH / 2 + 4)
            .attr('text-anchor', 'middle')
            .attr('fill', yr >= 0 ? '#3ecf8e' : '#ff5f6d')
            .text(pct(yr, 1));
        }
      });
    };

    paint();
    watchSize(node, paint);
  }

  function drawBars(selector, items, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const rows = (items || []).filter((d) => d && d.label != null && Number.isFinite(d.value));
    if (!rows.length) {
      emptyChart(node, opts.empty || 'No data.');
      return;
    }
    if (typeof d3 === 'undefined') {
      emptyChart(node, 'D3 failed to load.');
      return;
    }

    const paint = () => {
      const barH = 22;
      const longest = d3.max(rows, (d) => String(d.label).length) || 8;
      const left = Math.min(240, 12 + longest * 6.4);
      const w = Math.max(node.clientWidth || 640, 320);
      const margin = { top: 8, right: 64, bottom: 24, left };
      const innerH = rows.length * (barH + 8);
      const innerW = Math.max(w - margin.left - margin.right, 80);
      clearChart(node);

      const svg = d3.select(node).append('svg')
        .attr('width', w)
        .attr('height', margin.top + innerH + margin.bottom);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear()
        .domain(opts.diverging
          ? paddedDomain(rows.map((d) => d.value), { zeroBaseline: true, pad: 0.08 })
          : [0, (d3.max(rows, (d) => d.value) || 1) * 1.08])
        .range([0, innerW]);
      const y = d3.scaleBand().domain(rows.map((d) => d.label)).range([0, innerH]).padding(0.22);

      g.append('g').attr('class', 'axis axis-x')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(5).tickSize(-innerH).tickSizeOuter(0)
          .tickFormat(opts.format || compact));
      g.append('g').attr('class', 'axis axis-y')
        .call(d3.axisLeft(y).tickSize(0).tickPadding(8)
          .tickFormat((d) => (String(d).length > 28 ? `${String(d).slice(0, 26)}…` : d)));

      if (opts.diverging) {
        g.append('line').attr('class', 'zero-line')
          .attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', innerH);
      }

      const tip = ensureTooltip();
      g.selectAll('.bar').data(rows).join('rect')
        .attr('class', 'bar')
        .attr('x', (d) => (opts.diverging ? Math.min(x(0), x(d.value)) : 0))
        .attr('y', (d) => y(d.label))
        .attr('width', (d) => Math.abs(x(d.value) - (opts.diverging ? x(0) : 0)))
        .attr('height', y.bandwidth())
        .attr('rx', 3)
        .attr('fill', (d) => (opts.color ? opts.color(d)
          : (d.value >= 0 ? '#4c8dff' : '#ff5f6d')))
        .on('mousemove', (event, d) => {
          tip.style.opacity = '1';
          tip.innerHTML = `<div class="tip-date">${d.label}</div><div><b>${(opts.format || compact)(d.value)}</b></div>`;
          tip.style.left = `${event.clientX + 12}px`;
          tip.style.top = `${event.clientY + 12}px`;
        })
        .on('mouseleave', () => { tip.style.opacity = '0'; });

      g.selectAll('.bar-label').data(rows).join('text')
        .attr('class', 'bar-label')
        .attr('x', (d) => (opts.diverging
          ? x(d.value) + (d.value >= 0 ? 6 : -6)
          : x(d.value) + 6))
        .attr('y', (d) => y(d.label) + y.bandwidth() / 2 + 4)
        .attr('text-anchor', (d) => (opts.diverging && d.value < 0 ? 'end' : 'start'))
        .text((d) => (opts.format || compact)(d.value));
    };

    paint();
    watchSize(node, paint);
  }

  function drawXYLines(selector, rows, xKey, yKeys, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const pts = (rows || []).map((r) => {
      const o = { x: Number(r[xKey]) };
      yKeys.forEach((k) => { o[k] = Number(r[k]); });
      return o;
    }).filter((d) => Number.isFinite(d.x));
    if (!pts.length) {
      emptyChart(node, opts.empty || 'No data.');
      return;
    }

    const paint = () => {
      const maxima = yKeys.map((k) => d3.max(pts, (d) => Math.abs(d[k]) || 0)).filter((v) => v > 0);
      const ratio = maxima.length >= 2 ? d3.max(maxima) / d3.min(maxima) : 1;
      const dual = opts.dualY || (yKeys.length >= 2 && ratio >= 3);
      const box = chartBox(node, opts.height || 280, { right: dual ? 64 : 20, left: 64 });
      clearChart(node);
      const svg = d3.select(node).append('svg')
        .attr('width', box.w).attr('height', box.h)
        .attr('viewBox', `0 0 ${box.w} ${box.h}`);
      const g = svg.append('g')
        .attr('transform', `translate(${box.margin.left},${box.margin.top})`);

      const x = d3.scaleLinear()
        .domain(paddedDomain(pts.map((d) => d.x), { pad: 0.04 }))
        .range([0, box.innerW]);
      const fmtFor = (key, i) => {
        if (opts.yFormats && opts.yFormats[i]) return opts.yFormats[i];
        if (key === 'cagr') return (v) => pct(v, 1);
        return opts.yFormat || ((v) => num(v, 2));
      };

      const scales = {};
      if (dual) {
        yKeys.forEach((key) => {
          scales[key] = d3.scaleLinear()
            .domain(paddedDomain(pts.map((d) => d[key]).filter(Number.isFinite), { pad: 0.12 }))
            .range([box.innerH, 0]);
        });
        drawAxes(g, x, scales[yKeys[0]], box, fmtFor(yKeys[0], 0));
        g.append('g').attr('class', 'axis axis-y')
          .attr('transform', `translate(${box.innerW},0)`)
          .call(d3.axisRight(scales[yKeys[1] || yKeys[0]])
            .ticks(6).tickSize(0).tickSizeOuter(0)
            .tickFormat(fmtFor(yKeys[1], 1)));
      } else {
        const yVals = pts.flatMap((d) => yKeys.map((k) => d[k]).filter(Number.isFinite));
        const y = d3.scaleLinear().domain(paddedDomain(yVals, { pad: 0.12 })).range([box.innerH, 0]);
        yKeys.forEach((key) => { scales[key] = y; });
        drawAxes(g, x, y, box, fmtFor(yKeys[0], 0));
      }

      g.append('text').attr('class', 'axis-title')
        .attr('x', box.innerW / 2).attr('y', box.innerH + 30)
        .attr('text-anchor', 'middle').text(opts.xLabel || xKey);

      const tip = ensureTooltip();
      yKeys.forEach((key, i) => {
        const color = PALETTE[i % PALETTE.length];
        const y = scales[key];
        const ln = d3.line().curve(d3.curveLinear)
          .defined((d) => Number.isFinite(d[key]))
          .x((d) => x(d.x)).y((d) => y(d[key]));
        g.append('path').datum(pts).attr('class', 'line').attr('stroke', color).attr('d', ln);
        g.selectAll(`.pt-${i}`).data(pts.filter((d) => Number.isFinite(d[key]))).join('circle')
          .attr('cx', (d) => x(d.x)).attr('cy', (d) => y(d[key]))
          .attr('r', 3.5).attr('fill', color)
          .on('mousemove', (event, d) => {
            tip.style.opacity = '1';
            tip.innerHTML = `<div class="tip-date">${opts.xLabel || xKey} ${d.x}</div>`
              + yKeys.map((k, j) => `<div><span class="swatch" style="background:${PALETTE[j]}"></span>${pretty(k)} <b>${fmtFor(k, j)(d[k])}</b></div>`).join('');
            tip.style.left = `${event.clientX + 12}px`;
            tip.style.top = `${event.clientY + 12}px`;
          })
          .on('mouseleave', () => { tip.style.opacity = '0'; });
      });

      const legend = d3.select(node).insert('div', 'svg').attr('class', 'chart-legend');
      yKeys.forEach((key, i) => {
        const item = legend.append('span').attr('class', 'legend-item');
        item.append('span').attr('class', 'swatch')
          .style('background', PALETTE[i % PALETTE.length]);
        item.append('span').text(pretty(key) + (dual && i === 1 ? ' (right)' : dual && i === 0 ? ' (left)' : ''));
      });
    };

    paint();
    watchSize(node, paint);
  }

  function drawDonut(selector, items, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const rows = (items || []).filter((d) => d && Number(d.value) > 0);
    if (!rows.length) {
      emptyChart(node, opts.empty || 'No mix to show.');
      return;
    }
    const paint = () => {
      const size = Math.min(Math.max(node.clientWidth || 220, 180), 280);
      const r = size / 2 - 8;
      const ir = r * 0.62;
      clearChart(node);
      const svg = d3.select(node).append('svg').attr('width', size).attr('height', size);
      const g = svg.append('g').attr('transform', `translate(${size / 2},${size / 2})`);
      const pie = d3.pie().value((d) => d.value).sort(null);
      const arc = d3.arc().innerRadius(ir).outerRadius(r);
      const total = d3.sum(rows, (d) => d.value);
      const tip = ensureTooltip();
      g.selectAll('path').data(pie(rows)).join('path')
        .attr('d', arc)
        .attr('fill', (d, i) => d.data.color || PALETTE[i % PALETTE.length])
        .on('mousemove', (event, d) => {
          tip.style.opacity = '1';
          tip.innerHTML = `<div class="tip-date">${d.data.label}</div><div><b>${d.data.value}</b> (${((d.data.value / total) * 100).toFixed(0)}%)</div>`;
          tip.style.left = `${event.clientX + 12}px`;
          tip.style.top = `${event.clientY + 12}px`;
        })
        .on('mouseleave', () => { tip.style.opacity = '0'; });
      g.append('text').attr('class', 'donut-center')
        .attr('text-anchor', 'middle').attr('dy', '0.35em')
        .attr('fill', '#e6e8ee').style('font-size', '15px').style('font-weight', '600')
        .text(opts.center || total);
      const legend = d3.select(node).append('div').attr('class', 'chart-legend');
      rows.forEach((d, i) => {
        const item = legend.append('span').attr('class', 'legend-item');
        item.append('span').attr('class', 'swatch').style('background', d.color || PALETTE[i % PALETTE.length]);
        item.append('span').text(`${d.label} ${d.value}`);
      });
    };
    paint();
    watchSize(node, paint);
  }

  function drawGauge(selector, value, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const v = Math.max(0, Math.min(1, Number(value) || 0));
    const paint = () => {
      const w = Math.max(node.clientWidth || 220, 180);
      const h = opts.height || 110;
      clearChart(node);
      const svg = d3.select(node).append('svg').attr('width', w).attr('height', h);
      const r = Math.min(w / 2 - 10, h - 28);
      const g = svg.append('g').attr('transform', `translate(${w / 2},${h - 8})`);
      const arc = d3.arc().innerRadius(r * 0.68).outerRadius(r).startAngle(-Math.PI / 2);
      g.append('path').attr('d', arc.endAngle(Math.PI / 2)()).attr('fill', '#2a2f3a');
      g.append('path').attr('d', arc.endAngle(-Math.PI / 2 + Math.PI * v)()).attr('fill', opts.color || '#4c8dff');
      svg.append('text').attr('x', w / 2).attr('y', h - 22)
        .attr('text-anchor', 'middle').attr('fill', '#e6e8ee')
        .style('font-size', '18px').style('font-weight', '700')
        .text(opts.label || `${Math.round(v * 100)}%`);
    };
    paint();
    watchSize(node, paint);
  }

  function tableToObjects(table) {
    if (!table?.columns) return [];
    return table.rows.map((row) => {
      const o = {};
      table.columns.forEach((c, i) => { o[c] = row[i]; });
      return o;
    });
  }

  function drawSparkline(selector, values, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const xs = (values || []).map(Number).filter(Number.isFinite);
    clearChart(node);
    if (xs.length < 2) {
      emptyChart(node, opts.empty || 'No 1-year series');
      return;
    }
    const w = Math.max(node.clientWidth || 240, 120);
    const h = opts.height || 72;
    const pad = { t: 4, r: 4, b: 4, l: 4 };
    const min = d3.min(xs);
    const max = d3.max(xs);
    const x = d3.scaleLinear().domain([0, xs.length - 1]).range([pad.l, w - pad.r]);
    const y = d3.scaleLinear().domain([min, max]).range([h - pad.b, pad.t]);
    const svg = d3.select(node).append('svg').attr('width', w).attr('height', h);
    const line = d3.line().x((_, i) => x(i)).y((d) => y(d));
    const up = xs[xs.length - 1] >= xs[0];
    svg.append('path').attr('d', line(xs)).attr('fill', 'none')
      .attr('stroke', up ? '#3ecf8e' : '#ff5f6d').attr('stroke-width', 1.6);
  }

  return { drawLineChart, drawHeatmap, drawBars, drawXYLines, drawDonut, drawGauge, drawSparkline, tableToObjects, pct };
})();
