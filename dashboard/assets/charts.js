/* D3 charts for the Indian Equity Alpha dashboard. */

const Charts = (() => {
  const PALETTE = ['#ff9900', '#5dff6b', '#ffb000', '#ff4d4d', '#7ec8ff', '#c9a227'];
  const LABELS = {
    equity: 'Strategy',
    after_tax_equity: 'After tax',
    benchmark: 'Nifty 50 TRI',
    drawdown: 'Drawdown',
    sharpe: 'Sharpe',
    cagr: 'CAGR',
    excess: 'Cumulative excess vs Nifty',
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

  function placeTip(tip, event) {
    const pad = 12;
    const tw = 200;
    const th = 72;
    const x = Math.min(event.clientX + pad, window.innerWidth - tw - 8);
    const y = Math.min(event.clientY + pad, window.innerHeight - th - 8);
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${Math.max(8, y)}px`;
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
    const w = Math.max(node.clientWidth || 0, 160);
    const h = height || Math.max(node.clientHeight || 280, 180);
    const leftCap = Math.min(extra.left ?? 56, Math.max(40, w * 0.18));
    const margin = {
      top: extra.top ?? 16,
      right: extra.right ?? 16,
      bottom: extra.bottom ?? 32,
      left: leftCap,
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

      const box = chartBox(node, opts.height || 320, {
        left: shouldRebase ? 58 : (opts.percent ? 48 : 56),
        right: 12,
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
          placeTip(tip, event);
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
      const w = Math.max(node.clientWidth || 0, 220);
      const labelW = w < 420 ? 36 : 44;
      const totalW = w < 420 ? 44 : 52;
      const cellW = Math.max(18, (w - labelW - totalW) / months.length);
      const showNums = cellW >= 34;
      const cellH = showNums ? 26 : 20;
      const h = 24 + table.rows.length * cellH + 8;
      clearChart(node);

      const values = table.rows.flatMap((row) => monthIdx
        .map((i) => (i >= 0 ? row[i] : null))
        .filter((v) => typeof v === 'number'));
      const maxAbs = Math.max(0.02, d3.max(values, (v) => Math.abs(v)) || 0.02);
      const color = d3.scaleDiverging(
        d3.interpolateRgbBasis(['#6b1220', '#1a1610', '#0f4a28']),
      ).domain([-maxAbs, maxAbs]);

      const svg = d3.select(node).append('svg')
        .attr('width', w).attr('height', h)
        .attr('viewBox', `0 0 ${w} ${h}`);
      months.forEach((m, i) => {
        svg.append('text').attr('class', 'heat-label')
          .attr('x', labelW + i * cellW + cellW / 2).attr('y', 14)
          .attr('text-anchor', 'middle')
          .text(cellW < 28 ? m.slice(0, 1) : m.slice(0, 3));
      });
      svg.append('text').attr('class', 'heat-label')
        .attr('x', labelW + months.length * cellW + totalW / 2).attr('y', 14)
        .attr('text-anchor', 'middle').text(totalW < 48 ? 'Yr' : 'Year');

      const tip = ensureTooltip();
      table.rows.forEach((row, r) => {
        const yy = 20 + r * cellH;
        const yearLabel = String(row[yearIdx] ?? '');
        svg.append('text').attr('class', 'heat-label')
          .attr('x', labelW - 6).attr('y', yy + cellH / 2 + 3)
          .attr('text-anchor', 'end')
          .text(labelW < 40 ? yearLabel.slice(-2) : yearLabel);

        monthIdx.forEach((ci, i) => {
          const v = ci >= 0 ? row[ci] : null;
          const rect = svg.append('rect')
            .attr('x', labelW + i * cellW + 1)
            .attr('y', yy + 1)
            .attr('width', Math.max(1, cellW - 2))
            .attr('height', Math.max(1, cellH - 2))
            .attr('rx', 2)
            .attr('fill', typeof v === 'number' ? color(v) : '#14120c');
          if (typeof v === 'number') {
            if (showNums) {
              svg.append('text').attr('class', 'heat-cell')
                .attr('x', labelW + i * cellW + cellW / 2)
                .attr('y', yy + cellH / 2 + 3)
                .attr('text-anchor', 'middle')
                .attr('fill', '#e8dcc4')
                .text((v * 100).toFixed(cellW >= 44 ? 1 : 0));
            }
            rect.on('mousemove', (event) => {
              tip.style.opacity = '1';
              tip.innerHTML = `<div class="tip-date">${months[i]} ${row[yearIdx]}</div><div><b>${pct(v)}</b></div>`;
              placeTip(tip, event);
            }).on('mouseleave', () => { tip.style.opacity = '0'; });
          }
        });

        const yr = yearTotalIdx >= 0 ? row[yearTotalIdx] : null;
        if (typeof yr === 'number') {
          svg.append('text').attr('class', 'heat-year')
            .attr('x', labelW + months.length * cellW + totalW / 2)
            .attr('y', yy + cellH / 2 + 3)
            .attr('text-anchor', 'middle')
            .attr('fill', yr >= 0 ? '#5dff6b' : '#ff4d4d')
            .text(pct(yr, 0));
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
      const w = Math.max(node.clientWidth || 0, 180);
      const longest = d3.max(rows, (d) => String(d.label).length) || 8;
      const left = Math.min(Math.max(64, Math.min(w * 0.32, 12 + longest * 6.2)), 128);
      const right = opts.diverging ? 28 : 10;
      const margin = { top: 8, right, bottom: 22, left };
      const innerH = rows.length * (barH + 8);
      const innerW = Math.max(w - margin.left - margin.right, 60);
      const maxChars = Math.max(8, Math.floor((left - 10) / 6.4));
      clearChart(node);

      const svg = d3.select(node).append('svg')
        .attr('width', w)
        .attr('height', margin.top + innerH + margin.bottom)
        .attr('viewBox', `0 0 ${w} ${margin.top + innerH + margin.bottom}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

      const x = d3.scaleLinear()
        .domain(opts.diverging
          ? paddedDomain(rows.map((d) => d.value), { zeroBaseline: true, pad: 0.12 })
          : [0, (d3.max(rows, (d) => d.value) || 1) * 1.12])
        .range([0, innerW]);
      const y = d3.scaleBand().domain(rows.map((d) => d.label)).range([0, innerH]).padding(0.22);

      g.append('g').attr('class', 'axis axis-x')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).ticks(4).tickSize(-innerH).tickSizeOuter(0)
          .tickFormat(opts.format || compact));
      g.append('g').attr('class', 'axis axis-y')
        .call(d3.axisLeft(y).tickSize(0).tickPadding(6)
          .tickFormat((d) => (String(d).length > maxChars ? `${String(d).slice(0, maxChars - 1)}…` : d)));

      if (opts.diverging) {
        g.append('line').attr('class', 'zero-line')
          .attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', innerH);
      }

      const tip = ensureTooltip();
      const fmt = opts.format || compact;
      g.selectAll('.bar').data(rows).join('rect')
        .attr('class', 'bar')
        .attr('x', (d) => (opts.diverging ? Math.min(x(0), x(d.value)) : 0))
        .attr('y', (d) => y(d.label))
        .attr('width', (d) => Math.abs(x(d.value) - (opts.diverging ? x(0) : 0)))
        .attr('height', y.bandwidth())
        .attr('rx', 3)
        .attr('fill', (d) => (opts.color ? opts.color(d)
          : (d.value >= 0 ? '#ff9900' : '#ff4d4d')))
        .on('mousemove', (event, d) => {
          tip.style.opacity = '1';
          tip.innerHTML = `<div class="tip-date">${d.label}</div><div><b>${fmt(d.value)}</b></div>`;
          placeTip(tip, event);
        })
        .on('mouseleave', () => { tip.style.opacity = '0'; });

      g.selectAll('.bar-label').data(rows).join('text')
        .attr('class', 'bar-label')
        .attr('x', (d) => {
          const xv = x(d.value);
          if (opts.diverging) {
            if (d.value >= 0) return xv + 6 > innerW - 2 ? xv - 5 : xv + 5;
            return xv - 6 < 2 ? xv + 5 : xv - 5;
          }
          return xv > innerW - 36 ? xv - 5 : xv + 5;
        })
        .attr('y', (d) => y(d.label) + y.bandwidth() / 2 + 4)
        .attr('text-anchor', (d) => {
          const xv = x(d.value);
          if (opts.diverging) {
            if (d.value >= 0) return xv + 6 > innerW - 2 ? 'end' : 'start';
            return xv - 6 < 2 ? 'start' : 'end';
          }
          return xv > innerW - 36 ? 'end' : 'start';
        })
        .attr('fill', '#e8dcc4')
        .text((d) => fmt(d.value));
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
            placeTip(tip, event);
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
      const box = Math.max(node.clientWidth || 0, 140);
      const size = Math.min(box, 200);
      const r = size / 2 - 10;
      const ir = r * 0.62;
      clearChart(node);
      const svg = d3.select(node).append('svg')
        .attr('width', size).attr('height', size)
        .attr('viewBox', `0 0 ${size} ${size}`);
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
          tip.innerHTML = `<div class="tip-date">${d.data.label}</div><div><b>${opts.format ? opts.format(d.data.value) : d.data.value}</b> (${((d.data.value / total) * 100).toFixed(0)}%)</div>`;
          placeTip(tip, event);
        })
        .on('mouseleave', () => { tip.style.opacity = '0'; });
      g.append('text').attr('class', 'donut-center')
        .attr('text-anchor', 'middle').attr('dy', '0.35em')
        .attr('fill', '#e8dcc4').style('font-size', '14px').style('font-weight', '700')
        .text(opts.center || total);
      const legend = d3.select(node).append('div').attr('class', 'chart-legend');
      rows.forEach((d, i) => {
        const item = legend.append('span').attr('class', 'legend-item');
        item.append('span').attr('class', 'swatch').style('background', d.color || PALETTE[i % PALETTE.length]);
        item.append('span').text(`${d.label} ${opts.format ? opts.format(d.value) : (Number.isInteger(d.value) ? d.value : compact(d.value))}`);
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
      const w = Math.max(node.clientWidth || 0, 140);
      const h = opts.height || 100;
      clearChart(node);
      const svg = d3.select(node).append('svg')
        .attr('width', w).attr('height', h)
        .attr('viewBox', `0 0 ${w} ${h}`);
      const r = Math.max(24, Math.min(w / 2 - 12, h - 34));
      const g = svg.append('g').attr('transform', `translate(${w / 2},${h - 14})`);
      const arc = d3.arc().innerRadius(r * 0.68).outerRadius(r).startAngle(-Math.PI / 2);
      g.append('path').attr('d', arc.endAngle(Math.PI / 2)()).attr('fill', '#1a1610');
      g.append('path').attr('d', arc.endAngle(-Math.PI / 2 + Math.PI * v)()).attr('fill', opts.color || '#ff9900');
      svg.append('text').attr('x', w / 2).attr('y', h - 18)
        .attr('text-anchor', 'middle').attr('fill', '#e8dcc4')
        .style('font-size', '16px').style('font-weight', '700')
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
      .attr('stroke', up ? '#5dff6b' : '#ff4d4d').attr('stroke-width', 1.6);
  }

  function drawTreemap(selector, items, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const rows = (items || []).filter((d) => d && d.label && Number(d.value) > 0);
    if (!rows.length) {
      emptyChart(node, opts.empty || 'No sector mix yet.');
      return;
    }
    if (typeof d3 === 'undefined' || !d3.treemap) {
      emptyChart(node, 'D3 treemap unavailable.');
      return;
    }
    const paint = () => {
      const w = Math.max(node.clientWidth || 0, 200);
      const h = Math.min(opts.height || 280, Math.max(180, w * 0.62));
      clearChart(node);
      const root = d3.hierarchy({ children: rows })
        .sum((d) => Number(d.value) || 0)
        .sort((a, b) => (b.value || 0) - (a.value || 0));
      d3.treemap().size([w, h]).paddingInner(3).paddingOuter(0)(root);
      const svg = d3.select(node).append('svg')
        .attr('width', w).attr('height', h)
        .attr('viewBox', `0 0 ${w} ${h}`);
      const tip = ensureTooltip();
      const uid = `tm${Math.abs(w * h).toString(36)}`;
      const leaves = svg.selectAll('g').data(root.leaves()).join('g')
        .attr('transform', (d) => `translate(${d.x0},${d.y0})`);
      leaves.append('clipPath')
        .attr('id', (d, i) => `${uid}-${i}`)
        .append('rect')
        .attr('width', (d) => Math.max(0, d.x1 - d.x0))
        .attr('height', (d) => Math.max(0, d.y1 - d.y0));
      leaves.append('rect')
        .attr('width', (d) => Math.max(0, d.x1 - d.x0))
        .attr('height', (d) => Math.max(0, d.y1 - d.y0))
        .attr('fill', (d, i) => d.data.color || PALETTE[i % PALETTE.length])
        .on('mousemove', (event, d) => {
          tip.style.opacity = '1';
          tip.innerHTML = `<div class="tip-date">${d.data.label}</div><div><b>${d.data.value}</b> names</div>`;
          placeTip(tip, event);
        })
        .on('mouseleave', () => { tip.style.opacity = '0'; });
      leaves.append('text').attr('class', 'treemap-label')
        .attr('clip-path', (d, i) => `url(#${uid}-${i})`)
        .attr('x', 8).attr('y', 18)
        .text((d) => {
          const bw = d.x1 - d.x0;
          if (bw < 52 || d.y1 - d.y0 < 22) return '';
          const s = String(d.data.label);
          const max = Math.max(3, Math.floor((bw - 14) / 7));
          return s.length > max ? `${s.slice(0, max - 1)}…` : s;
        });
      leaves.append('text').attr('class', 'treemap-sub')
        .attr('clip-path', (d, i) => `url(#${uid}-${i})`)
        .attr('x', 8).attr('y', 34)
        .text((d) => ((d.x1 - d.x0) < 52 || (d.y1 - d.y0) < 40 ? '' : d.data.value));
    };
    paint();
    watchSize(node, paint);
  }

  function drawScatter(selector, items, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const rows = (items || []).filter((d) => d && Number.isFinite(d.x) && Number.isFinite(d.y));
    if (!rows.length) {
      emptyChart(node, opts.empty || 'No points.');
      return;
    }
    const paint = () => {
      const box = chartBox(node, opts.height || 280, { left: 48, right: 12, top: 18, bottom: 36 });
      clearChart(node);
      const svg = d3.select(node).append('svg')
        .attr('width', box.w).attr('height', box.h)
        .attr('viewBox', `0 0 ${box.w} ${box.h}`);
      const g = svg.append('g').attr('transform', `translate(${box.margin.left},${box.margin.top})`);
      const xs = rows.map((d) => d.x);
      const ys = rows.map((d) => d.y);
      const x = d3.scaleLinear()
        .domain(paddedDomain([Math.min(0, d3.min(xs)), Math.max(0.2, d3.max(xs))], { pad: 0.12 }))
        .range([0, box.innerW]);
      const y = d3.scaleLinear()
        .domain(paddedDomain([Math.min(0, d3.min(ys)), Math.max(0.02, d3.max(ys))], { pad: 0.14, zeroBaseline: true }))
        .range([box.innerH, 0]);
      const r = d3.scaleSqrt()
        .domain([0, d3.max(rows, (d) => Math.abs(d.size) || 0) || 1])
        .range([4, Math.min(16, box.innerW / 18)]);
      g.append('g').attr('class', 'axis axis-y')
        .call(d3.axisLeft(y).ticks(5).tickSize(-box.innerW).tickSizeOuter(0)
          .tickFormat(opts.yFormat || ((v) => pct(v, 0))));
      g.append('g').attr('class', 'axis axis-x')
        .attr('transform', `translate(0,${box.innerH})`)
        .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0)
          .tickFormat(opts.xFormat || ((v) => num(v, 1))));
      g.append('line').attr('class', 'zero-line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', box.innerH);
      g.append('line').attr('class', 'zero-line').attr('x1', 0).attr('x2', box.innerW).attr('y1', y(0)).attr('y2', y(0));
      g.append('text').attr('class', 'axis-title').attr('x', box.innerW / 2).attr('y', box.innerH + 30)
        .attr('text-anchor', 'middle').text(opts.xLabel || 'Sharpe');
      g.append('text').attr('class', 'axis-title').attr('transform', 'rotate(-90)')
        .attr('x', -box.innerH / 2).attr('y', -36).attr('text-anchor', 'middle')
        .text(opts.yLabel || 'Excess vs Nifty');
      const tip = ensureTooltip();
      g.selectAll('.dot').data(rows).join('circle')
        .attr('class', 'dot')
        .attr('cx', (d) => x(d.x)).attr('cy', (d) => y(d.y))
        .attr('r', (d) => r(Math.abs(d.size) || 0.05))
        .attr('fill', (d) => d.color || (d.status === 'pass' ? '#5dff6b' : '#ff4d4d'))
        .attr('fill-opacity', 0.85)
        .attr('stroke', '#050505').attr('stroke-width', 0.6)
        .on('mousemove', (event, d) => {
          tip.style.opacity = '1';
          tip.innerHTML = `<div class="tip-date">${d.label}</div>`
            + `<div>Sharpe <b>${num(Number.isFinite(d.sharpe) ? d.sharpe : d.x, 2)}</b></div>`
            + `<div>Excess <b>${pct(d.y, 1)}</b></div>`
            + (d.size != null ? `<div>CAGR <b>${pct(d.size, 1)}</b></div>` : '');
          placeTip(tip, event);
        })
        .on('mouseleave', () => { tip.style.opacity = '0'; });
      const labeled = rows.filter((d) => d.status === 'pass' && d.y > 0.04)
        .sort((a, b) => b.y - a.y).slice(0, 5);
      g.selectAll('.scatter-label').data(labeled).join('text')
        .attr('class', 'scatter-label')
        .attr('x', (d) => (x(d.x) > box.innerW * 0.62 ? x(d.x) - 8 : x(d.x) + 8))
        .attr('y', (d) => y(d.y) - 6)
        .attr('text-anchor', (d) => (x(d.x) > box.innerW * 0.62 ? 'end' : 'start'))
        .text((d) => {
          const s = String(d.label);
          return s.length > 16 ? `${s.slice(0, 15)}…` : s;
        });
      g.append('text').attr('class', 'quad-label')
        .attr('x', box.innerW - 4).attr('y', 12)
        .attr('text-anchor', 'end').text('PASS zone →');
    };
    paint();
    watchSize(node, paint);
  }

  function drawWaterfall(selector, items, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const rows = (items || []).filter((d) => d && d.label != null && Number.isFinite(d.value));
    if (!rows.length) {
      emptyChart(node, opts.empty || 'No waterfall.');
      return;
    }
    const paint = () => {
      const w = Math.max(node.clientWidth || 0, 160);
      const h = opts.height || 180;
      const margin = { top: 16, right: 8, bottom: 28, left: 42 };
      const innerW = Math.max(w - margin.left - margin.right, 40);
      const innerH = Math.max(h - margin.top - margin.bottom, 40);
      clearChart(node);
      const svg = d3.select(node).append('svg')
        .attr('width', w).attr('height', h).attr('viewBox', `0 0 ${w} ${h}`);
      const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);
      let cursor = 0;
      const steps = rows.map((d) => {
        const total = d.kind === 'total';
        const start = total ? 0 : cursor;
        const end = total ? d.value : cursor + d.value;
        if (!total) cursor = end;
        return { ...d, start, end };
      });
      const y = d3.scaleLinear()
        .domain(paddedDomain(steps.flatMap((d) => [d.start, d.end]), { zeroBaseline: true, pad: 0.12 }))
        .range([innerH, 0]);
      const x = d3.scaleBand().domain(steps.map((d) => d.label)).range([0, innerW]).padding(0.28);
      g.append('g').attr('class', 'axis axis-y')
        .call(d3.axisLeft(y).ticks(4).tickSize(-innerW).tickSizeOuter(0).tickFormat(opts.format || pct));
      g.append('g').attr('class', 'axis axis-x')
        .attr('transform', `translate(0,${innerH})`)
        .call(d3.axisBottom(x).tickSize(0).tickPadding(6));
      g.selectAll('.axis-x text').each(function wrapTick() {
        const t = d3.select(this);
        const s = String(t.text());
        if (s.length > 11) t.text(`${s.slice(0, 10)}…`);
      });
      g.append('line').attr('class', 'zero-line').attr('x1', 0).attr('x2', innerW).attr('y1', y(0)).attr('y2', y(0));
      const tip = ensureTooltip();
      g.selectAll('.wf').data(steps).join('rect')
        .attr('x', (d) => x(d.label))
        .attr('y', (d) => Math.min(y(d.start), y(d.end)))
        .attr('width', x.bandwidth())
        .attr('height', (d) => Math.max(1, Math.abs(y(d.end) - y(d.start))))
        .attr('rx', 2)
        .attr('fill', (d) => (d.kind === 'total' ? '#ff9900' : d.value >= 0 ? '#5dff6b' : '#ff4d4d'))
        .on('mousemove', (event, d) => {
          tip.style.opacity = '1';
          tip.innerHTML = `<div class="tip-date">${d.label}</div><div><b>${(opts.format || pct)(d.value)}</b></div>`;
          placeTip(tip, event);
        })
        .on('mouseleave', () => { tip.style.opacity = '0'; });
      g.selectAll('.wf-lab').data(steps).join('text')
        .attr('class', 'bar-label')
        .attr('x', (d) => x(d.label) + x.bandwidth() / 2)
        .attr('y', (d) => Math.min(y(d.start), y(d.end)) - 4)
        .attr('text-anchor', 'middle')
        .text((d) => (opts.format || pct)(d.value));
    };
    paint();
    watchSize(node, paint);
  }

  function drawPack(selector, items, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const rows = (items || []).filter((d) => d && d.label && Number(d.value) > 0);
    if (!rows.length) {
      emptyChart(node, opts.empty || 'No sector mix yet.');
      return;
    }
    if (typeof d3 === 'undefined' || !d3.pack) {
      drawTreemap(selector, items, opts);
      return;
    }
    const paint = () => {
      const w = Math.max(node.clientWidth || 0, 200);
      const h = Math.min(opts.height || 280, Math.max(200, w * 0.72));
      clearChart(node);
      const root = d3.hierarchy({ children: rows })
        .sum((d) => Number(d.value) || 0)
        .sort((a, b) => (b.value || 0) - (a.value || 0));
      d3.pack().size([w, h]).padding(3)(root);
      const svg = d3.select(node).append('svg')
        .attr('width', w).attr('height', h).attr('viewBox', `0 0 ${w} ${h}`);
      const tip = ensureTooltip();
      const g = svg.selectAll('g').data(root.leaves()).join('g')
        .attr('transform', (d) => `translate(${d.x},${d.y})`);
      g.append('circle')
        .attr('r', (d) => d.r)
        .attr('fill', (d, i) => PALETTE[i % PALETTE.length])
        .attr('fill-opacity', 0.92)
        .on('mousemove', (event, d) => {
          tip.style.opacity = '1';
          tip.innerHTML = `<div class="tip-date">${d.data.label}</div><div><b>${d.data.value}</b> names</div>`;
          placeTip(tip, event);
        })
        .on('mouseleave', () => { tip.style.opacity = '0'; });
      g.append('text').attr('class', 'pack-label')
        .attr('text-anchor', 'middle').attr('dy', '-0.15em')
        .text((d) => {
          if (d.r < 22) return '';
          const s = String(d.data.label);
          const max = Math.max(3, Math.floor((d.r * 1.6) / 6));
          return s.length > max ? `${s.slice(0, max - 1)}…` : s;
        });
      g.append('text').attr('class', 'pack-sub')
        .attr('text-anchor', 'middle').attr('dy', '1em')
        .text((d) => (d.r < 28 ? '' : d.data.value));
    };
    paint();
    watchSize(node, paint);
  }

  function drawRadar(selector, series, axes, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const keys = axes || [];
    const rows = (series || []).filter((d) => d && keys.every((k) => Number.isFinite(d[k])));
    if (!rows.length || !keys.length) {
      emptyChart(node, opts.empty || 'No radar data.');
      return;
    }
    const paint = () => {
      const w = Math.max(node.clientWidth || 0, 180);
      const h = opts.height || 260;
      const cx = w / 2;
      const cy = h / 2 - 10;
      const radius = Math.max(28, Math.min(cx, cy) - 30);
      clearChart(node);
      const svg = d3.select(node).append('svg')
        .attr('width', w).attr('height', h).attr('viewBox', `0 0 ${w} ${h}`);
      const angle = (i) => (Math.PI * 2 * i) / keys.length - Math.PI / 2;
      for (let lv = 1; lv <= 4; lv += 1) {
        const rr = radius * (lv / 4);
        const ring = keys.map((_, i) => `${cx + rr * Math.cos(angle(i))},${cy + rr * Math.sin(angle(i))}`);
        svg.append('polygon').attr('class', 'radar-grid').attr('points', ring.join(' '));
      }
      keys.forEach((k, i) => {
        svg.append('line').attr('class', 'radar-grid')
          .attr('x1', cx).attr('y1', cy)
          .attr('x2', cx + radius * Math.cos(angle(i)))
          .attr('y2', cy + radius * Math.sin(angle(i)));
        const label = opts.axisLabels?.[k] || pretty(k);
        svg.append('text').attr('class', 'radar-axis')
          .attr('x', cx + (radius + 14) * Math.cos(angle(i)))
          .attr('y', cy + (radius + 14) * Math.sin(angle(i)))
          .attr('dy', '0.35em')
          .attr('text-anchor', Math.abs(Math.cos(angle(i))) < 0.25 ? 'middle'
            : (Math.cos(angle(i)) > 0 ? 'start' : 'end'))
          .text(label);
      });
      const tip = ensureTooltip();
      const line = d3.lineRadial()
        .angle((_, i) => (Math.PI * 2 * i) / keys.length)
        .radius((v) => radius * Math.max(0, Math.min(1, v)))
        .curve(d3.curveLinearClosed);
      rows.forEach((row, ri) => {
        const vals = keys.map((k) => row[k]);
        const color = row.color || PALETTE[ri % PALETTE.length];
        svg.append('path')
          .attr('transform', `translate(${cx},${cy})`)
          .attr('d', line(vals))
          .attr('fill', color).attr('fill-opacity', 0.16)
          .attr('stroke', color).attr('stroke-width', 1.8)
          .on('mousemove', (event) => {
            tip.style.opacity = '1';
            tip.innerHTML = `<div class="tip-date">${row.label}</div>`
              + keys.map((k, i) => `<div>${opts.axisLabels?.[k] || k} <b>${num(vals[i], 2)}</b></div>`).join('');
            placeTip(tip, event);
          })
          .on('mouseleave', () => { tip.style.opacity = '0'; });
      });
      const legend = d3.select(node).append('div').attr('class', 'chart-legend');
      rows.forEach((d, i) => {
        const item = legend.append('span').attr('class', 'legend-item');
        item.append('span').attr('class', 'swatch').style('background', d.color || PALETTE[i % PALETTE.length]);
        item.append('span').text(d.label);
      });
    };
    paint();
    watchSize(node, paint);
  }

  function drawHistogram(selector, values, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const nums = (values || []).filter((v) => typeof v === 'number' && Number.isFinite(v));
    if (nums.length < 3) {
      emptyChart(node, opts.empty || 'Need more months.');
      return;
    }
    const paint = () => {
      const box = chartBox(node, opts.height || 180, { left: 32, right: 8, top: 10, bottom: 28 });
      clearChart(node);
      const svg = d3.select(node).append('svg')
        .attr('width', box.w).attr('height', box.h)
        .attr('viewBox', `0 0 ${box.w} ${box.h}`);
      const g = svg.append('g').attr('transform', `translate(${box.margin.left},${box.margin.top})`);
      const maxAbs = Math.max(0.04, d3.max(nums, (v) => Math.abs(v)));
      const x = d3.scaleLinear().domain([-maxAbs, maxAbs]).range([0, box.innerW]);
      const bins = d3.bin().domain(x.domain()).thresholds(7)(nums);
      const y = d3.scaleLinear().domain([0, d3.max(bins, (b) => b.length) || 1]).range([box.innerH, 0]);
      g.append('g').attr('class', 'axis axis-x')
        .attr('transform', `translate(0,${box.innerH})`)
        .call(d3.axisBottom(x).ticks(5).tickSizeOuter(0).tickFormat((v) => pct(v, 0)));
      g.append('g').attr('class', 'axis axis-y')
        .call(d3.axisLeft(y).ticks(3).tickSize(-box.innerW).tickSizeOuter(0).tickFormat((v) => String(v)));
      g.append('line').attr('class', 'zero-line').attr('x1', x(0)).attr('x2', x(0)).attr('y1', 0).attr('y2', box.innerH);
      const tip = ensureTooltip();
      g.selectAll('.hist').data(bins).join('rect')
        .attr('x', (d) => x(d.x0) + 1)
        .attr('y', (d) => y(d.length))
        .attr('width', (d) => Math.max(1, x(d.x1) - x(d.x0) - 2))
        .attr('height', (d) => box.innerH - y(d.length))
        .attr('fill', (d) => ((d.x0 + d.x1) / 2 >= 0 ? '#5dff6b' : '#ff4d4d'))
        .on('mousemove', (event, d) => {
          tip.style.opacity = '1';
          tip.innerHTML = `<div class="tip-date">${pct(d.x0, 1)} → ${pct(d.x1, 1)}</div><div><b>${d.length}</b> months</div>`;
          placeTip(tip, event);
        })
        .on('mouseleave', () => { tip.style.opacity = '0'; });
    };
    paint();
    watchSize(node, paint);
  }

  function drawStackedBar(selector, items, opts = {}) {
    const node = nodeOf(selector);
    if (!node) return;
    const rows = (items || []).filter((d) => d && Number(d.value) > 0);
    if (!rows.length) {
      emptyChart(node, opts.empty || 'No mix to show.');
      return;
    }
    const paint = () => {
      const w = Math.max(node.clientWidth || 0, 140);
      const h = opts.height || 52;
      const total = d3.sum(rows, (d) => d.value) || 1;
      clearChart(node);
      const svg = d3.select(node).append('svg')
        .attr('width', w).attr('height', 36).attr('viewBox', `0 0 ${w} 36`);
      let x0 = 0;
      const tip = ensureTooltip();
      rows.forEach((d, i) => {
        const bw = (d.value / total) * w;
        svg.append('rect')
          .attr('x', x0).attr('y', 8).attr('width', Math.max(0, bw)).attr('height', 20).attr('rx', 2)
          .attr('fill', d.color || PALETTE[i % PALETTE.length])
          .on('mousemove', (event) => {
            tip.style.opacity = '1';
            tip.innerHTML = `<div class="tip-date">${d.label}</div><div><b>${d.value}</b> (${((d.value / total) * 100).toFixed(0)}%)</div>`;
            placeTip(tip, event);
          })
          .on('mouseleave', () => { tip.style.opacity = '0'; });
        if (bw > 40) {
          svg.append('text').attr('class', 'stack-lab')
            .attr('x', x0 + bw / 2).attr('y', 22).attr('text-anchor', 'middle')
            .text(`${d.label} ${d.value}`);
        }
        x0 += bw;
      });
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

  return {
    drawLineChart, drawHeatmap, drawBars, drawXYLines, drawDonut, drawGauge,
    drawSparkline, drawTreemap, drawScatter, drawWaterfall, drawPack, drawRadar,
    drawHistogram, drawStackedBar, tableToObjects, pct,
  };
})();
