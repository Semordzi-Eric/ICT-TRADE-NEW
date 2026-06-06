/* ─────────────────────────────────────────────────────────────────────────
   ICT Trading Bot Dashboard — Main JS
   ───────────────────────────────────────────────────────────────────────── */

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  selectedSymbols: ['EURUSD'],
  timeframe: 'M15',
  backtestResults: {},   // symbol → result
  activeBacktestSym: null,
  equityCharts: {},
  foldChart: null,
  liveInterval: null,
};

// ── DOM helpers ────────────────────────────────────────────────────────────
const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// ── Tab navigation ─────────────────────────────────────────────────────────
$$('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const target = btn.dataset.tab;
    $$('.tab-btn').forEach(b => b.classList.toggle('active', b === btn));
    $$('.tab-panel').forEach(p => p.classList.toggle('active', p.id === target));
    if (target === 'tab-models') loadModels();
    if (target === 'tab-live') startLivePoll();
    else stopLivePoll();
  });
});

// ── Symbol chips ───────────────────────────────────────────────────────────
function initSymbolChips() {
  fetch('/api/symbols')
    .then(r => r.json())
    .then(({ symbols }) => {
      // Setup DL chips
      renderChips('setup-dl-symbol-chips', symbols, state.selectedSymbols, (sel) => {});
      // Backtest chips
      renderChips('bt-symbol-chips', symbols, state.selectedSymbols, (sel) => {
        state.selectedSymbols = sel;
      });
      // Train select
      const trainSel = $('#train-symbol');
      trainSel.innerHTML = symbols.map(s => `<option value="${s}">${s}</option>`).join('');
      // Live chips
      renderChips('live-symbol-chips', symbols, state.selectedSymbols, (sel) => {
        state.liveSymbols = sel;
      });
    });
}

function renderChips(containerId, symbols, selected, onChange) {
  const container = $(`#${containerId}`);
  if (!container) return;
  container.innerHTML = '';
  symbols.forEach(sym => {
    const chip = document.createElement('span');
    chip.className = 'chip' + (selected.includes(sym) ? ' selected' : '');
    chip.textContent = sym;
    chip.addEventListener('click', () => {
      chip.classList.toggle('selected');
      const sel = $$('.chip.selected', container).map(c => c.textContent);
      onChange(sel);
    });
    container.appendChild(chip);
  });
}

// ── Setup (MT5 & Data Download) ────────────────────────────────────────────
function initSetup() {
  $('#setup-mt5-connect-btn').addEventListener('click', connectMT5);
  $('#setup-dl-btn').addEventListener('click', downloadData);
  checkMT5Status();
}

function checkMT5Status() {
  fetch('/api/mt5/status')
    .then(r => r.json())
    .then(data => renderMT5Status(data))
    .catch(e => renderMT5Status({ connected: false, error: e.toString() }));
}

function connectMT5() {
  const btn = $('#setup-mt5-connect-btn');
  btn.disabled = true;
  btn.textContent = 'Connecting...';
  
  fetch('/api/mt5/connect', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      renderMT5Status(data);
      btn.disabled = false;
      btn.textContent = '🔗 Connect to MT5';
    })
    .catch(e => {
      renderMT5Status({ connected: false, error: e.toString() });
      btn.disabled = false;
      btn.textContent = '🔗 Connect to MT5';
    });
}

function renderMT5Status(data) {
  const statusEl = $('#setup-mt5-status');
  if (!statusEl) return;
  
  if (data.connected) {
    const acct = data.account || {};
    statusEl.innerHTML = `
      <div style="display: flex; align-items: center; margin-bottom: 10px;">
        <span class="status-dot status-ok" style="margin-right: 8px;"></span>
        <strong style="color: #e2e8f0; font-size: 1.1rem;">Connected to MT5</strong>
      </div>
      <div class="grid-3" style="gap: 10px; font-size: 0.85rem; color: var(--text2);">
        <div><strong>Account:</strong> ${acct.login}</div>
        <div><strong>Name:</strong> ${acct.name}</div>
        <div><strong>Server:</strong> ${acct.server}</div>
        <div><strong>Balance:</strong> ${acct.currency} ${Number(acct.balance).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
        <div><strong>Equity:</strong> ${acct.currency} ${Number(acct.equity).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</div>
        <div><strong>Leverage:</strong> 1:${acct.leverage}</div>
      </div>
    `;
  } else {
    statusEl.innerHTML = `
      <div style="display: flex; align-items: center;">
        <span class="status-dot status-bad" style="margin-right: 8px;"></span>
        <strong style="color: var(--red);">Disconnected</strong>
      </div>
      <div style="font-size: 0.85rem; color: var(--text3); margin-top: 5px;">
        ${data.error || 'MetaTrader 5 terminal is not open or not connected.'}
      </div>
    `;
  }
}

function downloadData() {
  const symbols = $$('#setup-dl-symbol-chips .chip.selected').map(c => c.textContent);
  if (!symbols.length) { alert('Select at least one symbol'); return; }

  const timeframe = $('#setup-dl-tf').value;
  const bars = parseInt($('#setup-dl-bars').value) || 200000;
  
  const btn = $('#setup-dl-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin">⟳</span> Downloading...';
  
  const logBox = $('#setup-dl-log');
  logBox.style.display = 'block';
  logBox.innerHTML = '';
  
  fetch('/api/data/download', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbols, timeframe, bars })
  }).then(res => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    function read() {
      reader.read().then(({ done, value }) => {
        if (done) {
          btn.disabled = false;
          btn.innerHTML = '⬇️ Download Data';
          loadDataStatus(); // refresh backtest tab data status
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          try {
            const ev = JSON.parse(line.slice(6));
            handleDownloadEvent(ev, logBox);
          } catch {}
        });
        read();
      });
    }
    read();
  }).catch(e => {
    logBox.innerHTML += `<div class="log-error">Error: ${e}</div>`;
    btn.disabled = false;
    btn.innerHTML = '⬇️ Download Data';
  });
}

function handleDownloadEvent(ev, logBox) {
  if (ev.type === 'log') {
    const level = ev.msg.startsWith('WARNING') ? 'warn' : ev.msg.startsWith('ERROR') ? 'error' : 'info';
    const div = document.createElement('div');
    div.className = `log-${level}`;
    div.textContent = ev.msg;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;
  } else if (ev.type === 'result') {
    const res = ev.data;
    const symbols = Object.keys(res);
    symbols.forEach(sym => {
       const info = res[sym];
       const div = document.createElement('div');
       if (info.error) {
         div.className = 'log-error';
         div.textContent = `${sym}: Error - ${info.error}`;
       } else {
         div.className = 'log-info';
         div.textContent = `${sym}: Saved ${info.bars.toLocaleString()} bars from ${info.source}`;
       }
       logBox.appendChild(div);
    });
    logBox.scrollTop = logBox.scrollHeight;
  } else if (ev.type === 'error') {
    logBox.innerHTML += `<div class="log-error">❌ ${ev.msg}</div>`;
  } else if (ev.type === 'done') {
    logBox.innerHTML += `<div class="log-done">✓ Download Complete</div>`;
  }
}

// ── Data status ────────────────────────────────────────────────────────────
function loadDataStatus() {
  fetch('/api/data/status')
    .then(r => r.json())
    .then(({ datasets }) => {
      const tbody = $('#data-table-body');
      if (!tbody) return;
      if (!datasets.length) {
        tbody.innerHTML = '<tr><td colspan="4" style="text-align:center;color:var(--text3)">No cached data found. Run --download-data first.</td></tr>';
        return;
      }
      tbody.innerHTML = datasets.map(d => {
        const cls = d.bars > 10000 ? 'status-ok' : d.bars > 2000 ? 'status-warn' : 'status-bad';
        const label = d.bars > 10000 ? 'Good' : d.bars > 2000 ? 'Low' : 'Insufficient';
        return `<tr>
          <td><span class="status-dot ${cls}"></span>${d.symbol}</td>
          <td>${d.timeframe}</td>
          <td>${d.bars.toLocaleString()}</td>
          <td><span style="color:var(--text3);font-size:0.8rem">${label}</span></td>
        </tr>`;
      }).join('');
    });
}

// ── Backtest ───────────────────────────────────────────────────────────────
function initBacktest() {
  $('#bt-run-btn').addEventListener('click', runBacktest);
  $('#bt-tf').addEventListener('change', e => state.timeframe = e.target.value);
}

function runBacktest() {
  const symbols = $$('#bt-symbol-chips .chip.selected').map(c => c.textContent);
  if (!symbols.length) { alert('Select at least one symbol'); return; }

  const btn = $('#bt-run-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin">⟳</span> Running…';

  const logBox = $('#bt-log');
  const balance = parseFloat($('#bt-balance').value) || 10000;
  logBox.innerHTML = '';

  // Clear previous results
  state.backtestResults = {};
  state.activeBacktestSym = null;
  $('#bt-sym-tabs').innerHTML = '';
  $('#bt-results-area').style.display = 'none';
  $('#bt-metrics-grid').innerHTML = '';

  const es = new EventSource(`/api/backtest?_t=${Date.now()}`);
  // Can't pass POST body to EventSource, use fetch with ReadableStream
  es.close();

  fetch('/api/backtest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbols, timeframe: state.timeframe, starting_balance: balance }),
  }).then(res => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    function read() {
      reader.read().then(({ done, value }) => {
        if (done) {
          btn.disabled = false;
          btn.innerHTML = '▶ Run Backtest';
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          try {
            const ev = JSON.parse(line.slice(6));
            handleBacktestEvent(ev, logBox);
          } catch {}
        });
        read();
      });
    }
    read();
  }).catch(e => {
    logBox.innerHTML += `<div class="log-error">Error: ${e}</div>`;
    btn.disabled = false;
    btn.innerHTML = '▶ Run Backtest';
  });
}

function handleBacktestEvent(ev, logBox) {
  if (ev.type === 'log') {
    const level = ev.msg.startsWith('WARNING') ? 'warn' : ev.msg.startsWith('ERROR') ? 'error' : 'info';
    const div = document.createElement('div');
    div.className = `log-${level}`;
    div.textContent = ev.msg;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;
  } else if (ev.type === 'result') {
    renderBacktestResults(ev.data);
  } else if (ev.type === 'error') {
    logBox.innerHTML += `<div class="log-error">❌ ${ev.msg}</div>`;
  } else if (ev.type === 'done') {
    logBox.innerHTML += `<div class="log-done">✓ Done</div>`;
  }
}

function renderBacktestResults(data) {
  state.backtestResults = data;
  const symbols = Object.keys(data).filter(s => !data[s].error);
  if (!symbols.length) return;

  const area = $('#bt-results-area');
  area.style.display = 'block';
  area.classList.add('fade-in');

  // Symbol tabs
  const tabsEl = $('#bt-sym-tabs');
  tabsEl.innerHTML = symbols.map((s, i) =>
    `<span class="sym-tab ${i === 0 ? 'active' : ''}" data-sym="${s}">${s}</span>`
  ).join('');
  $$('.sym-tab', tabsEl).forEach(t => {
    t.addEventListener('click', () => {
      $$('.sym-tab', tabsEl).forEach(x => x.classList.remove('active'));
      t.classList.add('active');
      showSymbolResults(t.dataset.sym);
    });
  });

  showSymbolResults(symbols[0]);
}

function showSymbolResults(symbol) {
  state.activeBacktestSym = symbol;
  const d = state.backtestResults[symbol];
  if (!d || d.error) return;

  const m = d.metrics;

  // Metric cards
  const metricsEl = $('#bt-metrics-grid');
  const fmt = (v, isPercent = false, dp = 2) => {
    if (v === undefined || v === null) return '—';
    return isPercent ? `${(v * 100).toFixed(dp)}%` : Number(v).toFixed(dp);
  };

  const goodBad = (v, gt, lt) => v > gt ? 'good' : v < lt ? 'bad' : 'neutral';

  metricsEl.innerHTML = `
    <div class="metric-card">
      <div class="metric-label">Total Return</div>
      <div class="metric-value ${m.total_return > 0 ? 'good' : 'bad'}">${fmt(m.total_return, true)}</div>
      <div class="metric-sub">Net PnL: $${fmt(m.net_pnl)}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Win Rate</div>
      <div class="metric-value ${goodBad(m.win_rate, 0.5, 0.35)}">${fmt(m.win_rate, true)}</div>
      <div class="metric-sub">${m.n_trades} trades</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Profit Factor</div>
      <div class="metric-value ${goodBad(m.profit_factor, 1.3, 1.0)}">${fmt(m.profit_factor)}</div>
      <div class="metric-sub">Gross P: $${fmt(m.gross_profit)} / L: $${fmt(m.gross_loss)}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Sharpe Ratio</div>
      <div class="metric-value ${goodBad(m.sharpe, 1.0, 0)}">${fmt(m.sharpe)}</div>
      <div class="metric-sub">Sortino: ${fmt(m.sortino)}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Max Drawdown</div>
      <div class="metric-value ${goodBad(-m.max_drawdown_pct, -0.1, -0.2)}">${fmt(m.max_drawdown_pct, true)}</div>
      <div class="metric-sub">Calmar: ${fmt(m.calmar)}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Avg R-Multiple</div>
      <div class="metric-value ${goodBad(m.avg_r_multiple, 0, -0.5)}">${fmt(m.avg_r_multiple)}</div>
      <div class="metric-sub">Expectancy: $${fmt(m.expectancy)}</div>
    </div>
  `;

  // Equity curve
  renderEquityCurve(symbol, d.equity_curve);

  // Trades table (last 50)
  const trades = (d.trades || []).slice(-50).reverse();
  const tbody = $('#bt-trades-body');
  if (tbody) {
    tbody.innerHTML = trades.map(t => {
      const win = t.pnl > 0;
      const pnlCls = win ? 'td-win' : 'td-loss';
      const exitIcon = t.exit_reason === 'tp' ? '🎯' : t.exit_reason === 'sl' ? '🛑' : '⏱';
      return `<tr>
        <td>${t.entry_time ? t.entry_time.slice(0,16) : '—'}</td>
        <td>${t.direction === 'long' ? '🟢 Long' : '🔴 Short'}</td>
        <td>${t.setup_type || '—'}</td>
        <td>${(t.entry_price || 0).toFixed(5)}</td>
        <td class="${pnlCls}">$${(t.pnl || 0).toFixed(2)}</td>
        <td class="${pnlCls}">${(t.r_multiple || 0).toFixed(2)}R</td>
        <td>${exitIcon} ${t.exit_reason || '—'}</td>
        <td>${t.bars_held || 0} bars</td>
      </tr>`;
    }).join('') || '<tr><td colspan="8" style="text-align:center;color:var(--text3)">No trades</td></tr>';
  }

  // Detection counts
  const counts = d.detection_counts || {};
  const detEl = $('#bt-detection-counts');
  if (detEl) {
    detEl.innerHTML = ['fvg', 'order_blocks', 'sweeps', 'bos', 'choch'].map(k =>
      `<div class="metric-card" style="padding:12px 16px">
        <div class="metric-label">${k.replace('_', ' ')}</div>
        <div class="metric-value neutral" style="font-size:1.3rem">${counts[k] ?? '—'}</div>
      </div>`
    ).join('');
  }
}

function renderEquityCurve(symbol, equityData) {
  const canvas = $('#bt-equity-chart');
  if (!canvas || !equityData) return;

  if (state.equityCharts[symbol]) {
    state.equityCharts[symbol].destroy();
  }

  const timestamps = equityData.timestamps;
  const values = equityData.values;

  // Compute drawdown
  let peak = values[0];
  const dd = values.map(v => {
    peak = Math.max(peak, v);
    return ((v - peak) / peak) * 100;
  });

  state.equityCharts[symbol] = new Chart(canvas, {
    type: 'line',
    data: {
      labels: timestamps,
      datasets: [
        {
          label: 'Equity',
          data: values,
          borderColor: '#00d4aa',
          backgroundColor: 'rgba(0,212,170,0.06)',
          borderWidth: 2,
          pointRadius: 0,
          fill: true,
          tension: 0.3,
          yAxisID: 'y',
        },
        {
          label: 'Drawdown %',
          data: dd,
          borderColor: 'rgba(244,63,94,0.7)',
          backgroundColor: 'rgba(244,63,94,0.07)',
          borderWidth: 1.5,
          pointRadius: 0,
          fill: true,
          tension: 0.2,
          yAxisID: 'y2',
        },
      ],
    },
    options: {
      responsive: true,
      animation: { duration: 400 },
      interaction: { mode: 'index', intersect: false },
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Inter' } } },
        tooltip: {
          borderWidth: 1,
          callbacks: {
            label: ctx => ctx.datasetIndex === 0
              ? `Equity: $${ctx.parsed.y.toFixed(2)}`
              : `Drawdown: ${ctx.parsed.y.toFixed(2)}%`,
          },
        },
      },
      scales: {
        x: { ticks: { color: '#64748b', maxTicksLimit: 8, font: { size: 11 } } },
        y: { ticks: { color: '#64748b', font: { size: 11 }, callback: v => '$' + v.toLocaleString() }, position: 'left' },
        y2: { ticks: { color: 'rgba(244,63,94,0.6)', font: { size: 10 }, callback: v => v.toFixed(1) + '%' }, grid: { display: false }, position: 'right', max: 0 },
      },
    },
  });
}

// ── Training ───────────────────────────────────────────────────────────────
function initTrain() {
  $('#train-btn').addEventListener('click', runTrain);
}

function runTrain() {
  const symbol = $('#train-symbol').value;
  const timeframe = $('#train-tf').value;
  const btn = $('#train-btn');
  btn.disabled = true;
  btn.innerHTML = '<span class="spin">⟳</span> Training…';

  const logBox = $('#train-log');
  logBox.innerHTML = '';
  $('#train-fold-area').style.display = 'none';

  fetch('/api/train', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbol, timeframe }),
  }).then(res => {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    function read() {
      reader.read().then(({ done, value }) => {
        if (done) {
          btn.disabled = false;
          btn.innerHTML = '🧠 Train Model';
          return;
        }
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        lines.forEach(line => {
          if (!line.startsWith('data: ')) return;
          try {
            const ev = JSON.parse(line.slice(6));
            handleTrainEvent(ev, logBox);
          } catch {}
        });
        read();
      });
    }
    read();
  }).catch(e => {
    logBox.innerHTML += `<div class="log-error">Error: ${e}</div>`;
    btn.disabled = false;
    btn.innerHTML = '🧠 Train Model';
  });
}

function handleTrainEvent(ev, logBox) {
  if (ev.type === 'log') {
    const level = ev.msg.startsWith('WARNING') ? 'warn' : ev.msg.startsWith('ERROR') ? 'error' : 'info';
    const div = document.createElement('div');
    div.className = `log-${level}`;
    div.textContent = ev.msg;
    logBox.appendChild(div);
    logBox.scrollTop = logBox.scrollHeight;
  } else if (ev.type === 'result') {
    renderFoldResults(ev.data.folds || []);
    loadModels();
  } else if (ev.type === 'error') {
    logBox.innerHTML += `<div class="log-error">❌ ${ev.msg}</div>`;
  } else if (ev.type === 'done') {
    logBox.innerHTML += `<div class="log-done">✓ Training complete — check Models tab</div>`;
  }
}

function renderFoldResults(folds) {
  if (!folds.length) return;
  const area = $('#train-fold-area');
  area.style.display = 'block';

  const canvas = $('#fold-auc-chart');
  if (state.foldChart) state.foldChart.destroy();

  state.foldChart = new Chart(canvas, {
    type: 'bar',
    data: {
      labels: folds.map(f => `Fold ${f.fold}`),
      datasets: [{
        label: 'AUC (test)',
        data: folds.map(f => f.auc_test),
        backgroundColor: folds.map(f => f.auc_test >= 0.55 ? 'rgba(0,212,170,0.7)' : 'rgba(244,63,94,0.6)'),
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: ctx => `AUC: ${ctx.parsed.y.toFixed(4)}` } },
      },
      scales: {
        x: { ticks: { color: '#64748b' } },
        y: { min: 0.4, max: 1, ticks: { color: '#64748b' } },
      },
    },
  });

  // Fold summary table
  const tbody = $('#fold-table-body');
  if (tbody) {
    tbody.innerHTML = folds.map(f => `<tr>
      <td>${f.fold}</td>
      <td>${f.train_start ? f.train_start.slice(0,10) : '—'}</td>
      <td>${f.train_end ? f.train_end.slice(0,10) : '—'}</td>
      <td>${f.val_end ? f.val_end.slice(0,10) : '—'}</td>
      <td class="${f.auc_test >= 0.55 ? 'td-win' : 'td-loss'}">${f.auc_test ? f.auc_test.toFixed(4) : '—'}</td>
      <td>${f.n_train?.toLocaleString() ?? '—'}</td>
      <td>${f.n_test?.toLocaleString() ?? '—'}</td>
    </tr>`).join('');
  }
}

// ── Models ─────────────────────────────────────────────────────────────────
function loadModels() {
  fetch('/api/models')
    .then(r => r.json())
    .then(({ models }) => {
      const grid = $('#models-grid');
      const empty = $('#models-empty');
      if (!models.length) {
        grid.innerHTML = '';
        if (empty) empty.style.display = 'block';
        return;
      }
      if (empty) empty.style.display = 'none';
      grid.innerHTML = models.map(m => renderModelCard(m)).join('');
      // Attach push-to-live buttons
      $$('.push-live-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const path = btn.dataset.path;
          const symbols = btn.dataset.symbol ? [btn.dataset.symbol] : ['EURUSD'];
          pushModelToLive(path, symbols);
        });
      });
    });
}

function renderModelCard(m) {
  const auc = m.avg_auc !== null ? m.avg_auc.toFixed(4) : '—';
  const aucColor = m.avg_auc >= 0.55 ? 'good' : m.avg_auc >= 0.5 ? 'neutral' : 'bad';
  const date = m.trained_at !== '?' ? m.trained_at.slice(0,16).replace('T',' ') : '—';
  const posRate = m.positive_rate !== null ? (m.positive_rate * 100).toFixed(1) + '%' : '—';
  const nSamples = m.n_samples ? m.n_samples.toLocaleString() : '—';
  return `
    <div class="model-card">
      <div class="model-header">
        <div>
          <div class="model-name">${m.symbol} ${m.timeframe}</div>
          <div style="font-size:0.75rem;color:var(--text3);margin-top:3px">${date}</div>
        </div>
        <span class="model-badge">Ensemble</span>
      </div>
      <div class="model-stats">
        <div class="model-stat"><div class="v ${aucColor}">${auc}</div><div class="l">Avg AUC</div></div>
        <div class="model-stat"><div class="v">${nSamples}</div><div class="l">Samples</div></div>
        <div class="model-stat"><div class="v">${posRate}</div><div class="l">Win Rate</div></div>
      </div>
      <div style="font-size:0.75rem;color:var(--text3);margin-bottom:10px">${m.folds.length} fold(s) trained</div>
      <div class="model-actions">
        <button class="btn btn-primary btn-sm push-live-btn"
          data-path="${m.path}" data-symbol="${m.symbol}">
          🚀 Push to Live
        </button>
        <button class="btn btn-secondary btn-sm" onclick="runBacktestWithModel('${m.path}', '${m.symbol}', '${m.timeframe}')">
          📊 Backtest
        </button>
      </div>
    </div>
  `;
}

function runBacktestWithModel(path, symbol, tf) {
  // Switch to backtest tab and pre-select
  $$('.tab-btn').forEach(b => b.classList.remove('active'));
  $$('.tab-panel').forEach(p => p.classList.remove('active'));
  $('[data-tab="tab-backtest"]').classList.add('active');
  $('#tab-backtest').classList.add('active');
  // Pre-select the symbol
  $$('#bt-symbol-chips .chip').forEach(c => {
    c.classList.toggle('selected', c.textContent === symbol);
  });
  state.selectedSymbols = [symbol];
  if (tf) $('#bt-tf').value = tf;
  setTimeout(runBacktest, 100);
}

function pushModelToLive(modelPath, symbols) {
  if (!confirm(`Push model from\n${modelPath}\nto LIVE on ${symbols.join(', ')}?`)) return;

  fetch('/api/live/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ symbols, model_path: modelPath, timeframe: 'M15' }),
  })
    .then(r => r.json())
    .then(d => {
      if (d.status === 'started') {
        alert('✅ Live executor started! Switch to the Live tab to monitor.');
        $$('.tab-btn').forEach(b => b.classList.remove('active'));
        $$('.tab-panel').forEach(p => p.classList.remove('active'));
        $('[data-tab="tab-live"]').classList.add('active');
        $('#tab-live').classList.add('active');
        startLivePoll();
      } else {
        alert('Already running or error: ' + JSON.stringify(d));
      }
    });
}

// ── Live monitoring ────────────────────────────────────────────────────────
function initLive() {
  $('#live-stop-btn').addEventListener('click', () => {
    fetch('/api/live/stop', { method: 'POST' }).then(() => {
      setTimeout(pollLiveStatus, 1000);
    });
  });
}

function startLivePoll() {
  if (state.liveInterval) return;
  pollLiveStatus();
  state.liveInterval = setInterval(pollLiveStatus, 3000);
}

function stopLivePoll() {
  clearInterval(state.liveInterval);
  state.liveInterval = null;
}

function pollLiveStatus() {
  fetch('/api/live/status')
    .then(r => r.json())
    .then(({ running, log }) => {
      // Update header badge
      const badge = $('#live-badge');
      if (badge) {
        badge.className = 'live-badge' + (running ? ' active' : '');
        badge.querySelector('.dot-text').textContent = running ? 'LIVE' : 'OFFLINE';
      }

      // Update live tab status
      const statusText = $('#live-status-text');
      if (statusText) statusText.textContent = running ? '🟢 Executor running' : '⚫ Executor stopped';

      const logPanel = $('#live-log-panel');
      if (logPanel) {
        logPanel.innerHTML = log.map(l => `<div>${l}</div>`).join('') || '<div style="color:var(--text3)">No activity yet</div>';
        logPanel.scrollTop = logPanel.scrollHeight;
      }

      const stopBtn = $('#live-stop-btn');
      if (stopBtn) stopBtn.disabled = !running;
    });
}

// ── Theme (Light/Dark Mode) ────────────────────────────────────────────────
function initTheme() {
  const toggle = $('#theme-toggle');
  if (!toggle) return;

  const currentTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.dataset.theme = currentTheme;
  toggle.textContent = currentTheme === 'light' ? '🌙' : '☀️';

  toggle.addEventListener('click', () => {
    const isLight = document.documentElement.dataset.theme === 'light';
    const newTheme = isLight ? 'dark' : 'light';
    document.documentElement.dataset.theme = newTheme;
    localStorage.setItem('theme', newTheme);
    toggle.textContent = newTheme === 'light' ? '🌙' : '☀️';
    
    // Update chart colors
    setChartDefaults();
    Object.values(state.equityCharts).forEach(c => c.update());
    if (state.foldChart) state.foldChart.update();
  });
}

// ── Chart.js global defaults ───────────────────────────────────────────────
function setChartDefaults() {
  const isLight = document.documentElement.dataset.theme === 'light';
  Chart.defaults.font.family = 'Inter';
  Chart.defaults.color = '#64748b';
  Chart.defaults.borderColor = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(30,41,64,0.5)';
  
  if (!Chart.defaults.plugins.tooltip) Chart.defaults.plugins.tooltip = {};
  Chart.defaults.plugins.tooltip.backgroundColor = isLight ? '#ffffff' : '#111622';
  Chart.defaults.plugins.tooltip.borderColor = isLight ? '#e2e8f0' : '#1e2940';
  Chart.defaults.plugins.tooltip.titleColor = isLight ? '#0f172a' : '#e2e8f0';
  Chart.defaults.plugins.tooltip.bodyColor = isLight ? '#475569' : '#94a3b8';
}

// ── Init ───────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  setChartDefaults();
  initSymbolChips();
  initSetup();
  loadDataStatus();
  initBacktest();
  initTrain();
  initLive();
  pollLiveStatus();

  // Start live poll if already on live tab
  if ($('#tab-live.active')) startLivePoll();
});
