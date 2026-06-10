/* ICT Trading Bot — Dashboard JavaScript v2 */

'use strict';

// ═══════════════════════════════════════════════════ UTILITIES
const $ = id => document.getElementById(id);
const el = (tag, cls, html) => { const e = document.createElement(tag); if(cls) e.className=cls; if(html!=null) e.innerHTML=html; return e; };

const fmt = {
  pct:   v => v != null ? (v*100).toFixed(1)+'%' : '—',
  float: (v,d=4) => v != null ? (+v).toFixed(d) : '—',
  int:   v => v != null ? Math.round(v).toLocaleString() : '—',
  pnl:   v => v != null ? (v>=0?'+':'')+v.toFixed(2) : '—',
  date:  s => s ? new Date(s).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'}) : '—',
  ago:   s => { if(!s) return '—'; const d=new Date(s),n=new Date(); const diff=Math.round((n-d)/1000); return diff<60?diff+'s ago':diff<3600?Math.round(diff/60)+'m ago':Math.round(diff/3600)+'h ago'; },
};

// ═══════════════════════════════════════════════════ APP STATE
const State = {
  symbols: [],            // [{symbol, category, session, pip_size}]
  mt5Connected: false,
  liveRunning: false,
  livePoller: null,
  intuitionEnabled: true,
  charts: {},             // chart instances keyed by id
};

// ═══════════════════════════════════════════════════ SSE STREAM HELPER
function streamSSE(url, body, onLog, onResult, onDone, onError) {
  // Use fetch + ReadableStream for POST SSE
  fetch(url, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify(body),
  }).then(resp => {
    if(!resp.ok) { onError && onError('HTTP '+resp.status); return; }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    function pump() {
      reader.read().then(({done, value}) => {
        if(done) { onDone && onDone(); return; }
        buf += decoder.decode(value, {stream:true});
        const lines = buf.split('\n');
        buf = lines.pop();
        for(const line of lines) {
          if(!line.startsWith('data:')) continue;
          try {
            const ev = JSON.parse(line.slice(5).trim());
            if(ev.type==='log')       onLog && onLog(ev.msg, 'info');
            else if(ev.type==='result') onResult && onResult(ev.data);
            else if(ev.type==='done')   onDone && onDone();
            else if(ev.type==='error')  onError && onError(ev.msg);
            else if(ev.type==='keepalive') {}
          } catch(e){}
        }
        pump();
      });
    }
    pump();
  }).catch(err => onError && onError(String(err)));
}

// ═══════════════════════════════════════════════════ LOG CONSOLE
function logAppend(consoleId, msg, level='info') {
  const c = $(consoleId);
  if(!c) return;
  if(c.querySelector('span[style]')) c.innerHTML='';
  const ts = new Date().toISOString().slice(11,19);
  const line = el('div');
  const isIntuition = msg && msg.includes('INTUITION');
  const cls = isIntuition ? 'log-intuition' : `log-${level.toLowerCase().includes('error')?'error':level.toLowerCase().includes('warn')?'warn':'info'}`;
  line.innerHTML = `<span class="log-ts">[${ts}]</span> <span class="${cls}">${escHtml(msg||'')}</span>`;
  c.appendChild(line);
  c.scrollTop = c.scrollHeight;
}
function escHtml(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

// ═══════════════════════════════════════════════════ UI HELPERS
const UI = {
  toggle(el, id) {
    el.classList.toggle('on');
  },
  selectAll(gridId) {
    document.querySelectorAll(`#${gridId} .symbol-chip`).forEach(c=>c.classList.add('selected'));
  },
  selectNone(gridId) {
    document.querySelectorAll(`#${gridId} .symbol-chip`).forEach(c=>c.classList.remove('selected'));
  },
  getSelected(gridId) {
    return [...document.querySelectorAll(`#${gridId} .symbol-chip.selected`)].map(c=>c.dataset.sym);
  },
  switchTab(btn, groupId) {
    const tabId = btn.dataset.tab;
    document.querySelectorAll(`#${groupId} .tab-panel`).forEach(p=>p.classList.remove('active'));
    btn.closest('.tab-bar').querySelectorAll('.tab-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active');
    $(`tab-${tabId.replace('tab-','')}`) && $(`tab-${tabId.replace('tab-','')}`).classList.add('active');
    // also handle id = tabId directly
    $(tabId) && $(tabId).classList.add('active');
  },
  buildSymbolGrid(gridId, symbols, defaultSelected=true) {
    const grid = $(gridId);
    if(!grid) return;
    grid.innerHTML='';
    symbols.forEach(s => {
      const chip = el('div', `symbol-chip cat-${s.category}${defaultSelected?' selected':''}`);
      chip.dataset.sym = s.symbol;
      chip.innerHTML = `<div class="cat-dot"></div>${s.symbol}`;
      chip.onclick = () => chip.classList.toggle('selected');
      grid.appendChild(chip);
    });
  },
  buildSymbolSelect(selectId, symbols) {
    const sel = $(selectId);
    if(!sel) return;
    sel.innerHTML = symbols.map(s=>`<option value="${s.symbol}">${s.symbol} (${s.category})</option>`).join('');
  },
  setLoading(btnId, loading, text='') {
    const b = $(btnId);
    if(!b) return;
    b.disabled = loading;
    if(loading) b.innerHTML = `<span class="spin">⏳</span> ${text||'Working…'}`;
  },
  catColor(cat) {
    return {fx:'#3b82f6', metal:'#f59e0b', index:'#8b5cf6', crypto:'#10b981'}[cat]||'#64748b';
  },
};

// ═══════════════════════════════════════════════════ CHART HELPER
function drawEquityChart(canvasId, timestamps, values) {
  if(State.charts[canvasId]) State.charts[canvasId].destroy();
  const ctx = $(canvasId);
  if(!ctx) return;
  const start = values[0]||0;
  const colors = values.map(v=>v>=start?'rgba(16,185,129,0.8)':'rgba(239,68,68,0.8)');
  State.charts[canvasId] = new Chart(ctx, {
    type:'line',
    data:{
      labels: timestamps.map(t=>t.slice(0,10)),
      datasets:[{
        data: values,
        borderColor: values[values.length-1]>=start?'#10b981':'#ef4444',
        backgroundColor: values[values.length-1]>=start?'rgba(16,185,129,0.06)':'rgba(239,68,68,0.06)',
        borderWidth: 1.5, fill: true, tension: 0.3, pointRadius: 0,
      }],
    },
    options:{
      responsive:true, maintainAspectRatio:false, animation:false,
      plugins:{legend:{display:false},tooltip:{mode:'index',intersect:false,callbacks:{label:ctx=>`$${ctx.parsed.y.toFixed(2)}`}}},
      scales:{
        x:{display:false,grid:{display:false}},
        y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#64748b',font:{size:10},callback:v=>`$${v.toFixed(0)}`}},
      },
    },
  });
}

// ═══════════════════════════════════════════════════ NAV
function initNav() {
  document.querySelectorAll('.nav-item[data-page]').forEach(btn => {
    btn.onclick = () => navigateTo(btn.dataset.page);
  });
}

function navigateTo(pageId) {
  document.querySelectorAll('.nav-item').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  const nav = $(`nav-${pageId}`);
  const page = $(`page-${pageId}`);
  if(nav)  nav.classList.add('active');
  if(page) page.classList.add('active');

  const titles = {
    overview:  ['Market Overview','All 16 markets — live status, bias & news'],
    sentiment: ['Sentiment Engine','Multi-source economic calendar & news scores'],
    markets:   ['Supported Markets','ICT compatibility guide for all asset classes'],
    data:      ['Data Management','Download historical OHLCV data'],
    train:     ['Model Training','Walk-forward ensemble — LightGBM + XGBoost + LSTM'],
    backtest:  ['Backtest','Simulate strategy on historical data'],
    benchmark: ['Market Benchmark','Ranked leaderboard of all champion models'],
    live:      ['Live Trading','24/7 multi-market automated execution'],
    registry:  ['Model Registry','Per-symbol champion models'],
    setup:     ['Setup / MT5','Connection & account configuration'],
  };
  const [title, sub] = titles[pageId]||['',''];
  $('page-title').textContent = title;
  $('page-sub').textContent = sub;

  // Lazy-load page data
  if(Pages[pageId]&&Pages[pageId].load) Pages[pageId].load();
}

// ═══════════════════════════════════════════════════ PAGES
const Pages = {};

// ─────────────────────────────────────────────── MT5 POLL
async function pollMT5() {
  try {
    const r = await fetch('/api/mt5/status');
    const d = await r.json();
    State.mt5Connected = d.connected;
    const dot = $('mt5-dot');
    const val = $('mt5-label-val');
    if(d.connected) {
      dot.classList.add('connected');
      val.textContent = `#${d.account.login}`;
    } else {
      dot.classList.remove('connected');
      val.textContent = 'Disconnected';
    }
  } catch(e) {}
}

// ─────────────────────────────────────────────── OVERVIEW
Pages.overview = {
  async load() {
    const [ovResp, sentResp] = await Promise.all([
      fetch('/api/market/overview').then(r=>r.json()).catch(()=>({overview:[]})),
      fetch('/api/sentiment').then(r=>r.json()).catch(()=>({sentiment:[]})),
    ]);
    const rows = ovResp.overview||[];
    const sentMap = {};
    (sentResp.sentiment||[]).forEach(s=>sentMap[s.symbol]=s);

    // Stats bar
    const total = rows.length;
    const withModel = rows.filter(r=>r.has_model).length;
    const withData  = rows.filter(r=>r.has_data).length;
    const blocked   = rows.filter(r=>!r.news_clear).length;
    $('overview-stats').innerHTML = `
      <div class="stat-card blue"><div class="stat-label">Markets</div><div class="stat-value">${total}</div><div class="stat-meta">Configured symbols</div></div>
      <div class="stat-card green"><div class="stat-label">Models Ready</div><div class="stat-value">${withModel}</div><div class="stat-meta">${total-withModel} need training</div></div>
      <div class="stat-card gold"><div class="stat-label">Data Available</div><div class="stat-value">${withData}</div><div class="stat-meta">${total-withData} need download</div></div>
      <div class="stat-card purple"><div class="stat-label">News Blocked</div><div class="stat-value">${blocked}</div><div class="stat-meta">High-impact event active</div></div>`;

    // Market cards
    const grid = $('market-grid');
    if(!rows.length) { grid.innerHTML='<div class="empty-state"><div class="empty-icon">⚙️</div><h3>Configure symbols in strategy_config.yaml</h3></div>'; return; }
    grid.innerHTML='';
    rows.forEach(r => {
      const sent = sentMap[r.symbol];
      const sentClass = (r.sentiment_score||0)>0.15?'bull':(r.sentiment_score||0)<-0.15?'bear':'neut';
      const statusCls = !r.news_clear?'blocked':r.drift_alert?'warn':'ok';
      const card = el('div', `market-card ${statusCls} fade-in`);
      const auc = r.champion_auc;
      const aucPct = auc?Math.round(auc*100):0;
      const aucColor = auc>=0.6?'#10b981':auc>=0.55?'#f59e0b':'#ef4444';
      card.innerHTML = `
        <div style="display:flex;justify-content:space-between;align-items:flex-start">
          <div>
            <div class="market-sym">${r.symbol}</div>
            <div class="market-meta">${r.category.toUpperCase()} · ${r.session}</div>
          </div>
          <span class="tag ${r.has_data?'tag-green':'tag-gray'}" style="font-size:10px">${r.has_data?'Data ✓':'No Data'}</span>
        </div>
        <div class="market-row">
          <span class="tag ${r.has_model?'tag-blue':'tag-gray'}">${r.has_model?'Model ✓':'No Model'}</span>
          <span class="market-sentiment ${sentClass}">${r.sentiment}</span>
        </div>
        <div class="auc-bar">
          <div class="auc-label"><span>Champion AUC</span><span style="color:${aucColor}">${auc?auc.toFixed(3):'—'}</span></div>
          <div class="progress-wrap"><div class="progress-bar" style="width:${aucPct}%;background:${aucColor}"></div></div>
        </div>
        ${!r.news_clear?`<div style="font-size:10px;color:#f87171;margin-top:6px">🔴 News blackout</div>`:''}
        ${r.drift_alert?`<div style="font-size:10px;color:#fbbf24;margin-top:6px">⚠ Model drift — retrain recommended</div>`:''}`;
      grid.appendChild(card);
    });
  }
};

// ─────────────────────────────────────────────── SENTIMENT
Pages.sentiment = {
  async load() {
    const d = await fetch('/api/sentiment').then(r=>r.json()).catch(()=>({sentiment:[],upcoming_events:[]}));
    const items = d.sentiment||[];
    const events = d.upcoming_events||[];

    // Sentiment bars
    const barsDiv = $('sentiment-bars');
    barsDiv.innerHTML='';
    items.forEach(s => {
      const score = s.score||0;
      const pct   = Math.abs(score)*50;
      const color  = score>0?'bull-bar':'bear-bar';
      const row = el('div','sentiment-row');
      row.innerHTML = `
        <div class="sentiment-sym">${s.symbol}</div>
        <div class="sentiment-bar-wrap">
          <div class="sentiment-mid-line"></div>
          <div class="sentiment-bar ${color}" style="width:${pct}%"></div>
        </div>
        <div class="sentiment-score-val ${score>0.15?'bull':score<-0.15?'bear':'neut'}">${score>=0?'+':''}${score.toFixed(2)}</div>
        <span class="tag ${s.is_blocked||s.trade_blocked?'tag-red':'tag-green'}" style="font-size:10px;width:70px;justify-content:center">${s.is_blocked||s.trade_blocked?'Blocked':'Clear'}</span>`;
      barsDiv.appendChild(row);
    });

    // Upcoming events
    const evDiv = $('upcoming-events');
    if(!events.length) { evDiv.innerHTML='<div class="empty-state"><div class="empty-icon">📭</div><h3>No high-impact events in next 24h</h3></div>'; }
    else {
      evDiv.innerHTML='';
      events.slice(0,20).forEach(e => {
        const row = el('div','event-item');
        const t = new Date(e.dt);
        const impCls = e.impact.toLowerCase()==='high'?'impact-high':e.impact.toLowerCase()==='medium'?'impact-med':'impact-low';
        row.innerHTML = `
          <div class="event-time">${t.toISOString().slice(11,16)}</div>
          <div class="event-ccy">${e.currency}</div>
          <div class="event-title">${escHtml(e.title)}</div>
          <span class="event-impact ${impCls}">${e.impact}</span>`;
        evDiv.appendChild(row);
      });
    }

    // Clearance table
    const tbody = $('clearance-body');
    tbody.innerHTML='';
    items.forEach(s => {
      const blocked = s.is_blocked||s.trade_blocked;
      const tr = el('tr');
      tr.innerHTML = `
        <td><strong>${s.symbol}</strong></td>
        <td>${blocked?'<span class="tag tag-red">🔴 Blocked</span>':s.pre_event_warning?'<span class="tag tag-gold">⚠ Pre-event</span>':'<span class="tag tag-green">✅ Clear</span>'}</td>
        <td style="font-family:var(--font-mono)">${(s.score>=0?'+':'')+s.score.toFixed(2)}</td>
        <td><span class="tag ${s.label.includes('bull')?'tag-green':s.label.includes('bear')?'tag-red':'tag-gray'}">${s.label}</span></td>
        <td>${s.events_today||0}</td>
        <td>${(s.sources||[]).join(', ')||'—'}</td>`;
      tbody.appendChild(tr);
    });
  }
};

// ─────────────────────────────────────────────── DATA
const dataPage = Pages.data = {
  async load() {
    await dataPage.loadStatus();
    if(!State.symbols.length) return;
    UI.buildSymbolGrid('dl-symbols', State.symbols, false);
  },
  async loadStatus() {
    const d = await fetch('/api/data/status').then(r=>r.json()).catch(()=>({datasets:[]}));
    const tbody = $('data-status-body');
    const rows = d.datasets||[];
    if(!rows.length) { tbody.innerHTML='<tr><td colspan="5" style="text-align:center;color:var(--text-3);padding:24px">No data downloaded yet</td></tr>'; return; }
    tbody.innerHTML = rows.map(r=>`
      <tr>
        <td><strong>${r.symbol}</strong></td>
        <td><span class="tag tag-blue">${r.timeframe}</span></td>
        <td style="font-family:var(--font-mono)">${r.bars.toLocaleString()}</td>
        <td>${r.size_mb} MB</td>
        <td style="color:var(--text-2)">${fmt.ago(r.modified)}</td>
      </tr>`).join('');
  },
  download() {
    const syms = UI.getSelected('dl-symbols');
    if(!syms.length) { alert('Select at least one symbol'); return; }
    const years = +$('dl-years').value;
    const tf    = $('dl-tf').value;
    $('dl-log').style.display='block';
    UI.setLoading('dl-btn', true, 'Downloading…');
    streamSSE('/api/data/download', {symbols:syms, timeframe:tf, data_years:years, bars:200000},
      (msg)=>logAppend('dl-log',msg),
      (data)=>{
        const res = Object.entries(data||{}).map(([s,v])=>`${s}: ${v.bars||0} bars (${v.source||'err'})`).join(' | ');
        logAppend('dl-log','Done: '+res,'ok');
        dataPage.loadStatus();
      },
      ()=>{ UI.setLoading('dl-btn',false); $('dl-btn').textContent='⬇ Download'; },
      (err)=>{ logAppend('dl-log','Error: '+err,'error'); UI.setLoading('dl-btn',false); $('dl-btn').textContent='⬇ Download'; }
    );
  }
};

// ─────────────────────────────────────────────── TRAIN
Pages.train = {
  load() {
    if(!State.symbols.length) return;
    UI.buildSymbolSelect('train-symbol', State.symbols);
    UI.buildSymbolGrid('train-all-symbols', State.symbols, true);
  },
  start() {
    const sym   = $('train-symbol').value;
    const tf    = $('train-tf').value;
    const years = +$('train-years').value;
    const autoP = $('auto-promote-toggle').classList.contains('on');
    $('train-log').innerHTML='';
    UI.setLoading('train-btn',true,'Training…');
    streamSSE('/api/train', {symbol:sym,timeframe:tf,data_years:years,auto_promote:autoP},
      (msg)=>logAppend('train-log',msg),
      (data)=>{
        const promoted = data.promoted?'✅ Promoted to registry!':'ℹ Not promoted (existing champion better)';
        logAppend('train-log',`avg_auc=${(data.avg_auc||0).toFixed(4)} gt=${(data.gt_score||0).toFixed(4)} ${promoted}`,'ok');
        Pages.train._renderFolds(data.folds||[],'train-result');
      },
      ()=>{ UI.setLoading('train-btn',false); $('train-btn').textContent='🚀 Start Training'; },
      (err)=>{ logAppend('train-log','Error: '+err,'error'); UI.setLoading('train-btn',false); $('train-btn').textContent='🚀 Start Training'; }
    );
  },
  startAll() {
    const syms  = UI.getSelected('train-all-symbols');
    const years = +$('train-all-years').value;
    $('train-all-log').innerHTML='';
    UI.setLoading('train-all-btn',true,'Training all…');
    streamSSE('/api/train/all',{symbols:syms,data_years:years},
      (msg)=>logAppend('train-all-log',msg),
      (data)=>{
        const summary = (data.summary||[]);
        const html = summary.map(s=>{
          const ok = s.status==='ok';
          const cls = ok?'tag-green':'tag-red';
          return `<span class="tag ${cls}" style="margin:2px">${s.symbol}: ${ok?'AUC '+s.avg_auc.toFixed(3):s.status}</span>`;
        }).join('');
        $('train-all-result').innerHTML=`<div style="padding-top:12px">${html}</div>`;
        logAppend('train-all-log','All done','ok');
      },
      ()=>{ UI.setLoading('train-all-btn',false); $('train-all-btn').textContent='🚀 Train All'; },
      (err)=>{ logAppend('train-all-log','Error: '+err,'error'); UI.setLoading('train-all-btn',false); $('train-all-btn').textContent='🚀 Train All'; }
    );
  },
  _renderFolds(folds, targetId) {
    if(!folds.length) return;
    const target = $(targetId);
    if(!target) return;
    const aucMin=0.45,aucMax=0.75;
    const cards = folds.map((f,i)=>{
      const auc = f.auc_test||0;
      const cls  = auc>=0.62?'fold-good':auc>=0.55?'fold-ok':'fold-bad';
      return `<div class="fold-card"><div class="fold-num">Fold ${i+1}</div><div class="fold-auc ${cls}">${auc.toFixed(3)}</div><div style="font-size:10px;color:var(--text-3)">${f.n_train||0} train / ${f.n_test||0} test</div></div>`;
    }).join('');
    target.innerHTML=`<div class="fold-grid">${cards}</div>`;
  }
};

// ─────────────────────────────────────────────── BACKTEST
Pages.backtest = {
  load() {
    if(!State.symbols.length) return;
    UI.buildSymbolGrid('bt-symbols', State.symbols, false);
  },
  run() {
    const syms = UI.getSelected('bt-symbols');
    if(!syms.length) { alert('Select at least one symbol'); return; }
    const tf      = $('bt-tf').value;
    const balance = +$('bt-balance').value;
    const useModel= $('bt-use-model-toggle').classList.contains('on');
    $('bt-log').innerHTML='';
    $('bt-results').innerHTML='';
    UI.setLoading('bt-btn',true,'Running…');
    streamSSE('/api/backtest',{symbols:syms,timeframe:tf,starting_balance:balance,use_model:useModel},
      (msg)=>logAppend('bt-log',msg),
      (data)=>Pages.backtest._render(data),
      ()=>{ UI.setLoading('bt-btn',false); $('bt-btn').textContent='▶ Run Backtest'; },
      (err)=>{ logAppend('bt-log','Error: '+err,'error'); UI.setLoading('bt-btn',false); $('bt-btn').textContent='▶ Run Backtest'; }
    );
  },
  _render(data) {
    const container = $('bt-results');
    container.innerHTML='';
    Object.entries(data||{}).forEach(([sym,res])=>{
      if(res.error) {
        container.innerHTML+=`<div class="alert alert-danger mb-16"><span>❌</span><div><strong>${sym}:</strong> ${res.error}</div></div>`;
        return;
      }
      const m = res.metrics||{};
      const wr   = m.win_rate!=null?fmt.pct(m.win_rate):'—';
      const pf   = m.profit_factor!=null?m.profit_factor.toFixed(2):'—';
      const sh   = m.sharpe!=null?m.sharpe.toFixed(2):'—';
      const dd   = m.max_drawdown!=null?fmt.pct(m.max_drawdown):'—';
      const pnl  = m.net_pnl!=null?fmt.pnl(m.net_pnl):'—';
      const pnlColor = (m.net_pnl||0)>=0?'var(--accent3)':'var(--danger)';
      const canvasId = `chart-${sym}`;
      const div = el('div','card mb-16');
      div.innerHTML=`
        <div class="card-header">
          <span style="font-size:16px">📈</span>
          <div><div class="card-title">${sym} Backtest</div><div class="card-sub">${m.trades||0} trades · ${res.model_used?'ML model ✓':'Rule-based'}</div></div>
        </div>
        <div class="grid-4 mb-16">
          <div class="stat-card green"><div class="stat-label">Win Rate</div><div class="stat-value" style="font-size:20px">${wr}</div></div>
          <div class="stat-card blue"><div class="stat-label">Profit Factor</div><div class="stat-value" style="font-size:20px">${pf}</div></div>
          <div class="stat-card purple"><div class="stat-label">Sharpe</div><div class="stat-value" style="font-size:20px">${sh}</div></div>
          <div class="stat-card gold"><div class="stat-label">Max DD</div><div class="stat-value" style="font-size:20px">${dd}</div></div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
          <div style="font-size:13px;color:var(--text-2)">Equity Curve</div>
          <div style="font-size:16px;font-weight:700;color:${pnlColor}">${pnl}</div>
        </div>
        <div class="chart-wrap"><canvas id="${canvasId}" class="equity-canvas"></canvas></div>
        <div style="margin-top:12px;font-size:12px;color:var(--text-2)">
          Detections — FVG: ${(res.detection_counts||{}).fvg||0} · OB: ${(res.detection_counts||{}).order_blocks||0} · Sweeps: ${(res.detection_counts||{}).sweeps||0} · BOS: ${(res.detection_counts||{}).bos||0} · CHoCH: ${(res.detection_counts||{}).choch||0}
        </div>`;
      container.appendChild(div);
      // draw chart
      if(res.equity_curve&&res.equity_curve.timestamps&&res.equity_curve.timestamps.length) {
        setTimeout(()=>drawEquityChart(canvasId, res.equity_curve.timestamps, res.equity_curve.values), 50);
      }
    });
  }
};

// ─────────────────────────────────────────────── BENCHMARK
Pages.benchmark = {
  load() {
    if(!State.symbols.length) return;
    UI.buildSymbolGrid('bench-symbols', State.symbols, true);
  },
  run() {
    const syms = UI.getSelected('bench-symbols');
    $('bench-log').innerHTML='';
    $('bench-results').innerHTML='';
    UI.setLoading('bench-btn',true,'Benchmarking…');
    streamSSE('/api/benchmark',{symbols:syms},
      (msg)=>logAppend('bench-log',msg),
      (data)=>Pages.benchmark._render(data.leaderboard||[]),
      ()=>{ UI.setLoading('bench-btn',false); $('bench-btn').textContent='▶ Run Benchmark'; },
      (err)=>{ logAppend('bench-log','Error: '+err,'error'); UI.setLoading('bench-btn',false); $('bench-btn').textContent='▶ Run Benchmark'; }
    );
  },
  _render(rows) {
    const container = $('bench-results');
    if(!rows.length) { container.innerHTML='<div class="alert alert-warn"><span>⚠️</span>No results — download data and train models first</div>'; return; }
    const medals = ['🥇','🥈','🥉'];
    const tableRows = rows.map((r,i)=>{
      if(r.error) return `<tr><td><strong>${r.symbol}</strong></td><td colspan="7" style="color:var(--danger)">${r.error}</td></tr>`;
      const wr = r.win_rate!=null?fmt.pct(r.win_rate):'—';
      const pf = r.profit_factor!=null?r.profit_factor.toFixed(2):'—';
      const sh = r.sharpe!=null?r.sharpe.toFixed(2):'—';
      const dd = r.max_dd!=null?fmt.pct(r.max_dd):'—';
      const pnl = r.net_pnl!=null?fmt.pnl(r.net_pnl):'—';
      const pnlColor = (r.net_pnl||0)>=0?'var(--accent3)':'var(--danger)';
      const auc = r.champion_auc!=null?r.champion_auc.toFixed(3):'—';
      const medal = medals[i]||`#${i+1}`;
      return `<tr>
        <td><span style="margin-right:6px">${medal}</span><strong>${r.symbol}</strong>${r.drift_alert?'<span class="tag tag-gold" style="margin-left:6px;font-size:10px">Drift</span>':''}</td>
        <td>${wr}</td><td>${pf}</td>
        <td style="font-family:var(--font-mono)">${sh}</td>
        <td>${dd}</td>
        <td style="color:${pnlColor};font-family:var(--font-mono)">${pnl}</td>
        <td>${fmt.int(r.n_trades)}</td>
        <td style="font-family:var(--font-mono)">${auc}</td>
      </tr>`;
    }).join('');
    container.innerHTML=`
      <div class="card">
        <div class="card-header"><span>🏆</span><div><div class="card-title">Results — Sorted by Sharpe</div><div class="card-sub">Champion models · ${rows.length} symbols</div></div></div>
        <div class="table-wrap"><table>
          <thead><tr><th>Symbol</th><th>Win Rate</th><th>Profit Factor</th><th>Sharpe</th><th>Max DD</th><th>Net PnL</th><th>Trades</th><th>AUC</th></tr></thead>
          <tbody>${tableRows}</tbody>
        </table></div>
      </div>`;
  }
};

// ─────────────────────────────────────────────── LIVE
Pages.live = {
  load() {
    if(!State.symbols.length) return;
    UI.buildSymbolGrid('live-symbols', State.symbols, true);
    Pages.live.pollStatus();
  },
  async start() {
    const syms = UI.getSelected('live-symbols');
    if(!syms.length) { alert('Select at least one symbol'); return; }
    const tf = $('live-tf').value;
    const intuition = $('intuition-toggle').classList.contains('on');
    const r = await fetch('/api/live/start',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({symbols:syms,timeframe:tf,intuition_enabled:intuition}),
    }).then(r=>r.json()).catch(()=>({}));
    if(r.status==='started'||r.status==='already_running') Pages.live.pollStatus();
    else alert(r.error||'Failed to start');
  },
  async stop() {
    await fetch('/api/live/stop',{method:'POST'});
  },
  async pollStatus() {
    const d = await fetch('/api/live/status').then(r=>r.json()).catch(()=>({running:false,log:[]}));
    State.liveRunning = d.running;
    const badge = $('live-badge');
    const badgeText = $('live-badge-text');
    if(d.running) {
      badge.classList.add('running'); badgeText.textContent='Live';
      $('live-start-btn').disabled=true; $('live-stop-btn').disabled=false;
      $('live-status-sub').textContent='Running — watching for signals…';
    } else {
      badge.classList.remove('running'); badgeText.textContent='Idle';
      $('live-start-btn').disabled=false; $('live-stop-btn').disabled=true;
      $('live-status-sub').textContent='Idle';
    }
    // Update log
    const log = d.log||[];
    if(log.length) {
      const console = $('live-log-console');
      const prevLen = +console.dataset.len||0;
      if(log.length>prevLen) {
        if(prevLen===0) console.innerHTML='';
        for(let i=prevLen;i<log.length;i++) {
          const msg=log[i]||'';
          const level=msg.includes('ERROR')||msg.includes('FATAL')?'error':msg.includes('INTUITION')?'ok':'info';
          logAppend('live-log-console',msg,level);
        }
        console.dataset.len=log.length;
      }
    }
    if(State.liveRunning) setTimeout(()=>Pages.live.pollStatus(), 3000);
  },
  clearLog() { $('live-log-console').innerHTML=''; $('live-log-console').dataset.len=0; },
  toggleIntuition() {
    const tog = $('intuition-toggle');
    tog.classList.toggle('on');
    const enabled = tog.classList.contains('on');
    fetch('/api/live/intuition',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled})});
  }
};

// ─────────────────────────────────────────────── REGISTRY
Pages.registry = {
  async load() {
    // Show loading state immediately, then replace
    const tbody = $('registry-body');
    tbody.innerHTML='<tr><td colspan="8" style="text-align:center;color:var(--text-3);padding:24px"><span class="spin">⏳</span> Loading…</td></tr>';
    const d = await fetch('/api/registry').then(r=>r.json()).catch(()=>({champions:[]}));
    const rows = d.champions||[];
    if(!rows.length) {
      tbody.innerHTML='<tr><td colspan="8" style="text-align:center;padding:40px"><div class="empty-state" style="padding:0"><div class="empty-icon">🗃️</div><h3>No champion models yet</h3><p style="margin-top:8px;color:var(--text-3)">Go to <strong>Train → Single Symbol</strong> or <strong>Train All Markets</strong> to train your first model</p></div></td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(r=>{
      const auc = r.avg_auc!=null?r.avg_auc.toFixed(3):'—';
      const gt  = r.gt_score!=null?r.gt_score.toFixed(3):'—';
      const aucColor = (r.avg_auc||0)>=0.60?'var(--accent3)':(r.avg_auc||0)>=0.55?'var(--gold)':'var(--danger)';
      const driftTag = r.drift_alert?'<span class="tag tag-gold">⚠ Drift</span>':'<span class="tag tag-green">OK</span>';
      const challBtn = r.has_challenger?`<button class="btn btn-warn btn-sm" onclick="Pages.registry.promote('${r.symbol}')">Promote Challenger</button>`:'';
      return `<tr>
        <td><strong>${r.symbol}</strong></td>
        <td style="color:${aucColor};font-family:var(--font-mono)">${auc}</td>
        <td style="font-family:var(--font-mono)">${gt}</td>
        <td>${r.n_folds||'—'}</td>
        <td style="color:var(--text-2);font-size:11px">${fmt.ago(r.promoted_at)}</td>
        <td style="font-size:11px;color:var(--text-2)">${r.data_start?r.data_start.slice(0,10):'—'} → ${r.data_end?r.data_end.slice(0,10):'—'}</td>
        <td>${driftTag}</td>
        <td style="display:flex;gap:4px;flex-wrap:wrap">${challBtn}<button class="btn btn-outline btn-sm" onclick="navigateTo('train')">Retrain</button></td>
      </tr>`;
    }).join('');
  },
  async promote(symbol) {
    if(!confirm(`Promote challenger for ${symbol} to champion?`)) return;
    const r = await fetch(`/api/registry/${symbol}/promote`,{method:'POST'}).then(r=>r.json()).catch(()=>({}));
    alert(r.promoted?'Promoted!':'No challenger to promote');
    Pages.registry.load();
  }
};

// ─────────────────────────────────────────────── SETUP
Pages.setup = {
  async load() {
    await pollMT5();
    const r = await fetch('/api/mt5/status').then(r=>r.json()).catch(()=>({connected:false}));
    Pages.setup._renderAccount(r);
  },
  _renderAccount(d) {
    const banner = $('mt5-status-banner');
    const info   = $('account-info');
    if(d.connected&&d.account) {
      banner.className='alert alert-ok mb-16';
      banner.innerHTML=`<span>✅</span><span>Connected to MetaTrader 5 — ${d.account.name} @ ${d.account.server}</span>`;
      info.innerHTML=`
        <div class="grid-2" style="gap:12px">
          <div class="stat-card green"><div class="stat-label">Balance</div><div class="stat-value" style="font-size:20px">${d.account.currency} ${(+d.account.balance).toLocaleString('en',{minimumFractionDigits:2})}</div></div>
          <div class="stat-card blue"><div class="stat-label">Equity</div><div class="stat-value" style="font-size:20px">${d.account.currency} ${(+d.account.equity).toLocaleString('en',{minimumFractionDigits:2})}</div></div>
          <div class="stat-card purple"><div class="stat-label">Login</div><div class="stat-value" style="font-size:16px">#${d.account.login}</div><div class="stat-meta">${d.account.name}</div></div>
          <div class="stat-card gold"><div class="stat-label">Leverage</div><div class="stat-value" style="font-size:16px">1:${d.account.leverage}</div><div class="stat-meta">${d.account.company}</div></div>
        </div>`;
    } else {
      banner.className='alert alert-warn mb-16';
      banner.innerHTML=`<span>⚠️</span><span>Not connected — open MetaTrader 5 on your desktop first</span>`;
      info.innerHTML='<div class="empty-state"><div class="empty-icon">🔌</div><h3>Connect MT5 to see account details</h3></div>';
    }
  },
  async connect() {
    const account  = $('mt5-account').value;
    const password = $('mt5-password').value;
    const server   = $('mt5-server').value;
    const r = await fetch('/api/mt5/connect',{
      method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({account:account||null,password:password||null,server:server||null}),
    }).then(r=>r.json()).catch(()=>({connected:false,error:'Network error'}));
    Pages.setup._renderAccount(r);
    pollMT5();
    if(!r.connected) alert(r.error||'Connection failed');
  }
};

// ═══════════════════════════════════════════════════ INIT
const App = {
  async init() {
    initNav();
    // Load symbols
    try {
      const d = await fetch('/api/symbols').then(r=>r.json());
      State.symbols = d.symbols||[];
    } catch(e) { State.symbols=[]; }

    // Populate all grids
    UI.buildSymbolGrid('dl-symbols', State.symbols, false);
    UI.buildSymbolGrid('bt-symbols', State.symbols, false);
    UI.buildSymbolGrid('bench-symbols', State.symbols, true);
    UI.buildSymbolGrid('live-symbols', State.symbols, true);
    UI.buildSymbolGrid('train-all-symbols', State.symbols, true);
    UI.buildSymbolSelect('train-symbol', State.symbols);

    // Initial page loads
    await pollMT5();
    Pages.overview.load();
    dataPage.loadStatus();

    // Polls
    setInterval(pollMT5, 15000);
    setInterval(()=>{ if(State.liveRunning) Pages.live.pollStatus(); }, 5000);
    setInterval(()=>{ const pg=document.querySelector('.nav-item.active'); if(pg&&pg.dataset.page==='overview') Pages.overview.load(); }, 30000);
  },
  refreshAll() {
    const activePage = document.querySelector('.nav-item.active');
    if(activePage&&Pages[activePage.dataset.page]&&Pages[activePage.dataset.page].load) {
      Pages[activePage.dataset.page].load();
    }
    pollMT5();
  }
};

window.addEventListener('DOMContentLoaded', ()=>App.init());
