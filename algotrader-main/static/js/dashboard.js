const API = {
    async get(url) {
        const res = await fetch(url);
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },
    async post(url, data) {
        const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },
    async delete_(url) {
        const res = await fetch(url, { method: 'DELETE' });
        if (!res.ok) throw new Error(await res.text());
        return res.json();
    },
};

let ws = null;
let statusInterval = null;
let portfolioInterval = null;
let portfolioLoaded = false;

async function loginKite() {
    try {
        const data = await API.get('/api/login-url');
        if (data.url) {
            try {
                const w = window.top || window.parent || window;
                w.open(data.url, '_blank');
            } catch(e) {
                window.open(data.url, '_blank');
            }
            document.getElementById('login-url-display').style.display = 'block';
            document.getElementById('login-url-link').href = data.url;
            document.getElementById('login-url-link').textContent = 'Click here to open Zerodha login';
        } else {
            showToast('Could not get login URL. Check API key configuration.', 'error');
        }
    } catch (e) {
        showToast('Login error: ' + e.message, 'error');
    }
}

async function manualAuthenticate() {
    const token = document.getElementById('manual-request-token').value.trim();
    if (!token) {
        showToast('Please paste the request_token from the redirect URL', 'error');
        return;
    }
    try {
        const res = await API.post('/api/authenticate', { request_token: token });
        if (res.error) {
            showToast('Authentication failed: ' + res.error, 'error');
            return;
        }
        showToast('Logged in to Zerodha successfully!', 'success');
        document.getElementById('login-url-display').style.display = 'none';
        loadStatus();
    } catch (e) {
        showToast('Authentication failed: ' + e.message, 'error');
    }
}

(function checkAuthResult() {
    const params = new URLSearchParams(window.location.search);
    if (params.get('auth') === 'success') {
        window.history.replaceState({}, '', '/');
        setTimeout(() => showToast('Logged in to Zerodha successfully!', 'success'), 500);
    }
    const authError = params.get('auth_error');
    if (authError) {
        window.history.replaceState({}, '', '/');
        setTimeout(() => showToast('Login failed: ' + authError, 'error'), 500);
    }
})();

function showToast(msg, type = 'info') {
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = msg;
    document.body.appendChild(el);
    requestAnimationFrame(() => el.classList.add('show'));
    setTimeout(() => {
        el.classList.remove('show');
        setTimeout(() => el.remove(), 300);
    }, 3000);
}

function switchTab(tabName) {
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(`tab-${tabName}`).classList.add('active');
    if (tabName === 'analysis' && !analysisData) {
        loadAnalysis();
    }
    // Auto-resolve token when switching to paper-trading tab
    if (tabName === 'paper-trading') {
        const tokenEl = document.getElementById('paper-token');
        if (tokenEl && !tokenEl.value) resolveToken('paper');
    }
}

function formatNum(n, decimals = 2) {
    if (n === null || n === undefined) return '0';
    return Number(n).toLocaleString('en-IN', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

function pnlClass(val) {
    if (val > 0) return 'positive';
    if (val < 0) return 'negative';
    return 'neutral';
}

async function loadFullStatus() {
    try {
        const data = await API.get('/api/status?full=true');
        updateOverviewCards(data);
        updateStrategiesList(data.strategies);
        updateRiskPanel(data.risk);
        updateJournalPanel(data.journal);
        updateExecutionPanel(data.execution);
        updateTicksTable(data.ticks);
        if (data.authenticated) {
            updatePortfolioSection(data);
            portfolioLoaded = true;
        }
        updateStatusBadge(data);
    } catch (e) {
        console.error('Full status load error:', e);
    }
}

async function loadStatus() {
    try {
        const data = await API.get('/api/status');
        updateOverviewCards(data);
        updateStrategiesList(data.strategies);
        updateRiskPanel(data.risk);
        updateJournalPanel(data.journal);
        updateExecutionPanel(data.execution);
        updateTicksTable(data.ticks);

        // Render signals from status response
        if (data.signals && data.signals.length > 0) {
            renderSignalsList(data.signals);
        }

        updateStatusBadge(data);

        // Auto-refresh market overview when authenticated
        if (data.authenticated) {
            loadMarketOverview();
        }
    } catch (e) {
        console.error('Status load error:', e);
    }
}

function renderSignalsList(signals) {
    const el = document.getElementById('signals-log');
    if (!signals || signals.length === 0) return;
    el.innerHTML = signals.slice().reverse().map(s => {
        const cls = s.transaction_type === 'BUY' ? 'buy' : 'sell';
        return `
            <div class="signal-entry ${cls}">
                <div class="signal-header">
                    <strong>${s.tradingsymbol} ${s.transaction_type}</strong>
                    <span class="signal-time">${new Date(s.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="signal-detail">
                    Strategy: ${s.strategy} | Qty: ${s.quantity} |
                    Price: ${formatNum(s.price)} | Confidence: ${formatNum(s.confidence)}%
                </div>
            </div>
        `;
    }).join('');
}

function updateStatusBadge(data) {
    const statusBadge = document.getElementById('status-badge');
    const btnLogin = document.getElementById('btn-login');
    const btnStart = document.getElementById('btn-start');
    const btnStop = document.getElementById('btn-stop');
    if (data.running) {
        statusBadge.textContent = 'LIVE';
        statusBadge.className = 'status-badge status-live';
        btnLogin.style.display = 'none';
        btnStart.style.display = 'none';
        btnStop.style.display = '';
    } else if (data.authenticated) {
        statusBadge.textContent = 'CONNECTED';
        statusBadge.className = 'status-badge status-connected';
        btnLogin.style.display = 'none';
        btnStart.style.display = '';
        btnStop.style.display = 'none';
    } else {
        statusBadge.textContent = 'OFFLINE';
        statusBadge.className = 'status-badge status-offline';
        btnLogin.style.display = '';
        btnStart.style.display = 'none';
        btnStop.style.display = 'none';
    }
}

function updatePortfolioSection(data) {
    const section = document.getElementById('portfolio-section');
    if (!section) return;
    section.style.display = 'block';

    if (data.profile) {
        document.getElementById('account-name').textContent = data.profile.user_name || data.profile.user_shortname || '--';
        document.getElementById('account-id').textContent = data.zerodha_id || '--';
    }

    if (data.margins) {
        const eq = data.margins.equity || {};
        const avail = (eq.available?.live_balance || eq.available?.cash || eq.net || 0);
        const used = (eq.utilised?.debits || 0);
        document.getElementById('available-margin').textContent = formatNum(avail, 0);
        document.getElementById('used-margin').textContent = 'Used: ' + formatNum(used, 0);
    }

    if (data.holdings) {
        let totalVal = 0;
        const holdings = data.holdings || [];
        holdings.forEach(h => { totalVal += (h.last_price || 0) * (h.quantity || 0); });
        document.getElementById('holdings-value').textContent = formatNum(totalVal, 0);
        document.getElementById('holdings-count').textContent = holdings.length + ' stocks';

        const hBody = document.getElementById('overview-holdings-body');
        if (hBody) {
            if (holdings.length) {
                hBody.innerHTML = holdings.slice(0, 20).map(h => {
                    const pnl = ((h.last_price || 0) - (h.average_price || 0)) * (h.quantity || 0);
                    const sym = h.tradingsymbol || '';
                    return `<tr>
                        <td><strong>${sym}</strong></td>
                        <td>${h.quantity}</td>
                        <td>${formatNum(h.average_price)}</td>
                        <td>${formatNum(h.last_price)}</td>
                        <td class="${pnlClass(pnl)}">${formatNum(pnl)}</td>
                        <td><button class="btn btn-outline" style="font-size:10px;padding:2px 6px" onclick="openDeepProfile('${sym}')">Review</button></td>
                    </tr>`;
                }).join('');
            } else {
                hBody.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted);text-align:center">No holdings</td></tr>';
            }
        }
    }

    if (data.positions) {
        const positions = data.positions.net || [];
        let totalPnl = 0;
        positions.forEach(p => { totalPnl += (p.pnl || 0); });
        document.getElementById('positions-pnl').textContent = formatNum(totalPnl);
        document.getElementById('positions-pnl').className = `metric-value ${pnlClass(totalPnl)}`;
        document.getElementById('positions-count').textContent = positions.length + ' positions';

        const pBody = document.getElementById('overview-positions-body');
        if (pBody) {
            if (positions.length) {
                pBody.innerHTML = positions.map(p => `<tr>
                    <td><strong>${p.tradingsymbol}</strong></td>
                    <td>${p.quantity}</td>
                    <td>${formatNum(p.average_price)}</td>
                    <td>${formatNum(p.last_price)}</td>
                    <td class="${pnlClass(p.pnl)}">${formatNum(p.pnl)}</td>
                </tr>`).join('');
            } else {
                pBody.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted);text-align:center">No positions</td></tr>';
            }
        }
    }
}

function updateOverviewCards(data) {
    const j = data.journal || {};
    const r = data.risk || {};
    document.getElementById('daily-pnl').textContent = formatNum(j.daily_pnl || 0);
    document.getElementById('daily-pnl').className = `metric-value ${pnlClass(j.daily_pnl || 0)}`;
    document.getElementById('total-pnl').textContent = formatNum(j.total_pnl || 0);
    document.getElementById('total-pnl').className = `metric-value ${pnlClass(j.total_pnl || 0)}`;
    document.getElementById('total-trades').textContent = j.total_trades || 0;
    document.getElementById('exposure').textContent = formatNum(r.total_exposure || 0, 0);
    const utilPct = r.utilization_pct || 0;
    document.getElementById('exposure-bar').style.width = `${Math.min(utilPct, 100)}%`;
    document.getElementById('exposure-bar').style.background =
        utilPct > 80 ? 'var(--accent-red)' : utilPct > 50 ? 'var(--accent-orange)' : 'var(--accent-green)';
}

function updateStrategiesList(strategies) {
    const el = document.getElementById('strategies-list');
    if (!strategies || strategies.length === 0) {
        el.innerHTML = '<div class="empty-state"><p>No strategies added</p></div>';
        return;
    }
    el.innerHTML = strategies.map(s => `
        <div class="strategy-card">
            <div class="strategy-info">
                <h4>${s.name}</h4>
                <span class="type-tag type-${s.type}">${s.type}</span>
                <span style="margin-left:8px;color:${s.is_active ? 'var(--accent-green)' : 'var(--accent-red)'}">
                    ${s.is_active ? 'Active' : 'Paused'}
                </span>
            </div>
            <div class="strategy-actions">
                <button class="btn btn-sm btn-outline" onclick="toggleStrategy('${s.name}')">
                    ${s.is_active ? 'Pause' : 'Resume'}
                </button>
                <button class="btn btn-sm btn-danger" onclick="removeStrategy('${s.name}')">Remove</button>
            </div>
        </div>
    `).join('');
}

function updateRiskPanel(risk) {
    if (!risk) return;
    document.getElementById('kill-switch-status').textContent = risk.kill_switch_active ? 'ACTIVE' : 'OFF';
    document.getElementById('kill-switch-status').className = risk.kill_switch_active ? 'negative' : 'positive';
    document.getElementById('risk-daily-pnl').textContent = formatNum(risk.daily_pnl);
    document.getElementById('risk-exposure').textContent = formatNum(risk.total_exposure, 0);
    document.getElementById('risk-max-loss').textContent = formatNum(risk.max_daily_loss, 0);
    document.getElementById('risk-max-exposure').textContent = formatNum(risk.max_exposure, 0);
    document.getElementById('risk-positions').textContent = risk.position_count;
}

function updateJournalPanel(journal) {
    if (!journal) return;
    const byStrat = journal.pnl_by_strategy || {};
    const el = document.getElementById('journal-by-strategy');
    el.innerHTML = Object.entries(byStrat).map(([k, v]) => `
        <tr><td>${k}</td><td class="${pnlClass(v)}">${formatNum(v)}</td></tr>
    `).join('') || '<tr><td colspan="2" style="color:var(--text-muted)">No data</td></tr>';
}

function updateExecutionPanel(exec) {
    if (!exec) return;
    document.getElementById('exec-pending').textContent = exec.pending_orders || 0;
    document.getElementById('exec-filled').textContent = exec.filled_orders || 0;
}

function getInstrumentName(token, serverName) {
    // 1. Try user-selected instruments map
    if (selectedInstruments.has(token)) return selectedInstruments.get(token);
    // 2. Try server-provided name
    if (serverName) return serverName;
    // 3. Try well-known presets from the HTML checkboxes
    const el = document.querySelector(`#preset-instruments input[value="${token}"]`);
    if (el) return el.getAttribute('data-name') || token;
    return token;
}

function updateTicksTable(ticks) {
    const el = document.getElementById('ticks-body');
    if (!ticks || Object.keys(ticks).length === 0) {
        // Fallback: show market overview quotes instead of empty table
        loadTicksFallback(el);
        return;
    }
    // Auto-populate selectedInstruments from server-provided names
    Object.values(ticks).forEach(t => {
        if (t.name && !selectedInstruments.has(t.token)) {
            selectedInstruments.set(t.token, t.name);
        }
    });
    el.innerHTML = Object.values(ticks).map(t => `
        <tr>
            <td><span style="font-weight:600;color:#00d4aa">${getInstrumentName(t.token, t.name)}</span><br><span style="font-size:10px;color:#5c6e7e">${t.token}</span></td>
            <td style="font-weight:600">${formatNum(t.ltp)}</td>
            <td class="${pnlClass(t.change)}">${formatNum(t.change)}</td>
            <td>${formatNum(t.high)} / ${formatNum(t.low)}</td>
            <td>${(t.volume || 0).toLocaleString()}</td>
            <td>${(t.buy_qty || 0).toLocaleString()} / ${(t.sell_qty || 0).toLocaleString()}</td>
        </tr>
    `).join('');
}

let _ticksFallbackLoaded = false;
async function loadTicksFallback(el) {
    if (_ticksFallbackLoaded) return; // avoid repeated fetches
    try {
        const data = await API.get('/api/market/overview');
        const symbols = data.market_overview || [];
        if (!symbols.length) {
            el.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center">No market data available</td></tr>';
            return;
        }
        _ticksFallbackLoaded = true;
        el.innerHTML = symbols.map(s => {
            const chgColor = (s.change || 0) >= 0 ? '#00d4aa' : '#ff5252';
            return `<tr>
                <td><span style="font-weight:600;color:#00d4aa">${s.symbol}</span><br><span style="font-size:10px;color:#5c6e7e">${s.exchange}</span></td>
                <td style="font-weight:600">${formatNum(s.last_price)}</td>
                <td style="color:${chgColor}">${formatNum(s.change)} (${formatNum(s.change_pct)}%)</td>
                <td>${formatNum(s.high)} / ${formatNum(s.low)}</td>
                <td>${(s.volume || 0).toLocaleString()}</td>
                <td>${(s.buy_quantity || 0).toLocaleString()} / ${(s.sell_quantity || 0).toLocaleString()}</td>
            </tr>`;
        }).join('');
    } catch (e) {
        el.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center">No live data - click Start Live to stream</td></tr>';
    }
}
// Reset fallback flag when live ticks start flowing
function resetTicksFallback() { _ticksFallbackLoaded = false; }

function updateSignalsLog(signal) {
    const el = document.getElementById('signals-log');
    const cls = signal.transaction_type === 'BUY' ? 'buy' : 'sell';
    const html = `
        <div class="signal-entry ${cls}">
            <div class="signal-header">
                <strong>${signal.tradingsymbol} ${signal.transaction_type}</strong>
                <span class="signal-time">${new Date(signal.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="signal-detail">
                Strategy: ${signal.strategy} | Qty: ${signal.quantity} |
                Price: ${formatNum(signal.price)} | Confidence: ${formatNum(signal.confidence)}%
            </div>
        </div>
    `;
    el.insertAdjacentHTML('afterbegin', html);
    while (el.children.length > 50) el.removeChild(el.lastChild);
}

let wsReconnectDelay = 1000;
let wsPingInterval = null;
let wsLastPong = Date.now();

function connectWS() {
    if (ws && (ws.readyState === WebSocket.CONNECTING || ws.readyState === WebSocket.OPEN)) return;

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${location.host}/ws/live`);

    ws.onopen = () => {
        wsReconnectDelay = 1000;
        wsLastPong = Date.now();
        if (wsPingInterval) clearInterval(wsPingInterval);
        wsPingInterval = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send('ping');
                if (Date.now() - wsLastPong > 45000) {
                    console.warn('WS: No pong in 45s, reconnecting');
                    ws.close();
                }
            }
        }, 15000);
    };

    ws.onmessage = (e) => {
        wsLastPong = Date.now();
        try {
            const msg = JSON.parse(e.data);
            if (msg.type === 'ticks') updateTicksTable(msg.data);
            else if (msg.type === 'signal') updateSignalsLog(msg.data);
            else if (msg.type === 'pong') { /* heartbeat reply */ }
            else if (msg.type === 'positions') {
                const pnlEl = document.getElementById('daily-pnl');
                if (pnlEl && msg.pnl !== undefined) {
                    pnlEl.textContent = formatNum(msg.pnl);
                    pnlEl.className = `metric-value ${pnlClass(msg.pnl)}`;
                }
            }
        } catch (err) { console.error('WS parse error:', err); }
    };

    ws.onclose = () => {
        if (wsPingInterval) { clearInterval(wsPingInterval); wsPingInterval = null; }
        console.log(`WS closed, reconnecting in ${wsReconnectDelay/1000}s`);
        setTimeout(connectWS, wsReconnectDelay);
        wsReconnectDelay = Math.min(wsReconnectDelay * 1.5, 30000);
    };

    ws.onerror = (err) => {
        console.error('WS error:', err);
        ws.close();
    };
}

async function toggleStrategy(name) {
    try {
        await API.post('/api/strategies/toggle', { name });
        showToast(`Strategy ${name} toggled`, 'success');
        loadStatus();
    } catch (e) { showToast(e.message, 'error'); }
}

async function removeStrategy(name) {
    try {
        await API.post('/api/strategies/remove', { name });
        showToast(`Strategy ${name} removed`, 'success');
        loadStatus();
    } catch (e) { showToast(e.message, 'error'); }
}

function showModal(id) {
    document.getElementById(id).classList.add('show');
    // Load tested strategies when opening the add-strategy modal
    if (id === 'modal-add-strategy') loadTestedStrategies();
}

function hideModal(id) {
    document.getElementById(id).classList.remove('show');
}

async function addStrategy() {
    const name = document.getElementById('strat-select').value;
    const timeframe = document.getElementById('strat-timeframe').value;
    const paramsStr = document.getElementById('strat-params').value.trim();
    let params = null;
    if (paramsStr) {
        try { params = JSON.parse(paramsStr); }
        catch { showToast('Invalid JSON params', 'error'); return; }
    }
    try {
        const res = await API.post('/api/strategies/add', { name, params, timeframe });
        if (res.error) {
            showToast(res.error, 'error');
            if (res.hint) showToast(res.hint, 'info');
            return;
        }
        showToast(`Added ${name} (${timeframe})`, 'success');
        hideModal('modal-add-strategy');
        loadStatus();
    } catch (e) { showToast(e.message, 'error'); }
}

// Load tested strategies when modal opens to show eligibility notice
async function loadTestedStrategies() {
    try {
        const res = await API.get('/api/strategies/tested');
        const notice = document.getElementById('strat-tested-notice');
        const listEl = document.getElementById('strat-tested-list');
        if (res.tested && res.tested.length > 0) {
            notice.style.display = 'block';
            listEl.textContent = 'Tested: ' + res.tested.join(', ');
            // Highlight tested options in the dropdown
            const select = document.getElementById('strat-select');
            for (const opt of select.options) {
                if (res.tested.includes(opt.value)) {
                    opt.textContent = opt.textContent.replace(/ ✓$/, '') + ' ✓';
                    opt.style.color = '#00d4aa';
                } else {
                    opt.textContent = opt.textContent.replace(/ ✓$/, '');
                    opt.style.color = '';
                }
            }
        } else {
            notice.style.display = 'block';
            listEl.textContent = 'No strategies tested yet. Run a backtest or paper trade first.';
        }
    } catch (e) { /* silently ignore */ }
}

let selectedInstruments = new Map();

function updateSelectedDisplay() {
    const container = document.getElementById('selected-instruments');
    const countEl = document.getElementById('selected-count');
    countEl.textContent = selectedInstruments.size;
    if (selectedInstruments.size === 0) {
        container.innerHTML = '<span style="color:#667788;font-size:12px">Select instruments above</span>';
        return;
    }
    container.innerHTML = Array.from(selectedInstruments.entries()).map(([token, name]) =>
        `<span class="inst-tag">${name} <span class="remove-tag" onclick="removeInstrument(${token})">&times;</span></span>`
    ).join('');
}

function removeInstrument(token) {
    selectedInstruments.delete(token);
    const cb = document.querySelector(`#preset-instruments input[value="${token}"]`);
    if (cb) cb.checked = false;
    saveSelectedInstruments();
    updateSelectedDisplay();
}

document.addEventListener('change', function(e) {
    if (e.target.matches('#preset-instruments input[type="checkbox"]')) {
        const token = parseInt(e.target.value);
        const name = e.target.getAttribute('data-name');
        if (e.target.checked) {
            selectedInstruments.set(token, name);
        } else {
            selectedInstruments.delete(token);
        }
        saveSelectedInstruments();
        updateSelectedDisplay();
    }
});

function toggleSearchInstrument(token, symbol, name) {
    if (selectedInstruments.has(token)) {
        selectedInstruments.delete(token);
    } else {
        selectedInstruments.set(token, symbol + (name && name !== symbol ? ` (${name})` : ''));
    }
    saveSelectedInstruments();
    updateSelectedDisplay();
}

function saveSelectedInstruments() {
    const data = Array.from(selectedInstruments.entries());
    try { localStorage.setItem('kite_selected_instruments', JSON.stringify(data)); } catch(e) {}
}

function loadSelectedInstruments() {
    try {
        const data = localStorage.getItem('kite_selected_instruments');
        if (data) {
            const entries = JSON.parse(data);
            entries.forEach(([k, v]) => selectedInstruments.set(k, v));
        }
    } catch(e) {}
}

loadSelectedInstruments();

async function searchInstruments() {
    const q = document.getElementById('search-inst-query').value.trim();
    const exchange = document.getElementById('search-exchange').value;
    const resultsDiv = document.getElementById('search-results');
    if (!q) { showToast('Enter a search term', 'error'); return; }
    resultsDiv.style.display = 'block';
    resultsDiv.innerHTML = '<div style="padding:10px;color:#8899aa;font-size:12px">Searching...</div>';
    try {
        const results = await API.get(`/api/instruments/search?q=${encodeURIComponent(q)}&exchange=${exchange}`);
        if (!results.length) {
            resultsDiv.innerHTML = '<div style="padding:10px;color:#8899aa;font-size:12px">No results found</div>';
            return;
        }
        resultsDiv.innerHTML = results.map(r => {
            const safeSymbol = r.symbol.replace(/'/g, "\\'");
            const safeName = (r.name || '').replace(/'/g, "\\'");
            const checked = selectedInstruments.has(r.token) ? 'checked' : '';
            const extra = r.expiry ? ` <span style="color:#ff9100;font-size:10px">${r.expiry}</span>` : '';
            return `
            <div class="search-result-item" onclick="toggleSearchInstrument(${r.token},'${safeSymbol}','${safeName}');this.querySelector('input').checked=!this.querySelector('input').checked;">
                <input type="checkbox" ${checked} style="accent-color:var(--accent-green);pointer-events:none">
                <span class="sr-token">${r.token}</span>
                <strong>${r.symbol}</strong>${extra}
                <span class="sr-name">${r.name || ''}</span>
                <span style="margin-left:auto;color:#5c6e7e;font-size:10px">${r.exchange}${r.type ? ' · ' + r.type : ''}</span>
            </div>`;
        }).join('');
    } catch (e) {
        resultsDiv.innerHTML = `<div style="padding:10px;color:var(--accent-red);font-size:12px">Error: ${e.message}</div>`;
    }
}

document.addEventListener('keydown', function(e) {
    if (e.target.id === 'search-inst-query' && e.key === 'Enter') searchInstruments();
});

// ─── Symbol Autocomplete ──────────────────────────────────────
// Shared typeahead for chart-symbol, bt-symbol, paper-symbol
(function initSymbolAutocomplete() {
    const configs = [
        { inputId: 'chart-symbol', exchangeId: 'chart-exchange' },
        { inputId: 'bt-symbol', exchangeId: 'bt-exchange' },
        { inputId: 'paper-symbol', exchangeId: 'paper-exchange' },
    ];
    let acDebounce = null;
    let activeAcIndex = -1;

    configs.forEach(cfg => {
        const input = document.getElementById(cfg.inputId);
        if (!input) return;

        // Wrap input in a position:relative container if not already
        let wrap = input.parentElement;
        if (!wrap.classList.contains('symbol-input-wrap')) {
            wrap = document.createElement('div');
            wrap.className = 'symbol-input-wrap';
            input.parentNode.insertBefore(wrap, input);
            wrap.appendChild(input);
        }

        // Create dropdown
        const dropdown = document.createElement('div');
        dropdown.className = 'symbol-autocomplete';
        dropdown.id = cfg.inputId + '-ac';
        wrap.appendChild(dropdown);

        input.setAttribute('autocomplete', 'off');

        input.addEventListener('input', function() {
            clearTimeout(acDebounce);
            const q = this.value.trim();
            if (q.length < 2) { dropdown.style.display = 'none'; return; }
            acDebounce = setTimeout(() => fetchAcResults(q, cfg, dropdown), 250);
        });

        input.addEventListener('keydown', function(e) {
            const items = dropdown.querySelectorAll('.ac-item');
            if (!items.length || dropdown.style.display === 'none') {
                if (e.key === 'Enter') return; // let default load behavior
                return;
            }
            if (e.key === 'ArrowDown') {
                e.preventDefault();
                activeAcIndex = Math.min(activeAcIndex + 1, items.length - 1);
                highlightAcItem(items);
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                activeAcIndex = Math.max(activeAcIndex - 1, 0);
                highlightAcItem(items);
            } else if (e.key === 'Enter' && activeAcIndex >= 0) {
                e.preventDefault();
                items[activeAcIndex].click();
            } else if (e.key === 'Escape') {
                dropdown.style.display = 'none';
                activeAcIndex = -1;
            }
        });

        input.addEventListener('blur', function() {
            setTimeout(() => { dropdown.style.display = 'none'; activeAcIndex = -1; }, 200);
        });
    });

    function highlightAcItem(items) {
        items.forEach((it, i) => it.classList.toggle('ac-active', i === activeAcIndex));
        if (activeAcIndex >= 0 && items[activeAcIndex]) {
            items[activeAcIndex].scrollIntoView({ block: 'nearest' });
        }
    }

    async function fetchAcResults(query, cfg, dropdown) {
        const exchange = document.getElementById(cfg.exchangeId)?.value || 'NSE';
        try {
            const results = await API.get('/api/instruments/search?q=' + encodeURIComponent(query) + '&exchange=' + exchange);
            if (!results || !results.length) {
                dropdown.innerHTML = '<div style="padding:8px 10px;color:#607d8b;font-size:11px">No matches for "' + query + '"</div>';
                dropdown.style.display = 'block';
                return;
            }
            activeAcIndex = -1;
            dropdown.innerHTML = results.slice(0, 15).map(r => {
                const extra = r.expiry ? ' <span style="color:#ff9100;font-size:10px">' + r.expiry + '</span>' : '';
                return '<div class="ac-item" data-symbol="' + r.symbol + '" data-token="' + r.token + '" data-exchange="' + (r.exchange||exchange) + '">' +
                    '<span class="ac-symbol">' + r.symbol + '</span>' + extra +
                    '<span class="ac-name">' + (r.name || '') + '</span>' +
                    '<span class="ac-exchange">' + (r.exchange||exchange) + (r.type ? ' · ' + r.type : '') + '</span>' +
                '</div>';
            }).join('');
            dropdown.style.display = 'block';

            // Click handler
            dropdown.querySelectorAll('.ac-item').forEach(item => {
                item.addEventListener('mousedown', function(e) {
                    e.preventDefault();
                    const sym = this.dataset.symbol;
                    const token = this.dataset.token;
                    const input = document.getElementById(cfg.inputId);
                    input.value = sym;
                    dropdown.style.display = 'none';
                    activeAcIndex = -1;

                    // If paper trading, fill the token
                    if (cfg.inputId === 'paper-symbol') {
                        const tokenEl = document.getElementById('paper-token');
                        const statusEl = document.getElementById('paper-token-status');
                        if (tokenEl) tokenEl.value = token;
                        if (statusEl) {
                            statusEl.textContent = '✓ ' + sym + ' → ' + token;
                            statusEl.style.color = '#00d4aa';
                        }
                    }
                });
            });
        } catch (e) {
            dropdown.innerHTML = '<div style="padding:8px 10px;color:#ff5252;font-size:11px">Search error</div>';
            dropdown.style.display = 'block';
        }
    }
})();

async function startLive() {
    const tokens = Array.from(selectedInstruments.keys());
    if (!tokens.length) { showToast('Select at least one instrument', 'error'); return; }
    const mode = document.getElementById('live-mode').value;
    // Build token→name map for server-side name resolution
    const names = {};
    selectedInstruments.forEach((name, token) => { names[token] = name; });
    try {
        const res = await API.post('/api/live/start', { tokens, mode, names });
        if (res.error) {
            showToast(res.error, 'error');
            if (res.hint) showToast(res.hint, 'info');
            return;
        }
        const stratCount = (res.strategies || []).length;
        showToast(`Live trading started with ${tokens.length} instruments and ${stratCount} strategies`, 'success');
        hideModal('modal-start-live');
        loadStatus();
    } catch (e) { showToast(e.message, 'error'); }
}

async function stopLive() {
    try {
        await API.post('/api/live/stop');
        showToast('Live trading stopped', 'info');
        loadStatus();
    } catch (e) { showToast(e.message, 'error'); }
}

async function toggleKillSwitch() {
    const risk = await API.get('/api/risk');
    const activate = !risk.kill_switch_active;
    if (activate && !confirm('Are you sure you want to activate the kill switch? This will stop all trading.')) return;
    try {
        await API.post('/api/risk/kill-switch', { activate });
        showToast(activate ? 'Kill switch ACTIVATED' : 'Kill switch deactivated', activate ? 'error' : 'success');
        loadStatus();
    } catch (e) { showToast(e.message, 'error'); }
}

async function loadSignals() {
    try {
        const signals = await API.get('/api/signals');
        const el = document.getElementById('signals-log');
        if (signals.length === 0) {
            el.innerHTML = '<div class="empty-state"><p>No signals generated yet</p></div>';
            return;
        }
        el.innerHTML = signals.reverse().map(s => {
            const cls = s.transaction_type === 'BUY' ? 'buy' : 'sell';
            return `
                <div class="signal-entry ${cls}">
                    <div class="signal-header">
                        <strong>${s.tradingsymbol} ${s.transaction_type}</strong>
                        <span class="signal-time">${new Date(s.timestamp).toLocaleTimeString()}</span>
                    </div>
                    <div class="signal-detail">
                        Strategy: ${s.strategy} | Qty: ${s.quantity} |
                        Price: ${formatNum(s.price)} | Confidence: ${formatNum(s.confidence)}%
                    </div>
                </div>
            `;
        }).join('');
    } catch (e) { console.error(e); }
}

async function loadOrders() {
    try {
        const orders = await API.get('/api/orders');
        const el = document.getElementById('orders-body');
        if (!orders || orders.length === 0) {
            el.innerHTML = '<tr><td colspan="8" style="color:var(--text-muted);text-align:center">No orders</td></tr>';
            return;
        }
        el.innerHTML = orders.map(o => `
            <tr>
                <td>${o.order_id?.slice(0, 12) || ''}</td>
                <td>${o.tradingsymbol}</td>
                <td class="${o.transaction_type === 'BUY' ? 'positive' : 'negative'}">${o.transaction_type}</td>
                <td>${o.quantity}</td>
                <td>${formatNum(o.price)}</td>
                <td>${o.order_type}</td>
                <td>${o.product}</td>
                <td><span class="status-badge ${o.status === 'COMPLETE' ? 'status-live' : 'status-offline'}">${o.status}</span></td>
            </tr>
        `).join('');
    } catch (e) { console.error(e); }
}

async function loadPositions() {
    try {
        const data = await API.get('/api/positions');
        const el = document.getElementById('positions-body');
        const positions = data.net || [];
        if (positions.length === 0) {
            el.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center">No positions</td></tr>';
            return;
        }
        el.innerHTML = positions.map(p => `
            <tr>
                <td>${p.tradingsymbol}</td>
                <td>${p.exchange}</td>
                <td>${p.quantity}</td>
                <td>${formatNum(p.average_price)}</td>
                <td>${formatNum(p.last_price)}</td>
                <td class="${pnlClass(p.pnl)}">${formatNum(p.pnl)}</td>
                <td>${p.product}</td>
            </tr>
        `).join('');
    } catch (e) { console.error(e); }
}

let oiWs = null;
let oiInitialized = false;
let cachedFutReport = null;
let cachedOptReport = {};

/* ─── OI Auto-refresh timers ─── */
let _oiAutoRefreshTimer = null;
let _oiActiveTab = 'options'; // track which OI sub-tab is active

function _clearOIAutoRefresh() {
    if (_oiAutoRefreshTimer) {
        clearInterval(_oiAutoRefreshTimer);
        _oiAutoRefreshTimer = null;
    }
}

function _startOIAutoRefresh(tab) {
    _clearOIAutoRefresh();
    _oiActiveTab = tab;
    const runNow = () => {
        if (tab === 'options') runOptionsOIScan();
        else if (tab === 'futures') runFuturesOIScan();
        else if (tab === 'strategy') _autoRefreshOIStrategy();
        else if (tab === 'live') loadOIData();
    };
    // Auto-refresh every 60 seconds
    _oiAutoRefreshTimer = setInterval(runNow, 60_000);
}

async function _autoRefreshOIStrategy() {
    // Scan whichever underlying is selected, then reload positions
    try {
        const sel = document.getElementById('oi-underlying-select').value;
        const st = document.getElementById('oi-strat-scan-status');
        st.textContent = `Auto-scanning ${sel}...`;
        const r = await fetch('/api/oi/strategy/scan', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({underlying: sel})});
        const d = await r.json();
        if (r.ok && d.signals) {
            st.textContent = `Auto: ${(d.signals||[]).length} signals @ ${new Date().toLocaleTimeString()}`;
            renderOIStrategyScanResult(d, sel);
        }
        loadOIStrategyPositions();
    } catch(e) { console.error('OI strategy auto-refresh error:', e); }
}

function initOITracker() {
    if (!oiInitialized) {
        oiInitialized = true;
        connectOIWebSocket();
    }
    document.getElementById('oi-underlying-select').addEventListener('change', () => {
        const activeView = document.querySelector('.oi-view.active');
        if (activeView && activeView.id === 'oi-view-options') loadCachedOptionsOI();
        if (activeView && activeView.id === 'oi-view-live') loadOIData();
    });
    // Auto-load the default active tab (Options OI)
    _startOIAutoRefresh('options');
    runOptionsOIScan();
}

/* ─── OI Sub-tab switching ─── */
function switchOITab(tab) {
    document.querySelectorAll('.oi-tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.oi-view').forEach(v => v.classList.remove('active'));
    document.querySelector(`.oi-tab-btn[data-oi-tab="${tab}"]`).classList.add('active');
    document.getElementById('oi-view-' + tab).classList.add('active');

    // Start auto-refresh for the active tab
    _startOIAutoRefresh(tab);

    // Immediate load on tab switch
    if (tab === 'options') {
        if (cachedOptReport[document.getElementById('oi-underlying-select').value]) {
            loadCachedOptionsOI();
        } else {
            runOptionsOIScan();
        }
    }
    if (tab === 'futures') {
        if (cachedFutReport) {
            renderFuturesOI(cachedFutReport);
        } else {
            runFuturesOIScan();
        }
    }
    if (tab === 'strategy') loadOIStrategyOnSwitch();
    if (tab === 'live') startLiveOIStream();
}

/* ═══ OI STRATEGY FUNCTIONS ═══ */
let _oiStratLastScan = null;

function loadOIStrategyOnSwitch() {
    loadOIStrategyPositions();
    loadOIStrategyConfig();
    // Also auto-scan on switch if no previous scan data
    if (!_oiStratLastScan) {
        const sel = document.getElementById('oi-underlying-select').value;
        runOIStrategyScan(sel);
    }
}

async function runOIStrategyScan(underlying) {
    const st = document.getElementById('oi-strat-scan-status');
    st.textContent = `Scanning ${underlying}...`;
    try {
        const r = await fetch('/api/oi/strategy/scan', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({underlying})});
        const d = await r.json();
        if (!r.ok) { st.textContent = d.detail || 'Scan failed'; return; }
        st.textContent = `${underlying} scan complete — ${(d.signals||[]).length} signals`;
        _oiStratLastScan = d;
        renderOIStrategyScanResult(d, underlying);
        loadOIStrategyPositions();
    } catch(e) { st.textContent = 'Error: ' + e.message; }
}

async function runOIStrategyScanBoth() {
    const st = document.getElementById('oi-strat-scan-status');
    st.textContent = 'Scanning NIFTY + SENSEX...';
    try {
        const r = await fetch('/api/oi/strategy/scan-both', {method:'POST'});
        const d = await r.json();
        if (!r.ok) { st.textContent = d.detail || 'Scan failed'; return; }
        const nSigs = (d.nifty?.signals||[]).length;
        const sSigs = (d.sensex?.signals||[]).length;
        st.textContent = `Both scanned — NIFTY: ${nSigs}, SENSEX: ${sSigs} signals`;
        // Merge signals for display
        const allSignals = [...(d.nifty?.signals||[]), ...(d.sensex?.signals||[])];
        const merged = {signals: allSignals, scan_time: new Date().toISOString()};
        _oiStratLastScan = merged;
        renderOIStrategyScanResult(merged, 'BOTH');
        // Show summary cards from last response
        if (d.nifty) updateOIStratScanDetail(d.nifty, 'NIFTY');
        loadOIStrategyPositions();
    } catch(e) { st.textContent = 'Error: ' + e.message; }
}

function renderOIStrategyScanResult(data, index) {
    // Update summary cards
    const signals = data.signals || [];
    const bullish = signals.filter(s => s.direction === 'BULLISH').length;
    const bearish = signals.filter(s => s.direction === 'BEARISH').length;
    const dir = bullish > bearish ? 'BULLISH' : bearish > bullish ? 'BEARISH' : 'NEUTRAL';
    const dirEl = document.getElementById('oi-strat-direction');
    dirEl.textContent = dir;
    dirEl.style.color = dir === 'BULLISH' ? 'var(--success)' : dir === 'BEARISH' ? 'var(--danger)' : 'var(--text-muted)';
    
    const avgConf = signals.length > 0 ? (signals.reduce((a,s) => a + (s.confidence||0), 0) / signals.length).toFixed(0) + '%' : '--';
    document.getElementById('oi-strat-confidence').textContent = avgConf;
    document.getElementById('oi-strat-signal-count').textContent = signals.length;

    // Update scan detail if single index
    if (index !== 'BOTH' && data.pcr_oi != null) {
        updateOIStratScanDetail(data, index);
    }

    // Render signals table
    const tbody = document.getElementById('oi-strat-signals-body');
    if (signals.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" style="color:var(--text-muted);text-align:center">No signals detected</td></tr>';
        return;
    }
    tbody.innerHTML = signals.map(s => {
        const dirColor = s.direction === 'BULLISH' ? 'var(--success)' : s.direction === 'BEARISH' ? 'var(--danger)' : 'var(--text-muted)';
        const confPct = (s.confidence||0).toFixed(0) + '%';
        const typeLabel = (s.signal_type||'').replace(/_/g,' ');
        return `<tr>
            <td>${s.underlying||index}</td>
            <td style="font-size:11px">${typeLabel}</td>
            <td style="color:${dirColor};font-weight:600">${s.direction||'--'}</td>
            <td>${s.strike||'--'}</td>
            <td>${s.option_type||'--'}</td>
            <td>${confPct}</td>
            <td style="font-size:11px">${(s.action||'').replace(/_/g,' ')}</td>
            <td>${s.entry_price ? s.entry_price.toFixed(1) : '--'}</td>
            <td>${s.stop_loss ? s.stop_loss.toFixed(1) : '--'}</td>
            <td>${s.target ? s.target.toFixed(1) : '--'}</td>
            <td><button class="btn btn-sm btn-primary" onclick="executeOISignal('${s.id}')">Execute</button></td>
        </tr>`;
    }).join('');
}

function updateOIStratScanDetail(data, index) {
    const panel = document.getElementById('oi-strat-scan-detail');
    panel.style.display = 'block';
    document.getElementById('oi-strat-scan-index').textContent = index;
    document.getElementById('oi-strat-scan-time').textContent = data.timestamp ? new Date(data.timestamp).toLocaleTimeString() : '--';
    document.getElementById('oi-strat-pcr').textContent = data.pcr_oi != null ? data.pcr_oi.toFixed(2) : '--';
    document.getElementById('oi-strat-maxpain').textContent = data.max_pain || '--';
    document.getElementById('oi-strat-ivskew').textContent = data.iv_skew != null ? data.iv_skew.toFixed(1) + '%' : '--';
    document.getElementById('oi-strat-straddle').textContent = data.straddle_premium != null ? data.straddle_premium.toFixed(1) : '--';
}

async function executeOISignal(signalId) {
    if (!confirm('Execute this OI signal as a position?')) return;
    try {
        const r = await fetch('/api/oi/strategy/execute', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({signal_id: signalId})});
        const d = await r.json();
        if (!r.ok) { alert(d.detail || 'Execute failed'); return; }
        showToast('Position created from signal', 'success');
        loadOIStrategyPositions();
    } catch(e) { alert('Error: ' + e.message); }
}

async function closeOIPosition(positionId) {
    const priceStr = prompt('Enter current/exit price:');
    if (!priceStr) return;
    const exitPrice = parseFloat(priceStr);
    if (isNaN(exitPrice)) { alert('Invalid price'); return; }
    try {
        const r = await fetch('/api/oi/strategy/close', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({position_id: positionId, exit_price: exitPrice})});
        const d = await r.json();
        if (!r.ok) { alert(d.detail || 'Close failed'); return; }
        showToast('Position closed', 'info');
        loadOIStrategyPositions();
    } catch(e) { alert('Error: ' + e.message); }
}

async function loadOIStrategyPositions() {
    try {
        const r = await fetch('/api/oi/strategy/positions');
        const d = await r.json();
        renderOIStratOpenPositions(d.open || []);
        renderOIStratClosedPositions(d.closed || []);
        // Update P&L card
        const totalPnl = (d.open||[]).reduce((a, p) => a + (p.pnl||0), 0);
        const pnlEl = document.getElementById('oi-strat-pnl');
        pnlEl.textContent = totalPnl !== 0 ? (totalPnl > 0 ? '+' : '') + totalPnl.toFixed(0) : '--';
        pnlEl.style.color = totalPnl > 0 ? 'var(--success)' : totalPnl < 0 ? 'var(--danger)' : '';
    } catch(e) { console.error('Load positions error:', e); }
}

function renderOIStratOpenPositions(positions) {
    const tbody = document.getElementById('oi-strat-open-body');
    if (positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="10" style="color:var(--text-muted);text-align:center">No open positions</td></tr>';
        return;
    }
    tbody.innerHTML = positions.map(p => {
        const pnl = p.pnl || 0;
        const pnlColor = pnl > 0 ? 'var(--success)' : pnl < 0 ? 'var(--danger)' : '';
        return `<tr>
            <td>${p.underlying||'--'}</td>
            <td style="font-size:11px">${(p.signal_type||'').replace(/_/g,' ')}</td>
            <td>${(p.direction||'').replace(/_/g,' ')}</td>
            <td>${p.strike||'--'}</td>
            <td>${p.entry_price ? p.entry_price.toFixed(1) : '--'}</td>
            <td>${p.current_price ? p.current_price.toFixed(1) : '--'}</td>
            <td style="color:${pnlColor};font-weight:600">${pnl !== 0 ? (pnl > 0 ? '+' : '') + pnl.toFixed(1) : '--'}</td>
            <td>${p.stop_loss ? p.stop_loss.toFixed(1) : '--'}</td>
            <td>${p.target ? p.target.toFixed(1) : '--'}</td>
            <td><button class="btn btn-sm" style="background:var(--danger);color:#fff" onclick="closeOIPosition('${p.id}')">Close</button></td>
        </tr>`;
    }).join('');
}

function renderOIStratClosedPositions(positions) {
    const tbody = document.getElementById('oi-strat-closed-body');
    if (positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="color:var(--text-muted);text-align:center">No closed positions</td></tr>';
        return;
    }
    tbody.innerHTML = positions.map(p => {
        const pnl = p.pnl || 0;
        const pnlColor = pnl > 0 ? 'var(--success)' : pnl < 0 ? 'var(--danger)' : '';
        const dur = p.duration || '--';
        return `<tr>
            <td>${p.underlying||'--'}</td>
            <td style="font-size:11px">${(p.signal_type||'').replace(/_/g,' ')}</td>
            <td>${(p.direction||'').replace(/_/g,' ')}</td>
            <td>${p.strike||'--'}</td>
            <td>${p.entry_price ? p.entry_price.toFixed(1) : '--'}</td>
            <td>${p.exit_price ? p.exit_price.toFixed(1) : '--'}</td>
            <td style="color:${pnlColor};font-weight:600">${pnl !== 0 ? (pnl > 0 ? '+' : '') + pnl.toFixed(1) : '--'}</td>
            <td>${p.exit_reason||'--'}</td>
            <td>${dur}</td>
        </tr>`;
    }).join('');
}

async function loadOIStrategySummary() {
    try {
        const r = await fetch('/api/oi/strategy/summary');
        const d = await r.json();
        const panel = document.getElementById('oi-strat-perf-card');
        panel.style.display = 'block';
        document.getElementById('oi-strat-perf-total').textContent = d.closed_positions || 0;
        document.getElementById('oi-strat-perf-winrate').textContent = d.win_rate != null ? d.win_rate.toFixed(0) + '%' : '--';
        const tp = d.total_pnl || 0;
        const tpEl = document.getElementById('oi-strat-perf-pnl');
        tpEl.textContent = tp !== 0 ? (tp > 0 ? '+' : '') + tp.toFixed(0) : '0';
        tpEl.style.color = tp > 0 ? 'var(--success)' : tp < 0 ? 'var(--danger)' : '';
        const avgPnl = d.closed_positions > 0 ? tp / d.closed_positions : 0;
        document.getElementById('oi-strat-perf-avg').textContent = d.closed_positions > 0 ? avgPnl.toFixed(1) : '--';
        // By signal type breakdown
        const byType = d.by_signal_type || {};
        const typeDiv = document.getElementById('oi-strat-perf-by-type');
        const types = Object.entries(byType);
        if (types.length > 0) {
            typeDiv.innerHTML = '<div style="font-size:12px;color:var(--text-muted);margin-bottom:6px">By Signal Type</div>' +
                '<table class="data-table"><thead><tr><th>Type</th><th>Trades</th><th>Win%</th><th>P&L</th></tr></thead><tbody>' +
                types.map(([t, v]) => {
                    const wp = v.win_rate != null ? v.win_rate.toFixed(0)+'%' : '--';
                    const pnl = v.pnl || 0;
                    return `<tr><td style="font-size:11px">${t.replace(/_/g,' ')}</td><td>${v.trades||0}</td><td>${wp}</td><td style="color:${pnl>0?'var(--success)':pnl<0?'var(--danger)':''}">${pnl.toFixed(0)}</td></tr>`;
                }).join('') + '</tbody></table>';
        } else { typeDiv.innerHTML = ''; }
    } catch(e) { console.error('Summary error:', e); }
}

function toggleOIStrategyConfig() {
    const panel = document.getElementById('oi-strat-config-panel');
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
}

async function loadOIStrategyConfig() {
    try {
        const r = await fetch('/api/oi/strategy/config');
        const d = await r.json();
        document.getElementById('oi-cfg-pcr-bull').value = d.pcr_extreme_low || 0.5;
        document.getElementById('oi-cfg-pcr-bear').value = d.pcr_extreme_high || 1.5;
        document.getElementById('oi-cfg-wall-prox').value = d.wall_proximity_pct || 1.5;
        document.getElementById('oi-cfg-straddle-col').value = d.straddle_collapse_pct || 20;
        document.getElementById('oi-cfg-iv-skew').value = d.iv_skew_threshold || 5.0;
        document.getElementById('oi-cfg-min-conf').value = d.min_confidence || 0.6;
        document.getElementById('oi-cfg-max-pos').value = d.max_open_positions || 5;
        document.getElementById('oi-cfg-sl').value = d.stop_loss_pct || 30;
        document.getElementById('oi-cfg-target').value = d.target_pct || 60;
    } catch(e) { console.error('Load config error:', e); }
}

async function saveOIStrategyConfig() {
    const updates = {
        pcr_extreme_low: parseFloat(document.getElementById('oi-cfg-pcr-bull').value),
        pcr_extreme_high: parseFloat(document.getElementById('oi-cfg-pcr-bear').value),
        wall_proximity_pct: parseFloat(document.getElementById('oi-cfg-wall-prox').value),
        straddle_collapse_pct: parseFloat(document.getElementById('oi-cfg-straddle-col').value),
        iv_skew_threshold: parseFloat(document.getElementById('oi-cfg-iv-skew').value),
        min_confidence: parseFloat(document.getElementById('oi-cfg-min-conf').value),
        max_open_positions: parseInt(document.getElementById('oi-cfg-max-pos').value),
        stop_loss_pct: parseFloat(document.getElementById('oi-cfg-sl').value),
        target_pct: parseFloat(document.getElementById('oi-cfg-target').value)
    };
    try {
        const r = await fetch('/api/oi/strategy/config', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(updates)});
        const d = await r.json();
        if (r.ok) showToast('Config saved', 'success');
        else alert(d.detail || 'Save failed');
    } catch(e) { alert('Error: ' + e.message); }
}

function switchFutView(view) {
    document.querySelectorAll('.fut-view-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.fut-view').forEach(v => v.classList.remove('active'));
    const btn = document.querySelector(`.fut-view-btn[onclick*="'${view}'"]`);
    if (btn) btn.classList.add('active');
    document.getElementById('fut-view-' + view).classList.add('active');
}

/* ─── WebSocket (live stream) ─── */
let _liveStreamStarted = false;
let _liveOIRestPollTimer = null;

async function startLiveOIStream() {
    // Show loading state
    const statusEl = document.getElementById('oi-stream-status');
    statusEl.textContent = 'STARTING...';
    statusEl.className = 'status-badge status-pending';

    // First load current data via REST
    await loadOIData();

    // Then start OI tracking subscription on the backend (instruments + tick subscription)
    if (!_liveStreamStarted) {
        _liveStreamStarted = true;
        try {
            const res = await API.post('/api/oi/start', {
                nifty_spot: 0,  // will be fetched by backend
                sensex_spot: 0,
            });
            console.log('OI tracking started:', res);
            statusEl.textContent = 'STREAMING';
            statusEl.className = 'status-badge status-live';
        } catch(e) {
            console.error('Failed to start OI tracking:', e);
            statusEl.textContent = 'POLLING';
            statusEl.className = 'status-badge status-live';
        }
    }
    // Start REST polling fallback every 10s — supplements WebSocket stream
    _startLiveOIRestPoll();
}

function _startLiveOIRestPoll() {
    if (_liveOIRestPollTimer) return; // already running
    _liveOIRestPollTimer = setInterval(async () => {
        // Only poll if we're on the Live tab
        if (_oiActiveTab !== 'live') return;
        try {
            const sel = document.getElementById('oi-underlying-select').value;
            const data = await API.get(`/api/oi/summary?underlying=${sel}`);
            if (data && (data.total_ce_oi > 0 || data.total_pe_oi > 0)) {
                renderLiveOIData(data);
                document.getElementById('oi-last-update').textContent = 'Poll: ' + new Date().toLocaleTimeString();
            }
        } catch (e) { /* ignore poll errors */ }
    }, 10000);
}

function connectOIWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    oiWs = new WebSocket(`${protocol}//${location.host}/ws/oi`);
    oiWs.onopen = () => {
        document.getElementById('oi-stream-status').textContent = 'CONNECTED';
        document.getElementById('oi-stream-status').className = 'status-badge status-live';
    };
    oiWs.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            if (msg.type === 'oi_update') {
                const sel = document.getElementById('oi-underlying-select').value;
                const data = sel === 'NIFTY' ? msg.nifty : msg.sensex;
                if (data && (data.total_ce_oi > 0 || data.total_pe_oi > 0)) {
                    renderLiveOIData(data);
                    document.getElementById('oi-last-update').textContent = 'Live: ' + new Date().toLocaleTimeString();
                }
            }
        } catch (err) { console.error('OI WS parse error:', err); }
    };
    oiWs.onclose = () => {
        document.getElementById('oi-stream-status').textContent = 'RECONNECTING';
        document.getElementById('oi-stream-status').className = 'status-badge status-offline';
        setTimeout(connectOIWebSocket, 5000);
    };
    oiWs.onerror = () => oiWs.close();
}

async function loadOIData() {
    const sel = document.getElementById('oi-underlying-select').value;
    try {
        const data = await API.get(`/api/oi/summary?underlying=${sel}`);
        renderLiveOIData(data);
    } catch (e) { console.error('OI load error:', e); }
}

/* ─── Helpers ─── */
function formatLargeNum(n) {
    if (n == null) return '--';
    n = Number(n);
    if (isNaN(n)) return '--';
    if (Math.abs(n) >= 10000000) return (n / 10000000).toFixed(2) + ' Cr';
    if (Math.abs(n) >= 100000) return (n / 100000).toFixed(2) + ' L';
    if (Math.abs(n) >= 1000) return (n / 1000).toFixed(1) + ' K';
    return n.toLocaleString();
}

function buildupBadge(b) {
    if (!b) return '<span class="buildup-badge buildup-neutral">--</span>';
    const m = {
        'LONG_BUILDUP': ['Long Buildup', 'buildup-long'],
        'SHORT_BUILDUP': ['Short Buildup', 'buildup-short'],
        'LONG_UNWINDING': ['Long Unwinding', 'buildup-unwinding'],
        'SHORT_COVERING': ['Short Covering', 'buildup-covering'],
    };
    const [label, cls] = m[b] || [b, 'buildup-neutral'];
    return `<span class="buildup-badge ${cls}">${label}</span>`;
}

function signalBadge(s) {
    const m = {
        'CE_WRITING': ['CE Writing', 'signal-ce-writing'],
        'CE_BUYING': ['CE Buying', 'signal-ce-buying'],
        'PE_WRITING': ['PE Writing', 'signal-pe-writing'],
        'PE_BUYING': ['PE Buying', 'signal-pe-buying'],
    };
    const [label, cls] = m[s] || [s, 'signal-ce-writing'];
    return `<span class="signal-badge ${cls}">${label}</span>`;
}

function biasColor(bias) {
    if (!bias) return 'var(--text-muted)';
    const b = bias.toUpperCase();
    if (b.includes('BULLISH')) return '#00d4aa';
    if (b.includes('BEARISH')) return '#ff5252';
    return 'var(--text-secondary)';
}

function chgClass(v) {
    if (v == null) return 'neutral';
    return v > 0 ? 'positive' : v < 0 ? 'negative' : 'neutral';
}

function fmtPct(v) {
    if (v == null) return '--';
    return (v > 0 ? '+' : '') + Number(v).toFixed(2) + '%';
}

/* ════════════════════════════════════════════
   OPTIONS OI  (Near-ATM Analysis)
   ════════════════════════════════════════════ */
async function runOptionsOIScan() {
    const sel = document.getElementById('oi-underlying-select').value;
    const statusEl = document.getElementById('opt-oi-status');
    statusEl.textContent = 'Scanning ' + sel + '...';
    statusEl.style.color = 'var(--accent-blue)';
    try {
        const data = await API.post('/api/oi/options/scan', { underlying: sel });
        if (data.error) throw new Error(data.error);
        cachedOptReport[sel] = data;
        const ts = new Date().toLocaleTimeString();
        statusEl.textContent = 'Scanned at ' + ts;
        statusEl.style.color = 'var(--accent-green)';
        const globalTs = document.getElementById('oi-last-update');
        if (globalTs) globalTs.textContent = ts;
        renderOptionsOI(data);
    } catch (e) {
        statusEl.textContent = 'Error: ' + e.message;
        statusEl.style.color = 'var(--accent-red)';
    }
}

async function loadCachedOptionsOI() {
    const sel = document.getElementById('oi-underlying-select').value;
    if (cachedOptReport[sel]) { renderOptionsOI(cachedOptReport[sel]); return; }
    try {
        const data = await API.get(`/api/oi/options/report?underlying=${sel}`);
        if (!data.error) { cachedOptReport[sel] = data; renderOptionsOI(data); }
    } catch (e) { /* ignore */ }
}

function renderOptionsOI(d) {
    // Summary cards
    document.getElementById('opt-spot').textContent = formatNum(d.spot_price || 0);
    document.getElementById('opt-atm').textContent = formatNum(d.atm_strike || 0, 0);
    document.getElementById('opt-expiry').textContent = d.expiry || '--';

    const pcr = d.pcr_oi || 0;
    const pcrEl = document.getElementById('opt-pcr');
    pcrEl.textContent = formatNum(pcr, 3);
    pcrEl.style.color = pcr > 1 ? '#00d4aa' : pcr < 0.7 ? '#ff5252' : 'var(--text-primary)';
    document.getElementById('opt-pcr-vol').textContent = formatNum(d.pcr_volume || 0, 3);

    document.getElementById('opt-maxpain').textContent = formatNum(d.max_pain || 0, 0);
    document.getElementById('opt-straddle').textContent = formatNum(d.atm_straddle_premium || 0);

    const biasEl = document.getElementById('opt-bias');
    biasEl.textContent = d.bias || '--';
    biasEl.style.color = biasColor(d.bias);
    const reasonsEl = document.getElementById('opt-bias-reasons');
    reasonsEl.innerHTML = (d.bias_reasons || []).map(r => `• ${r}`).join('<br>') || '--';

    // OI Walls
    document.getElementById('opt-pe-wall').textContent = d.max_pe_oi_strike ? formatNum(d.max_pe_oi_strike, 0) : '--';
    document.getElementById('opt-ce-wall').textContent = d.max_ce_oi_strike ? formatNum(d.max_ce_oi_strike, 0) : '--';

    // Find wall OI values
    const strikes = d.strikes || [];
    const ceWallStrike = strikes.find(s => s.strike === d.max_ce_oi_strike);
    const peWallStrike = strikes.find(s => s.strike === d.max_pe_oi_strike);
    document.getElementById('opt-ce-wall-oi').textContent = 'OI: ' + formatLargeNum(ceWallStrike?.ce_oi || d.max_ce_oi || 0);
    document.getElementById('opt-pe-wall-oi').textContent = 'OI: ' + formatLargeNum(peWallStrike?.pe_oi || d.max_pe_oi || 0);

    // IV Skew
    const skew = d.iv_skew || 0;
    const skewEl = document.getElementById('opt-iv-skew');
    skewEl.textContent = (skew > 0 ? '+' : '') + formatNum(skew, 2) + '%';
    skewEl.style.color = skew > 2 ? '#ff5252' : skew < -2 ? '#00d4aa' : 'var(--text-primary)';
    document.getElementById('opt-ce-iv').textContent = d.avg_ce_iv ? formatNum(d.avg_ce_iv, 1) + '%' : '--';
    document.getElementById('opt-pe-iv').textContent = d.avg_pe_iv ? formatNum(d.avg_pe_iv, 1) + '%' : '--';

    // Totals
    document.getElementById('opt-ce-total').textContent = formatLargeNum(d.total_ce_oi || 0);
    document.getElementById('opt-pe-total').textContent = formatLargeNum(d.total_pe_oi || 0);
    document.getElementById('opt-ce-vol').textContent = formatLargeNum(d.total_ce_volume || 0);
    document.getElementById('opt-pe-vol').textContent = formatLargeNum(d.total_pe_volume || 0);

    // Strike chain table
    renderOptionsChain(strikes, d.atm_strike || 0);

    // Top OI additions
    renderTopOIAdds(d.top_ce_oi_additions || [], d.top_pe_oi_additions || []);

    // Buildup signals
    renderOptBuildupSignals(d.buildup_signals || []);
}

function renderOptionsChain(strikes, atm) {
    const body = document.getElementById('opt-chain-body');
    if (!strikes || strikes.length === 0) {
        body.innerHTML = '<tr><td colspan="12" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    // Find max OI for bars
    let maxOI = 1;
    strikes.forEach(s => { maxOI = Math.max(maxOI, s.ce_oi || 0, s.pe_oi || 0); });

    body.innerHTML = strikes.map(s => {
        const isATM = s.is_atm || s.strike === atm;
        const rowCls = isATM ? 'chain-atm-row' : '';
        const ceBar = ((s.ce_oi || 0) / maxOI * 100).toFixed(0);
        const peBar = ((s.pe_oi || 0) / maxOI * 100).toFixed(0);
        const ceChgCls = chgClass(s.ce_oi_change);
        const peChgCls = chgClass(s.pe_oi_change);
        const netChg = (s.pe_oi_change || 0) - (s.ce_oi_change || 0);
        const netCls = chgClass(netChg);

        return `<tr class="${rowCls}">
            <td class="${ceChgCls}" style="text-align:right;font-size:11px">${formatLargeNum(s.ce_oi_change || 0)}</td>
            <td style="text-align:right;font-size:11px">${formatLargeNum(s.ce_volume || 0)}</td>
            <td style="text-align:right;position:relative"><div class="oi-bar-wrap"><div class="oi-bar oi-bar-ce" style="width:${ceBar}%"></div></div>${formatLargeNum(s.ce_oi || 0)}</td>
            <td style="text-align:right">${formatNum(s.ce_ltp || 0)}</td>
            <td style="text-align:right;font-size:11px">${s.ce_iv ? formatNum(s.ce_iv, 1) + '%' : '--'}</td>
            <td style="text-align:center;font-weight:700;font-size:13px;${isATM ? 'color:var(--accent-blue)' : ''}">${formatNum(s.strike, 0)}${isATM ? ' ★' : ''}</td>
            <td style="font-size:11px">${s.pe_iv ? formatNum(s.pe_iv, 1) + '%' : '--'}</td>
            <td>${formatNum(s.pe_ltp || 0)}</td>
            <td style="position:relative"><div class="oi-bar-wrap"><div class="oi-bar oi-bar-pe" style="width:${peBar}%"></div></div>${formatLargeNum(s.pe_oi || 0)}</td>
            <td style="font-size:11px">${formatLargeNum(s.pe_volume || 0)}</td>
            <td class="${peChgCls}" style="font-size:11px">${formatLargeNum(s.pe_oi_change || 0)}</td>
            <td class="${netCls}" style="font-size:11px;font-weight:600">${formatLargeNum(netChg)}</td>
        </tr>`;
    }).join('');
}

function renderTopOIAdds(ceAdds, peAdds) {
    const ceEl = document.getElementById('opt-top-ce-adds');
    const peEl = document.getElementById('opt-top-pe-adds');
    ceEl.innerHTML = ceAdds.length === 0 ? '--' : ceAdds.map(a =>
        `<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid var(--border)">
            <span>${formatNum(a.strike, 0)}</span>
            <span class="positive">+${formatLargeNum(a.oi_change || a.ce_oi_change || 0)}</span>
        </div>`
    ).join('');
    peEl.innerHTML = peAdds.length === 0 ? '--' : peAdds.map(a =>
        `<div style="display:flex;justify-content:space-between;padding:2px 0;border-bottom:1px solid var(--border)">
            <span>${formatNum(a.strike, 0)}</span>
            <span class="positive">+${formatLargeNum(a.oi_change || a.pe_oi_change || 0)}</span>
        </div>`
    ).join('');
}

function renderOptBuildupSignals(signals) {
    const el = document.getElementById('opt-buildup-signals');
    if (!signals || signals.length === 0) {
        el.innerHTML = '<div class="empty-state"><p>No signals</p></div>';
        return;
    }
    el.innerHTML = signals.map(s => {
        return `<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid var(--border)">
            ${signalBadge(s.pattern || s.signal || s.type || '')}
            <span style="font-weight:600">${formatNum(s.strike, 0)}</span>
            <span style="font-size:11px;color:var(--text-muted)">OIΔ: ${formatLargeNum(s.oi_change || 0)} | ${s.sentiment || s.type || ''}</span>
        </div>`;
    }).join('');
}

/* ════════════════════════════════════════════
   FUTURES OI  (Index + Stock Futures)
   ════════════════════════════════════════════ */
async function runFuturesOIScan() {
    const statusEl = document.getElementById('fut-oi-status');
    statusEl.textContent = 'Scanning all futures OI...';
    statusEl.style.color = 'var(--accent-blue)';
    try {
        const data = await API.post('/api/oi/futures/scan', {});
        if (data.error) throw new Error(data.error);
        cachedFutReport = data;
        const ts = new Date().toLocaleTimeString();
        statusEl.textContent = 'Scanned at ' + ts;
        statusEl.style.color = 'var(--accent-green)';
        document.getElementById('oi-last-update').textContent = 'Futures: ' + ts;
        renderFuturesOI(data);
    } catch (e) {
        statusEl.textContent = 'Error: ' + e.message;
        statusEl.style.color = 'var(--accent-red)';
    }
}

function renderFuturesOI(d) {
    // Market Sentiment
    const s = d.market_sentiment || {};
    const biasEl = document.getElementById('fut-bias');
    biasEl.textContent = s.bias || '--';
    biasEl.style.color = biasColor(s.bias);
    document.getElementById('fut-confidence').textContent = formatNum(s.confidence || 0, 0) + '%';
    document.getElementById('fut-bullish-count').textContent = (s.long_buildup || 0) + ' + ' + (s.short_covering || 0);
    document.getElementById('fut-bearish-count').textContent = (s.short_buildup || 0) + ' + ' + (s.long_unwinding || 0);
    document.getElementById('fut-reasons').innerHTML = (s.reasons || []).map(r => `• ${r}`).join('<br>') || '--';

    // Index Futures
    renderFutIndexTable(d.index_futures || []);

    // Buildup tables
    renderFutBuildupTable(d.top_long_buildup || [], 'fut-long-buildup-body');
    renderFutBuildupTable(d.top_short_buildup || [], 'fut-short-buildup-body');
    renderFutBuildupTable(d.top_long_unwinding || [], 'fut-long-unwind-body');
    renderFutBuildupTable(d.top_short_covering || [], 'fut-short-cover-body');

    // All stocks
    renderFutAllStocks(d.stock_futures || []);

    // Rollover
    renderFutRollover(d.rollover_data || []);

    // Sector
    renderFutSector(d.sector_summary || []);
}

function renderFutIndexTable(entries) {
    const body = document.getElementById('fut-index-body');
    if (!entries || entries.length === 0) {
        body.innerHTML = '<tr><td colspan="9" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    body.innerHTML = entries.map(e => `<tr>
        <td style="font-weight:600">${e.symbol || ''}</td>
        <td style="font-size:11px">${e.expiry || ''}</td>
        <td>${formatNum(e.ltp || 0)}</td>
        <td class="${chgClass(e.change_pct)}">${fmtPct(e.change_pct)}</td>
        <td>${formatLargeNum(e.oi || 0)}</td>
        <td class="${chgClass(e.oi_change)}">${formatLargeNum(e.oi_change || 0)}</td>
        <td class="${chgClass(e.oi_change_pct)}">${fmtPct(e.oi_change_pct)}</td>
        <td>${formatLargeNum(e.volume || 0)}</td>
        <td>${buildupBadge(e.buildup)}</td>
    </tr>`).join('');
}

function renderFutBuildupTable(entries, bodyId) {
    const body = document.getElementById(bodyId);
    if (!entries || entries.length === 0) {
        body.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center">None</td></tr>';
        return;
    }
    body.innerHTML = entries.map(e => `<tr>
        <td style="font-weight:600;font-size:12px">${e.symbol || ''}</td>
        <td style="font-size:11px;color:var(--text-muted)">${e.sector || '--'}</td>
        <td>${formatNum(e.ltp || 0)}</td>
        <td class="${chgClass(e.change_pct)}" style="font-weight:600">${fmtPct(e.change_pct)}</td>
        <td class="${chgClass(e.oi_change_pct)}" style="font-weight:600">${fmtPct(e.oi_change_pct)}</td>
        <td>${formatLargeNum(e.volume || 0)}</td>
    </tr>`).join('');
}

function renderFutAllStocks(entries) {
    const body = document.getElementById('fut-all-body');
    if (!entries || entries.length === 0) {
        body.innerHTML = '<tr><td colspan="11" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    body.innerHTML = entries.map(e => `<tr>
        <td style="font-weight:600;font-size:12px">${e.symbol || ''}</td>
        <td style="font-size:11px;color:var(--text-muted)">${e.sector || '--'}</td>
        <td style="font-size:11px">${e.expiry || ''}</td>
        <td>${formatNum(e.ltp || 0)}</td>
        <td class="${chgClass(e.change_pct)}">${fmtPct(e.change_pct)}</td>
        <td>${formatLargeNum(e.oi || 0)}</td>
        <td class="${chgClass(e.oi_change)}">${formatLargeNum(e.oi_change || 0)}</td>
        <td class="${chgClass(e.oi_change_pct)}">${fmtPct(e.oi_change_pct)}</td>
        <td>${formatLargeNum(e.volume || 0)}</td>
        <td style="font-size:11px">${formatNum(e.value_lakhs || 0, 0)}</td>
        <td>${buildupBadge(e.buildup)}</td>
    </tr>`).join('');
}

function renderFutRollover(entries) {
    const body = document.getElementById('fut-rollover-body');
    if (!entries || entries.length === 0) {
        body.innerHTML = '<tr><td colspan="8" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    body.innerHTML = entries.map(e => {
        const sig = (e.signal || '').toUpperCase();
        const sigCls = sig.includes('POSITIVE') ? 'positive' : sig.includes('NEGATIVE') ? 'negative' : 'neutral';
        return `<tr>
            <td style="font-weight:600">${e.symbol || ''}</td>
            <td style="font-size:11px">${e.current_expiry || ''}</td>
            <td style="font-size:11px">${e.next_expiry || ''}</td>
            <td>${formatLargeNum(e.current_oi || 0)}</td>
            <td>${formatLargeNum(e.next_oi || 0)}</td>
            <td style="font-weight:600">${formatNum(e.rollover_pct || 0)}%</td>
            <td>${fmtPct(e.basis_pct)}</td>
            <td class="${sigCls}" style="font-weight:600">${e.signal || '--'}</td>
        </tr>`;
    }).join('');
}

function renderFutSector(sectorList) {
    const heatmap = document.getElementById('fut-sector-heatmap');
    const body = document.getElementById('fut-sector-body');
    if (!sectorList || sectorList.length === 0) {
        heatmap.innerHTML = '<div style="color:var(--text-muted)">No sector data</div>';
        body.innerHTML = '<tr><td colspan="8" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }

    // Heatmap tiles
    heatmap.innerHTML = sectorList.map(s => {
        const chg = s.avg_change_pct || 0;
        const tileCls = s.bias === 'BULLISH' ? 'sect-bull' : s.bias === 'BEARISH' ? 'sect-bear' : 'sect-neutral';
        return `<div class="sector-oi-tile ${tileCls}">
            <div class="s-name">${s.sector || 'Other'}</div>
            <div class="s-chg" style="color:${chg >= 0 ? '#00d4aa' : '#ff5252'}">${fmtPct(chg)}</div>
            <div class="s-bias">${s.bias || 'NEUTRAL'}</div>
        </div>`;
    }).join('');

    // Table
    body.innerHTML = sectorList.map(s => `<tr>
        <td style="font-weight:600;font-size:12px">${s.sector || 'Other'}</td>
        <td class="${chgClass(s.avg_change_pct)}" style="font-weight:600">${fmtPct(s.avg_change_pct)}</td>
        <td style="color:#00d4aa">${s.long_buildup || 0}</td>
        <td style="color:#ff5252">${s.short_buildup || 0}</td>
        <td style="color:#ff9100">${s.long_unwinding || 0}</td>
        <td style="color:#64b5f6">${s.short_covering || 0}</td>
        <td style="font-weight:600;color:${biasColor(s.bias)}">${s.bias || 'NEUTRAL'}</td>
        <td style="font-size:11px;color:var(--text-muted)">${s.stock_count || 0}</td>
    </tr>`).join('');
}

/* ════════════════════════════════════════════
   LIVE STREAM (legacy WebSocket data)
   ════════════════════════════════════════════ */
function renderLiveOIData(data) {
    document.getElementById('oi-nifty-spot').textContent = formatNum(data.spot_price || 0);
    document.getElementById('oi-nifty-atm').textContent = formatNum(data.atm_strike || 0, 0);

    const pcr = data.pcr_oi || 0;
    const pcrEl = document.getElementById('oi-nifty-pcr');
    pcrEl.textContent = formatNum(pcr, 3);
    pcrEl.className = `metric-value ${pcr > 1 ? 'positive' : pcr < 0.7 ? 'negative' : 'neutral'}`;

    document.getElementById('oi-nifty-ce-total').textContent = formatLargeNum(data.total_ce_oi || 0);
    document.getElementById('oi-nifty-pe-total').textContent = formatLargeNum(data.total_pe_oi || 0);
    document.getElementById('oi-nifty-ce-vol').textContent = formatLargeNum(data.total_ce_volume || 0);
    document.getElementById('oi-nifty-pe-vol').textContent = formatLargeNum(data.total_pe_volume || 0);

    renderNearATMTable(data.near_atm_data || [], 'CE', 'oi-atm-ce-body');
    renderNearATMTable(data.near_atm_data || [], 'PE', 'oi-atm-pe-body');

    const allActive = [...(data.most_active_ce || []), ...(data.most_active_pe || [])];
    allActive.sort((a, b) => (b.activity || 0) - (a.activity || 0));
    renderMostActiveTable(allActive.slice(0, 15));

    const allLosers = [...(data.oi_losers_ce || []), ...(data.oi_losers_pe || [])];
    allLosers.sort((a, b) => (a.oi_change || 0) - (b.oi_change || 0));
    // Show entries with most negative OI change; include zeros if no negatives yet
    const negLosers = allLosers.filter(l => l.oi_change < 0);
    renderOILosersTable((negLosers.length > 0 ? negLosers : allLosers).slice(0, 15));

    renderBuildupSignals(data.oi_buildup_signals || []);
}

function renderNearATMTable(entries, optType, elId) {
    const el = document.getElementById(elId);
    const filtered = entries.filter(e => e.type === optType);
    if (filtered.length === 0) {
        el.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    filtered.sort((a, b) => a.strike - b.strike);
    el.innerHTML = filtered.map(e => {
        const oiChgClass = e.oi_change > 0 ? 'positive' : e.oi_change < 0 ? 'negative' : 'neutral';
        const ltpChgClass = e.ltp_change > 0 ? 'positive' : e.ltp_change < 0 ? 'negative' : 'neutral';
        const isAtm = e.atm_dist === 0;
        const rowStyle = isAtm ? 'background:rgba(29,161,242,0.1);font-weight:600' : '';
        return `
            <tr style="${rowStyle}">
                <td>${formatNum(e.strike, 0)}${isAtm ? ' ATM' : ''}</td>
                <td>${formatLargeNum(e.oi)}</td>
                <td class="${oiChgClass}">${e.oi_change > 0 ? '+' : ''}${formatLargeNum(e.oi_change)}</td>
                <td class="${oiChgClass}">${e.oi_change_pct > 0 ? '+' : ''}${formatNum(e.oi_change_pct)}%</td>
                <td>${formatLargeNum(e.volume)}</td>
                <td>${formatNum(e.ltp)}</td>
                <td class="${ltpChgClass}">${formatNum(e.ltp_change)}</td>
            </tr>
        `;
    }).join('');
}

function renderMostActiveTable(entries) {
    const el = document.getElementById('oi-most-active-body');
    if (!entries || entries.length === 0) {
        el.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    el.innerHTML = entries.map(e => `
        <tr>
            <td style="font-size:11px">${e.symbol || ''}</td>
            <td><span class="type-tag ${e.type === 'CE' ? 'type-equity' : 'type-option'}">${e.type}</span></td>
            <td>${formatNum(e.strike, 0)}</td>
            <td>${formatLargeNum(e.oi)}</td>
            <td>${formatLargeNum(e.volume)}</td>
            <td style="font-weight:600">${formatLargeNum(e.activity)}</td>
        </tr>
    `).join('');
}

function renderOILosersTable(entries) {
    const el = document.getElementById('oi-losers-body');
    if (!entries || entries.length === 0) {
        el.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    el.innerHTML = entries.map(e => `
        <tr>
            <td style="font-size:11px">${e.symbol || ''}</td>
            <td><span class="type-tag ${e.type === 'CE' ? 'type-equity' : 'type-option'}">${e.type}</span></td>
            <td>${formatNum(e.strike, 0)}</td>
            <td>${formatLargeNum(e.oi)}</td>
            <td class="negative">${formatLargeNum(e.oi_change)}</td>
            <td class="negative">${formatNum(e.oi_change_pct)}%</td>
        </tr>
    `).join('');
}

function renderBuildupSignals(signals) {
    const el = document.getElementById('oi-buildup-signals');
    if (!signals || signals.length === 0) {
        el.innerHTML = '<div class="empty-state"><p>No buildup signals detected yet</p></div>';
        return;
    }
    el.innerHTML = signals.map(s => {
        const isBullish = s.sentiment === 'BULLISH';
        const borderColor = isBullish ? 'var(--accent-green)' : 'var(--accent-red)';
        const sentColor = isBullish ? 'positive' : 'negative';
        return `
            <div class="signal-entry" style="border-left-color:${borderColor}">
                <div class="signal-header">
                    <strong>${s.symbol} <span class="${sentColor}">${s.signal}</span></strong>
                    <span class="type-tag ${s.type === 'CE' ? 'type-equity' : 'type-option'}" style="margin-left:6px">${s.type}</span>
                </div>
                <div class="signal-detail">
                    Strike: ${formatNum(s.strike, 0)} |
                    OI Change: <span class="${s.oi_change > 0 ? 'positive' : 'negative'}">${s.oi_change > 0 ? '+' : ''}${formatLargeNum(s.oi_change)} (${s.oi_change_pct > 0 ? '+' : ''}${formatNum(s.oi_change_pct)}%)</span> |
                    LTP: ${formatNum(s.ltp)} |
                    Vol: ${formatLargeNum(s.volume)} |
                    <span class="${sentColor}" style="font-weight:600">${s.sentiment}</span>
                </div>
            </div>
        `;
    }).join('');
}

let analysisData = null;

function switchAnalysisView(view) {
    document.querySelectorAll('.analysis-view').forEach(v => v.classList.remove('active'));
    document.querySelectorAll('.analysis-view-btn').forEach(b => b.classList.remove('active'));
    const el = document.getElementById('analysis-view-' + view);
    if (el) el.classList.add('active');
    const btn = document.querySelector(`.analysis-view-btn[data-view="${view}"]`);
    if (btn) btn.classList.add('active');
    // Load sector data on-demand
    if (view === 'sectors' && !document.getElementById('analysis-sectors-body').innerHTML.trim()) {
        loadSectorAnalysis();
    }
}

async function loadAnalysis() {
    try {
        const data = await API.get('/api/analysis/results');
        if (data.scanned) {
            analysisData = data;
            renderAnalysis(data);
        }
    } catch (e) {}
}

async function runAnalysisScan() {
    const btn = document.getElementById('btn-run-scan');
    const status = document.getElementById('scan-status');
    const universe = document.getElementById('scan-universe').value || 'nifty50';
    const label = universe === 'nifty500' ? 'NIFTY 500' : 'NIFTY 50';
    const estTime = universe === 'nifty500' ? '~5 minutes' : '~30 seconds';
    btn.disabled = true;
    btn.textContent = 'Scanning...';
    status.textContent = `Running deep analysis on ${label} stocks (${estTime})...`;
    status.style.color = '#ff9100';
    try {
        const data = await API.post('/api/analysis/scan', { universe });
        if (data.permission_error) {
            status.textContent = data.error || 'Historical Data API not enabled on your Kite Connect app.';
            status.style.color = '#ff9100';
            showToast('Historical Data API subscription required', 'error');
        } else if (data.auth_error) {
            status.textContent = data.error || 'Kite session expired. Please re-login to Zerodha.';
            status.style.color = '#ff5252';
            showToast('Session expired - please re-login to Zerodha', 'error');
        } else if (data.scanned) {
            analysisData = data;
            renderAnalysis(data);
            status.textContent = `Deep analysis complete: ${data.total_stocks} stocks analyzed`;
            status.style.color = '#00d4aa';
            showToast(`${label} scan complete`, 'success');
        } else {
            status.textContent = data.error || data.message || 'Scan failed';
            status.style.color = '#ff5252';
        }
    } catch (e) {
        status.textContent = 'Scan failed: ' + e.message;
        status.style.color = '#ff5252';
        showToast('Scan failed', 'error');
    }
    btn.disabled = false;
    btn.textContent = 'Run Full Scan';
}

function fmtPct(v) { return v != null ? (v >= 0 ? '+' : '') + v.toFixed(2) + '%' : '--'; }
function fmtNum(v) { return v != null ? v.toFixed(2) : '--'; }
function pctClass(v) { return v > 0 ? 'text-green' : v < 0 ? 'text-red' : ''; }
function gradeClass(g) { return g === 'A+' || g === 'A' ? 'text-green' : g === 'B' ? 'text-yellow' : 'text-red'; }
function stageColor(st) { return st === 2 ? '#00d4aa' : st === 1 ? '#ffe066' : st === 3 ? '#ff9100' : st === 4 ? '#ff5252' : '#8899aa'; }
function accumBadge(acc) {
    if (!acc) return '<span style="color:var(--text-muted);font-size:11px">--</span>';
    const s = acc.status;
    if (s === 'strong_accumulation') return '<span style="color:#00d4aa;font-weight:700;font-size:11px">++ Accum</span>';
    if (s === 'accumulation') return '<span style="color:#00d4aa;font-size:11px">+ Accum</span>';
    if (s === 'strong_distribution') return '<span style="color:#ff5252;font-weight:700;font-size:11px">-- Distrib</span>';
    if (s === 'distribution') return '<span style="color:#ff5252;font-size:11px">- Distrib</span>';
    return '<span style="color:var(--text-muted);font-size:11px">Neutral</span>';
}
function detailBtn(sym) {
    return `<button class="btn btn-outline" style="font-size:10px;padding:2px 8px" onclick="openDeepProfile('${sym}')">View</button>`;
}

function renderTriggerBadges(triggers) {
    if (!triggers || !triggers.length) return '<span style="color:var(--text-muted);font-size:11px">None</span>';
    return triggers.slice(0, 4).map(t => {
        const color = t.type === 'bullish' ? '#00d4aa' : t.type === 'bearish' ? '#ff5252' : '#ff9100';
        const icon = t.strength === 'strong' ? '●' : '○';
        return `<span style="display:inline-block;background:${color}22;color:${color};padding:1px 6px;border-radius:3px;font-size:10px;margin:1px">${icon} ${t.signal}</span>`;
    }).join(' ') + (triggers.length > 4 ? `<span style="color:var(--text-muted);font-size:10px"> +${triggers.length-4} more</span>` : '');
}

function renderAnalysis(data) {
    if (data.scan_time) {
        document.getElementById('scan-time').textContent = 'Last scan: ' + new Date(data.scan_time).toLocaleString();
    }

    // Super Performers (enhanced)
    const superBody = document.getElementById('analysis-super-body');
    if (data.super_performers && data.super_performers.length) {
        superBody.innerHTML = data.super_performers.map(s => `<tr>
            <td><strong>${s.symbol}</strong></td>
            <td style="font-size:11px;color:var(--text-muted)">${s.sector || '--'}</td>
            <td>${fmtNum(s.ltp)}</td>
            <td><span class="${gradeClass(s.super_performance?.grade)}" style="font-weight:700">${s.super_performance?.grade || '--'}</span></td>
            <td>${s.super_performance?.score || 0}/${s.super_performance?.total || 13}</td>
            <td style="color:${stageColor(s.stage)}">${s.stage_name || 'S' + (s.stage||'?')}</td>
            <td style="font-weight:600">${s.rs_rating != null ? s.rs_rating.toFixed(0) : '--'}</td>
            <td>${fmtNum(s.trend_score)}</td>
            <td class="${pctClass(s.rsi_14 > 50 ? 1 : -1)}">${fmtNum(s.rsi_14)}</td>
            <td>${fmtNum(s.adx_14)}</td>
            <td class="${pctClass(s.change_1d)}">${fmtPct(s.change_1d)}</td>
            <td>${accumBadge(s.accumulation)}</td>
            <td>${renderTriggerBadges(s.triggers)}</td>
            <td>${detailBtn(s.symbol)}</td>
        </tr>`).join('');
    } else {
        superBody.innerHTML = '<tr><td colspan="14" style="color:var(--text-muted);text-align:center">No super performance stocks found in current scan</td></tr>';
    }

    // Fallback: show top trending if no super performers
    const topTrendBody = data.top_by_trend_score || [];
    if (!data.super_performers?.length && topTrendBody.length) {
        superBody.innerHTML += '<tr><td colspan="14" style="color:#ff9100;text-align:center;font-size:12px;padding-top:12px">Top Trending Stocks (by score):</td></tr>';
        superBody.innerHTML += topTrendBody.slice(0, 5).map(s => `<tr>
            <td><strong>${s.symbol}</strong></td>
            <td style="font-size:11px;color:var(--text-muted)">${s.sector || '--'}</td>
            <td>${fmtNum(s.ltp)}</td>
            <td><span class="${gradeClass(s.super_performance?.grade)}">${s.super_performance?.grade || '--'}</span></td>
            <td>${s.super_performance?.score || 0}/${s.super_performance?.total || 13}</td>
            <td style="color:${stageColor(s.stage)}">${s.stage_name || '--'}</td>
            <td>${s.rs_rating != null ? s.rs_rating.toFixed(0) : '--'}</td>
            <td>${fmtNum(s.trend_score)}</td>
            <td>${fmtNum(s.rsi_14)}</td>
            <td>${fmtNum(s.adx_14)}</td>
            <td class="${pctClass(s.change_1d)}">${fmtPct(s.change_1d)}</td>
            <td>${accumBadge(s.accumulation)}</td>
            <td>${renderTriggerBadges(s.triggers)}</td>
            <td>${detailBtn(s.symbol)}</td>
        </tr>`).join('');
    }

    // Stage 2 Advancing
    renderStage2(data.stage_2_advancing);
    // RS Leaders
    renderRSLeaders(data.rs_leaders);
    // VCP Setups
    renderVCPSetups(data.vcp_setups);
    // Accumulation / Distribution
    renderAccumulation(data.accumulating, data.distributing);

    // Existing views
    renderGainersLosers('analysis-daily-gainers', data.daily?.top_gainers, 'change_1d');
    renderGainersLosers('analysis-daily-losers', data.daily?.top_losers, 'change_1d');
    renderGainersLosers('analysis-weekly-gainers', data.weekly?.top_gainers, 'change_5d');
    renderGainersLosers('analysis-weekly-losers', data.weekly?.top_losers, 'change_5d');
    renderMonthly('analysis-monthly-gainers', data.monthly?.top_gainers);
    renderMonthly('analysis-monthly-losers', data.monthly?.top_losers);
    renderTriggers('analysis-bullish-triggers', data.bullish_triggers, 'bullish');
    renderTriggers('analysis-bearish-triggers', data.bearish_triggers, 'bearish');
    renderVolumeSurges(data.volume_surges);
    render52w(data.near_52w_high, data.near_52w_low);
    renderAllStocks(data.all_stocks);
}

function renderStage2(stocks) {
    const el = document.getElementById('analysis-stage2-body');
    if (!el) return;
    if (!stocks || !stocks.length) {
        el.innerHTML = '<tr><td colspan="13" style="color:var(--text-muted);text-align:center">No Stage 2 stocks found</td></tr>';
        return;
    }
    el.innerHTML = stocks.map(s => `<tr>
        <td><strong>${s.symbol}</strong></td>
        <td style="font-size:11px;color:var(--text-muted)">${s.sector || '--'}</td>
        <td>${fmtNum(s.ltp)}</td>
        <td style="font-weight:600">${s.rs_rating != null ? s.rs_rating.toFixed(0) : '--'}</td>
        <td>${fmtNum(s.trend_score)}</td>
        <td>${fmtNum(s.adx_14)}</td>
        <td>${s.vol_ratio ? s.vol_ratio.toFixed(1) + 'x' : '--'}</td>
        <td class="${pctClass(s.change_1d)}">${fmtPct(s.change_1d)}</td>
        <td class="${pctClass(s.change_20d)}">${fmtPct(s.change_20d)}</td>
        <td class="${pctClass(s.change_60d)}">${fmtPct(s.change_60d)}</td>
        <td>${s.vcp_detected ? '<span style="color:#e040fb;font-weight:700">VCP ✓</span>' : '--'}</td>
        <td>${accumBadge(s.accumulation)}</td>
        <td>${detailBtn(s.symbol)}</td>
    </tr>`).join('');
}

function renderRSLeaders(stocks) {
    const el = document.getElementById('analysis-rs-body');
    if (!el) return;
    if (!stocks || !stocks.length) {
        el.innerHTML = '<tr><td colspan="11" style="color:var(--text-muted);text-align:center">No RS data available</td></tr>';
        return;
    }
    el.innerHTML = stocks.map(s => `<tr>
        <td><strong>${s.symbol}</strong></td>
        <td style="font-size:11px;color:var(--text-muted)">${s.sector || '--'}</td>
        <td>${fmtNum(s.ltp)}</td>
        <td style="font-weight:700;color:${s.rs_rating >= 80 ? '#00d4aa' : s.rs_rating >= 50 ? '#ffe066' : '#ff5252'}">${s.rs_rating != null ? s.rs_rating.toFixed(0) : '--'}</td>
        <td style="color:${stageColor(s.stage)}">${s.stage_name || '--'}</td>
        <td>${fmtNum(s.trend_score)}</td>
        <td class="${pctClass(s.change_1d)}">${fmtPct(s.change_1d)}</td>
        <td class="${pctClass(s.change_20d)}">${fmtPct(s.change_20d)}</td>
        <td class="${pctClass(s.change_60d)}">${fmtPct(s.change_60d)}</td>
        <td class="${pctClass(s.change_250d)}">${fmtPct(s.change_250d)}</td>
        <td>${detailBtn(s.symbol)}</td>
    </tr>`).join('');
}

function renderVCPSetups(stocks) {
    const el = document.getElementById('analysis-vcp-body');
    if (!el) return;
    if (!stocks || !stocks.length) {
        el.innerHTML = '<tr><td colspan="11" style="color:var(--text-muted);text-align:center">No VCP patterns detected</td></tr>';
        return;
    }
    el.innerHTML = stocks.map(s => `<tr>
        <td><strong>${s.symbol}</strong></td>
        <td style="font-size:11px;color:var(--text-muted)">${s.sector || '--'}</td>
        <td>${fmtNum(s.ltp)}</td>
        <td style="color:#e040fb;font-weight:600">${fmtNum(s.vcp_pivot)}</td>
        <td>${s.vcp_tightness != null ? (s.vcp_tightness * 100).toFixed(0) + '%' : '--'}</td>
        <td style="font-weight:600">${s.rs_rating != null ? s.rs_rating.toFixed(0) : '--'}</td>
        <td style="color:${stageColor(s.stage)}">${s.stage_name || '--'}</td>
        <td>${s.vol_ratio ? s.vol_ratio.toFixed(1) + 'x' : '--'}</td>
        <td>${s.pocket_pivot ? '<span style="color:#00d4aa;font-weight:700">PP ✓</span>' : '--'}</td>
        <td>${accumBadge(s.accumulation)}</td>
        <td>${detailBtn(s.symbol)}</td>
    </tr>`).join('');
}

function renderAccumulation(accumulating, distributing) {
    const accEl = document.getElementById('analysis-accumulating-body');
    const distEl = document.getElementById('analysis-distributing-body');
    if (accEl) {
        if (accumulating && accumulating.length) {
            accEl.innerHTML = accumulating.map(s => `<tr>
                <td><strong>${s.symbol}</strong></td>
                <td>${fmtNum(s.ltp)}</td>
                <td style="color:#00d4aa;font-weight:700">${s.accumulation?.score || 0}</td>
                <td>${s.obv_trend || '--'}</td>
                <td>${s.mfi_14 != null ? s.mfi_14.toFixed(0) : '--'}</td>
                <td style="font-size:10px">${(s.accumulation?.signals || []).slice(0,3).join('; ')}</td>
                <td>${detailBtn(s.symbol)}</td>
            </tr>`).join('');
        } else {
            accEl.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center">No accumulation signals</td></tr>';
        }
    }
    if (distEl) {
        if (distributing && distributing.length) {
            distEl.innerHTML = distributing.map(s => `<tr>
                <td><strong>${s.symbol}</strong></td>
                <td>${fmtNum(s.ltp)}</td>
                <td style="color:#ff5252;font-weight:700">${s.accumulation?.score || 0}</td>
                <td>${s.obv_trend || '--'}</td>
                <td>${s.mfi_14 != null ? s.mfi_14.toFixed(0) : '--'}</td>
                <td style="font-size:10px">${(s.accumulation?.signals || []).slice(0,3).join('; ')}</td>
                <td>${detailBtn(s.symbol)}</td>
            </tr>`).join('');
        } else {
            distEl.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center">No distribution signals</td></tr>';
        }
    }
}

async function loadSectorAnalysis() {
    try {
        const data = await API.get('/api/analysis/sectors');
        if (data.error) {
            document.getElementById('analysis-sectors-body').innerHTML = `<tr><td colspan="10" style="color:#ff5252;text-align:center">${data.error}</td></tr>`;
            return;
        }
        renderSectors(data);
    } catch (e) {
        document.getElementById('analysis-sectors-body').innerHTML = '<tr><td colspan="10" style="color:#ff5252;text-align:center">Failed to load sectors</td></tr>';
    }
}

function renderSectors(data) {
    const sectors = data.sectors || {};
    // Heatmap
    const heatmap = document.getElementById('sector-heatmap');
    if (heatmap) {
        heatmap.innerHTML = Object.entries(sectors).map(([name, s]) => {
            const chg = s.avg_change_20d || 0;
            const bg = chg > 2 ? '#00d4aa' : chg > 0 ? '#00d4aa88' : chg > -2 ? '#ff525288' : '#ff5252';
            return `<div style="background:${bg}22;border:1px solid ${bg};border-radius:8px;padding:12px 16px;min-width:120px;text-align:center">
                <div style="font-weight:700;color:${bg}">${name}</div>
                <div style="font-size:20px;font-weight:700;color:${bg}">${chg >= 0 ? '+' : ''}${chg.toFixed(1)}%</div>
                <div style="font-size:11px;color:var(--text-muted)">${s.stock_count} stocks · RS ${s.avg_rs_rating?.toFixed(0) || '--'}</div>
                <div style="font-size:10px;color:var(--text-muted)">Trend: ${s.avg_trend_score?.toFixed(0) || '--'}</div>
            </div>`;
        }).join('');
    }
    // Table
    const tbody = document.getElementById('analysis-sectors-body');
    if (tbody) {
        tbody.innerHTML = Object.entries(sectors).map(([name, s]) => `<tr>
            <td><strong>${name}</strong></td>
            <td>${s.stock_count}</td>
            <td class="${pctClass(s.avg_change_1d)}">${fmtPct(s.avg_change_1d)}</td>
            <td class="${pctClass(s.avg_change_5d)}">${fmtPct(s.avg_change_5d)}</td>
            <td class="${pctClass(s.avg_change_20d)}">${fmtPct(s.avg_change_20d)}</td>
            <td class="${pctClass(s.avg_change_60d)}">${fmtPct(s.avg_change_60d)}</td>
            <td style="font-weight:600">${s.avg_rs_rating?.toFixed(0) || '--'}</td>
            <td>${s.avg_trend_score?.toFixed(0) || '--'}</td>
            <td>${s.super_performers || 0}</td>
            <td style="color:#00d4aa">${s.best_stock?.symbol || '--'} (${s.best_stock?.trend_score?.toFixed(0) || '--'})</td>
        </tr>`).join('');
    }
}

function renderGainersLosers(id, stocks, changeKey) {
    const el = document.getElementById(id);
    if (!el) return;
    if (!stocks || !stocks.length) {
        el.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    el.innerHTML = stocks.map(s => `<tr>
        <td><strong>${s.symbol}</strong></td>
        <td>${fmtNum(s.ltp)}</td>
        <td class="${pctClass(s[changeKey])}" style="font-weight:600">${fmtPct(s[changeKey])}</td>
        <td>${fmtNum(s.rsi_14)}</td>
        <td>${s.vol_ratio ? s.vol_ratio.toFixed(1) + 'x' : '--'}</td>
        <td>${fmtNum(s.trend_score)}</td>
    </tr>`).join('');
}

function renderMonthly(id, stocks) {
    const el = document.getElementById(id);
    if (!el) return;
    if (!stocks || !stocks.length) {
        el.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    el.innerHTML = stocks.map(s => `<tr>
        <td><strong>${s.symbol}</strong></td>
        <td>${fmtNum(s.ltp)}</td>
        <td class="${pctClass(s.change_20d)}" style="font-weight:600">${fmtPct(s.change_20d)}</td>
        <td>${fmtNum(s.sma_50)}</td>
        <td>${fmtNum(s.sma_200)}</td>
        <td>${fmtNum(s.trend_score)}</td>
    </tr>`).join('');
}

function renderTriggers(id, stocks, type) {
    const el = document.getElementById(id);
    if (!el) return;
    if (!stocks || !stocks.length) {
        el.innerHTML = '<tr><td colspan="5" style="color:var(--text-muted);text-align:center">No triggers detected</td></tr>';
        return;
    }
    el.innerHTML = stocks.map(s => {
        const filtered = (s.triggers || []).filter(t => t.type === type);
        return filtered.map(t => `<tr>
            <td><strong>${s.symbol}</strong></td>
            <td>${fmtNum(s.ltp)}</td>
            <td class="${pctClass(s.change_1d)}">${fmtPct(s.change_1d)}</td>
            <td>${t.signal}</td>
            <td><span style="color:${t.strength === 'strong' ? '#ff9100' : '#8899aa'};font-weight:600">${t.strength.toUpperCase()}</span></td>
        </tr>`).join('');
    }).join('');
}

function renderVolumeSurges(stocks) {
    const el = document.getElementById('analysis-volume-body');
    if (!el) return;
    if (!stocks || !stocks.length) {
        el.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center">No volume surges</td></tr>';
        return;
    }
    el.innerHTML = stocks.map(s => `<tr>
        <td><strong>${s.symbol}</strong></td>
        <td>${fmtNum(s.ltp)}</td>
        <td class="${pctClass(s.change_1d)}">${fmtPct(s.change_1d)}</td>
        <td style="color:#ff9100;font-weight:600">${s.vol_ratio ? s.vol_ratio.toFixed(1) + 'x' : '--'}</td>
        <td>${fmtNum(s.rsi_14)}</td>
        <td class="${pctClass(s.macd_histogram)}">${fmtNum(s.macd_histogram)}</td>
        <td>${fmtNum(s.trend_score)}</td>
    </tr>`).join('');
}

function render52w(highStocks, lowStocks) {
    const highEl = document.getElementById('analysis-52w-high-body');
    const lowEl = document.getElementById('analysis-52w-low-body');
    if (highEl) {
        if (highStocks && highStocks.length) {
            highEl.innerHTML = highStocks.map(s => `<tr>
                <td><strong>${s.symbol}</strong></td>
                <td>${fmtNum(s.ltp)}</td>
                <td>${fmtNum(s.high_52w)}</td>
                <td class="${pctClass(s.dist_from_52w_high)}">${fmtPct(s.dist_from_52w_high)}</td>
                <td>${fmtNum(s.rsi_14)}</td>
                <td>${fmtNum(s.trend_score)}</td>
            </tr>`).join('');
        } else {
            highEl.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        }
    }
    if (lowEl) {
        if (lowStocks && lowStocks.length) {
            lowEl.innerHTML = lowStocks.map(s => `<tr>
                <td><strong>${s.symbol}</strong></td>
                <td>${fmtNum(s.ltp)}</td>
                <td>${fmtNum(s.low_52w)}</td>
                <td class="${pctClass(-Math.abs(s.dist_from_52w_low || 0))}">${fmtPct(s.dist_from_52w_low)}</td>
                <td>${fmtNum(s.rsi_14)}</td>
                <td>${fmtNum(s.trend_score)}</td>
            </tr>`).join('');
        } else {
            lowEl.innerHTML = '<tr><td colspan="6" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        }
    }
}

function renderAllStocks(allStocks) {
    const el = document.getElementById('analysis-all-body');
    if (!el) return;
    if (!allStocks || !Object.keys(allStocks).length) {
        el.innerHTML = '<tr><td colspan="15" style="color:var(--text-muted);text-align:center">No data</td></tr>';
        return;
    }
    const sorted = Object.values(allStocks).sort((a, b) => (b.trend_score || 0) - (a.trend_score || 0));
    el.innerHTML = sorted.map(s => `<tr>
        <td><strong>${s.symbol}</strong></td>
        <td style="font-size:11px;color:var(--text-muted)">${s.sector || '--'}</td>
        <td>${fmtNum(s.ltp)}</td>
        <td class="${pctClass(s.change_1d)}">${fmtPct(s.change_1d)}</td>
        <td class="${pctClass(s.change_5d)}">${fmtPct(s.change_5d)}</td>
        <td class="${pctClass(s.change_20d)}">${fmtPct(s.change_20d)}</td>
        <td style="color:${stageColor(s.stage)};font-size:11px">${s.stage_name || '--'}</td>
        <td style="font-weight:600">${s.rs_rating != null ? s.rs_rating.toFixed(0) : '--'}</td>
        <td>${fmtNum(s.rsi_14)}</td>
        <td>${fmtNum(s.adx_14)}</td>
        <td>${s.vol_ratio ? s.vol_ratio.toFixed(1) + 'x' : '--'}</td>
        <td><span class="${gradeClass(s.super_performance?.grade)}" style="font-weight:700">${s.super_performance?.grade || '--'}</span></td>
        <td>${fmtNum(s.trend_score)}</td>
        <td>${accumBadge(s.accumulation)}</td>
        <td>${detailBtn(s.symbol)}</td>
    </tr>`).join('');
}

// ─── Deep Profile Modal ───────────────────────────────────────

async function openDeepProfile(symbol) {
    const modal = document.getElementById('deep-profile-modal');
    const body = document.getElementById('deep-profile-body');
    const title = document.getElementById('deep-profile-title');
    title.textContent = symbol + ' — Deep Analysis Profile';
    body.innerHTML = '<div style="text-align:center;padding:40px;color:var(--text-muted)">Loading deep analysis...</div>';
    modal.style.display = 'flex';
    try {
        const data = await API.get('/api/analysis/deep/' + symbol);
        if (data.error) {
            body.innerHTML = `<div style="color:#ff5252;text-align:center">${data.error}</div>`;
            return;
        }
        renderDeepProfile(body, data);
    } catch (e) {
        body.innerHTML = `<div style="color:#ff5252;text-align:center">Failed to load: ${e.message}</div>`;
    }
}

function closeDeepProfile() {
    document.getElementById('deep-profile-modal').style.display = 'none';
}

function renderDeepProfile(el, d) {
    const v = d.verdict || {};
    const verdictColor = v.action?.includes('BUY') ? '#00d4aa' : v.action?.includes('SELL') ? '#ff5252' : '#ff9100';
    el.innerHTML = `
        <!-- Verdict Banner -->
        <div style="background:${verdictColor}15;border:1px solid ${verdictColor};border-radius:8px;padding:16px;margin-bottom:16px;text-align:center">
            <div style="font-size:24px;font-weight:700;color:${verdictColor}">${v.action || 'HOLD'}</div>
            <div style="font-size:13px;color:var(--text-muted)">Confidence: ${v.confidence || 50}% | Bull: ${v.bull_points||0} vs Bear: ${v.bear_points||0}</div>
            ${v.reasons?.length ? `<div style="margin-top:8px;font-size:11px;color:var(--text-muted)">${v.reasons.join(' · ')}</div>` : ''}
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:16px">
            <!-- Stage -->
            <div class="card" style="padding:12px;text-align:center">
                <div style="font-size:11px;color:var(--text-muted)">Weinstein Stage</div>
                <div style="font-size:22px;font-weight:700;color:${stageColor(d.stage)}">${d.stage_name || '--'}</div>
                <div style="font-size:11px;color:var(--text-muted)">Confidence: ${d.stage_confidence || 0}%</div>
            </div>
            <!-- RS Rating -->
            <div class="card" style="padding:12px;text-align:center">
                <div style="font-size:11px;color:var(--text-muted)">RS Rating (vs NIFTY 50)</div>
                <div style="font-size:22px;font-weight:700;color:${(d.rs_rating||50)>=70?'#00d4aa':(d.rs_rating||50)>=40?'#ffe066':'#ff5252'}">${d.rs_rating != null ? d.rs_rating.toFixed(0) : '--'}</div>
                <div style="font-size:11px;color:var(--text-muted)">Scale: 1-99</div>
            </div>
            <!-- Trend Score -->
            <div class="card" style="padding:12px;text-align:center">
                <div style="font-size:11px;color:var(--text-muted)">Trend Score</div>
                <div style="font-size:22px;font-weight:700;color:${(d.trend_score||0)>=60?'#00d4aa':(d.trend_score||0)>=30?'#ffe066':'#ff5252'}">${fmtNum(d.trend_score)}</div>
                <div style="font-size:11px;color:var(--text-muted)">Scale: 0-100</div>
            </div>
        </div>

        <!-- Super Performance -->
        <div class="card" style="padding:12px;margin-bottom:12px">
            <div style="font-weight:600;margin-bottom:8px">Super Performance (Minervini) — Grade: <span class="${gradeClass(d.super_performance?.grade)}" style="font-size:18px">${d.super_performance?.grade || '--'}</span> (${d.super_performance?.score||0}/${d.super_performance?.total||13})</div>
            <div style="display:flex;flex-wrap:wrap;gap:6px">
                ${Object.entries(d.super_performance?.checks || {}).map(([k, v]) => 
                    `<span style="background:${v?'#00d4aa22':'#ff525222'};color:${v?'#00d4aa':'#ff5252'};padding:2px 8px;border-radius:4px;font-size:10px">${v?'✓':'✗'} ${k.replace(/_/g,' ')}</span>`
                ).join('')}
            </div>
        </div>

        <!-- Patterns -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
            <div class="card" style="padding:12px">
                <div style="font-weight:600;margin-bottom:6px">VCP Pattern</div>
                ${d.vcp_detected ? `<div style="color:#e040fb;font-weight:700">Detected ✓</div>
                <div style="font-size:11px;color:var(--text-muted)">Tightness: ${((d.vcp_tightness||0)*100).toFixed(0)}% | Pivot: ₹${fmtNum(d.vcp_pivot)}</div>` 
                : '<div style="color:var(--text-muted)">Not detected</div>'}
            </div>
            <div class="card" style="padding:12px">
                <div style="font-weight:600;margin-bottom:6px">Accumulation</div>
                <div>${accumBadge(d.accumulation)}</div>
                <div style="font-size:10px;color:var(--text-muted);margin-top:4px">${(d.accumulation?.signals || []).join('; ') || 'No signals'}</div>
            </div>
        </div>

        <!-- Multi-Timeframe -->
        <div class="card" style="padding:12px;margin-bottom:12px">
            <div style="font-weight:600;margin-bottom:8px">Multi-Timeframe Analysis</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px">
                <div>
                    <div style="font-size:11px;color:var(--text-muted)">Daily</div>
                    <div class="${pctClass(d.change_1d)}">${fmtPct(d.change_1d)} (1D)</div>
                    <div class="${pctClass(d.change_5d)}">${fmtPct(d.change_5d)} (5D)</div>
                </div>
                <div>
                    <div style="font-size:11px;color:var(--text-muted)">Weekly</div>
                    <div style="color:${d.weekly_trend?.trend==='bullish'?'#00d4aa':d.weekly_trend?.trend==='bearish'?'#ff5252':'#8899aa'}">${d.weekly_trend?.trend || '--'} (${d.weekly_trend?.strength || 0}%)</div>
                    <div class="${pctClass(d.weekly_trend?.change_pct)}">${fmtPct(d.weekly_trend?.change_pct)}</div>
                </div>
                <div>
                    <div style="font-size:11px;color:var(--text-muted)">Monthly</div>
                    <div style="color:${d.monthly_trend?.trend==='bullish'?'#00d4aa':d.monthly_trend?.trend==='bearish'?'#ff5252':'#8899aa'}">${d.monthly_trend?.trend || '--'} (${d.monthly_trend?.strength || 0}%)</div>
                    <div class="${pctClass(d.monthly_trend?.change_pct)}">${fmtPct(d.monthly_trend?.change_pct)}</div>
                </div>
            </div>
        </div>

        <!-- Performance -->
        <div class="card" style="padding:12px;margin-bottom:12px">
            <div style="font-weight:600;margin-bottom:8px">Performance Timeline</div>
            <div style="display:flex;gap:16px;flex-wrap:wrap">
                <div><span style="color:var(--text-muted);font-size:11px">1D</span> <span class="${pctClass(d.change_1d)}" style="font-weight:600">${fmtPct(d.change_1d)}</span></div>
                <div><span style="color:var(--text-muted);font-size:11px">5D</span> <span class="${pctClass(d.change_5d)}" style="font-weight:600">${fmtPct(d.change_5d)}</span></div>
                <div><span style="color:var(--text-muted);font-size:11px">1M</span> <span class="${pctClass(d.change_20d)}" style="font-weight:600">${fmtPct(d.change_20d)}</span></div>
                <div><span style="color:var(--text-muted);font-size:11px">3M</span> <span class="${pctClass(d.change_60d)}" style="font-weight:600">${fmtPct(d.change_60d)}</span></div>
                <div><span style="color:var(--text-muted);font-size:11px">6M</span> <span class="${pctClass(d.change_120d)}" style="font-weight:600">${fmtPct(d.change_120d)}</span></div>
                <div><span style="color:var(--text-muted);font-size:11px">1Y</span> <span class="${pctClass(d.change_250d)}" style="font-weight:600">${fmtPct(d.change_250d)}</span></div>
            </div>
        </div>

        <!-- Key Technicals -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px">
            <div class="card" style="padding:12px">
                <div style="font-weight:600;margin-bottom:8px">Momentum Indicators</div>
                <table style="width:100%;font-size:12px">
                    <tr><td style="color:var(--text-muted)">RSI (14)</td><td style="font-weight:600">${fmtNum(d.rsi_14)}</td></tr>
                    <tr><td style="color:var(--text-muted)">Stochastic K/D</td><td>${fmtNum(d.stoch_k)} / ${fmtNum(d.stoch_d)}</td></tr>
                    <tr><td style="color:var(--text-muted)">Williams %R</td><td>${fmtNum(d.williams_r)}</td></tr>
                    <tr><td style="color:var(--text-muted)">CCI (20)</td><td>${fmtNum(d.cci_20)}</td></tr>
                    <tr><td style="color:var(--text-muted)">ROC (12)</td><td class="${pctClass(d.roc_12)}">${fmtNum(d.roc_12)}</td></tr>
                    <tr><td style="color:var(--text-muted)">MFI (14)</td><td>${fmtNum(d.mfi_14)}</td></tr>
                </table>
            </div>
            <div class="card" style="padding:12px">
                <div style="font-weight:600;margin-bottom:8px">Trend & Volatility</div>
                <table style="width:100%;font-size:12px">
                    <tr><td style="color:var(--text-muted)">ADX (14)</td><td style="font-weight:600">${fmtNum(d.adx_14)}</td></tr>
                    <tr><td style="color:var(--text-muted)">+DI / -DI</td><td>${fmtNum(d.plus_di)} / ${fmtNum(d.minus_di)}</td></tr>
                    <tr><td style="color:var(--text-muted)">Ichimoku</td><td style="color:${d.ichimoku_signal==='bullish'?'#00d4aa':d.ichimoku_signal==='bearish'?'#ff5252':'#8899aa'}">${d.ichimoku_signal || '--'}</td></tr>
                    <tr><td style="color:var(--text-muted)">LR Slope (20)</td><td class="${pctClass(d.lr_slope_20)}">${fmtNum(d.lr_slope_20)}</td></tr>
                    <tr><td style="color:var(--text-muted)">ATR (14)</td><td>${fmtNum(d.atr_14)}</td></tr>
                    <tr><td style="color:var(--text-muted)">BB Bandwidth</td><td>${fmtNum(d.bb_bandwidth)}%</td></tr>
                    <tr><td style="color:var(--text-muted)">Hist. Volatility</td><td>${fmtNum(d.hist_volatility)}%</td></tr>
                </table>
            </div>
        </div>

        <!-- Key Levels -->
        <div class="card" style="padding:12px;margin-bottom:12px">
            <div style="font-weight:600;margin-bottom:8px">Key Levels</div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;font-size:12px">
                <div>
                    <div style="color:var(--text-muted)">Moving Averages</div>
                    <div>SMA 20: ${fmtNum(d.sma_20)}</div>
                    <div>SMA 50: ${fmtNum(d.sma_50)}</div>
                    <div>SMA 150: ${fmtNum(d.sma_150)}</div>
                    <div>SMA 200: ${fmtNum(d.sma_200)}</div>
                    <div>EMA 9: ${fmtNum(d.ema_9)}</div>
                    <div>EMA 21: ${fmtNum(d.ema_21)}</div>
                </div>
                <div>
                    <div style="color:var(--text-muted)">Support / Resistance</div>
                    ${(d.support_resistance?.resistance || []).map(r => `<div style="color:#ff5252">R: ₹${r.toFixed(2)}</div>`).join('')}
                    <div style="color:var(--text-muted)">LTP: ₹${fmtNum(d.ltp)}</div>
                    ${(d.support_resistance?.support || []).map(s => `<div style="color:#00d4aa">S: ₹${s.toFixed(2)}</div>`).join('')}
                </div>
                <div>
                    <div style="color:var(--text-muted)">Pivot Points / 52W</div>
                    <div>PP: ${fmtNum(d.pivot_pp)}</div>
                    <div style="color:#ff5252">R1: ${fmtNum(d.pivot_r1)}</div>
                    <div style="color:#00d4aa">S1: ${fmtNum(d.pivot_s1)}</div>
                    <div>52W High: ${fmtNum(d.high_52w)} (${fmtPct(d.dist_from_52w_high)})</div>
                    <div>52W Low: ${fmtNum(d.low_52w)} (${fmtPct(d.dist_from_52w_low)})</div>
                </div>
            </div>
        </div>

        <!-- Triggers -->
        ${d.triggers?.length ? `<div class="card" style="padding:12px">
            <div style="font-weight:600;margin-bottom:8px">Active Triggers (${d.triggers.length})</div>
            <div>${renderTriggerBadges(d.triggers)}</div>
        </div>` : ''}
    `;
}

async function logoutUser() {
    try {
        await API.post('/auth/logout', {});
    } catch (e) {}
    window.location.href = '/login';
}

// ─── Symbol ↔ Token Resolution ────────────────────────────────

/**
 * Resolve an instrument_token from a tradingsymbol via server API.
 * Keeps the token field locked (readonly) so users can't desync it.
 * @param {string} prefix - element ID prefix: 'paper' reads paper-symbol/exchange/token
 */
async function resolveToken(prefix) {
    const symbolEl = document.getElementById(prefix + '-symbol');
    const exchangeEl = document.getElementById(prefix + '-exchange');
    const tokenEl = document.getElementById(prefix + '-token');
    const statusEl = document.getElementById(prefix + '-token-status');
    const symbol = (symbolEl?.value || '').trim().toUpperCase();
    const exchange = exchangeEl?.value || 'NSE';
    if (!symbol) {
        if (tokenEl) tokenEl.value = '';
        if (statusEl) { statusEl.textContent = '⚠ enter a symbol'; statusEl.style.color = '#ffb74d'; }
        return;
    }
    if (statusEl) { statusEl.textContent = 'resolving...'; statusEl.style.color = '#607d8b'; }
    try {
        const resp = await API.get('/api/market/resolve-token?symbol=' + encodeURIComponent(symbol) + '&exchange=' + exchange);
        if (tokenEl) tokenEl.value = resp.instrument_token;
        if (statusEl) { statusEl.textContent = '✓ ' + exchange + ':' + symbol + ' → ' + resp.instrument_token; statusEl.style.color = '#00d4aa'; }
    } catch (e) {
        if (tokenEl) tokenEl.value = '';
        if (statusEl) { statusEl.textContent = '✗ not found'; statusEl.style.color = '#ff5252'; }
    }
}

// ─── Paper Trading ────────────────────────────────────────────

let paperEquityChart = null;

async function runPaperTrade() {
    const btn = document.getElementById('btn-run-paper');
    const loading = document.getElementById('paper-loading');
    btn.disabled = true;
    loading.style.display = 'block';
    loading.style.color = 'var(--accent-green)';
    loading.textContent = 'Fetching historical data & running simulation...';
    const strategy = document.getElementById('paper-strategy').value;
    const isFnO = isFnOStrategy(strategy);

    try {
        let result;
        if (isFnOCustomStrategy(strategy)) {
            // Custom F&O strategies route through the builder test endpoint
            result = await API.post('/api/fno-builder/test', {
                name: stripFnOCustomPrefix(strategy),
                underlying: document.getElementById('paper-underlying')?.value || 'NIFTY',
                bars: 500,
                capital: parseFloat(document.getElementById('paper-capital').value) || 500000,
            });
            if (result && result.error) {
                loading.textContent = 'Error: ' + result.error;
                loading.style.color = '#ff5252';
                return;
            }
            displayFnOPaperResult(result);
        } else if (isFnO) {
            result = await API.post('/api/fno-paper-trade/run', {
                strategy: stripFnOPrefix(strategy),
                underlying: document.getElementById('paper-underlying').value,
                capital: parseFloat(document.getElementById('paper-capital').value) || 500000,
                days: parseInt(document.getElementById('paper-days').value) || 60,
                delta_target: parseFloat(document.getElementById('paper-delta').value) || 0.16,
                max_positions: parseInt(document.getElementById('paper-max-pos').value) || 3,
                profit_target_pct: parseFloat(document.getElementById('paper-profit-target').value) || 50,
                stop_loss_pct: parseFloat(document.getElementById('paper-stop-loss').value) || 100,
            });
            if (result && result.error) {
                loading.textContent = 'Error: ' + result.error;
                loading.style.color = '#ff5252';
                return;
            }
            displayFnOPaperResult(result);
        } else {
            const symbol = document.getElementById('paper-symbol').value.trim().toUpperCase();
            if (!symbol) {
                loading.textContent = 'Error: Please enter a symbol (e.g. RELIANCE)';
                loading.style.color = '#ff5252';
                return;
            }
            result = await API.post('/api/paper-trade/run', {
                strategy: strategy,
                tradingsymbol: symbol,
                exchange: document.getElementById('paper-exchange').value,
                interval: document.getElementById('paper-interval').value,
                days: parseInt(document.getElementById('paper-days').value) || 60,
                capital: parseFloat(document.getElementById('paper-capital').value) || 100000,
                commission_pct: parseFloat(document.getElementById('paper-commission').value) || 0.03,
                slippage_pct: parseFloat(document.getElementById('paper-slippage').value) || 0.05,
            });
            if (result && result.error) {
                loading.textContent = 'Error: ' + result.error;
                loading.style.color = '#ff5252';
                return;
            }
            displayPaperResult(result);
        }
    } catch (e) {
        loading.textContent = 'Error: ' + (e.message || 'Failed');
        loading.style.color = '#ff5252';
    } finally {
        btn.disabled = false;
    }
}

async function runPaperTradeSample() {
    const btn = document.getElementById('btn-run-paper');
    const loading = document.getElementById('paper-loading');
    btn.disabled = true;
    loading.style.display = 'block';
    loading.style.color = 'var(--accent-green)';
    loading.textContent = 'Running quick test with sample data...';
    const strategy = document.getElementById('paper-strategy').value;
    const isFnO = isFnOStrategy(strategy);

    try {
        let result;
        if (isFnOCustomStrategy(strategy)) {
            result = await API.post('/api/fno-builder/test', {
                name: stripFnOCustomPrefix(strategy),
                underlying: document.getElementById('paper-underlying')?.value || 'NIFTY',
                bars: 500,
                capital: parseFloat(document.getElementById('paper-capital').value) || 500000,
            });
            if (result && result.error) {
                loading.textContent = 'Error: ' + result.error;
                loading.style.color = '#ff5252';
                return;
            }
            displayFnOPaperResult(result);
        } else if (isFnO) {
            result = await API.post('/api/fno-paper-trade/sample', {
                strategy: stripFnOPrefix(strategy),
                underlying: document.getElementById('paper-underlying')?.value || 'NIFTY',
                capital: parseFloat(document.getElementById('paper-capital').value) || 500000,
                bars: 500,
                delta_target: parseFloat(document.getElementById('paper-delta')?.value) || 0.16,
                max_positions: parseInt(document.getElementById('paper-max-pos')?.value) || 3,
                profit_target_pct: parseFloat(document.getElementById('paper-profit-target')?.value) || 50,
                stop_loss_pct: parseFloat(document.getElementById('paper-stop-loss')?.value) || 100,
            });
            if (result && result.error) {
                loading.textContent = 'Error: ' + result.error;
                loading.style.color = '#ff5252';
                return;
            }
            displayFnOPaperResult(result);
        } else {
            result = await API.post('/api/paper-trade/sample', {
                strategy: strategy,
                tradingsymbol: document.getElementById('paper-symbol')?.value?.trim()?.toUpperCase() || 'SAMPLE',
                bars: 500,
                capital: parseFloat(document.getElementById('paper-capital').value) || 100000,
                interval: document.getElementById('paper-interval')?.value || '5minute',
            });
            if (result && result.error) {
                loading.textContent = 'Error: ' + result.error;
                loading.style.color = '#ff5252';
                return;
            }
            displayPaperResult(result);
        }
    } catch (e) {
        loading.textContent = 'Error: ' + (e.message || 'Failed');
        loading.style.color = '#ff5252';
    } finally {
        btn.disabled = false;
    }
}

function displayFnOPaperResult(r) {
    const loading = document.getElementById('paper-loading');
    loading.style.display = 'none';

    if (r.error) {
        loading.style.display = 'block';
        loading.textContent = 'Error: ' + r.error;
        loading.style.color = '#ff5252';
        return;
    }

    const pnlClass = (r.total_pnl||0) >= 0 ? 'positive' : 'negative';
    document.getElementById('paper-summary').innerHTML = `
        <div class="summary-grid">
            <div class="summary-item"><span class="summary-label">Strategy</span><span class="summary-value">${r.strategy_name||r.strategy||'--'}</span></div>
            <div class="summary-item"><span class="summary-label">Underlying</span><span class="summary-value">${r.underlying||'--'}</span></div>
            <div class="summary-item"><span class="summary-label">Initial Capital</span><span class="summary-value">₹${fmtNum(r.initial_capital||0)}</span></div>
            <div class="summary-item"><span class="summary-label">Final Capital</span><span class="summary-value ${pnlClass}">₹${fmtNum(r.final_capital||0)}</span></div>
            <div class="summary-item"><span class="summary-label">Total P&L</span><span class="summary-value ${pnlClass}">₹${fmtNum(r.total_pnl||0)}</span></div>
            <div class="summary-item"><span class="summary-label">Total Costs</span><span class="summary-value negative">₹${fmtNum(r.total_costs||0)}</span></div>
            <div class="summary-item"><span class="summary-label">Return</span><span class="summary-value ${pnlClass}">${fmtPct(r.total_return_pct||0)}</span></div>
            <div class="summary-item"><span class="summary-label">Total Trades</span><span class="summary-value">${r.total_trades||0}</span></div>
            <div class="summary-item"><span class="summary-label">Win Rate</span><span class="summary-value">${fmtPct(r.win_rate||0)}</span></div>
            <div class="summary-item"><span class="summary-label">Max Drawdown</span><span class="summary-value negative">${fmtPct(r.max_drawdown_pct||0)}</span></div>
            <div class="summary-item"><span class="summary-label">Sharpe Ratio</span><span class="summary-value">${(r.sharpe_ratio||0).toFixed(4)}</span></div>
            <div class="summary-item"><span class="summary-label">Profit Factor</span><span class="summary-value">${(r.profit_factor||0).toFixed(2)}</span></div>
        </div>`;

    // Show results section
    const section = document.getElementById('paper-results-section');
    section.style.display = 'block';
    document.getElementById('paper-fno-results').style.display = '';

    // Equity curve
    if (r.equity_curve && r.equity_curve.length > 1) drawEquityCurve(r.equity_curve);

    // Health report
    if (r.health_report) displayHealthReport(r.health_report, 'paper');

    // Greeks info
    const greeks = r.greeks_history || [];
    if (greeks.length > 0) {
        const last = greeks[greeks.length - 1] || {};
        document.getElementById('paper-greeks-info').innerHTML =
            '<div class="grid grid-2" style="gap:4px">' +
            metricCard('Net Delta', (last.net_delta||0).toFixed(4)) +
            metricCard('Net Gamma', (last.net_gamma||0).toFixed(6)) +
            metricCard('Net Theta', '₹' + formatNum(last.net_theta||0)) +
            metricCard('Net Vega', (last.net_vega||0).toFixed(4)) +
            '</div>';
    }

    // Margin info
    const margins = r.margin_history || [];
    if (margins.length > 0) {
        const lastM = margins[margins.length - 1] || {};
        const peakM = Math.max(...margins.map(m => m.margin_used || m.total_margin || 0));
        document.getElementById('paper-margin-info').innerHTML =
            '<div class="grid grid-2" style="gap:4px">' +
            metricCard('Current Margin', '₹' + formatNum(lastM.margin_used||lastM.total_margin||0)) +
            metricCard('Peak Margin', '₹' + formatNum(peakM), '#ffb74d') +
            '</div>';
    }

    // F&O positions table
    const positions = r.positions || r.trades || [];
    const tbody = document.getElementById('paper-fno-positions-body');
    if (positions.length > 0) {
        tbody.innerHTML = positions.map((p, i) => {
            const pnlColor = (p.pnl||0) >= 0 ? '#00d4aa' : '#ff5252';
            const entryDate = (p.entry_time||p.entry||'--').toString().slice(0, 10);
            return '<tr>' +
                '<td>' + (i+1) + '</td>' +
                '<td>' + (p.structure||p.type||'--') + '</td>' +
                '<td>' + (p.legs||p.num_legs||'--') + '</td>' +
                '<td>₹' + formatNum(p.net_premium||0) + '</td>' +
                '<td style="color:' + pnlColor + '">₹' + formatNum(p.pnl||0) + '</td>' +
                '<td>' + (p.regime||'--') + '</td>' +
                '<td>' + entryDate + '</td>' +
                '</tr>';
        }).join('');
    } else {
        tbody.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center">No F&O positions</td></tr>';
    }

    // Also populate equity trade log
    const tradeBody = document.getElementById('paper-trades-body');
    if (positions.length > 0) {
        tradeBody.innerHTML = positions.map(p => {
            const pnlCl = (p.pnl||0) >= 0 ? 'text-green' : 'text-red';
            const entryT = (p.entry_time||p.entry||'--').toString().slice(0, 10);
            const exitT = (p.exit_time||p.exit||'--').toString().slice(0, 10);
            const exitR = p.exit_reason||p.reason||'--';
            return `<tr>
                <td>${p.structure||p.type||'--'}</td>
                <td>₹${fmtNum(p.net_premium||0)}</td>
                <td>${(p.pnl||0) >= 0 ? 'Profit' : 'Loss'}</td>
                <td>${p.qty||p.quantity||'--'}</td>
                <td class="${pnlCl}">₹${fmtNum(p.pnl||0)}</td>
                <td>${entryT}</td>
                <td>${exitT}</td>
                <td>${exitR}</td>
            </tr>`;
        }).join('');
    }
}

function displayPaperResult(r) {
    const loading = document.getElementById('paper-loading');
    loading.style.display = 'none';

    if (r.error) {
        loading.style.display = 'block';
        loading.textContent = 'Error: ' + r.error;
        loading.style.color = '#ff5252';
        return;
    }

    // Summary
    const pnlClass = r.total_pnl >= 0 ? 'positive' : 'negative';
    const paperSrcLabel = r.data_source ? (' <span style="color:#ffaa00">(' + r.data_source + ')</span>') : '';
    document.getElementById('paper-summary').innerHTML = `
        <div class="summary-grid">
            <div class="summary-item"><span class="summary-label">Strategy</span><span class="summary-value">${r.strategy_name}</span></div>
            <div class="summary-item"><span class="summary-label">Symbol</span><span class="summary-value">${r.tradingsymbol}${paperSrcLabel}</span></div>
            <div class="summary-item"><span class="summary-label">Initial Capital</span><span class="summary-value">₹${fmtNum(r.initial_capital)}</span></div>
            <div class="summary-item"><span class="summary-label">Final Capital</span><span class="summary-value ${pnlClass}">₹${fmtNum(r.final_capital)}</span></div>
            <div class="summary-item"><span class="summary-label">Total P&L</span><span class="summary-value ${pnlClass}">₹${fmtNum(r.total_pnl)}</span></div>
            <div class="summary-item"><span class="summary-label">Return</span><span class="summary-value ${pnlClass}">${fmtPct(r.total_return_pct)}</span></div>
            <div class="summary-item"><span class="summary-label">Total Trades</span><span class="summary-value">${r.total_trades}</span></div>
            <div class="summary-item"><span class="summary-label">Win Rate</span><span class="summary-value">${fmtPct(r.win_rate)}</span></div>
            <div class="summary-item"><span class="summary-label">Profit Factor</span><span class="summary-value">${fmtNum(r.profit_factor)}</span></div>
            <div class="summary-item"><span class="summary-label">Max Drawdown</span><span class="summary-value negative">${fmtPct(r.max_drawdown_pct)}</span></div>
            <div class="summary-item"><span class="summary-label">Sharpe Ratio</span><span class="summary-value">${r.sharpe_ratio?.toFixed(4) || '--'}</span></div>
            <div class="summary-item"><span class="summary-label">Avg Win</span><span class="summary-value positive">₹${fmtNum(r.avg_win)}</span></div>
            <div class="summary-item"><span class="summary-label">Avg Loss</span><span class="summary-value negative">₹${fmtNum(r.avg_loss)}</span></div>
            <div class="summary-item"><span class="summary-label">Timeframe</span><span class="summary-value">${r.timeframe}</span></div>
        </div>`;

    // Equity curve chart
    const section = document.getElementById('paper-results-section');
    section.style.display = 'block';

    if (r.equity_curve && r.equity_curve.length > 0) {
        drawEquityCurve(r.equity_curve);
    }

    // Health report
    if (r.health_report) displayHealthReport(r.health_report, 'paper');

    // Trade log
    const tbody = document.getElementById('paper-trades-body');
    const tradeData = r.positions || r.trades || [];
    // For equity backtests, filter to exit trades only (skip entries)
    const exitData = tradeData.filter(p => p.pnl !== undefined && (!p.type || !p.type.includes('ENTRY')));
    if (exitData.length > 0) {
        tbody.innerHTML = exitData.map(p => {
            const pnlCl = p.pnl >= 0 ? 'text-green' : 'text-red';
            const entryVal = p.entry_price || p.net_premium || p.entry;
            const exitVal = p.exit_price || p.exit;
            const entryDisp = (typeof entryVal === 'number' || (typeof entryVal === 'string' && !isNaN(entryVal))) ? '₹' + fmtNum(entryVal) : (entryVal || '--');
            const exitDisp = (typeof exitVal === 'number' || (typeof exitVal === 'string' && !isNaN(exitVal))) ? '₹' + fmtNum(exitVal) : (exitVal || '--');
            return `<tr>
                <td>${p.type||p.structure||'--'}</td>
                <td>${entryDisp}</td>
                <td>${exitDisp}</td>
                <td>${p.qty||p.quantity||'--'}</td>
                <td class="${pnlCl}">₹${fmtNum(p.pnl)}</td>
                <td>${p.entry_time||p.entry||'--'}</td>
                <td>${p.exit_time||p.exit||'--'}</td>
                <td>${p.reason||p.exit_reason||'--'}</td>
            </tr>`;
        }).join('');
    } else {
        tbody.innerHTML = '<tr><td colspan="8" style="color:var(--text-muted);text-align:center">No trades</td></tr>';
    }
}

function drawEquityCurve(data) {
    const canvas = document.getElementById('paper-equity-chart');
    const ctx = canvas.getContext('2d');
    const w = canvas.parentElement.clientWidth - 40;
    const h = 200;
    canvas.width = w;
    canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    if (data.length < 2) return;

    const min = Math.min(...data) * 0.995;
    const max = Math.max(...data) * 1.005;
    const range = max - min || 1;
    const stepX = w / (data.length - 1);

    // Grid lines
    ctx.strokeStyle = '#1a2332';
    ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const y = (i / 4) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
        const val = max - (i / 4) * range;
        ctx.fillStyle = '#667788';
        ctx.font = '10px Inter';
        ctx.fillText('₹' + val.toFixed(0), 4, y - 4);
    }

    // Line
    const lastVal = data[data.length - 1];
    const firstVal = data[0];
    ctx.strokeStyle = lastVal >= firstVal ? '#00d4aa' : '#ff5252';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
        const x = i * stepX;
        const y = h - ((data[i] - min) / range) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Fill
    ctx.lineTo((data.length - 1) * stepX, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = lastVal >= firstVal ? 'rgba(0,212,170,0.08)' : 'rgba(255,82,82,0.08)';
    ctx.fill();
}

// ─── Strategy Builder ─────────────────────────────────────────

let builderIndicators = [];

async function loadBuilderIndicators() {
    if (builderIndicators.length > 0) return;
    try {
        builderIndicators = await API.get('/api/strategy-builder/indicators');
    } catch (e) {
        console.error('Failed to load indicators', e);
    }
    loadCustomStrategiesList();
    loadCustomStrategiesDropdowns();
    initFnOBuilder();
}

function addBuilderRule(type) {
    const container = document.getElementById(`builder-${type}-rules`);
    const ruleId = 'rule_' + Date.now();
    const div = document.createElement('div');
    div.className = 'builder-rule';
    div.id = ruleId;

    const indicatorOpts = builderIndicators.map(ind =>
        `<option value="${ind.id}">${ind.name}</option>`
    ).join('');

    div.innerHTML = `
        <select class="rule-indicator" onchange="updateRuleOptions('${ruleId}')">
            <option value="">Select Indicator</option>
            ${indicatorOpts}
        </select>
        <select class="rule-condition"></select>
        <span class="rule-params-container"></span>
        <span class="rule-value-container"></span>
        <button class="rule-remove" onclick="this.parentElement.remove()">&times;</button>
    `;
    container.appendChild(div);
}

function updateRuleOptions(ruleId) {
    const ruleEl = document.getElementById(ruleId);
    const indId = ruleEl.querySelector('.rule-indicator').value;
    const ind = builderIndicators.find(i => i.id === indId);
    if (!ind) return;

    // Conditions
    const condSelect = ruleEl.querySelector('.rule-condition');
    condSelect.innerHTML = ind.conditions.map(c =>
        `<option value="${c}">${c.replace(/_/g, ' ')}</option>`
    ).join('');

    // Params
    const paramsContainer = ruleEl.querySelector('.rule-params-container');
    paramsContainer.innerHTML = (ind.params || []).map(p =>
        `<label style="font-size:11px;color:var(--text-muted);margin-left:4px">${p.label}:</label>
         <input class="rule-param" data-param="${p.name}" type="number" value="${p.default}" min="${p.min || 1}" max="${p.max || 500}" step="${p.type === 'float' ? 0.1 : 1}">`
    ).join('');

    // Value (for indicators that need a threshold)
    const valContainer = ruleEl.querySelector('.rule-value-container');
    if (ind.value_required) {
        valContainer.innerHTML = `<label style="font-size:11px;color:var(--text-muted);margin-left:4px">${ind.value_label}:</label>
            <input class="rule-value" type="number" value="${ind.value_default}" min="${ind.value_min || 0}" max="${ind.value_max || 100}" step="0.1">`;
    } else {
        valContainer.innerHTML = '';
    }
}

function collectRules(type) {
    const container = document.getElementById(`builder-${type}-rules`);
    const rules = [];
    container.querySelectorAll('.builder-rule').forEach(ruleEl => {
        const indicator = ruleEl.querySelector('.rule-indicator')?.value;
        const condition = ruleEl.querySelector('.rule-condition')?.value;
        if (!indicator || !condition) return;

        const params = {};
        ruleEl.querySelectorAll('.rule-param').forEach(inp => {
            const name = inp.dataset.param;
            params[name] = parseFloat(inp.value) || parseInt(inp.value);
        });

        const valueInput = ruleEl.querySelector('.rule-value');
        const rule = { indicator, condition, params };
        if (valueInput) {
            rule.value = parseFloat(valueInput.value);
        }
        rules.push(rule);
    });
    return rules;
}

function buildStrategyConfig() {
    return {
        name: document.getElementById('builder-name').value.trim().replace(/\s+/g, '_').toLowerCase(),
        description: document.getElementById('builder-desc').value.trim(),
        entry_rules: collectRules('entry'),
        exit_rules: collectRules('exit'),
        entry_logic: document.getElementById('builder-entry-logic').value,
        exit_logic: document.getElementById('builder-exit-logic').value,
        quantity: parseInt(document.getElementById('builder-qty').value) || 10,
        exchange: document.getElementById('builder-exchange').value,
        stop_loss_pct: parseFloat(document.getElementById('builder-sl').value) || 0,
        target_pct: parseFloat(document.getElementById('builder-target').value) || 0,
        use_atr_sl: document.getElementById('builder-atr-sl').checked,
    };
}

async function saveCustomStrategy() {
    const config = buildStrategyConfig();
    if (!config.name) {
        showBuilderStatus('Strategy name is required', true);
        return;
    }
    if (config.entry_rules.length === 0) {
        showBuilderStatus('Add at least one entry rule', true);
        return;
    }

    try {
        const result = await API.post('/api/strategy-builder/save', config);
        if (result.error) {
            showBuilderStatus(result.error, true);
        } else {
            showBuilderStatus('Strategy "' + config.name + '" saved!', false);
            loadCustomStrategiesList();
            loadCustomStrategiesDropdowns();
        }
    } catch (e) {
        showBuilderStatus('Save failed: ' + e.message, true);
    }
}

async function testCustomStrategy() {
    const config = buildStrategyConfig();
    if (!config.name) {
        showBuilderStatus('Strategy name is required', true);
        return;
    }
    if (config.entry_rules.length === 0) {
        showBuilderStatus('Add at least one entry rule', true);
        return;
    }

    showBuilderStatus('Saving & testing with sample data...', false);

    try {
        const result = await API.post('/api/strategy-builder/test', {
            config: config,
            bars: 500,
            capital: 100000,
        });
        if (result.error) {
            showBuilderStatus(result.error, true);
        } else {
            showBuilderStatus('Test complete! Strategy saved.', false);
            loadCustomStrategiesList();
            loadCustomStrategiesDropdowns();
            // Show test result in builder
            const metrics = document.getElementById('builder-test-metrics');
            const section = document.getElementById('builder-test-result');
            section.style.display = 'block';
            const pnlCl = result.total_pnl >= 0 ? 'positive' : 'negative';
            metrics.innerHTML = `
                <div class="summary-grid">
                    <div class="summary-item"><span class="summary-label">Trades</span><span class="summary-value">${result.total_trades}</span></div>
                    <div class="summary-item"><span class="summary-label">Win Rate</span><span class="summary-value">${fmtPct(result.win_rate)}</span></div>
                    <div class="summary-item"><span class="summary-label">P&L</span><span class="summary-value ${pnlCl}">₹${fmtNum(result.total_pnl)}</span></div>
                    <div class="summary-item"><span class="summary-label">Return</span><span class="summary-value ${pnlCl}">${fmtPct(result.total_return_pct)}</span></div>
                    <div class="summary-item"><span class="summary-label">Profit Factor</span><span class="summary-value">${fmtNum(result.profit_factor)}</span></div>
                    <div class="summary-item"><span class="summary-label">Max DD</span><span class="summary-value negative">${fmtPct(result.max_drawdown_pct)}</span></div>
                </div>`;
        }
    } catch (e) {
        showBuilderStatus('Test failed: ' + e.message, true);
    }
}

async function loadCustomStrategiesList() {
    try {
        const strategies = await API.get('/api/strategy-builder/list');
        const container = document.getElementById('custom-strategies-list');
        if (!strategies || strategies.length === 0) {
            container.innerHTML = '<div class="empty-state"><p>No custom strategies saved</p></div>';
            return;
        }
        container.innerHTML = strategies.map(s => `
            <div class="custom-strat-item">
                <div>
                    <div class="cs-name">${s.name}</div>
                    <div class="cs-meta">${s.description || ''} &bull; ${s.entry_rules} entry rules, ${s.exit_rules} exit rules</div>
                </div>
                <div class="cs-actions">
                    <button class="btn btn-outline btn-sm" onclick="paperTradeCustom('${s.name}')">Test</button>
                    <button class="btn btn-outline btn-sm" style="border-color:#ff5252;color:#ff5252" onclick="deleteCustomStrategy('${s.name}')">Delete</button>
                </div>
            </div>
        `).join('');
    } catch (e) {
        console.error('Failed to load custom strategies', e);
    }
}

async function loadCustomStrategiesDropdowns() {
    try {
        const strategies = await API.get('/api/strategy-builder/list');
        const html = (strategies || []).map(s =>
            `<option value="${s.name}">${s.name}</option>`
        ).join('');
        // Paper trading dropdown
        const paperGroup = document.getElementById('paper-custom-strategies');
        if (paperGroup) paperGroup.innerHTML = html;
        // Backtest dropdown
        const btGroup = document.getElementById('bt-custom-strategies');
        if (btGroup) btGroup.innerHTML = html;
        // Add strategy modal dropdown
        const modalGroup = document.getElementById('modal-custom-strategies');
        if (modalGroup) modalGroup.innerHTML = html;
    } catch (e) {}
}

async function deleteCustomStrategy(name) {
    if (!confirm(`Delete strategy "${name}"?`)) return;
    try {
        await API.delete_('/api/strategy-builder/' + encodeURIComponent(name));
        loadCustomStrategiesList();
        loadCustomStrategiesDropdowns();
        showBuilderStatus('Deleted "' + name + '"', false);
    } catch (e) {
        showBuilderStatus('Delete failed: ' + e.message, true);
    }
}

async function paperTradeCustom(name) {
    // Switch to paper trading tab and run
    switchTab('paper-trading');
    document.getElementById('paper-strategy').value = name;
    await runPaperTradeSample();
}

function showBuilderStatus(msg, isError) {
    const el = document.getElementById('builder-status');
    el.style.display = 'block';
    el.style.color = isError ? '#ff5252' : '#00d4aa';
    el.textContent = msg;
    if (!isError) setTimeout(() => { el.style.display = 'none'; }, 5000);
}

// ===== F&O STRATEGY BUILDER =====

const FNO_LOT_SIZES = { NIFTY: 50, BANKNIFTY: 15, FINNIFTY: 25, MIDCPNIFTY: 50, SENSEX: 10 };
const FNO_STRIKE_GAPS = { NIFTY: 50, BANKNIFTY: 100, FINNIFTY: 50, MIDCPNIFTY: 25, SENSEX: 100 };
let fnoLegCounter = 0;

function updateFnOBuilderLotSize() {
    const u = document.getElementById('fno-builder-underlying').value;
    const lot = FNO_LOT_SIZES[u] || 50;
    document.getElementById('fno-builder-lot-info').textContent = `Lot: ${lot}`;
}

function addFnOBuilderLeg(preset) {
    fnoLegCounter++;
    const id = fnoLegCounter;
    const container = document.getElementById('fno-builder-legs');
    const action = preset?.action || 'sell';
    const optType = preset?.option_type || 'CE';
    const mode = preset?.strike_mode || 'offset';
    const val = preset?.strike_value ?? 0;
    const lots = preset?.lots || 1;

    const div = document.createElement('div');
    div.id = `fno-leg-${id}`;
    div.style.cssText = 'display:grid;grid-template-columns:80px 70px 100px 120px 60px 30px;gap:6px;align-items:center;background:#0d1821;padding:8px;border-radius:6px;border:1px solid #2a3a4a';
    div.innerHTML = `
        <select class="fno-leg-action" style="padding:4px 6px;background:#131f2e;border:1px solid #2a3a4a;color:#e0e6ed;border-radius:4px;font-size:12px">
            <option value="buy" ${action==='buy'?'selected':''}>BUY</option>
            <option value="sell" ${action==='sell'?'selected':''}>SELL</option>
        </select>
        <select class="fno-leg-type" style="padding:4px 6px;background:#131f2e;border:1px solid #2a3a4a;color:#e0e6ed;border-radius:4px;font-size:12px">
            <option value="CE" ${optType==='CE'?'selected':''}>CE</option>
            <option value="PE" ${optType==='PE'?'selected':''}>PE</option>
        </select>
        <select class="fno-leg-mode" style="padding:4px 6px;background:#131f2e;border:1px solid #2a3a4a;color:#e0e6ed;border-radius:4px;font-size:12px">
            <option value="absolute" ${mode==='absolute'?'selected':''}>Strike</option>
            <option value="offset" ${mode==='offset'?'selected':''}>ATM Offset</option>
            <option value="delta" ${mode==='delta'?'selected':''}>Delta</option>
        </select>
        <input class="fno-leg-value" type="number" value="${val}" step="${mode==='delta'?'0.01':'50'}" style="padding:4px 6px;background:#131f2e;border:1px solid #2a3a4a;color:#e0e6ed;border-radius:4px;font-size:12px" placeholder="${mode==='delta'?'0.20':'25500'}">
        <input class="fno-leg-lots" type="number" value="${lots}" min="1" max="20" style="padding:4px 6px;background:#131f2e;border:1px solid #2a3a4a;color:#e0e6ed;border-radius:4px;font-size:12px">
        <button onclick="document.getElementById('fno-leg-${id}').remove()" style="background:none;border:none;color:#ff5252;cursor:pointer;font-size:16px" title="Remove leg">✕</button>
    `;
    container.appendChild(div);
}

function collectFnOLegs() {
    const legs = [];
    document.querySelectorAll('#fno-builder-legs > div').forEach(div => {
        const action = div.querySelector('.fno-leg-action')?.value;
        const option_type = div.querySelector('.fno-leg-type')?.value;
        const strike_mode = div.querySelector('.fno-leg-mode')?.value;
        const strike_value = parseFloat(div.querySelector('.fno-leg-value')?.value) || 0;
        const lots = parseInt(div.querySelector('.fno-leg-lots')?.value) || 1;
        if (action && option_type) {
            legs.push({ action, option_type, strike_mode, strike_value, lots });
        }
    });
    return legs;
}

function buildFnOConfig() {
    return {
        name: document.getElementById('fno-builder-name').value.trim().replace(/\s+/g, '_').toLowerCase(),
        description: document.getElementById('fno-builder-desc').value.trim(),
        underlying: document.getElementById('fno-builder-underlying').value,
        strategy_type: document.getElementById('fno-builder-type').value,
        legs: collectFnOLegs(),
        profit_target_pct: parseFloat(document.getElementById('fno-builder-profit').value) || 50,
        stop_loss_pct: parseFloat(document.getElementById('fno-builder-sl').value) || 100,
        max_positions: parseInt(document.getElementById('fno-builder-max-pos').value) || 1,
        lot_size: FNO_LOT_SIZES[document.getElementById('fno-builder-underlying').value] || 50,
        entry_dte_min: parseInt(document.getElementById('fno-builder-dte-min').value) || 15,
        entry_dte_max: parseInt(document.getElementById('fno-builder-dte-max').value) || 45,
    };
}

async function saveFnOStrategy() {
    const config = buildFnOConfig();
    if (!config.name) { showFnOBuilderStatus('Strategy name is required', true); return; }
    if (!config.legs.length) { showFnOBuilderStatus('Add at least one leg', true); return; }
    try {
        const result = await API.post('/api/fno-builder/save', config);
        if (result.error) { showFnOBuilderStatus(result.error, true); return; }
        showFnOBuilderStatus('Saved "' + config.name + '"', false);
        loadFnOCustomStrategiesList();
        loadFnOCustomStrategiesDropdowns();
    } catch (e) {
        showFnOBuilderStatus('Save failed: ' + e.message, true);
    }
}

async function testFnOStrategy() {
    const config = buildFnOConfig();
    if (!config.name) { showFnOBuilderStatus('Strategy name is required', true); return; }
    if (!config.legs.length) { showFnOBuilderStatus('Add at least one leg', true); return; }
    showFnOBuilderStatus('Saving & testing...', false);
    try {
        const result = await API.post('/api/fno-builder/test', { ...config, bars: 500, capital: 500000 });
        if (result.error) { showFnOBuilderStatus('Test error: ' + result.error, true); return; }
        showFnOBuilderStatus('Saved & tested "' + config.name + '"', false);
        loadFnOCustomStrategiesList();
        loadFnOCustomStrategiesDropdowns();
    } catch (e) {
        showFnOBuilderStatus('Test failed: ' + e.message, true);
    }
}

async function computePayoff() {
    const config = buildFnOConfig();
    if (!config.legs.length) { showFnOBuilderStatus('Add at least one leg', true); return; }
    const spot = parseFloat(document.getElementById('fno-builder-spot').value) || 25000;
    try {
        const result = await API.post('/api/fno-builder/payoff', { config, spot_price: spot });
        drawFnOPayoffChart(result, spot);
        const stats = document.getElementById('fno-payoff-stats');
        stats.innerHTML = `
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">
                <div>Max Profit: <strong style="color:#00d4aa">₹${fmtNum(result.max_profit)}</strong></div>
                <div>Max Loss: <strong style="color:#ff5252">₹${fmtNum(result.max_loss)}</strong></div>
                <div>Net Premium: <strong>₹${fmtNum(result.net_premium)}</strong></div>
                <div>Breakeven: <strong>${(result.breakeven || []).map(b => '₹'+b.toFixed(0)).join(', ') || '--'}</strong></div>
            </div>`;
    } catch (e) {
        showFnOBuilderStatus('Payoff failed: ' + e.message, true);
    }
}

function drawFnOPayoffChart(data, spot) {
    const canvas = document.getElementById('fno-payoff-chart');
    const ctx = canvas.getContext('2d');
    const w = canvas.parentElement.clientWidth - 40;
    const h = 220;
    canvas.width = w; canvas.height = h;
    ctx.clearRect(0, 0, w, h);

    const prices = data.prices || [];
    const payoff = data.payoff || [];
    if (!prices.length) return;

    const minP = Math.min(...payoff);
    const maxP = Math.max(...payoff);
    const range = (maxP - minP) || 1;
    const padding = range * 0.1;
    const yMin = minP - padding;
    const yMax = maxP + padding;
    const yRange = yMax - yMin;
    const stepX = w / (prices.length - 1);

    // Grid + zero line
    ctx.strokeStyle = '#1a2332'; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
        const y = (i / 4) * h;
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
    // Zero line
    const zeroY = h - ((0 - yMin) / yRange) * h;
    if (zeroY > 0 && zeroY < h) {
        ctx.strokeStyle = '#667788'; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(0, zeroY); ctx.lineTo(w, zeroY); ctx.stroke();
        ctx.setLineDash([]);
    }
    // Spot price line
    const spotIdx = prices.findIndex(p => p >= spot);
    if (spotIdx > 0) {
        const spotX = spotIdx * stepX;
        ctx.strokeStyle = '#ffaa00'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        ctx.beginPath(); ctx.moveTo(spotX, 0); ctx.lineTo(spotX, h); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#ffaa00'; ctx.font = '10px Inter';
        ctx.fillText('Spot', spotX + 4, 12);
    }

    // Payoff line
    ctx.beginPath();
    for (let i = 0; i < payoff.length; i++) {
        const x = i * stepX;
        const y = h - ((payoff[i] - yMin) / yRange) * h;
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = '#00d4aa'; ctx.lineWidth = 2; ctx.stroke();

    // Profit fill (above zero) and loss fill (below zero)
    for (let i = 0; i < payoff.length - 1; i++) {
        const x1 = i * stepX, x2 = (i + 1) * stepX;
        const y1 = h - ((payoff[i] - yMin) / yRange) * h;
        const y2 = h - ((payoff[i + 1] - yMin) / yRange) * h;
        const z = h - ((0 - yMin) / yRange) * h;
        ctx.fillStyle = payoff[i] >= 0 ? 'rgba(0,212,170,0.12)' : 'rgba(255,82,82,0.12)';
        ctx.beginPath();
        ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineTo(x2, z); ctx.lineTo(x1, z);
        ctx.closePath(); ctx.fill();
    }

    // Labels
    ctx.fillStyle = '#667788'; ctx.font = '10px Inter';
    ctx.fillText('₹' + prices[0].toFixed(0), 4, h - 4);
    ctx.fillText('₹' + prices[prices.length - 1].toFixed(0), w - 60, h - 4);
    ctx.fillText('₹' + maxP.toFixed(0), 4, 12);
    ctx.fillText('₹' + minP.toFixed(0), 4, h - 18);
}

async function loadFnOTemplates() {
    try {
        const data = await API.get('/api/fno-builder/templates');
        const sel = document.getElementById('fno-builder-template');
        if (sel && data.templates) {
            sel.innerHTML = '<option value="">— Select a template —</option>' +
                data.templates.map(t =>
                    `<option value="${t.name}">${t.description}</option>`
                ).join('');
        }
    } catch (e) {}
}

function loadFnOTemplate() {
    const sel = document.getElementById('fno-builder-template');
    const name = sel.value;
    if (!name) return;
    // Fetch templates and populate legs
    API.get('/api/fno-builder/templates').then(data => {
        const tmpl = (data.templates || []).find(t => t.name === name);
        if (!tmpl) return;
        document.getElementById('fno-builder-type').value = tmpl.strategy_type || 'custom';
        // Clear existing legs
        document.getElementById('fno-builder-legs').innerHTML = '';
        fnoLegCounter = 0;
        // Add template legs
        (tmpl.legs || []).forEach(leg => addFnOBuilderLeg(leg));
    }).catch(() => {});
}

async function loadFnOCustomStrategiesList() {
    try {
        const strategies = await API.get('/api/fno-builder/list');
        const container = document.getElementById('fno-custom-strategies-list');
        if (!strategies || !strategies.length) {
            container.innerHTML = '<div class="empty-state"><p>No custom F&O strategies saved</p></div>';
            return;
        }
        container.innerHTML = strategies.map(s => `
            <div style="display:flex;justify-content:space-between;align-items:center;padding:8px;background:#0d1821;border-radius:6px;margin-bottom:6px;border:1px solid #2a3a4a">
                <div>
                    <strong style="color:#ff9100">${s.name}</strong>
                    <span style="color:#667788;font-size:11px;margin-left:8px">${s.underlying} · ${s.strategy_type} · ${(s.legs||[]).length} legs</span>
                    <div style="font-size:11px;color:#8899aa;margin-top:2px">${s.description || ''}</div>
                </div>
                <div style="display:flex;gap:6px">
                    <button class="btn btn-outline btn-sm" onclick="loadFnOBuilderStrategy('${s.name}')" style="font-size:10px;padding:2px 8px">Edit</button>
                    <button class="btn btn-outline btn-sm" onclick="deleteFnOCustomStrategy('${s.name}')" style="font-size:10px;padding:2px 8px;color:#ff5252;border-color:#ff5252">Delete</button>
                </div>
            </div>
        `).join('');
    } catch (e) {}
}

async function loadFnOCustomStrategiesDropdowns() {
    try {
        const strategies = await API.get('/api/fno-builder/list');
        const html = (strategies || []).map(s =>
            `<option value="fno_custom_${s.name}">⚙ ${s.name}</option>`
        ).join('');
        ['bt-fno-custom-strategies', 'paper-fno-custom-strategies'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.innerHTML = html;
        });
    } catch (e) {}
}

async function loadFnOBuilderStrategy(name) {
    try {
        const cfg = await API.get('/api/fno-builder/get/' + encodeURIComponent(name));
        document.getElementById('fno-builder-name').value = cfg.name || '';
        document.getElementById('fno-builder-desc').value = cfg.description || '';
        document.getElementById('fno-builder-underlying').value = cfg.underlying || 'NIFTY';
        document.getElementById('fno-builder-type').value = cfg.strategy_type || 'custom';
        document.getElementById('fno-builder-max-pos').value = cfg.max_positions || 1;
        document.getElementById('fno-builder-profit').value = cfg.profit_target_pct || 50;
        document.getElementById('fno-builder-sl').value = cfg.stop_loss_pct || 100;
        document.getElementById('fno-builder-dte-min').value = cfg.entry_dte_min || 15;
        document.getElementById('fno-builder-dte-max').value = cfg.entry_dte_max || 45;
        updateFnOBuilderLotSize();
        // Reload legs
        document.getElementById('fno-builder-legs').innerHTML = '';
        fnoLegCounter = 0;
        (cfg.legs || []).forEach(leg => addFnOBuilderLeg(leg));
    } catch (e) {
        showFnOBuilderStatus('Failed to load: ' + e.message, true);
    }
}

async function deleteFnOCustomStrategy(name) {
    if (!confirm(`Delete F&O strategy "${name}"?`)) return;
    try {
        await API.delete_('/api/fno-builder/' + encodeURIComponent(name));
        loadFnOCustomStrategiesList();
        loadFnOCustomStrategiesDropdowns();
        showFnOBuilderStatus('Deleted "' + name + '"', false);
    } catch (e) {
        showFnOBuilderStatus('Delete failed: ' + e.message, true);
    }
}

function showFnOBuilderStatus(msg, isError) {
    const el = document.getElementById('fno-builder-status');
    el.style.display = 'block';
    el.style.color = isError ? '#ff5252' : '#00d4aa';
    el.textContent = msg;
    if (!isError) setTimeout(() => { el.style.display = 'none'; }, 5000);
}

// Load F&O builder data on tab switch
function initFnOBuilder() {
    loadFnOTemplates();
    loadFnOCustomStrategiesList();
    loadFnOCustomStrategiesDropdowns();
    updateFnOBuilderLotSize();
}

// ===== CHARTS TAB =====
// ── Chart indicator checkbox config ──
const CHART_IND_CONFIG = {
    sma_20:  { color: '#ffb74d', dash: false, overlay: true },
    sma_50:  { color: '#42a5f5', dash: false, overlay: true },
    sma_200: { color: '#ce93d8', dash: false, overlay: true },
    ema_9:   { color: '#ffd54f', dash: true,  overlay: true },
    ema_21:  { color: '#4dd0e1', dash: true,  overlay: true },
    bb:      { color: '#607d8b', dash: true,  overlay: true }, // maps to bb_upper + bb_lower
    vwap:    { color: '#ff7043', dash: false, overlay: true },
    rsi_14:  { color: '#ab47bc', dash: false, overlay: false },
    vol_sma: { color: '#66bb6a', dash: false, overlay: false },
};

function isIndEnabled(key) {
    const cb = document.querySelector('.chart-ind-cb[data-ind="' + key + '"]');
    return cb ? cb.checked : false;
}

// Wire checkboxes to re-render chart without re-fetching data
document.addEventListener('change', function(e) {
    if (e.target.classList.contains('chart-ind-cb') && window._chartData) {
        renderPriceChart(window._chartData);
        renderRsiChart(window._chartData);
        renderVolumeChart(window._chartData);
    }
});

async function loadChart() {
    const symbol = document.getElementById('chart-symbol').value.trim().toUpperCase();
    const exchange = document.getElementById('chart-exchange').value;
    const interval = document.getElementById('chart-interval').value;
    const days = parseInt(document.getElementById('chart-days').value) || 365;
    if (!symbol) return;
    const status = document.getElementById('chart-status');
    status.textContent = 'Loading chart for ' + symbol + '...';
    status.style.color = '';
    try {
        const data = await API.get('/api/chart/historical/' + encodeURIComponent(symbol) +
            '?exchange=' + exchange + '&interval=' + interval + '&days=' + days +
            '&include_indicators=true');
        window._chartData = data; // cache for re-renders
        const tz = document.getElementById('chart-timezone').value;
        status.textContent = symbol + ' · ' + interval + ' · ' + (data.candles ? data.candles.length : 0) + ' candles · TZ: ' + tz;
        renderPriceChart(data);
        renderRsiChart(data);
        renderVolumeChart(data);
        if (data.indicators && Object.keys(data.indicators).length) {
            renderIndicatorCards(data.indicators);
            document.getElementById('chart-indicators-panel').style.display = '';
        } else {
            document.getElementById('chart-indicators-panel').style.display = 'none';
        }
    } catch (e) {
        status.textContent = 'Error: ' + e.message;
        status.style.color = '#ff5252';
    }
}

// ── Timezone-aware date formatter ──
function formatChartDate(dateStr) {
    const tz = document.getElementById('chart-timezone')?.value || 'Asia/Kolkata';
    try {
        const d = new Date(dateStr);
        return d.toLocaleString('en-IN', { timeZone: tz, month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', hour12: false });
    } catch { return dateStr; }
}
function formatChartDateShort(dateStr) {
    const tz = document.getElementById('chart-timezone')?.value || 'Asia/Kolkata';
    try {
        const d = new Date(dateStr);
        return d.toLocaleDateString('en-IN', { timeZone: tz, month: 'short', day: 'numeric' });
    } catch { return ''; }
}

function renderPriceChart(data) {
    const canvas = document.getElementById('price-chart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 2;
    const h = 400;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    if (!data.candles || !data.candles.length) {
        ctx.fillStyle = '#607d8b'; ctx.font = '14px sans-serif';
        ctx.fillText('No data available', w/2-50, h/2);
        return;
    }

    const candles = data.candles;
    const pad = {top: 20, right: 60, bottom: 30, left: 10};
    const cw = (w - pad.left - pad.right) / candles.length;
    let minP = Infinity, maxP = -Infinity;
    candles.forEach(c => { if(c.low < minP) minP = c.low; if(c.high > maxP) maxP = c.high; });
    const range = maxP - minP || 1;
    const yScale = (v) => pad.top + (1 - (v - minP) / range) * (h - pad.top - pad.bottom);

    // Grid + Y axis labels
    ctx.strokeStyle = '#1a2332'; ctx.lineWidth = 0.5;
    for(let i = 0; i <= 5; i++) {
        const y = pad.top + i * (h - pad.top - pad.bottom) / 5;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w-pad.right, y); ctx.stroke();
        ctx.fillStyle = '#607d8b'; ctx.font = '10px monospace';
        const val = maxP - i * range / 5;
        ctx.fillText(val.toFixed(2), w - pad.right + 4, y + 3);
    }

    // X axis date labels (timezone aware)
    const labelCount = Math.min(6, candles.length);
    const step = Math.floor(candles.length / labelCount);
    ctx.fillStyle = '#607d8b'; ctx.font = '9px monospace'; ctx.textAlign = 'center';
    for (let i = 0; i < candles.length; i += step) {
        if (candles[i].date || candles[i].timestamp) {
            const x = pad.left + i * cw + cw / 2;
            ctx.fillText(formatChartDateShort(candles[i].date || candles[i].timestamp), x, h - 4);
        }
    }
    ctx.textAlign = 'start';

    // Candles
    candles.forEach((c, i) => {
        const x = pad.left + i * cw + cw / 2;
        const bw = Math.max(1, cw * 0.7);
        const isGreen = c.close >= c.open;
        ctx.strokeStyle = isGreen ? '#00d4aa' : '#ff5252';
        ctx.fillStyle = isGreen ? '#00d4aa' : '#ff5252';
        // Wick
        ctx.beginPath(); ctx.moveTo(x, yScale(c.high)); ctx.lineTo(x, yScale(c.low));
        ctx.lineWidth = 1; ctx.stroke();
        // Body
        const oY = yScale(c.open), cY = yScale(c.close);
        const bodyTop = Math.min(oY, cY), bodyH = Math.max(Math.abs(oY - cY), 1);
        ctx.fillRect(x - bw/2, bodyTop, bw, bodyH);
    });

    // Overlay indicators — only draw enabled ones
    if (data.indicators) {
        const drawLine = (arr, color, dashed) => {
            if (!arr || !arr.length) return;
            ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.2;
            if (dashed) ctx.setLineDash([4, 4]); else ctx.setLineDash([]);
            let started = false;
            arr.forEach((v, i) => {
                if (v === null || v === undefined) return;
                const x = pad.left + i * cw + cw/2, y = yScale(v);
                if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
            });
            ctx.stroke(); ctx.setLineDash([]);
        };
        if (isIndEnabled('sma_20'))  drawLine(data.indicators.sma_20,  '#ffb74d', false);
        if (isIndEnabled('sma_50'))  drawLine(data.indicators.sma_50,  '#42a5f5', false);
        if (isIndEnabled('sma_200')) drawLine(data.indicators.sma_200, '#ce93d8', false);
        if (isIndEnabled('ema_9'))   drawLine(data.indicators.ema_9,   '#ffd54f', true);
        if (isIndEnabled('ema_21'))  drawLine(data.indicators.ema_21,  '#4dd0e1', true);
        if (isIndEnabled('bb')) {
            drawLine(data.indicators.bb_upper, '#607d8b', true);
            drawLine(data.indicators.bb_lower, '#607d8b', true);
        }
        if (isIndEnabled('vwap'))    drawLine(data.indicators.vwap,    '#ff7043', false);
    }

    // Legend (enabled indicators only)
    const legendItems = [];
    if (isIndEnabled('sma_20'))  legendItems.push({label:'SMA 20', color:'#ffb74d'});
    if (isIndEnabled('sma_50'))  legendItems.push({label:'SMA 50', color:'#42a5f5'});
    if (isIndEnabled('sma_200')) legendItems.push({label:'SMA 200', color:'#ce93d8'});
    if (isIndEnabled('ema_9'))   legendItems.push({label:'EMA 9', color:'#ffd54f'});
    if (isIndEnabled('ema_21'))  legendItems.push({label:'EMA 21', color:'#4dd0e1'});
    if (isIndEnabled('bb'))      legendItems.push({label:'BB', color:'#607d8b'});
    if (isIndEnabled('vwap'))    legendItems.push({label:'VWAP', color:'#ff7043'});
    if (legendItems.length) {
        let lx = pad.left + 6;
        ctx.font = '10px sans-serif';
        legendItems.forEach(it => {
            ctx.fillStyle = it.color;
            ctx.fillRect(lx, 6, 12, 3);
            lx += 15;
            ctx.fillText(it.label, lx, 10);
            lx += ctx.measureText(it.label).width + 10;
        });
    }
}

// ── RSI Sub-panel ──
function renderRsiChart(data) {
    const canvas = document.getElementById('rsi-chart');
    if (!canvas) return;
    const showRsi = isIndEnabled('rsi_14') && data.indicators && data.indicators.rsi_14;
    if (!showRsi) {
        canvas.style.display = 'none'; canvas.style.height = '0px';
        return;
    }
    canvas.style.display = ''; canvas.style.height = '100px';
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 2, h = 100;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const rsi = data.indicators.rsi_14;
    const pad = {left: 10, right: 60, top: 8, bottom: 8};
    const cw = (w - pad.left - pad.right) / rsi.length;
    const yScale = v => pad.top + (1 - v / 100) * (h - pad.top - pad.bottom);

    // Background zones
    ctx.fillStyle = 'rgba(255,82,82,0.08)';
    ctx.fillRect(pad.left, yScale(100), w - pad.left - pad.right, yScale(70) - yScale(100));
    ctx.fillStyle = 'rgba(0,212,170,0.08)';
    ctx.fillRect(pad.left, yScale(30), w - pad.left - pad.right, yScale(0) - yScale(30));

    // Grid lines at 30, 50, 70
    ctx.strokeStyle = '#1a2332'; ctx.lineWidth = 0.5; ctx.setLineDash([3, 3]);
    [30, 50, 70].forEach(lv => {
        const y = yScale(lv);
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        ctx.fillStyle = '#607d8b'; ctx.font = '9px monospace';
        ctx.fillText(lv.toString(), w - pad.right + 4, y + 3);
    });
    ctx.setLineDash([]);

    // RSI line
    ctx.beginPath(); ctx.strokeStyle = '#ab47bc'; ctx.lineWidth = 1.5;
    let started = false;
    rsi.forEach((v, i) => {
        if (v === null || v === undefined) return;
        const x = pad.left + i * cw + cw / 2, y = yScale(v);
        if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Label
    ctx.fillStyle = '#ab47bc'; ctx.font = '10px sans-serif';
    ctx.fillText('RSI 14', pad.left + 4, pad.top + 10);
}

function renderVolumeChart(data) {
    const canvas = document.getElementById('volume-chart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 2, h = 80;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);
    if (!data.candles || !data.candles.length) return;
    const candles = data.candles;
    const pad = {left: 10, right: 60};
    const cw = (w - pad.left - pad.right) / candles.length;
    const maxVol = Math.max(...candles.map(c => c.volume || 0)) || 1;
    candles.forEach((c, i) => {
        const x = pad.left + i * cw;
        const bw = Math.max(1, cw * 0.7);
        const bh = (c.volume / maxVol) * (h - 10);
        ctx.fillStyle = c.close >= c.open ? 'rgba(0,212,170,0.4)' : 'rgba(255,82,82,0.4)';
        ctx.fillRect(x + (cw - bw)/2, h - bh, bw, bh);
    });

    // Volume SMA overlay
    if (isIndEnabled('vol_sma') && data.indicators && data.indicators.volume_sma_20) {
        const volSma = data.indicators.volume_sma_20;
        ctx.beginPath(); ctx.strokeStyle = '#66bb6a'; ctx.lineWidth = 1.2;
        let started = false;
        volSma.forEach((v, i) => {
            if (v === null || v === undefined) return;
            const x = pad.left + i * cw + cw / 2;
            const y = h - (v / maxVol) * (h - 10);
            if (!started) { ctx.moveTo(x, y); started = true; } else ctx.lineTo(x, y);
        });
        ctx.stroke();
    }
}

function renderIndicatorCards(indicators) {
    const el = document.getElementById('chart-indicator-cards');
    const latest = {};
    Object.entries(indicators).forEach(([k, arr]) => {
        if (!arr || !arr.length) return;
        for (let i = arr.length - 1; i >= 0; i--) {
            if (arr[i] !== null && arr[i] !== undefined) { latest[k] = arr[i]; break; }
        }
    });
    el.innerHTML = Object.entries(latest).map(([k, v]) => {
        return '<div class="card stat-card"><div class="stat-label">' + k.replace(/_/g, ' ').toUpperCase() +
            '</div><div class="stat-value" style="font-size:14px">' + (typeof v === 'number' ? v.toFixed(2) : v) + '</div></div>';
    }).join('');
}

async function loadMarketOverview() {
    try {
        const data = await API.get('/api/market/overview');
        const body = document.getElementById('market-overview-body');
        const symbols = data.market_overview || [];
        if (!symbols.length) {
            body.innerHTML = '<tr><td colspan="8" style="color:var(--text-muted);text-align:center">No data</td></tr>';
            return;
        }
        body.innerHTML = symbols.map(s => {
            const chgColor = (s.change || 0) >= 0 ? '#00d4aa' : '#ff5252';
            return '<tr><td><strong>' + s.symbol + '</strong></td>' +
                '<td>' + (s.last_price || '--') + '</td>' +
                '<td style="color:' + chgColor + '">' + (s.change || 0).toFixed(2) + '</td>' +
                '<td style="color:' + chgColor + '">' + (s.change_pct || 0).toFixed(2) + '%</td>' +
                '<td>' + (s.open || '--') + '</td>' +
                '<td>' + (s.high || '--') + '</td>' +
                '<td>' + (s.low || '--') + '</td>' +
                '<td>' + formatNum(s.volume || 0) + '</td></tr>';
        }).join('');
    } catch (e) {
        document.getElementById('market-overview-body').innerHTML =
            '<tr><td colspan="8" style="color:#ff5252">' + e.message + '</td></tr>';
    }
}

// ===== BACKTEST TAB =====
function isFnOStrategy(val) { return val && val.startsWith('fno_'); }
function isFnOCustomStrategy(val) { return val && val.startsWith('fno_custom_'); }
function stripFnOPrefix(val) { return val.replace(/^fno_/, ''); }
function stripFnOCustomPrefix(val) { return val.replace(/^fno_custom_/, ''); }

function toggleBTMode() {
    const strategy = document.getElementById('bt-strategy').value;
    const isFnO = isFnOStrategy(strategy);
    document.getElementById('bt-fno-fields').style.display = isFnO ? '' : 'none';
    // Show/hide equity-specific fields based on strategy type
    const live = document.getElementById('bt-live-fields');
    const src = document.getElementById('bt-source').value;
    if (live) {
        live.style.display = (isFnO || src === 'sample') ? 'none' : '';
    }
    // Show/hide sample fields
    const sample = document.getElementById('bt-sample-fields');
    if (sample) sample.style.display = src === 'sample' ? '' : 'none';
    // Adjust default capital for F&O
    const cap = document.getElementById('bt-capital');
    if (isFnO && parseFloat(cap.value) < 200000) cap.value = 500000;
    if (!isFnO && parseFloat(cap.value) >= 500000) cap.value = 100000;
}

function togglePaperMode() {
    const strategy = document.getElementById('paper-strategy').value;
    const isFnO = isFnOStrategy(strategy);
    document.getElementById('paper-fno-fields').style.display = isFnO ? '' : 'none';
    // Show/hide equity-only fields (symbol, token, exchange, interval)
    ['paper-symbol', 'paper-token', 'paper-exchange', 'paper-interval', 'paper-commission', 'paper-slippage'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            const grp = el.closest('.form-group');
            if (grp) grp.style.display = isFnO ? 'none' : '';
        }
    });
    const tokenStatus = document.getElementById('paper-token-status');
    if (tokenStatus) tokenStatus.parentElement.closest('.form-group').style.display = isFnO ? 'none' : '';
    const cap = document.getElementById('paper-capital');
    if (isFnO && parseFloat(cap.value) < 200000) cap.value = 500000;
    if (!isFnO && parseFloat(cap.value) >= 500000) cap.value = 100000;
}

function toggleBTSource() {
    const src = document.getElementById('bt-source').value;
    const strategy = document.getElementById('bt-strategy').value;
    const isFnO = isFnOStrategy(strategy);
    // In F&O mode live fields stay hidden
    document.getElementById('bt-live-fields').style.display = (src === 'live' && !isFnO) ? '' : 'none';
    document.getElementById('bt-sample-fields').style.display = src === 'sample' ? '' : 'none';
}

async function runBacktest() {
    const btn = document.getElementById('btn-run-bt');
    btn.disabled = true; btn.textContent = 'Running...';
    const src = document.getElementById('bt-source').value;
    const strategy = document.getElementById('bt-strategy').value;
    const isFnO = isFnOStrategy(strategy);
    try {
        let data;
        if (isFnO) {
            // F&O backtest — always sample-based (synthetic chain)
            data = await runFnOBacktestInner(strategy, src);
        } else if (src === 'sample') {
            data = await runBacktestSampleInner();
        } else {
            const symbol = document.getElementById('bt-symbol').value.trim().toUpperCase();
            if (!symbol) {
                document.getElementById('bt-summary').innerHTML = '<span style="color:#ff5252">Please enter a symbol (e.g. RELIANCE)</span>';
                return;
            }
            const exchange = document.getElementById('bt-exchange').value;
            const body = {
                strategy: strategy,
                tradingsymbol: symbol,
                exchange: exchange,
                interval: document.getElementById('bt-interval').value,
                days: parseInt(document.getElementById('bt-days').value) || 365,
                capital: parseFloat(document.getElementById('bt-capital').value) || 100000,
                position_sizing: document.getElementById('bt-sizing').value,
                slippage_pct: parseFloat(document.getElementById('bt-slippage').value) || 0.05,
                use_indian_costs: document.getElementById('bt-indian-costs').checked
            };
            data = await API.post('/api/backtest/run', body);
        }
        if (data && data.error) {
            document.getElementById('bt-summary').innerHTML = '<span style="color:#ff5252">Error: ' + data.error + '</span>';
            return;
        }
        if (isFnO) {
            displayFnOBacktestResult(data);
        } else {
            displayBacktestResult(data);
        }
    } catch (e) {
        document.getElementById('bt-summary').innerHTML = '<span style="color:#ff5252">Error: ' + e.message + '</span>';
    } finally {
        btn.disabled = false; btn.textContent = 'Run Backtest';
    }
}

async function runBacktestSample() {
    const btn = document.getElementById('btn-run-bt');
    btn.disabled = true; btn.textContent = 'Running Sample...';
    const strategy = document.getElementById('bt-strategy').value;
    const isFnO = isFnOStrategy(strategy);
    try {
        let data;
        if (isFnO) {
            data = await runFnOBacktestInner(strategy, 'sample');
        } else {
            data = await runBacktestSampleInner();
        }
        if (data && data.error) {
            document.getElementById('bt-summary').innerHTML = '<span style="color:#ff5252">Error: ' + data.error + '</span>';
            return;
        }
        if (isFnO) {
            displayFnOBacktestResult(data);
        } else {
            displayBacktestResult(data);
        }
    } catch (e) {
        document.getElementById('bt-summary').innerHTML = '<span style="color:#ff5252">Error: ' + e.message + '</span>';
    } finally {
        btn.disabled = false; btn.textContent = 'Run Backtest';
    }
}

async function runFnOBacktestInner(strategy, src) {
    // Custom F&O strategies built via the F&O Builder use a different API
    if (isFnOCustomStrategy(strategy)) {
        const custName = stripFnOCustomPrefix(strategy);
        return await API.post('/api/fno-builder/test', {
            name: custName,
            underlying: document.getElementById('bt-underlying').value,
            bars: parseInt(document.getElementById('bt-bars')?.value) || 500,
            capital: parseFloat(document.getElementById('bt-capital').value) || 500000,
        });
    }
    const body = {
        strategy: stripFnOPrefix(strategy),
        underlying: document.getElementById('bt-underlying').value,
        capital: parseFloat(document.getElementById('bt-capital').value) || 500000,
        days: parseInt(document.getElementById('bt-days')?.value) || 365,
        delta_target: parseFloat(document.getElementById('bt-delta').value) || 0.16,
        max_positions: parseInt(document.getElementById('bt-max-pos').value) || 3,
        profit_target_pct: parseFloat(document.getElementById('bt-profit-target').value) || 50,
        stop_loss_pct: parseFloat(document.getElementById('bt-stop-loss').value) || 100,
        entry_dte_min: parseInt(document.getElementById('bt-dte-min').value) || 15,
        entry_dte_max: parseInt(document.getElementById('bt-dte-max').value) || 45,
        use_regime_filter: document.getElementById('bt-regime-filter')?.checked ?? true,
    };
    if (src === 'sample') {
        body.bars = parseInt(document.getElementById('bt-bars')?.value) || 500;
        return await API.post('/api/fno-backtest/sample', body);
    } else {
        return await API.post('/api/fno-backtest/run', body);
    }
}

async function runBacktestSampleInner() {
    return await API.post('/api/backtest/sample', {
        strategy: document.getElementById('bt-strategy').value,
        bars: parseInt(document.getElementById('bt-bars')?.value) || 500,
        capital: parseFloat(document.getElementById('bt-capital').value) || 100000,
        interval: document.getElementById('bt-interval')?.value || 'day'
    });
}

function displayBacktestResult(data) {
    if (!data) return;
    // Metrics are at root level (spread via **metrics in Python)
    const m = data;
    const eq = data.equity_curve || [];
    const trades = data.trades || [];
    const mc = data.monte_carlo || {};
    const costs = data.total_costs || {};
    const analytics = data.trade_analytics || {};
    const bench = data.benchmark || {};

    const profitColor = (m.total_return_pct || 0) >= 0 ? '#00d4aa' : '#ff5252';
    const tfLabel = m.timeframe || 'day';
    const srcLabel = m.data_source ? (' <span style="color:#ffaa00">(' + m.data_source + ')</span>') : '';
    document.getElementById('bt-summary').innerHTML =
        '<div style="margin-bottom:8px;font-size:12px;color:#8899aa">' +
        'Timeframe: <strong style="color:#e0e0e0">' + tfLabel + '</strong>' + srcLabel + '</div>' +
        '<div class="grid grid-3" style="margin-bottom:16px">' +
        metricCard('Total Return', (m.total_return_pct||0).toFixed(2) + '%', profitColor) +
        metricCard('Net P&L', '₹' + formatNum(m.total_pnl||0), profitColor) +
        metricCard('Sharpe Ratio', (m.sharpe_ratio||0).toFixed(2)) +
        metricCard('Max Drawdown', (m.max_drawdown_pct||0).toFixed(2) + '%', '#ff5252') +
        metricCard('Win Rate', (m.win_rate||0).toFixed(1) + '%') +
        metricCard('Total Trades', m.total_trades||0) +
        metricCard('Sortino', (m.sortino_ratio||0).toFixed(2)) +
        metricCard('Calmar', (m.calmar_ratio||0).toFixed(2)) +
        metricCard('Expectancy', '₹' + (m.expectancy||0).toFixed(2)) +
        metricCard('Profit Factor', (m.profit_factor||0).toFixed(2)) +
        metricCard('Recovery Factor', (m.recovery_factor||0).toFixed(2)) +
        metricCard('Payoff Ratio', (m.payoff_ratio||0).toFixed(2)) +
        '</div>' +
        (bench.alpha !== undefined ? '<div class="grid grid-3" style="margin-bottom:0">' +
            metricCard('Alpha', (bench.alpha||0).toFixed(4)) +
            metricCard('Beta', (bench.beta||0).toFixed(4)) +
            metricCard('Info Ratio', (bench.information_ratio||0).toFixed(4)) +
        '</div>' : '');

    // Equity curve
    if (eq.length) drawBtEquityCurve(eq);

    // Health report
    if (data.health_report) displayHealthReport(data.health_report);

    // Underwater
    const rolling = data.rolling || {};
    if (rolling.underwater_curve && rolling.underwater_curve.length) drawUnderwaterCurve(rolling.underwater_curve);
    // Monte Carlo
    const mcFC = mc.final_capital || {};
    if (mc && mc.simulations) {
        const initCap = m.initial_capital || 100000;
        const medianReturn = initCap > 0 ? ((mcFC.median || initCap) / initCap - 1) * 100 : 0;
        const p5Return = initCap > 0 ? ((mcFC.p5 || initCap) / initCap - 1) * 100 : 0;
        const p95Return = initCap > 0 ? ((mcFC.p95 || initCap) / initCap - 1) * 100 : 0;
        document.getElementById('bt-monte-carlo').innerHTML =
            '<div class="grid grid-2" style="margin-bottom:0">' +
            metricCard('Median Return', medianReturn.toFixed(2) + '%') +
            metricCard('CI 5th', p5Return.toFixed(2) + '%', '#ff5252') +
            metricCard('CI 95th', p95Return.toFixed(2) + '%', '#00d4aa') +
            metricCard('Ruin Prob', (mc.ruin_probability||0).toFixed(2) + '%', (mc.ruin_probability||0) > 20 ? '#ff5252' : '#00d4aa') +
            '</div>';
    } else if (mc && mc.error) {
        document.getElementById('bt-monte-carlo').innerHTML =
            `<div style="color:#ffaa00;font-size:13px;padding:8px">${mc.error}</div>`;
    } else {
        document.getElementById('bt-monte-carlo').innerHTML =
            '<div style="color:#667788;font-size:13px;padding:8px">Not enough trades for Monte Carlo simulation</div>';
    }
    // Trade analytics
    const slStats = data.stop_loss_stats || {};
    const tpStats = { tp_exits: slStats.tp_exits || 0, tp_total_pnl: slStats.tp_total_pnl || 0 };
    document.getElementById('bt-trade-analytics').innerHTML =
        '<div class="grid grid-2" style="margin-bottom:8px">' +
        metricCard('Avg Win', '₹' + (m.avg_win||0).toFixed(2)) +
        metricCard('Avg Loss', '₹' + (m.avg_loss||0).toFixed(2)) +
        metricCard('Max Win Streak', analytics.max_win_streak || 0) +
        metricCard('Max Lose Streak', analytics.max_lose_streak || 0) +
        metricCard('Avg Hold Bars', (analytics.avg_hold_bars||0).toFixed(1)) +
        metricCard('Wins / Losses', (m.winning_trades||0) + ' / ' + (m.losing_trades||0)) +
        '</div>' +
        (slStats.sl_exits > 0 ? '<div style="margin-top:4px;padding:8px;background:#1a1a2e;border-radius:6px;border:1px solid #2a3a4a">' +
            '<div style="color:#ffaa00;font-size:12px;margin-bottom:6px;font-weight:600">Stop-Loss Execution</div>' +
            '<div class="grid grid-3" style="margin-bottom:0">' +
            metricCard('SL Exits', slStats.sl_exits) +
            metricCard('Signal Exits', slStats.signal_exits || 0) +
            metricCard('Final Exits', slStats.final_exits || 0) +
            metricCard('SL P&L', '₹' + formatNum(slStats.sl_total_pnl||0), (slStats.sl_total_pnl||0) >= 0 ? '#00d4aa' : '#ff5252') +
            metricCard('Signal P&L', '₹' + formatNum(slStats.signal_total_pnl||0), (slStats.signal_total_pnl||0) >= 0 ? '#00d4aa' : '#ff5252') +
            '</div></div>' : '') +
        (tpStats && tpStats.tp_exits > 0 ? '<div style="margin-top:4px;padding:8px;background:#1a2e1a;border-radius:6px;border:1px solid #2a4a3a">' +
            '<div style="color:#00d4aa;font-size:12px;margin-bottom:6px;font-weight:600">Profit Target Execution</div>' +
            '<div class="grid grid-3" style="margin-bottom:0">' +
            metricCard('TP Exits', tpStats.tp_exits) +
            metricCard('TP P&L', '₹' + formatNum(tpStats.tp_total_pnl||0), '#00d4aa') +
            '</div></div>' : '');
    // Cost breakdown
    if (costs && costs.total !== undefined) {
        document.getElementById('bt-cost-breakdown').innerHTML =
            '<div class="grid grid-3" style="margin-bottom:0">' +
            metricCard('Total Costs', '₹' + formatNum(costs.total||0)) +
            metricCard('Brokerage', '₹' + formatNum(costs.brokerage||0)) +
            metricCard('STT', '₹' + formatNum(costs.stt||0)) +
            metricCard('Exchange Txn', '₹' + formatNum(costs.exchange_txn||0)) +
            metricCard('GST', '₹' + formatNum(costs.gst||0)) +
            metricCard('Stamp Duty', '₹' + formatNum(costs.stamp_duty||0)) +
            metricCard('SEBI', '₹' + formatNum(costs.sebi||0)) +
            metricCard('Slippage', '₹' + formatNum(costs.slippage||0)) +
            '</div>';
    }
    // Trade log — merge entry+exit pairs into round-trip rows
    const tbody = document.getElementById('bt-trades-body');
    const roundTrips = [];
    let pendingEntry = null;
    for (const t of trades) {
        if (t.type && t.type.includes('ENTRY')) {
            pendingEntry = t;
        } else if (t.pnl !== undefined) {
            roundTrips.push({
                symbol: (pendingEntry && pendingEntry.tradingsymbol) || '',
                direction: (pendingEntry && pendingEntry.type) || t.type || '',
                entry_price: (pendingEntry && pendingEntry.entry_price) || 0,
                exit_price: t.exit_price || 0,
                quantity: (pendingEntry && pendingEntry.quantity) || 0,
                pnl: t.pnl || 0,
                costs: t.costs || 0,
                hold_bars: t.hold_bars,
                mae_pct: t.mae_pct,
                mfe_pct: t.mfe_pct,
            });
            pendingEntry = null;
        }
    }
    if (roundTrips.length) {
        tbody.innerHTML = roundTrips.map((rt, i) => {
            const color = rt.pnl >= 0 ? '#00d4aa' : '#ff5252';
            const dir = rt.direction.replace('_ENTRY','');
            return '<tr><td>' + (i+1) + '</td><td>' + dir + '</td>' +
                '<td>' + rt.entry_price.toFixed(2) + ' → ' + rt.exit_price.toFixed(2) + '</td>' +
                '<td>' + rt.quantity + '</td>' +
                '<td style="color:' + color + '">₹' + rt.pnl.toFixed(2) + '</td>' +
                '<td>₹' + rt.costs.toFixed(2) + '</td>' +
                '<td>' + (rt.hold_bars||'--') + '</td>' +
                '<td>' + (rt.mae_pct != null ? rt.mae_pct.toFixed(2) + '%' : '--') + '</td>' +
                '<td>' + (rt.mfe_pct != null ? rt.mfe_pct.toFixed(2) + '%' : '--') + '</td></tr>';
        }).join('');
    } else {
        tbody.innerHTML = '<tr><td colspan="9" style="color:var(--text-muted);text-align:center">No trades</td></tr>';
    }
}

function metricCard(label, value, color) {
    return '<div class="card stat-card" style="padding:10px"><div class="stat-label" style="font-size:10px">' + label +
        '</div><div class="stat-value" style="font-size:15px;' + (color ? 'color:' + color : '') + '">' + value + '</div></div>';
}

function drawBtEquityCurve(eq) {
    const canvas = document.getElementById('bt-equity-chart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 32, h = 300;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
    const pad = {top: 20, right: 60, bottom: 25, left: 10};
    const minV = Math.min(...eq), maxV = Math.max(...eq);
    const range = maxV - minV || 1;
    const pw = (w - pad.left - pad.right) / (eq.length - 1 || 1);
    const yS = v => pad.top + (1 - (v - minV) / range) * (h - pad.top - pad.bottom);

    // Zero line / initial capital line
    ctx.strokeStyle = '#1a2332'; ctx.lineWidth = 0.5;
    for(let i = 0; i <= 4; i++) {
        const y = pad.top + i * (h - pad.top - pad.bottom) / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        ctx.fillStyle = '#607d8b'; ctx.font = '10px monospace';
        ctx.fillText(formatNum(maxV - i * range / 4), w - pad.right + 4, y + 3);
    }

    // Fill area
    ctx.beginPath();
    ctx.moveTo(pad.left, yS(eq[0]));
    eq.forEach((v, i) => ctx.lineTo(pad.left + i * pw, yS(v)));
    ctx.lineTo(pad.left + (eq.length - 1) * pw, h - pad.bottom);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(0,212,170,0.3)');
    grad.addColorStop(1, 'rgba(0,212,170,0)');
    ctx.fillStyle = grad; ctx.fill();

    // Line
    ctx.beginPath(); ctx.strokeStyle = '#00d4aa'; ctx.lineWidth = 1.5;
    eq.forEach((v, i) => { i === 0 ? ctx.moveTo(pad.left, yS(v)) : ctx.lineTo(pad.left + i * pw, yS(v)); });
    ctx.stroke();
}

function drawUnderwaterCurve(uw) {
    const canvas = document.getElementById('bt-underwater-chart');
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 32, h = 120;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
    const pad = {top: 10, right: 60, bottom: 20, left: 10};
    const minV = Math.min(...uw, 0);
    const range = Math.abs(minV) || 1;
    const pw = (w - pad.left - pad.right) / (uw.length - 1 || 1);
    const yS = v => pad.top + (-v / range) * (h - pad.top - pad.bottom);

    // Zero line
    ctx.strokeStyle = '#607d8b'; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(pad.left, pad.top); ctx.lineTo(w - pad.right, pad.top); ctx.stroke();
    ctx.fillStyle = '#607d8b'; ctx.font = '10px monospace';
    ctx.fillText('0%', w - pad.right + 4, pad.top + 3);
    ctx.fillText(minV.toFixed(1) + '%', w - pad.right + 4, h - pad.bottom + 3);

    // Fill
    ctx.beginPath();
    ctx.moveTo(pad.left, pad.top);
    uw.forEach((v, i) => ctx.lineTo(pad.left + i * pw, yS(v)));
    ctx.lineTo(pad.left + (uw.length-1) * pw, pad.top);
    ctx.closePath();
    ctx.fillStyle = 'rgba(255,82,82,0.3)'; ctx.fill();
    // Line
    ctx.beginPath(); ctx.strokeStyle = '#ff5252'; ctx.lineWidth = 1;
    uw.forEach((v,i) => { i === 0 ? ctx.moveTo(pad.left, yS(v)) : ctx.lineTo(pad.left + i * pw, yS(v)); });
    ctx.stroke();
}

// ===== F&O BACKTEST RESULT DISPLAY =====
function displayFnOBacktestResult(data) {
    if (!data) return;
    const m = data;
    const eq = data.equity_curve || [];
    const trades = data.trades || [];
    const greeksHist = data.greeks_history || [];
    const marginHist = data.margin_history || [];
    const regimeHist = data.regime_history || [];
    const settings = data.settings || {};

    // Show/hide appropriate sections
    document.getElementById('bt-fno-results').style.display = '';

    const profitColor = (m.total_return_pct || 0) >= 0 ? '#00d4aa' : '#ff5252';
    const fnoSrcLabel = m.data_source ? (' <span style="color:#ffaa00">(' + m.data_source + ')</span>') : '';
    document.getElementById('bt-summary').innerHTML =
        (fnoSrcLabel ? '<div style="margin-bottom:8px;font-size:12px;color:#8899aa">Data: ' + fnoSrcLabel + '</div>' : '') +
        '<div class="grid grid-3" style="margin-bottom:16px">' +
        metricCard('Total Return', (m.total_return_pct||0).toFixed(2) + '%', profitColor) +
        metricCard('Net P&L', '₹' + formatNum(m.total_pnl||0), profitColor) +
        metricCard('Sharpe Ratio', (m.sharpe_ratio||0).toFixed(2)) +
        metricCard('Win Rate', (m.win_rate||0).toFixed(1) + '%') +
        metricCard('Total Trades', m.total_trades||0) +
        metricCard('Max Drawdown', (m.max_drawdown_pct||0).toFixed(2) + '%', '#ff5252') +
        metricCard('Profit Factor', (m.profit_factor||0).toFixed(2)) +
        metricCard('Total Costs', '₹' + formatNum(m.total_costs||0), '#ffb74d') +
        metricCard('Final Capital', '₹' + formatNum(m.final_capital||0), profitColor) +
        '</div>';

    // Draw equity curve on existing canvas
    if (eq.length > 1) drawBtEquityCurve(eq);

    // Health report
    if (data.health_report) displayHealthReport(data.health_report);

    // Draw greeks history chart
    if (greeksHist.length > 0) drawFnOGreeksChart(greeksHist, 'bt-greeks-chart');

    // Draw margin history chart
    if (marginHist.length > 0) drawFnOMarginChart(marginHist, 'bt-margin-chart');

    // Regime info
    if (regimeHist.length > 0) {
        const regimeCounts = {};
        regimeHist.forEach(r => { regimeCounts[r.regime] = (regimeCounts[r.regime]||0) + 1; });
        const total = regimeHist.length;
        let regimeHtml = '<div class="grid grid-2" style="gap:4px">';
        for (const [regime, count] of Object.entries(regimeCounts)) {
            const pct = ((count/total)*100).toFixed(1);
            regimeHtml += metricCard(regime.replace(/_/g,' '), pct + '%');
        }
        regimeHtml += '</div>';
        document.getElementById('bt-regime-info').innerHTML = regimeHtml;
    }

    // FnO settings
    document.getElementById('bt-fno-settings').innerHTML =
        '<div style="line-height:1.8">' +
        '<div><b>Strategy:</b> ' + (m.strategy_name||settings.strategy||'--') + '</div>' +
        '<div><b>Underlying:</b> ' + (m.underlying||settings.underlying||'--') + '</div>' +
        '<div><b>Delta Target:</b> ' + (settings.delta_target||'--') + '</div>' +
        '<div><b>Max Positions:</b> ' + (settings.max_positions||'--') + '</div>' +
        '<div><b>Profit Target:</b> ' + (settings.profit_target_pct||'--') + '%</div>' +
        '<div><b>Stop Loss:</b> ' + (settings.stop_loss_pct||'--') + '%</div>' +
        '<div><b>DTE Range:</b> ' + (settings.entry_dte_min||'--') + '-' + (settings.entry_dte_max||'--') + ' days</div>' +
        '<div><b>Regime Filter:</b> ' + (settings.use_regime_filter ? 'ON' : 'OFF') + '</div></div>';

    // Positions summary
    if (trades.length > 0) {
        let posHtml = '<div style="overflow-x:auto"><table style="width:100%;font-size:12px"><thead><tr>' +
            '<th>#</th><th>Structure</th><th>Legs</th><th>Net Premium</th><th>P&L</th><th>Max Profit</th><th>Max Loss</th>' +
            '</tr></thead><tbody>';
        trades.forEach((t, i) => {
            const pnlColor = (t.pnl||0) >= 0 ? '#00d4aa' : '#ff5252';
            posHtml += '<tr><td>' + (i+1) + '</td>' +
                '<td>' + (t.structure||t.type||'--') + '</td>' +
                '<td>' + (t.legs||t.num_legs||'--') + '</td>' +
                '<td>₹' + formatNum(t.net_premium||0) + '</td>' +
                '<td style="color:' + pnlColor + '">₹' + formatNum(t.pnl||0) + '</td>' +
                '<td>₹' + formatNum(t.max_profit||0) + '</td>' +
                '<td>₹' + formatNum(t.max_loss||0) + '</td></tr>';
        });
        posHtml += '</tbody></table></div>';
        document.getElementById('bt-fno-positions').innerHTML = posHtml;
    }

    // Reuse trade log for F&O
    const tbody = document.getElementById('bt-trades-body');
    // Filter to exit trades only (skip ENTRY) and render with enriched fields
    const exitTrades = trades.filter(t => t.type !== 'ENTRY' && t.pnl !== undefined);
    if (exitTrades.length > 0) {
        tbody.innerHTML = exitTrades.map((t, i) => {
            const color = (t.pnl||0) >= 0 ? '#00d4aa' : '#ff5252';
            return '<tr><td>' + (i+1) + '</td><td>' + (t.structure||t.type||'--') + '</td>' +
                '<td>' + (t.entry_time||t.date||'--') + ' → ' + (t.exit_time||t.date||'--') + '</td>' +
                '<td>' + (t.qty||t.legs||'--') + '</td>' +
                '<td style="color:' + color + '">₹' + formatNum(t.pnl||0) + '</td>' +
                '<td>₹' + formatNum(t.costs||0) + '</td>' +
                '<td>' + (t.hold_bars||'--') + '</td>' +
                '<td>' + (t.regime||'--') + '</td>' +
                '<td>' + (t.exit_reason||t.type||'--') + '</td></tr>';
        }).join('');
    } else {
        tbody.innerHTML = '<tr><td colspan="9" style="color:var(--text-muted);text-align:center">No F&O trades</td></tr>';
    }
}

function drawFnOGreeksChart(history, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 32, h = 200;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
    const pad = {top: 20, right: 70, bottom: 25, left: 10};

    const deltas = history.map(g => g.net_delta || 0);
    const gammas = history.map(g => g.net_gamma || 0);
    const thetas = history.map(g => g.net_theta || 0);

    const series = [{data: deltas, color: '#00d4aa', label: 'Delta'},
                     {data: gammas, color: '#42a5f5', label: 'Gamma'},
                     {data: thetas, color: '#ff5252', label: 'Theta'}];
    const allVals = [...deltas, ...gammas, ...thetas];
    const minV = Math.min(...allVals), maxV = Math.max(...allVals);
    const range = maxV - minV || 1;
    const pw = (w - pad.left - pad.right) / (history.length - 1 || 1);
    const yS = v => pad.top + (1 - (v - minV) / range) * (h - pad.top - pad.bottom);

    // Grid
    ctx.strokeStyle = '#1a2332'; ctx.lineWidth = 0.5;
    for(let i = 0; i <= 4; i++) {
        const y = pad.top + i * (h - pad.top - pad.bottom) / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        ctx.fillStyle = '#607d8b'; ctx.font = '9px monospace';
        ctx.fillText((maxV - i * range / 4).toFixed(2), w - pad.right + 4, y + 3);
    }

    // Lines
    series.forEach(s => {
        ctx.beginPath(); ctx.strokeStyle = s.color; ctx.lineWidth = 1.5;
        s.data.forEach((v, i) => { i === 0 ? ctx.moveTo(pad.left, yS(v)) : ctx.lineTo(pad.left + i * pw, yS(v)); });
        ctx.stroke();
    });

    // Legend
    series.forEach((s, i) => {
        ctx.fillStyle = s.color; ctx.font = '10px Inter';
        ctx.fillText('● ' + s.label, pad.left + i * 70, h - 4);
    });
}

function drawFnOMarginChart(history, canvasId) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.parentElement.getBoundingClientRect();
    const w = rect.width - 32, h = 200;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.width = w + 'px'; canvas.style.height = h + 'px';
    ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
    const pad = {top: 20, right: 70, bottom: 25, left: 10};

    const vals = history.map(m => m.margin_used || m.total_margin || 0);
    const minV = 0, maxV = Math.max(...vals) * 1.1 || 1;
    const range = maxV - minV || 1;
    const pw = (w - pad.left - pad.right) / (vals.length - 1 || 1);
    const yS = v => pad.top + (1 - (v - minV) / range) * (h - pad.top - pad.bottom);

    // Grid
    ctx.strokeStyle = '#1a2332'; ctx.lineWidth = 0.5;
    for(let i = 0; i <= 4; i++) {
        const y = pad.top + i * (h - pad.top - pad.bottom) / 4;
        ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
        ctx.fillStyle = '#607d8b'; ctx.font = '9px monospace';
        ctx.fillText('₹' + formatNum(maxV - i * range / 4), w - pad.right + 4, y + 3);
    }

    // Area fill
    ctx.beginPath();
    ctx.moveTo(pad.left, yS(vals[0]));
    vals.forEach((v, i) => ctx.lineTo(pad.left + i * pw, yS(v)));
    ctx.lineTo(pad.left + (vals.length - 1) * pw, h - pad.bottom);
    ctx.lineTo(pad.left, h - pad.bottom);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, 0, 0, h);
    grad.addColorStop(0, 'rgba(255,183,77,0.3)');
    grad.addColorStop(1, 'rgba(255,183,77,0)');
    ctx.fillStyle = grad; ctx.fill();

    // Line
    ctx.beginPath(); ctx.strokeStyle = '#ffb74d'; ctx.lineWidth = 1.5;
    vals.forEach((v, i) => { i === 0 ? ctx.moveTo(pad.left, yS(v)) : ctx.lineTo(pad.left + i * pw, yS(v)); });
    ctx.stroke();

    ctx.fillStyle = '#ffb74d'; ctx.font = '10px Inter';
    ctx.fillText('● Margin Used', pad.left, h - 4);
}

// ===== REPORTS TAB =====
let currentReportTab = 'pnl';
async function loadReports() {
    switchReportTab('pnl');
}

function switchReportTab(tab) {
    currentReportTab = tab;
    document.querySelectorAll('.report-tab').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.report-view').forEach(v => v.style.display = 'none');
    const btn = document.querySelector('.report-tab[onclick*="' + tab + '"]');
    if (btn) btn.classList.add('active');
    const view = document.getElementById('report-view-' + tab);
    if (view) view.style.display = '';
    // Load data
    const loaders = {
        pnl: loadPnLReport, holdings: loadHoldingsReport, positions: loadPositionsReport,
        trades: loadTradesReport, orders: loadOrdersReport, margins: loadMarginsReport
    };
    if (loaders[tab]) loaders[tab]();
}

async function loadPnLReport() {
    try {
        const data = await API.get('/api/reports/pnl');
        const cards = document.getElementById('pnl-summary-cards');
        const det = document.getElementById('pnl-details');
        const posSummary = data.positions_summary || {};
        const holdSummary = data.holdings_summary || {};
        const rpnl = posSummary.realized_pnl || 0;
        const upnl = posSummary.unrealized_pnl || 0;
        const dayPnl = data.day_pnl || 0;
        const investPnl = data.investment_pnl || 0;
        const total = data.combined_pnl || (dayPnl + investPnl);
        const c = total >= 0 ? '#00d4aa' : '#ff5252';
        cards.innerHTML =
            metricCard('Combined P&L', '₹' + formatNum(total), c) +
            metricCard('Day P&L', '₹' + formatNum(dayPnl), dayPnl >= 0 ? '#00d4aa' : '#ff5252') +
            metricCard('Investment P&L', '₹' + formatNum(investPnl), investPnl >= 0 ? '#00d4aa' : '#ff5252') +
            metricCard('Realized', '₹' + formatNum(rpnl), rpnl >= 0 ? '#00d4aa' : '#ff5252') +
            metricCard('Unrealized', '₹' + formatNum(upnl), upnl >= 0 ? '#00d4aa' : '#ff5252');
        let html = '<div class="card-title">P&L Breakdown</div>';
        html += '<div style="padding:10px;color:var(--text-muted)"><p>Positions P&L: ₹' + formatNum(dayPnl) + ' | Holdings P&L: ₹' + formatNum(investPnl) + '</p>';
        const tradesSummary = data.trades_summary || {};
        if (tradesSummary.total_trades) {
            html += '<p>Trades today: ' + tradesSummary.total_trades + ' (Buy: ₹' + formatNum(tradesSummary.buy_value||0) + ', Sell: ₹' + formatNum(tradesSummary.sell_value||0) + ')</p>';
        }
        html += '<p style="font-size:0.85em">Report time: ' + (data.report_time || 'N/A') + '</p></div>';
        det.innerHTML = html;
    } catch (e) {
        document.getElementById('pnl-details').innerHTML = '<div style="color:#ff5252">' + e.message + '</div>';
    }
}

async function loadHoldingsReport() {
    try {
        const data = await API.get('/api/reports/holdings');
        const summary = data.summary || {};
        const dayPnl = (data.holdings || []).reduce((s, h) => s + (h.day_change || 0), 0);
        const cards = document.getElementById('holdings-summary-cards');
        cards.innerHTML =
            metricCard('Total Value', '₹' + formatNum(summary.current_value || 0)) +
            metricCard('Day P&L', '₹' + formatNum(dayPnl), dayPnl>=0?'#00d4aa':'#ff5252') +
            metricCard('Total P&L', '₹' + formatNum(summary.total_pnl || 0), (summary.total_pnl||0)>=0?'#00d4aa':'#ff5252') +
            metricCard('Holdings', summary.holdings_count || 0);
        const body = document.getElementById('report-holdings-body');
        if (data.holdings && data.holdings.length) {
            body.innerHTML = data.holdings.map(h => {
                const c = (h.pnl||0) >= 0 ? '#00d4aa' : '#ff5252';
                const sym = h.tradingsymbol || '';
                return '<tr><td><strong>' + sym + '</strong></td>' +
                    '<td>' + (h.quantity||0) + '</td><td>' + (h.average_price||0).toFixed(2) + '</td>' +
                    '<td>' + (h.last_price||0).toFixed(2) + '</td>' +
                    '<td>₹' + formatNum(h.current_value||0) + '</td>' +
                    '<td style="color:' + c + '">₹' + (h.pnl||0).toFixed(2) + '</td>' +
                    '<td style="color:' + c + '">' + (h.pnl_pct||0).toFixed(2) + '%</td>' +
                    '<td>' + (h.weight_pct||0).toFixed(1) + '%</td>' +
                    '<td><button class="btn btn-outline" style="font-size:10px;padding:2px 8px" onclick="openDeepProfile(\'' + sym + '\')">Review</button></td></tr>';
            }).join('');
        } else {
            body.innerHTML = '<tr><td colspan="9" style="color:var(--text-muted);text-align:center">No holdings</td></tr>';
        }
    } catch (e) {
        document.getElementById('report-holdings-body').innerHTML = '<tr><td colspan="9" style="color:#ff5252">' + e.message + '</td></tr>';
    }
}

async function loadPositionsReport() {
    try {
        const data = await API.get('/api/reports/positions');
        const summary = data.summary || {};
        const cards = document.getElementById('positions-summary-cards');
        cards.innerHTML =
            metricCard('Realized', '₹' + formatNum(summary.realized_pnl||0), (summary.realized_pnl||0)>=0?'#00d4aa':'#ff5252') +
            metricCard('Unrealized', '₹' + formatNum(summary.unrealized_pnl||0), (summary.unrealized_pnl||0)>=0?'#00d4aa':'#ff5252') +
            metricCard('Open', summary.open_positions||0) +
            metricCard('Closed', summary.closed_positions||0);
        const body = document.getElementById('report-positions-body');
        const positions = data.positions || [];
        if (positions.length) {
            body.innerHTML = positions.map(p => {
                const c = (p.pnl||0) >= 0 ? '#00d4aa' : '#ff5252';
                return '<tr><td><strong>' + (p.tradingsymbol||'') + '</strong></td>' +
                    '<td>' + (p.direction || (p.buy_quantity > 0 ? 'LONG' : 'SHORT')) + '</td>' +
                    '<td>' + (p.quantity||0) + '</td>' +
                    '<td>' + (p.average_price||0).toFixed(2) + '</td>' +
                    '<td>' + (p.last_price||0).toFixed(2) + '</td>' +
                    '<td style="color:' + c + '">₹' + (p.pnl||0).toFixed(2) + '</td>' +
                    '<td>' + (p.product||'') + '</td></tr>';
            }).join('');
        } else {
            body.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center">No positions</td></tr>';
        }
    } catch (e) {
        document.getElementById('report-positions-body').innerHTML = '<tr><td colspan="7" style="color:#ff5252">' + e.message + '</td></tr>';
    }
}

async function loadTradesReport() {
    try {
        const data = await API.get('/api/reports/trades');
        const summary = data.summary || {};
        const cards = document.getElementById('trades-summary-cards');
        cards.innerHTML =
            metricCard('Trades', summary.total_trades||0) +
            metricCard('Buy Turnover', '₹' + formatNum(summary.buy_value||0)) +
            metricCard('Sell Turnover', '₹' + formatNum(summary.sell_value||0));
        const body = document.getElementById('report-trades-body');
        if (data.trades && data.trades.length) {
            body.innerHTML = data.trades.map(t => {
                return '<tr><td>' + (t.trade_id||'') + '</td>' +
                    '<td><strong>' + (t.tradingsymbol||'') + '</strong></td>' +
                    '<td>' + (t.transaction_type||'') + '</td>' +
                    '<td>' + (t.quantity||0) + '</td>' +
                    '<td>' + (t.average_price||0).toFixed(2) + '</td>' +
                    '<td>₹' + formatNum((t.average_price||0) * (t.quantity||0)) + '</td>' +
                    '<td>' + (t.product||'') + '</td></tr>';
            }).join('');
        } else {
            body.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center">No trades today</td></tr>';
        }
    } catch (e) {
        document.getElementById('report-trades-body').innerHTML = '<tr><td colspan="7" style="color:#ff5252">' + e.message + '</td></tr>';
    }
}

async function loadOrdersReport() {
    try {
        const data = await API.get('/api/reports/orders');
        const summary = data.summary || {};
        const cards = document.getElementById('orders-summary-cards');
        cards.innerHTML =
            metricCard('Total', summary.total_orders||0) +
            metricCard('Completed', summary.filled||0, '#00d4aa') +
            metricCard('Pending', summary.pending||0, '#ffb74d') +
            metricCard('Rejected', summary.rejected||0, '#ff5252');
        const body = document.getElementById('report-orders-body');
        if (data.orders && data.orders.length) {
            body.innerHTML = data.orders.map(o => {
                const sc = o.status === 'COMPLETE' ? '#00d4aa' : o.status === 'REJECTED' ? '#ff5252' : '#ffb74d';
                return '<tr><td>' + (o.order_id||'') + '</td>' +
                    '<td><strong>' + (o.tradingsymbol||'') + '</strong></td>' +
                    '<td>' + (o.transaction_type||'') + '</td>' +
                    '<td>' + (o.quantity||0) + '</td>' +
                    '<td>' + (o.price||o.average_price||0).toFixed(2) + '</td>' +
                    '<td style="color:' + sc + '">' + (o.status||'') + '</td>' +
                    '<td>' + (o.order_timestamp||'') + '</td></tr>';
            }).join('');
        } else {
            body.innerHTML = '<tr><td colspan="7" style="color:var(--text-muted);text-align:center">No orders</td></tr>';
        }
    } catch (e) {
        document.getElementById('report-orders-body').innerHTML = '<tr><td colspan="7" style="color:#ff5252">' + e.message + '</td></tr>';
    }
}

async function loadMarginsReport() {
    try {
        const data = await API.get('/api/reports/margins');
        const margins = data.margins || data;
        const cards = document.getElementById('margins-cards');
        let html = '';
        ['equity', 'commodity'].forEach(seg => {
            const m = margins[seg];
            if (!m) return;
            const used = (m.utilised && m.utilised.debits) || 0;
            const available = m.available?.live_balance || m.net || 0;
            html += metricCard(seg.charAt(0).toUpperCase() + seg.slice(1) + ' Net', '₹' + formatNum(m.net||0));
            html += metricCard(seg.charAt(0).toUpperCase() + seg.slice(1) + ' Available', '₹' + formatNum(available));
            html += metricCard(seg.charAt(0).toUpperCase() + seg.slice(1) + ' Used', '₹' + formatNum(used));
        });
        cards.innerHTML = html || '<div class="card" style="padding:20px;color:var(--text-muted)">No margin data</div>';
    } catch (e) {
        document.getElementById('margins-cards').innerHTML = '<div style="color:#ff5252">' + e.message + '</div>';
    }
}

/* ========================  JOURNAL  ======================== */
let _jAnalytics = null;
let _jPage = 1;
const _jPageSize = 50;

function switchJournalTab(id, btn) {
    document.querySelectorAll('.j-subtab').forEach(d => d.style.display = 'none');
    const el = document.getElementById(id);
    if (el) el.style.display = '';
    document.querySelectorAll('#j-subtabs .btn').forEach(b => { b.className = 'btn btn-sm btn-outline'; });
    if (btn) btn.className = 'btn btn-sm btn-primary';
}

function _jFilters() {
    const p = {};
    const s = v => v || undefined;
    const strategy = document.getElementById('j-filter-strategy').value;
    const instrument = document.getElementById('j-filter-instrument').value;
    const source = document.getElementById('j-filter-source').value;
    const direction = document.getElementById('j-filter-direction').value;
    const trade_type = document.getElementById('j-filter-type').value;
    const days = document.getElementById('j-filter-days').value;
    if (strategy) p.strategy = strategy;
    if (instrument) p.instrument = instrument;
    if (source) p.source = source;
    if (direction) p.direction = direction;
    if (trade_type) p.trade_type = trade_type;
    if (days) p.days = days;
    return p;
}

function _qs(params) {
    return Object.entries(params).filter(([,v]) => v !== undefined).map(([k,v]) => `${k}=${encodeURIComponent(v)}`).join('&');
}

async function loadJournal() {
    // Always load entries independently — don't let analytics failures block trade list
    _jPage = 1;
    loadJournalEntries();
    try {
        const f = _jFilters();
        const qs = _qs(f);
        // Use allSettled so one API failure doesn't block everything
        const results = await Promise.allSettled([
            fetch('/api/pro-journal/filters').then(r => r.json()),
            fetch('/api/pro-journal/summary' + (qs ? '?' + qs : '')).then(r => r.json()),
            fetch('/api/pro-journal/analytics' + (qs ? '?' + qs : '')).then(r => r.json()),
            fetch('/api/pro-journal/daily-pnl' + (qs ? '?' + qs : '')).then(r => r.json()),
            fetch('/api/pro-journal/strategy-breakdown' + (qs ? '?' + qs : '')).then(r => r.json()),
            fetch('/api/pro-journal/regime-matrix' + (qs ? '?' + qs : '')).then(r => r.json())
        ]);
        const filtersR = results[0].status === 'fulfilled' ? results[0].value : {};
        const summaryR = results[1].status === 'fulfilled' ? results[1].value : {};
        const analyticsR = results[2].status === 'fulfilled' ? results[2].value : {};
        const dailyR = results[3].status === 'fulfilled' ? results[3].value : {};
        const strategyR = results[4].status === 'fulfilled' ? results[4].value : {};
        const regimeR = results[5].status === 'fulfilled' ? results[5].value : {};
        // Populate filter dropdowns
        _populateJournalFilters(filtersR);
        // Summary cards
        _renderJournalSummary(summaryR, analyticsR);
        // Analytics
        _jAnalytics = analyticsR;
        // Unwrap nested API responses
        const dailyArr = dailyR.daily_pnl || dailyR || [];
        const stratArr = strategyR.strategies || strategyR || [];
        const regimeArr = regimeR.matrix || regimeR || [];
        _renderJournalOverview(analyticsR, dailyArr, stratArr, regimeArr);
        _renderExecutionTab(analyticsR);
        _renderEdgeTab(analyticsR);
    } catch (e) {
        console.error('Journal load error:', e);
    }
}

function _populateJournalFilters(data) {
    if (!data || data.error) return;
    const sSel = document.getElementById('j-filter-strategy');
    const iSel = document.getElementById('j-filter-instrument');
    const curS = sSel.value, curI = iSel.value;
    // Keep first option, add rest
    while (sSel.options.length > 1) sSel.remove(1);
    while (iSel.options.length > 1) iSel.remove(1);
    (data.strategies || []).forEach(s => { const o = new Option(s, s); sSel.add(o); });
    (data.instruments || []).forEach(i => { const o = new Option(i, i); iSel.add(o); });
    sSel.value = curS; iSel.value = curI;
}

function _renderJournalSummary(summary, analytics) {
    const s = summary || {};
    const a = analytics || {};
    const core = a.core_metrics || {};
    const risk = a.risk_adjusted || {};
    const con = a.consistency_score || {};
    document.getElementById('j-stat-total').textContent = s.total_trades || 0;
    const wr = core.win_rate != null ? (core.win_rate * 100).toFixed(1) + '%' : '—';
    document.getElementById('j-stat-winrate').textContent = wr;
    _colorVal('j-stat-winrate', (core.win_rate || 0) >= 0.5);
    document.getElementById('j-stat-expectancy').textContent = core.expectancy != null ? '₹' + core.expectancy.toFixed(0) : '—';
    _colorVal('j-stat-expectancy', (core.expectancy || 0) >= 0);
    document.getElementById('j-stat-pf').textContent = core.profit_factor != null ? core.profit_factor.toFixed(2) : '—';
    document.getElementById('j-stat-sharpe').textContent = risk.sharpe_ratio != null ? risk.sharpe_ratio.toFixed(2) : '—';
    document.getElementById('j-stat-sortino').textContent = risk.sortino_ratio != null ? risk.sortino_ratio.toFixed(2) : '—';
    const dd = risk.max_drawdown_pct != null ? risk.max_drawdown_pct.toFixed(1) + '%' : '—';
    document.getElementById('j-stat-dd').textContent = dd;
    const cs = con.score != null ? con.score.toFixed(0) + ' ' + (con.rating || '') : '—';
    document.getElementById('j-stat-consistency').textContent = cs;
}

function _colorVal(id, positive) {
    const el = document.getElementById(id);
    if (el) el.style.color = positive ? '#00d4aa' : '#ff5252';
}

function _renderJournalOverview(analytics, dailyData, strategyData, regimeData) {
    // Daily PnL chart
    _drawDailyPnLChart(dailyData);
    // Equity curve
    _drawEquityChart(dailyData);
    // Strategy breakdown table
    const sb = document.getElementById('j-strategy-body');
    sb.innerHTML = '';
    (strategyData || []).forEach(s => {
        const pnlColor = (s.total_pnl || 0) >= 0 ? '#00d4aa' : '#ff5252';
        const wr = s.win_rate != null ? (s.win_rate * 100).toFixed(1) + '%' : '—';
        sb.innerHTML += `<tr>
            <td>${s.strategy || s.strategy_name || '—'}</td>
            <td>${s.trades || 0}</td>
            <td>${wr}</td>
            <td style="color:${pnlColor}">₹${(s.total_pnl || 0).toFixed(0)}</td>
            <td>₹${(s.avg_pnl || 0).toFixed(0)}</td>
            <td>${s.profit_factor != null ? s.profit_factor.toFixed(2) : '—'}</td>
        </tr>`;
    });
    if (!strategyData || strategyData.length === 0) sb.innerHTML = '<tr><td colspan="6" style="color:#667788">No data</td></tr>';
    // Regime matrix
    const rb = document.getElementById('j-regime-body');
    rb.innerHTML = '';
    (regimeData || []).forEach(r => {
        const pnlColor = (r.total_pnl || 0) >= 0 ? '#00d4aa' : '#ff5252';
        const wr = r.win_rate != null ? (r.win_rate * 100).toFixed(1) + '%' : '—';
        rb.innerHTML += `<tr>
            <td>${r.regime || '—'}</td>
            <td>${r.trades || 0}</td>
            <td>${wr}</td>
            <td style="color:${pnlColor}">₹${(r.total_pnl || 0).toFixed(0)}</td>
            <td>₹${(r.avg_pnl || 0).toFixed(0)}</td>
        </tr>`;
    });
    if (!regimeData || regimeData.length === 0) rb.innerHTML = '<tr><td colspan="5" style="color:#667788">No data</td></tr>';
}

function _drawDailyPnLChart(data) {
    const canvas = document.getElementById('j-daily-pnl-chart');
    if (!canvas || !data || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 32;
    const H = canvas.height = 220;
    ctx.clearRect(0, 0, W, H);
    const pnls = data.map(d => d.pnl || 0);
    const maxAbs = Math.max(Math.abs(Math.min(...pnls)), Math.abs(Math.max(...pnls)), 1);
    const pad = { t: 20, b: 40, l: 60, r: 20 };
    const cW = W - pad.l - pad.r;
    const cH = H - pad.t - pad.b;
    const barW = Math.max(2, (cW / pnls.length) - 1);
    const midY = pad.t + cH / 2;
    // Zero line
    ctx.strokeStyle = '#2a3a4a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, midY); ctx.lineTo(W - pad.r, midY); ctx.stroke();
    // Bars
    pnls.forEach((p, i) => {
        const x = pad.l + (i / pnls.length) * cW;
        const h = (Math.abs(p) / maxAbs) * (cH / 2);
        ctx.fillStyle = p >= 0 ? '#00d4aa' : '#ff5252';
        if (p >= 0) ctx.fillRect(x, midY - h, barW, h);
        else ctx.fillRect(x, midY, barW, h);
    });
    // Y axis labels
    ctx.fillStyle = '#667788'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText('+₹' + maxAbs.toFixed(0), pad.l - 4, pad.t + 10);
    ctx.fillText('0', pad.l - 4, midY + 4);
    ctx.fillText('-₹' + maxAbs.toFixed(0), pad.l - 4, H - pad.b);
    // X axis: first and last date
    ctx.textAlign = 'center';
    if (data.length > 0) {
        ctx.fillText(data[0].date || '', pad.l + 20, H - 10);
        ctx.fillText(data[data.length - 1].date || '', W - pad.r - 20, H - 10);
    }
}

function _drawEquityChart(dailyData) {
    const canvas = document.getElementById('j-equity-chart');
    if (!canvas || !dailyData || dailyData.length === 0) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 32;
    const H = canvas.height = 220;
    ctx.clearRect(0, 0, W, H);
    // Build cumulative equity
    let cum = 0;
    const eqs = dailyData.map(d => { cum += (d.pnl || 0); return cum; });
    const mn = Math.min(...eqs, 0);
    const mx = Math.max(...eqs, 1);
    const range = mx - mn || 1;
    const pad = { t: 20, b: 40, l: 60, r: 20 };
    const cW = W - pad.l - pad.r;
    const cH = H - pad.t - pad.b;
    // Gradient fill
    const grad = ctx.createLinearGradient(0, pad.t, 0, H - pad.b);
    grad.addColorStop(0, 'rgba(0,212,170,0.3)');
    grad.addColorStop(1, 'rgba(0,212,170,0.02)');
    ctx.beginPath();
    eqs.forEach((v, i) => {
        const x = pad.l + (i / (eqs.length - 1 || 1)) * cW;
        const y = pad.t + cH - ((v - mn) / range) * cH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    // Fill under curve
    const lastX = pad.l + cW;
    ctx.lineTo(lastX, pad.t + cH); ctx.lineTo(pad.l, pad.t + cH); ctx.closePath();
    ctx.fillStyle = grad; ctx.fill();
    // Line
    ctx.beginPath();
    eqs.forEach((v, i) => {
        const x = pad.l + (i / (eqs.length - 1 || 1)) * cW;
        const y = pad.t + cH - ((v - mn) / range) * cH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#00d4aa'; ctx.lineWidth = 2; ctx.stroke();
    // Y labels
    ctx.fillStyle = '#667788'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText('₹' + mx.toFixed(0), pad.l - 4, pad.t + 10);
    ctx.fillText('₹' + mn.toFixed(0), pad.l - 4, H - pad.b);
}

async function loadJournalEntries() {
    try {
        const f = _jFilters();
        f.page = _jPage; f.page_size = _jPageSize;
        const review = document.getElementById('j-trades-review').value;
        if (review) f.review_status = review;
        const qs = _qs(f);
        const data = await fetch('/api/pro-journal/entries?' + qs).then(r => r.json());
        _renderTradesTable(data);
    } catch (e) {
        console.error('Load entries error:', e);
    }
}

function _renderTradesTable(data) {
    const tb = document.getElementById('j-trades-body');
    tb.innerHTML = '';
    const entries = Array.isArray(data.entries) ? data.entries : Array.isArray(data) ? data : [];
    if (entries.length === 0) {
        tb.innerHTML = '<tr><td colspan="14" style="color:#667788;text-align:center">No journal entries yet. Trades from backtests, paper trading, and live trading will appear here.</td></tr>';
        document.getElementById('j-trades-pagination').innerHTML = '';
        return;
    }
    entries.forEach(e => {
        const pnl = e.net_pnl || e.gross_pnl || 0;
        const pnlColor = pnl >= 0 ? '#00d4aa' : '#ff5252';
        const ex = e.execution || {};
        const slip = ex.entry_slippage_pct != null ? Math.abs(ex.entry_slippage_pct).toFixed(2) + '%' : '—';
        const mae = e.mae_pct != null ? e.mae_pct.toFixed(1) + '%' : '—';
        const mfe = e.mfe_pct != null ? e.mfe_pct.toFixed(1) + '%' : '—';
        const edge = e.edge_ratio != null ? e.edge_ratio.toFixed(2) : '—';
        const review = e.review_status || 'unreviewed';
        const reviewClass = review === 'reviewed' ? 'color:#00d4aa' : review === 'flagged' ? 'color:#ff5252' : 'color:#ffaa00';
        const entryT = e.entry_time ? new Date(e.entry_time).toLocaleString('en-IN', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit' }) : '—';
        const entryPrice = ex.actual_fill_price || 0;
        const exitPrice = ex.actual_exit_price || 0;
        tb.innerHTML += `<tr>
            <td style="font-size:11px">${entryT}</td>
            <td>${e.strategy_name || e.strategy || '—'}</td>
            <td>${e.instrument || '—'}</td>
            <td>${e.direction || '—'}</td>
            <td>${e.trade_type || '—'}</td>
            <td>₹${entryPrice.toFixed(2)}</td>
            <td>₹${exitPrice.toFixed(2)}</td>
            <td style="color:${pnlColor};font-weight:600">₹${pnl.toFixed(0)}</td>
            <td>${slip}</td>
            <td>${mae}</td>
            <td>${mfe}</td>
            <td>${edge}</td>
            <td style="${reviewClass};font-size:11px">${review}</td>
            <td><button class="btn btn-outline btn-sm" onclick="viewJournalEntry('${e.entry_id || e.trade_id || e.id}')">View</button></td>
        </tr>`;
    });
    // Pagination
    const pag = document.getElementById('j-trades-pagination');
    const total = data.total || entries.length;
    const pages = Math.ceil(total / _jPageSize);
    let html = '';
    for (let p = 1; p <= Math.min(pages, 10); p++) {
        const cls = p === _jPage ? 'btn btn-primary btn-sm' : 'btn btn-outline btn-sm';
        html += `<button class="${cls}" onclick="_jPage=${p};loadJournalEntries()">${p}</button>`;
    }
    pag.innerHTML = html;
}

async function viewJournalEntry(id) {
    try {
        const data = await fetch(`/api/pro-journal/entry/${id}`).then(r => r.json());
        if (data.error) { alert(data.error); return; }
        let html = '<div style="max-width:700px;max-height:80vh;overflow-y:auto;padding:16px;background:#0d1821;border-radius:8px;border:1px solid #2a3a4a">';
        html += `<h3 style="color:#00d4aa;margin-bottom:12px">Trade: ${data.trade_id || id}</h3>`;
        html += `<div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;color:#c0c8d0">`;
        html += `<div><b>Strategy:</b> ${data.strategy || '—'}</div>`;
        html += `<div><b>Instrument:</b> ${data.instrument || '—'}</div>`;
        html += `<div><b>Direction:</b> ${data.direction || '—'}</div>`;
        html += `<div><b>Source:</b> ${data.source || '—'}</div>`;
        html += `<div><b>Entry:</b> ₹${(data.entry_price || 0).toFixed(2)}</div>`;
        html += `<div><b>Exit:</b> ₹${(data.exit_price || 0).toFixed(2)}</div>`;
        const pnl = data.net_pnl || 0;
        html += `<div><b>P&L:</b> <span style="color:${pnl >= 0 ? '#00d4aa' : '#ff5252'}">₹${pnl.toFixed(2)}</span></div>`;
        html += `<div><b>MAE/MFE:</b> ${(data.mae_pct || 0).toFixed(1)}% / ${(data.mfe_pct || 0).toFixed(1)}%</div>`;
        html += '</div>';
        // Execution details
        if (data.execution) {
            const ex = data.execution;
            html += '<div style="margin-top:12px;padding:10px;background:#111c26;border-radius:6px;font-size:12px;color:#a0aab4">';
            html += '<b style="color:#ffaa00">Execution Quality</b><br>';
            html += `Slippage: ${ex.slippage_pct != null ? ex.slippage_pct.toFixed(3) + '%' : '—'} | `;
            html += `Alpha: ${ex.execution_alpha != null ? ex.execution_alpha.toFixed(3) + '%' : '—'} | `;
            html += `Latency: ${ex.signal_to_fill_ms || '—'}ms`;
            html += '</div>';
        }
        // Strategy context
        if (data.strategy_context) {
            const sc = data.strategy_context;
            html += '<div style="margin-top:8px;padding:10px;background:#111c26;border-radius:6px;font-size:12px;color:#a0aab4">';
            html += '<b style="color:#00d4aa">Strategy Context</b><br>';
            html += `IV: ${sc.iv_level || '—'} (${sc.iv_percentile || '—'}%ile) | `;
            html += `Regime: ${sc.trend_regime || '—'} | VIX: ${sc.vix_level || '—'} | `;
            html += `Confidence: ${sc.signal_confidence || '—'}`;
            html += '</div>';
        }
        // Notes / Tags
        html += '<div style="margin-top:12px">';
        html += `<div style="font-size:12px;color:#667788;margin-bottom:4px">Tags: ${(data.tags || []).join(', ') || 'none'}</div>`;
        html += `<div style="font-size:12px;color:#667788">Notes: ${data.notes || 'none'}</div>`;
        html += '</div>';
        html += '<div style="margin-top:12px;display:flex;gap:8px">';
        html += `<button class="btn btn-outline btn-sm" onclick="updateJournalReview('${data.trade_id || id}','reviewed')">Mark Reviewed</button>`;
        html += `<button class="btn btn-sm" style="background:#ff5252;color:#fff" onclick="updateJournalReview('${data.trade_id || id}','flagged')">Flag</button>`;
        html += '</div></div>';
        // Show as overlay
        let overlay = document.getElementById('j-entry-modal');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.id = 'j-entry-modal';
            overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;z-index:999';
            overlay.onclick = e => { if (e.target === overlay) overlay.style.display = 'none'; };
            document.body.appendChild(overlay);
        }
        overlay.innerHTML = html;
        overlay.style.display = 'flex';
    } catch (e) {
        alert('Error loading entry: ' + e.message);
    }
}

async function updateJournalReview(id, status) {
    try {
        await fetch(`/api/pro-journal/entry/${id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ review_status: status })
        });
        const modal = document.getElementById('j-entry-modal');
        if (modal) modal.style.display = 'none';
        loadJournalEntries();
    } catch (e) {
        alert('Update failed: ' + e.message);
    }
}

function _renderExecutionTab(analytics) {
    const ex = analytics.execution_quality || {};
    document.getElementById('j-exec-avg-slip').textContent = ex.avg_slippage_pct != null ? ex.avg_slippage_pct.toFixed(3) + '%' : '—';
    document.getElementById('j-exec-max-slip').textContent = ex.max_slippage_pct != null ? ex.max_slippage_pct.toFixed(3) + '%' : '—';
    document.getElementById('j-exec-alpha').textContent = ex.avg_execution_alpha != null ? ex.avg_execution_alpha.toFixed(3) + '%' : '—';
    _colorVal('j-exec-alpha', (ex.avg_execution_alpha || 0) >= 0);
    document.getElementById('j-exec-latency').textContent = ex.avg_latency_ms != null ? ex.avg_latency_ms.toFixed(0) + 'ms' : '—';
    document.getElementById('j-exec-partial').textContent = ex.avg_partial_fills != null ? ex.avg_partial_fills.toFixed(1) : '—';
    document.getElementById('j-exec-cost-pnl').textContent = ex.avg_cost_pct_of_pnl != null ? ex.avg_cost_pct_of_pnl.toFixed(1) + '%' : '—';
    // Cost breakdown
    const cost = analytics.cost_analysis || {};
    const ct = document.getElementById('j-cost-table');
    if (cost.breakdown) {
        const b = cost.breakdown;
        ct.innerHTML = `<div style="font-size:12px;color:#c0c8d0;display:grid;grid-template-columns:1fr 1fr;gap:4px">
            <div>Brokerage: ₹${(b.brokerage || 0).toFixed(0)}</div>
            <div>STT: ₹${(b.stt || 0).toFixed(0)}</div>
            <div>Exchange: ₹${(b.exchange_txn || b.exchange || 0).toFixed(0)}</div>
            <div>GST: ₹${(b.gst || 0).toFixed(0)}</div>
            <div>Stamp: ₹${(b.stamp || 0).toFixed(0)}</div>
            <div>Slippage: ₹${(b.slippage || 0).toFixed(0)}</div>
            <div style="grid-column:span 2;margin-top:4px;color:#ffaa00">Edge Erosion: ${(cost.edge_erosion_pct || 0).toFixed(1)}%</div>
        </div>`;
    }
    // Slippage drift chart
    _loadSlippageDriftChart();
    // Cost pie chart
    _drawCostChart(cost);
}

async function _loadSlippageDriftChart() {
    try {
        const f = _jFilters();
        const qs = _qs(f);
        const data = await fetch('/api/pro-journal/slippage-drift' + (qs ? '?' + qs : '')).then(r => r.json());
        _drawSlippageDrift(data.history || []);
    } catch (e) { console.error('Slippage drift error:', e); }
}

function _drawSlippageDrift(data) {
    const canvas = document.getElementById('j-slippage-chart');
    if (!canvas || !data || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 32;
    const H = canvas.height = 220;
    ctx.clearRect(0, 0, W, H);
    const vals = data.map(d => d.slippage_pct || d.avg_slippage || 0);
    const mn = Math.min(...vals, 0);
    const mx = Math.max(...vals, 0.01);
    const range = mx - mn || 0.01;
    const pad = { t: 20, b: 40, l: 60, r: 20 };
    const cW = W - pad.l - pad.r;
    const cH = H - pad.t - pad.b;
    // Line
    ctx.beginPath();
    vals.forEach((v, i) => {
        const x = pad.l + (i / (vals.length - 1 || 1)) * cW;
        const y = pad.t + cH - ((v - mn) / range) * cH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#ffaa00'; ctx.lineWidth = 2; ctx.stroke();
    // Rolling avg line if available
    if (data[0] && data[0].rolling_avg != null) {
        ctx.beginPath();
        data.forEach((d, i) => {
            if (d.rolling_avg == null) return;
            const x = pad.l + (i / (data.length - 1 || 1)) * cW;
            const y = pad.t + cH - ((d.rolling_avg - mn) / range) * cH;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.strokeStyle = '#ff5252'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]); ctx.stroke();
        ctx.setLineDash([]);
    }
    // Labels
    ctx.fillStyle = '#667788'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText(mx.toFixed(3) + '%', pad.l - 4, pad.t + 10);
    ctx.fillText(mn.toFixed(3) + '%', pad.l - 4, H - pad.b);
}

function _drawCostChart(cost) {
    const canvas = document.getElementById('j-cost-chart');
    if (!canvas || !cost || !cost.breakdown) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 32;
    const H = canvas.height = 220;
    ctx.clearRect(0, 0, W, H);
    const b = cost.breakdown;
    const items = [
        { label: 'Brokerage', val: b.brokerage || 0, color: '#00d4aa' },
        { label: 'STT', val: b.stt || 0, color: '#3399ff' },
        { label: 'Exchange', val: b.exchange_txn || b.exchange || 0, color: '#ffaa00' },
        { label: 'GST', val: b.gst || 0, color: '#ff5252' },
        { label: 'Stamp', val: b.stamp || 0, color: '#aa66ff' },
        { label: 'Slippage', val: b.slippage || 0, color: '#ff77aa' }
    ].filter(i => i.val > 0);
    const total = items.reduce((s, i) => s + i.val, 0) || 1;
    const cx = W / 2, cy = H / 2, r = Math.min(cx, cy) - 30;
    let angle = -Math.PI / 2;
    items.forEach(item => {
        const slice = (item.val / total) * Math.PI * 2;
        ctx.beginPath(); ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r, angle, angle + slice);
        ctx.fillStyle = item.color; ctx.fill();
        // Label
        const midAngle = angle + slice / 2;
        const lx = cx + (r + 18) * Math.cos(midAngle);
        const ly = cy + (r + 18) * Math.sin(midAngle);
        ctx.fillStyle = '#c0c8d0'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(item.label, lx, ly);
        angle += slice;
    });
}

function _renderEdgeTab(analytics) {
    const edge = analytics.edge_stability || {};
    const decay = analytics.edge_decay || {};
    const dist = analytics.trade_distribution || {};
    const mfe = analytics.mae_mfe_analysis || {};
    // Edge cards
    document.getElementById('j-edge-stability').textContent = edge.expectancy_stability_score != null ? edge.expectancy_stability_score.toFixed(1) + '%' : '—';
    _colorVal('j-edge-stability', (edge.expectancy_stability_score || 0) >= 70);
    const decayPct = decay.expectancy_decay_pct != null ? decay.expectancy_decay_pct.toFixed(1) + '%' : '—';
    document.getElementById('j-edge-decay').textContent = decayPct;
    _colorVal('j-edge-decay', (decay.expectancy_decay_pct || 0) <= 20);
    // Best/worst month
    const monthly = edge.monthly_performance || [];
    if (monthly.length > 0) {
        const best = monthly.reduce((a, b) => (b.pnl || 0) > (a.pnl || 0) ? b : a);
        const worst = monthly.reduce((a, b) => (b.pnl || 0) < (a.pnl || 0) ? b : a);
        document.getElementById('j-edge-best-month').textContent = best.month ? `${best.month}: ₹${(best.pnl || 0).toFixed(0)}` : '—';
        document.getElementById('j-edge-worst-month').textContent = worst.month ? `${worst.month}: ₹${(worst.pnl || 0).toFixed(0)}` : '—';
    }
    // Monthly table
    const mb = document.getElementById('j-monthly-body');
    mb.innerHTML = '';
    monthly.forEach(m => {
        const c = (m.pnl || 0) >= 0 ? '#00d4aa' : '#ff5252';
        mb.innerHTML += `<tr>
            <td>${m.month || '—'}</td>
            <td>${m.trades || 0}</td>
            <td>${m.win_rate != null ? (m.win_rate * 100).toFixed(1) + '%' : '—'}</td>
            <td style="color:${c}">₹${(m.pnl || 0).toFixed(0)}</td>
            <td>₹${(m.expectancy || 0).toFixed(0)}</td>
            <td>${m.profit_factor != null ? m.profit_factor.toFixed(2) : '—'}</td>
        </tr>`;
    });
    if (monthly.length === 0) mb.innerHTML = '<tr><td colspan="6" style="color:#667788">No data</td></tr>';
    // Rolling Sharpe chart
    _drawRollingSharpe(edge.rolling_sharpe || edge.rolling_sharpe_20 || []);
    // Day of week chart
    _drawBarChart('j-dow-chart', dist.by_day_of_week || dist.by_day || {}, 'Day of Week P&L');
    // Session chart
    _drawBarChart('j-session-chart', dist.by_session || {}, 'Session P&L');
    // MAE/MFE stats
    const mfeStat = document.getElementById('j-mae-mfe-stats');
    if (mfe) {
        mfeStat.innerHTML = `
            <b>Winners MFE Capture:</b> ${mfe.winners_mfe_capture_pct != null ? mfe.winners_mfe_capture_pct.toFixed(1) + '%' : '—'}<br>
            <b>Losers MAE→Loss Ratio:</b> ${mfe.losers_mae_to_loss_pct != null ? mfe.losers_mae_to_loss_pct.toFixed(2) + '%' : '—'}<br>
            <b>Interpretation:</b> ${mfe.interpretation || '—'}
        `;
    }
    // Confidence buckets
    const cb = document.getElementById('j-confidence-body');
    cb.innerHTML = '';
    const buckets = dist.by_confidence || {};
    Object.entries(buckets).forEach(([bk, bv]) => {
        const c = (bv.total_pnl || 0) >= 0 ? '#00d4aa' : '#ff5252';
        cb.innerHTML += `<tr>
            <td>${bk}</td>
            <td>${bv.trades || 0}</td>
            <td>${bv.win_rate != null ? (bv.win_rate * 100).toFixed(1) + '%' : '—'}</td>
            <td style="color:${c}">₹${(bv.total_pnl || 0).toFixed(0)}</td>
            <td>₹${(bv.avg_pnl || 0).toFixed(0)}</td>
        </tr>`;
    });
    if (Object.keys(buckets).length === 0) cb.innerHTML = '<tr><td colspan="5" style="color:#667788">No data</td></tr>';
}

function _drawRollingSharpe(data) {
    const canvas = document.getElementById('j-rolling-sharpe-chart');
    if (!canvas || !data || data.length === 0) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 32;
    const H = canvas.height = 220;
    ctx.clearRect(0, 0, W, H);
    const vals = data.map(d => typeof d === 'number' ? d : d.sharpe || 0);
    const mn = Math.min(...vals, 0);
    const mx = Math.max(...vals, 1);
    const range = mx - mn || 1;
    const pad = { t: 20, b: 30, l: 50, r: 20 };
    const cW = W - pad.l - pad.r;
    const cH = H - pad.t - pad.b;
    // Zero line
    const zeroY = pad.t + cH - ((0 - mn) / range) * cH;
    ctx.strokeStyle = '#2a3a4a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, zeroY); ctx.lineTo(W - pad.r, zeroY); ctx.stroke();
    // Line
    ctx.beginPath();
    vals.forEach((v, i) => {
        const x = pad.l + (i / (vals.length - 1 || 1)) * cW;
        const y = pad.t + cH - ((v - mn) / range) * cH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#3399ff'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#667788'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText(mx.toFixed(2), pad.l - 4, pad.t + 10);
    ctx.fillText(mn.toFixed(2), pad.l - 4, H - pad.b);
}

function _drawBarChart(canvasId, dataObj, title) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 32;
    const H = canvas.height = 200;
    ctx.clearRect(0, 0, W, H);
    const keys = Object.keys(dataObj);
    if (keys.length === 0) return;
    const vals = keys.map(k => {
        const v = dataObj[k];
        return typeof v === 'number' ? v : (v.total_pnl || v.pnl || v.avg_pnl || 0);
    });
    const maxAbs = Math.max(Math.abs(Math.min(...vals)), Math.abs(Math.max(...vals)), 1);
    const pad = { t: 20, b: 50, l: 50, r: 20 };
    const cW = W - pad.l - pad.r;
    const cH = H - pad.t - pad.b;
    const barW = Math.max(20, cW / keys.length - 8);
    const midY = pad.t + cH / 2;
    ctx.strokeStyle = '#2a3a4a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, midY); ctx.lineTo(W - pad.r, midY); ctx.stroke();
    keys.forEach((k, i) => {
        const x = pad.l + (i / keys.length) * cW + 4;
        const h = (Math.abs(vals[i]) / maxAbs) * (cH / 2);
        ctx.fillStyle = vals[i] >= 0 ? '#00d4aa' : '#ff5252';
        if (vals[i] >= 0) ctx.fillRect(x, midY - h, barW, h);
        else ctx.fillRect(x, midY, barW, h);
        // Label
        ctx.fillStyle = '#667788'; ctx.font = '10px sans-serif'; ctx.textAlign = 'center';
        ctx.save(); ctx.translate(x + barW / 2, H - 10);
        ctx.fillText(k.substring(0, 8), 0, 0);
        ctx.restore();
    });
}

async function loadPortfolioHealth() {
    try {
        const [health, history] = await Promise.all([
            fetch('/api/pro-journal/portfolio-health').then(r => r.json()),
            fetch('/api/pro-journal/portfolio-health/history').then(r => r.json())
        ]);
        document.getElementById('j-port-equity').textContent = health.equity != null ? '₹' + health.equity.toFixed(0) : '—';
        const dp = health.daily_pnl || 0;
        document.getElementById('j-port-day-pnl').textContent = '₹' + dp.toFixed(0);
        _colorVal('j-port-day-pnl', dp >= 0);
        const dd = health.rolling_drawdown || 0;
        document.getElementById('j-port-dd').textContent = dd.toFixed(1) + '%';
        document.getElementById('j-port-open').textContent = health.open_positions || 0;
        document.getElementById('j-port-margin').textContent = health.margin_utilization != null ? health.margin_utilization.toFixed(1) + '%' : '—';
        // Exposure breakdown
        const exposure = health.exposure_by_symbol || {};
        const expEl = document.getElementById('j-exposure-content');
        if (Object.keys(exposure).length > 0) {
            let html = '<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px">';
            Object.entries(exposure).forEach(([sym, val]) => {
                html += `<div><b>${sym}:</b> ₹${val.toFixed(0)}</div>`;
            });
            html += '</div>';
            expEl.innerHTML = html;
        } else {
            expEl.innerHTML = 'No open exposure';
        }
        // Drawdown chart & capital table from history
        _drawDrawdownChart(history);
        _renderCapitalHistory(history);
    } catch (e) {
        console.error('Portfolio health error:', e);
    }
}

function _drawDrawdownChart(history) {
    const canvas = document.getElementById('j-dd-chart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 32;
    const H = canvas.height = 220;
    ctx.clearRect(0, 0, W, H);
    const snapshots = history.snapshots || [];
    if (snapshots.length === 0) return;
    const dds = snapshots.map(s => s.drawdown_pct || s.rolling_drawdown || 0);
    const mx = Math.max(...dds.map(Math.abs), 1);
    const pad = { t: 20, b: 30, l: 50, r: 20 };
    const cW = W - pad.l - pad.r;
    const cH = H - pad.t - pad.b;
    ctx.fillStyle = 'rgba(255,82,82,0.2)';
    ctx.beginPath();
    ctx.moveTo(pad.l, pad.t);
    dds.forEach((d, i) => {
        const x = pad.l + (i / (dds.length - 1 || 1)) * cW;
        const y = pad.t + (Math.abs(d) / mx) * cH;
        ctx.lineTo(x, y);
    });
    ctx.lineTo(pad.l + cW, pad.t); ctx.closePath(); ctx.fill();
    ctx.beginPath();
    dds.forEach((d, i) => {
        const x = pad.l + (i / (dds.length - 1 || 1)) * cW;
        const y = pad.t + (Math.abs(d) / mx) * cH;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = '#ff5252'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#667788'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
    ctx.fillText('0%', pad.l - 4, pad.t + 4);
    ctx.fillText('-' + mx.toFixed(1) + '%', pad.l - 4, H - pad.b);
}

function _renderCapitalHistory(history) {
    const tb = document.getElementById('j-capital-body');
    tb.innerHTML = '';
    const snaps = (history.snapshots || []).slice(-20).reverse();
    snaps.forEach(s => {
        tb.innerHTML += `<tr>
            <td style="font-size:11px">${s.timestamp ? new Date(s.timestamp).toLocaleString('en-IN', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit' }) : '—'}</td>
            <td>₹${(s.equity || 0).toFixed(0)}</td>
            <td>${(s.risk_pct || 0).toFixed(1)}%</td>
            <td>${(s.margin_pct || 0).toFixed(1)}%</td>
            <td style="color:#ff5252">${(s.drawdown_pct || 0).toFixed(1)}%</td>
            <td>${s.open_positions || 0}</td>
        </tr>`;
    });
    if (snaps.length === 0) tb.innerHTML = '<tr><td colspan="6" style="color:#667788">No snapshots</td></tr>';
}

async function loadFnOAnalytics() {
    try {
        const f = _jFilters();
        const qs = _qs(f);
        const data = await fetch('/api/pro-journal/fno-analytics' + (qs ? '?' + qs : '')).then(r => r.json());
        const fno = data.fno_analytics || data || {};
        // IV buckets
        const ivBody = document.getElementById('j-fno-iv-body');
        ivBody.innerHTML = '';
        const ivBuckets = fno.iv_percentile_analysis || [];
        ivBuckets.forEach(b => {
            const c = (b.total_pnl || 0) >= 0 ? '#00d4aa' : '#ff5252';
            ivBody.innerHTML += `<tr>
                <td>${b.bucket || '—'}</td>
                <td>${b.trades || 0}</td>
                <td>${b.win_rate != null ? (b.win_rate * 100).toFixed(1) + '%' : '—'}</td>
                <td>₹${(b.avg_pnl || 0).toFixed(0)}</td>
                <td style="color:${c}">₹${(b.total_pnl || 0).toFixed(0)}</td>
            </tr>`;
        });
        if (ivBuckets.length === 0) ivBody.innerHTML = '<tr><td colspan="5" style="color:#667788">No F&O data</td></tr>';
        // DTE buckets
        const dteBody = document.getElementById('j-fno-dte-body');
        dteBody.innerHTML = '';
        const dteBuckets = fno.dte_analysis || [];
        dteBuckets.forEach(b => {
            const c = (b.total_pnl || 0) >= 0 ? '#00d4aa' : '#ff5252';
            dteBody.innerHTML += `<tr>
                <td>${b.bucket || '—'}</td>
                <td>${b.trades || 0}</td>
                <td>${b.win_rate != null ? (b.win_rate * 100).toFixed(1) + '%' : '—'}</td>
                <td>₹${(b.avg_pnl || 0).toFixed(0)}</td>
                <td style="color:${c}">₹${(b.total_pnl || 0).toFixed(0)}</td>
            </tr>`;
        });
        if (dteBuckets.length === 0) dteBody.innerHTML = '<tr><td colspan="5" style="color:#667788">No F&O data</td></tr>';
        // Greeks attribution
        const greeks = fno.greeks_attribution || {};
        const gs = document.getElementById('j-greeks-stats');
        gs.innerHTML = `
            <b>Delta P&L:</b> ₹${(greeks.delta_pnl || 0).toFixed(0)}<br>
            <b>Gamma P&L:</b> ₹${(greeks.gamma_pnl || 0).toFixed(0)}<br>
            <b>Theta P&L:</b> ₹${(greeks.theta_pnl || 0).toFixed(0)}<br>
            <b>Vega P&L:</b> ₹${(greeks.vega_pnl || 0).toFixed(0)}
        `;
        _drawGreeksPnLChart(greeks);
    } catch (e) {
        console.error('F&O analytics error:', e);
    }
}

function _drawGreeksPnLChart(greeks) {
    const canvas = document.getElementById('j-greeks-pnl-chart');
    if (!canvas || !greeks) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width = canvas.parentElement.clientWidth - 32;
    const H = canvas.height = 220;
    ctx.clearRect(0, 0, W, H);
    const items = [
        { label: 'Delta', val: greeks.delta_pnl || 0, color: '#00d4aa' },
        { label: 'Gamma', val: greeks.gamma_pnl || 0, color: '#3399ff' },
        { label: 'Theta', val: greeks.theta_pnl || 0, color: '#ffaa00' },
        { label: 'Vega', val: greeks.vega_pnl || 0, color: '#aa66ff' }
    ];
    const maxAbs = Math.max(...items.map(i => Math.abs(i.val)), 1);
    const pad = { t: 20, b: 40, l: 60, r: 20 };
    const cW = W - pad.l - pad.r;
    const cH = H - pad.t - pad.b;
    const barW = Math.min(60, cW / items.length - 16);
    const midY = pad.t + cH / 2;
    ctx.strokeStyle = '#2a3a4a'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, midY); ctx.lineTo(W - pad.r, midY); ctx.stroke();
    items.forEach((item, i) => {
        const x = pad.l + (i / items.length) * cW + (cW / items.length - barW) / 2;
        const h = (Math.abs(item.val) / maxAbs) * (cH / 2);
        ctx.fillStyle = item.color;
        if (item.val >= 0) ctx.fillRect(x, midY - h, barW, h);
        else ctx.fillRect(x, midY, barW, h);
        ctx.fillStyle = '#c0c8d0'; ctx.font = '11px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(item.label, x + barW / 2, H - 10);
        ctx.fillText('₹' + item.val.toFixed(0), x + barW / 2, item.val >= 0 ? midY - h - 6 : midY + h + 14);
    });
}

async function loadSystemEvents() {
    try {
        const [events, stats] = await Promise.all([
            fetch('/api/pro-journal/system-events?limit=50').then(r => r.json()),
            fetch('/api/pro-journal/db-stats').then(r => r.json())
        ]);
        // System event counters
        let disconnects = 0, rejections = 0, recoveries = 0;
        (events || []).forEach(e => {
            if (e.event_type === 'ws_disconnect') disconnects++;
            else if (e.event_type === 'broker_rejection') rejections++;
            else if (e.event_type === 'crash_recovery') recoveries++;
        });
        document.getElementById('j-sys-disconnects').textContent = disconnects;
        document.getElementById('j-sys-rejections').textContent = rejections;
        document.getElementById('j-sys-recoveries').textContent = recoveries;
        // DB stats
        if (stats && stats.db_size_mb != null) {
            document.getElementById('j-sys-dbsize').textContent = stats.db_size_mb.toFixed(1) + ' MB';
        }
        // Events table
        const tb = document.getElementById('j-events-body');
        tb.innerHTML = '';
        (events || []).forEach(e => {
            const sevColor = e.severity === 'critical' ? '#ff5252' : e.severity === 'warning' ? '#ffaa00' : '#667788';
            const t = e.timestamp ? new Date(e.timestamp).toLocaleString('en-IN', { day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit', second:'2-digit' }) : '—';
            tb.innerHTML += `<tr>
                <td style="font-size:11px">${t}</td>
                <td>${e.event_type || '—'}</td>
                <td style="color:${sevColor}">${e.severity || '—'}</td>
                <td style="font-size:11px">${e.details || '—'}</td>
            </tr>`;
        });
        if (!events || events.length === 0) {
            tb.innerHTML = '<tr><td colspan="4" style="color:#667788">No system events</td></tr>';
        }
    } catch (e) {
        console.error('System events error:', e);
    }
}

async function exportJournal() {
    try {
        const data = await fetch('/api/pro-journal/export').then(r => r.json());
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url; a.download = 'journal_export_' + new Date().toISOString().slice(0, 10) + '.json';
        a.click(); URL.revokeObjectURL(url);
    } catch (e) {
        alert('Export failed: ' + e.message);
    }
}

// ────────────────────────────────────────────────────────────
// Strategy Health Report Rendering
// ────────────────────────────────────────────────────────────

function displayHealthReport(health, prefix) {
    prefix = prefix || 'bt';
    const panel = document.getElementById(prefix + '-health-report');
    if (!panel || !health || health.error) {
        if (panel) panel.style.display = 'none';
        return;
    }
    panel.style.display = '';

    const score = health.overall_score || 0;
    const verdict = health.overall_verdict || 'FAIL';
    const grade = health.grade || '--';
    const ready = health.execution_ready;

    // Badge
    const badge = document.getElementById(prefix + '-health-badge') || document.getElementById('health-badge');
    const badgeColors = { PASS: '#00d4aa', WARN: '#ffc107', FAIL: '#ff5252' };
    if (badge) {
        badge.style.background = badgeColors[verdict] || '#666';
        badge.style.color = verdict === 'WARN' ? '#000' : '#fff';
        badge.textContent = grade + ' (' + score.toFixed(0) + '/100)';
    }

    // Overall & execution readiness
    const readyIcon = ready ? '✅' : '🚫';
    const readyText = ready ? 'Strategy is execution-ready' : 'Strategy NOT execution-ready — resolve blockers';
    const readyColor = ready ? '#00d4aa' : '#ff5252';
    const overallEl = document.getElementById(prefix + '-health-overall') || document.getElementById('health-overall');
    if (overallEl) overallEl.innerHTML =
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">' +
        '<span style="font-size:28px;font-weight:800;color:' + badgeColors[verdict] + '">' + score.toFixed(0) + '</span>' +
        '<div><div style="font-size:14px;font-weight:600;color:' + readyColor + '">' + readyIcon + ' ' + readyText + '</div>' +
        '<div style="font-size:11px;color:var(--text-muted)">Score threshold: 60 | No critical pillar failures</div></div></div>';

    // 8 Pillars
    const pillars = health.pillars || {};
    const pillarOrder = ['profitability','drawdown','trade_quality','robustness','execution','capital_efficiency','risk_architecture','psychological'];
    const pillarLabels = {
        profitability: 'Core Profitability',
        drawdown: 'Drawdown Control',
        trade_quality: 'Trade Quality',
        robustness: 'Robustness',
        execution: 'Execution Reality',
        capital_efficiency: 'Capital Efficiency',
        risk_architecture: 'Risk Architecture',
        psychological: 'Psychological'
    };
    const pillarIcons = {
        profitability: '💰', drawdown: '📉', trade_quality: '🎯',
        robustness: '🛡️', execution: '⚡', capital_efficiency: '💹',
        risk_architecture: '🏗️', psychological: '🧠'
    };

    let pillarsHtml = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:8px">';
    for (const key of pillarOrder) {
        const p = pillars[key];
        if (!p) continue;
        const pScore = p.score || 0;
        const pVerdict = p.verdict || 'FAIL';
        const vColor = badgeColors[pVerdict] || '#666';
        const barWidth = Math.min(100, Math.max(0, pScore));

        // Key metrics for this pillar
        const metrics = p.metrics || {};
        let metricsHtml = '';
        const metricDisplay = _getKeyMetricsForPillar(key, metrics);
        if (metricDisplay.length > 0) {
            metricsHtml = '<div style="font-size:10px;color:var(--text-muted);margin-top:4px;line-height:1.5">' +
                metricDisplay.map(m => '<span>' + m.label + ': <b>' + m.value + '</b></span>').join(' · ') + '</div>';
        }

        // Notes
        let notesHtml = '';
        if (p.notes && p.notes.length > 0) {
            notesHtml = '<div style="font-size:10px;color:' + vColor + ';margin-top:3px">' + p.notes[0] + '</div>';
        }

        pillarsHtml += '<div style="background:var(--card-bg);border:1px solid var(--border);border-radius:8px;padding:10px;border-left:3px solid ' + vColor + '">' +
            '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">' +
            '<span style="font-size:12px;font-weight:600">' + (pillarIcons[key]||'') + ' ' + (pillarLabels[key]||key) + '</span>' +
            '<span style="font-size:11px;font-weight:700;color:' + vColor + '">' + pVerdict + '</span></div>' +
            '<div style="height:6px;background:var(--border);border-radius:3px;overflow:hidden">' +
            '<div style="height:100%;width:' + barWidth + '%;background:' + vColor + ';border-radius:3px;transition:width 0.5s"></div></div>' +
            '<div style="font-size:10px;color:var(--text-muted);margin-top:2px">' + pScore.toFixed(0) + '/100</div>' +
            metricsHtml + notesHtml + '</div>';
    }
    pillarsHtml += '</div>';
    const pillarsEl = document.getElementById(prefix + '-health-pillars') || document.getElementById('health-pillars');
    if (pillarsEl) pillarsEl.innerHTML = pillarsHtml;

    // Blockers
    const blockers = health.blockers || [];
    const blockersEl = document.getElementById(prefix + '-health-blockers') || document.getElementById('health-blockers');
    if (blockersEl) {
        if (blockers.length > 0) {
            blockersEl.innerHTML =
                '<div style="background:#ff525220;border:1px solid #ff5252;border-radius:6px;padding:8px;margin-bottom:6px">' +
                '<div style="font-size:12px;font-weight:700;color:#ff5252;margin-bottom:4px">🚫 Blockers (must fix before execution)</div>' +
                blockers.map(b => '<div style="font-size:11px;color:#ff5252;padding:2px 0">• ' + b + '</div>').join('') + '</div>';
        } else {
            blockersEl.innerHTML = '';
        }
    }

    // Warnings
    const warns = health.warnings || [];
    const warnsEl = document.getElementById(prefix + '-health-warnings') || document.getElementById('health-warnings');
    if (warnsEl) {
        if (warns.length > 0) {
            warnsEl.innerHTML =
                '<div style="background:#ffc10720;border:1px solid #ffc107;border-radius:6px;padding:8px;margin-bottom:6px">' +
                '<div style="font-size:12px;font-weight:700;color:#ffc107;margin-bottom:4px">⚠️ Warnings</div>' +
                warns.map(w => '<div style="font-size:11px;color:#e0a800;padding:2px 0">• ' + w + '</div>').join('') + '</div>';
        } else {
            warnsEl.innerHTML = '';
        }
    }

    // Summary
    const summaryLines = health.summary || [];
    const summaryEl = document.getElementById(prefix + '-health-summary') || document.getElementById('health-summary');
    if (summaryEl && summaryLines.length > 0) {
        summaryEl.innerHTML =
            '<div style="font-size:12px;color:var(--text-muted);line-height:1.6">' +
            summaryLines.map(s => '<div>• ' + s + '</div>').join('') + '</div>';
    }
}

function _getKeyMetricsForPillar(key, metrics) {
    const m = [];
    const fmt = (v, d=2) => v != null ? Number(v).toFixed(d) : '--';
    switch(key) {
        case 'profitability':
            if (metrics.expectancy != null) m.push({label:'Expectancy', value:'₹'+fmt(metrics.expectancy)});
            if (metrics.expectancy_r != null) m.push({label:'Exp/R', value:fmt(metrics.expectancy_r,3)});
            if (metrics.profit_factor != null) m.push({label:'PF', value:fmt(metrics.profit_factor)});
            if (metrics.sharpe_ratio != null) m.push({label:'Sharpe', value:fmt(metrics.sharpe_ratio)});
            if (metrics.sortino_ratio != null) m.push({label:'Sortino', value:fmt(metrics.sortino_ratio)});
            break;
        case 'drawdown':
            if (metrics.max_drawdown_pct != null) m.push({label:'Max DD', value:fmt(metrics.max_drawdown_pct,1)+'%'});
            if (metrics.recovery_factor != null) m.push({label:'Recovery', value:fmt(metrics.recovery_factor)});
            if (metrics.equity_curve_r2 != null) m.push({label:'R²', value:fmt(metrics.equity_curve_r2,4)});
            if (metrics.ulcer_index != null) m.push({label:'Ulcer', value:fmt(metrics.ulcer_index)});
            break;
        case 'trade_quality':
            if (metrics.win_rate != null) m.push({label:'Win%', value:fmt(metrics.win_rate,1)+'%'});
            if (metrics.payoff_ratio != null) m.push({label:'Payoff', value:fmt(metrics.payoff_ratio)});
            if (metrics.win_loss_ratio != null) m.push({label:'W/L', value:fmt(metrics.win_loss_ratio)});
            if (metrics.total_trades != null) m.push({label:'Trades', value:metrics.total_trades});
            if (metrics.max_losing_streak != null) m.push({label:'Max Lose Streak', value:metrics.max_losing_streak});
            break;
        case 'robustness':
            if (metrics.monthly_win_rate != null) m.push({label:'Monthly Win%', value:fmt(metrics.monthly_win_rate,1)+'%'});
            if (metrics.top3_month_concentration != null) m.push({label:'Top3 Conc.', value:fmt(metrics.top3_month_concentration,1)+'%'});
            break;
        case 'execution':
            if (metrics.cost_drag_pct != null) m.push({label:'Cost Drag', value:fmt(metrics.cost_drag_pct,1)+'%'});
            if (metrics.slippage_pct != null) m.push({label:'Slippage', value:fmt(metrics.slippage_pct,1)+'%'});
            break;
        case 'capital_efficiency':
            if (metrics.cagr != null) m.push({label:'CAGR', value:fmt(metrics.cagr,1)+'%'});
            if (metrics.return_on_margin != null && metrics.return_on_margin !== null) m.push({label:'ROM', value:fmt(metrics.return_on_margin,1)+'%'});
            if (metrics.total_return_pct != null) m.push({label:'Return', value:fmt(metrics.total_return_pct,1)+'%'});
            break;
        case 'risk_architecture':
            if (metrics.position_sizing) m.push({label:'Sizing', value:metrics.position_sizing});
            if (metrics.has_stop_loss != null) m.push({label:'SL', value:metrics.has_stop_loss?'Yes':'No'});
            if (metrics.has_profit_target != null) m.push({label:'PT', value:metrics.has_profit_target?'Yes':'No'});
            break;
        case 'psychological':
            if (metrics.max_losing_streak != null) m.push({label:'Max Losing', value:metrics.max_losing_streak+' trades'});
            if (metrics.underwater_pct != null) m.push({label:'Underwater', value:fmt(metrics.underwater_pct,1)+'%'});
            if (metrics.worst_trade != null) m.push({label:'Worst', value:'₹'+fmt(metrics.worst_trade)});
            break;
    }
    return m;
}

document.addEventListener('DOMContentLoaded', () => {
    loadFullStatus();
    loadSignals();
    loadMarketOverview();
    connectWS();
    statusInterval = setInterval(loadStatus, 10000);
    portfolioInterval = setInterval(loadFullStatus, 30000);
    // Load custom strategies into all dropdowns early
    loadCustomStrategiesDropdowns();
    loadFnOCustomStrategiesDropdowns();
});
