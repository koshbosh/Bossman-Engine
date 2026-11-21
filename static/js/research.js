const STORAGE_KEY = 'bossman-research-desk-v1';
const INITIAL_STATE = {
  ticker: 'MSFT',
  watchlist: ['MSFT', 'NVDA', 'PLTR', 'AAPL'],
  page: 'overview',
  notes: {},
  scenarios: {}
};

const PAGE_TITLES = {
  overview: 'Company overview',
  filings: 'Filings & sources',
  thesis: 'Thesis builder',
  macro: 'Macro monitor'
};

let state = loadState();
let toastTimer;
let researchStatus = null;
let researchBusy = false;
let researchAnswers = {};
let researchErrors = {};

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.side-link').forEach(button => {
    button.addEventListener('click', () => showPage(button.dataset.page));
  });
  document.getElementById('company-search').addEventListener('submit', event => {
    event.preventDefault();
    addCompany(document.getElementById('search-input').value);
  });
  document.getElementById('add-watchlist').addEventListener('click', () => {
    const input = document.getElementById('search-input');
    input.focus();
    input.placeholder = 'Type a ticker, then press Enter…';
  });
  document.getElementById('dismiss-banner').addEventListener('click', () => {
    document.querySelector('.demo-banner').remove();
  });
  render();
  loadResearchStatus();
});

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    if (!saved || typeof saved !== 'object') return { ...INITIAL_STATE };
    const watchlist = Array.isArray(saved.watchlist)
      ? saved.watchlist.filter(ticker => isValidTicker(ticker) && ticker !== 'NSTR').slice(0, 20)
      : [...INITIAL_STATE.watchlist];
    const ticker = isValidTicker(saved.ticker) && saved.ticker !== 'NSTR'
      ? saved.ticker
      : INITIAL_STATE.ticker;
    if (!watchlist.includes(ticker)) watchlist.unshift(ticker);
    return {
      ...INITIAL_STATE,
      ticker,
      watchlist: watchlist.length ? watchlist : [...INITIAL_STATE.watchlist],
      page: Object.hasOwn(PAGE_TITLES, saved.page) ? saved.page : 'overview',
      notes: saved.notes && typeof saved.notes === 'object' ? saved.notes : {},
      scenarios: saved.scenarios && typeof saved.scenarios === 'object' ? saved.scenarios : {}
    };
  } catch (error) {
    console.warn('Could not restore the research workspace:', error);
    return { ...INITIAL_STATE };
  }
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
    return true;
  } catch (error) {
    console.error('Could not save the research workspace:', error);
    showToast('Your browser could not save this workspace. Check storage permissions.');
    return false;
  }
}

function isValidTicker(value) {
  return typeof value === 'string' && /^[A-Z0-9.-]{1,10}$/.test(value);
}

function addCompany(rawTicker) {
  const ticker = String(rawTicker || '').trim().toUpperCase();
  if (!isValidTicker(ticker)) {
    showToast('Enter a ticker using 1–10 letters, numbers, dots, or hyphens.');
    return;
  }
  if (!state.watchlist.includes(ticker)) {
    if (state.watchlist.length >= 20) {
      showToast('Your watchlist is full. Remove a ticker before adding another.');
      return;
    }
    state.watchlist.push(ticker);
  }
  state.ticker = ticker;
  state.page = 'overview';
  saveState();
  render();
  document.getElementById('search-input').value = '';
  showToast(`${ticker} added. Ask a question to retrieve SEC research.`);
}

function showPage(page) {
  if (!Object.hasOwn(PAGE_TITLES, page)) return;
  state.page = page;
  saveState();
  render();
}

function selectCompany(ticker) {
  state.ticker = ticker;
  state.page = 'overview';
  saveState();
  render();
}

function render() {
  document.getElementById('breadcrumb-current').textContent = PAGE_TITLES[state.page];
  document.querySelectorAll('.side-link').forEach(button => {
    const active = button.dataset.page === state.page;
    button.classList.toggle('active', active);
    button.setAttribute('aria-current', active ? 'page' : 'false');
  });
  renderWatchlist();
  const content = document.getElementById('page-content');
  if (state.page === 'overview') content.innerHTML = renderOverview();
  if (state.page === 'filings') content.innerHTML = renderFilings();
  if (state.page === 'thesis') content.innerHTML = renderThesis();
  if (state.page === 'macro') content.innerHTML = renderMacro();
  bindPageEvents();
}

function renderWatchlist() {
  const watchlist = document.getElementById('watchlist');
  watchlist.replaceChildren(...state.watchlist.map(ticker => {
    const row = document.createElement('div');
    row.className = `watch-row${ticker === state.ticker ? ' active' : ''}`;
    const button = document.createElement('button');
    button.className = 'watch-item';
    button.type = 'button';
    button.setAttribute('aria-label', `Open ${ticker} research`);
    const symbol = document.createElement('span');
    symbol.className = 'watch-symbol';
    symbol.textContent = ticker.slice(0, 4);
    const copy = document.createElement('span');
    copy.className = 'watch-copy';
    const label = document.createElement('strong');
    label.textContent = ticker;
    const name = document.createElement('small');
    name.textContent = 'SEC-backed research';
    copy.append(label, name);
    button.append(symbol, copy);
    button.addEventListener('click', () => selectCompany(ticker));
    const remove = document.createElement('button');
    remove.className = 'watch-remove';
    remove.type = 'button';
    remove.setAttribute('aria-label', `Remove ${ticker} from watchlist`);
    remove.textContent = '×';
    remove.addEventListener('click', () => removeCompany(ticker));
    row.append(button, remove);
    return row;
  }));
}

function removeCompany(ticker) {
  if (state.watchlist.length === 1) {
    showToast('Keep at least one company in your watchlist.');
    return;
  }
  state.watchlist = state.watchlist.filter(item => item !== ticker);
  if (state.ticker === ticker) state.ticker = state.watchlist[0];
  saveState();
  render();
}

function renderOverview() {
  const ticker = escapeHtml(state.ticker);
  const answer = researchAnswers[state.ticker];
  const statusMessage = researchStatus
    ? researchStatus.ready
      ? `Configured · SEC + ${escapeHtml(researchStatus.model)} · access token required`
      : `Needs setup · ${researchStatus.missing.map(escapeHtml).join(' and ')}`
    : 'Checking server configuration…';
  const answerHtml = answer
    ? `${answer.warnings && answer.warnings.length ? `<div class="research-error" role="note">${answer.warnings.map(escapeHtml).join('<br>')}</div>` : ''}
       ${answer.marketAnalysis ? renderMarketAnalysis(answer.marketAnalysis) : ''}
       <div class="research-answer">${renderAnswerText(answer.answer, answer.sources)}</div>
       <div class="research-sources"><div class="eyebrow">Sources · retrieved ${escapeHtml(answer.retrievedAt)}</div>
         ${answer.sources.map(source => `<a class="research-source" id="source-${escapeHtml(source.id)}" href="${escapeHtml(source.url)}" target="_blank" rel="noopener noreferrer"><span class="source-type">${escapeHtml(source.type)}</span><span><strong>${escapeHtml(source.title)}</strong><small>${escapeHtml(source.filingDate || 'SEC XBRL data')}</small></span><span class="source-arrow">↗</span></a>`).join('')}
       </div>`
    : researchErrors[state.ticker]
      ? `<div class="research-error" role="alert"><strong>Research request failed</strong><p>${escapeHtml(researchErrors[state.ticker])}</p></div>`
      : `<div class="research-empty"><span class="research-spark">✳</span><strong>Ask a company research question</strong><p>Answers use SEC company facts and recent 10-K/10-Q excerpts, with links back to each source.</p>
        <div class="prompt-suggestions">
          <button type="button" data-question="Summarize the business model and the most important disclosed risks.">Business &amp; risks</button>
          <button type="button" data-question="Summarize recent financial performance and cash generation, citing the reporting periods.">Financial performance</button>
          <button type="button" data-question="Build a bull, base, and bear thesis from the available evidence. State the key assumptions and what would invalidate each case.">Bull / base / bear</button>
          <button type="button" data-question="What can the available SEC evidence tell me about an entry point, and what market data is missing?">Entry-point context</button>
        </div>
      </div>`;
  return `<div class="page-heading"><div><div class="eyebrow">Company workspace / source-backed analysis</div><h1>${ticker} Research</h1></div><div class="heading-note">SEC primary sources · AI-assisted</div></div>
    <section class="company-hero">
      <div><div class="company-title"><div class="company-logo">${ticker.slice(0, 4)}</div><div><h2>${ticker}</h2><p>Ask a question to retrieve SEC filings and build a cited research brief.</p></div></div>
        <div class="company-tags"><span class="tag">${ticker}</span><span class="tag">SEC filings</span><span class="tag">Source citations</span></div>
      </div>
      <div class="hero-aside"><strong>◉ ${statusMessage}</strong><span>API credentials stay on the server</span></div>
    </section>
    <section class="card research-card">
      <div class="card-header"><h3>Research assistant</h3><span>Evidence is cached temporarily in memory · answers are not stored server-side</span></div>
      <div class="card-body">
        ${researchStatus && !researchStatus.ready ? `<div class="research-setup"><strong>One-time server setup required</strong><span>Set ${researchStatus.missing.map(escapeHtml).join(' and ')} in the server environment, then restart the app. Keep the AI API key private; setup steps are in <code>RESEARCH_AI_SETUP.md</code>.</span></div>` : ''}
        <div class="research-access-row"><label for="research-access-token">Research access token</label><input id="research-access-token" type="password" autocomplete="current-password" placeholder="Enter the token configured on the server"><span>Kept in this browser tab; never sent to the AI provider.</span></div>
        <form class="research-form" id="research-form">
          <label class="sr-only" for="research-question">Ask a company research question</label>
          <textarea id="research-question" maxlength="1200" placeholder="Ask about business drivers, financial trends, risks, catalysts, or a scenario thesis…"></textarea>
          <div class="research-form-footer"><span>Up to 1,200 characters · SEC evidence is sent to the configured AI provider</span><button class="button primary" type="submit" ${researchBusy ? 'disabled' : ''}>${researchBusy ? 'Researching…' : 'Ask research assistant ↗'}</button></div>
        </form>
        <div class="research-result" id="research-result" aria-live="polite">${answerHtml}</div>
      </div>
    </section>
    <div class="lower-grid">
      <section class="card"><div class="card-header"><h3>Analyst notes</h3><span>Saved in this browser</span></div><div class="card-body"><textarea class="note-area" id="company-notes" placeholder="Capture observations, questions, and source references…">${escapeHtml(state.notes[state.ticker] || '')}</textarea><div class="save-hint"><span>Private to this browser</span><span id="note-status">Saved automatically</span></div></div></section>
      <section class="card"><div class="card-header"><h3>Scenario snapshot</h3><button class="button small" data-open-page="thesis">Edit scenarios ↗</button></div><div class="card-body"><div class="scenario-summary">${scenarioSummary()}</div></div></section>
    </div>`;
}

function renderMarketAnalysis(analysis) {
  const zones = Array.isArray(analysis.zones) ? analysis.zones : [];
  return `<section class="market-entry-analysis">
    <div class="market-entry-header"><div><strong>Volatility-based entry zones</strong><small>Quant model · ${escapeHtml(analysis.horizon_days)} trading days</small></div>
      <span>Spot $${Number(analysis.spot_price).toFixed(2)} · ${escapeHtml(analysis.price_as_of || 'date unavailable')}</span></div>
    <div class="market-zone-grid">${zones.map(zone => `<div class="market-zone">
      <span>${escapeHtml(zone.center_sigma)}σ band</span>
      <strong>$${Number(zone.lower_price).toFixed(2)}–$${Number(zone.upper_price).toFixed(2)}</strong>
      <small>${zone.touch_probability == null ? 'Touch estimate unavailable' : `${(Number(zone.touch_probability) * 100).toFixed(1)}% modeled touch estimate`}</small>
    </div>`).join('')}</div>
    <p>These are volatility bands and historical-bar touch estimates—not fair value, expected returns, or a recommendation. Quote source and methodology are cited below.</p>
  </section>`;
}

function renderAnswerText(answer, sources) {
  const sourceIds = new Set(sources.map(source => source.id));
  return escapeHtml(answer)
    .replace(/\n/g, '<br>')
    .replace(/\[(S\d+)\]/g, (match, sourceId) => (
      sourceIds.has(sourceId)
        ? `<a class="inline-citation" href="#source-${sourceId}">${sourceId}</a>`
        : match
    ));
}

function sourceRow(type, title, detail, href) {
  return `<div class="source-row"><span class="source-type">${type}</span><div class="source-info"><strong>${title}</strong><small>${detail}</small></div><a href="${href}" target="_blank" rel="noopener noreferrer" aria-label="Open ${title}">↗</a></div>`;
}

function renderFilings() {
  const ticker = escapeHtml(state.ticker);
  return `<div class="page-heading"><div><div class="eyebrow">Primary-source research</div><h1>Filings &amp; sources</h1></div><div class="heading-note">Company: ${ticker}</div></div>
    <section class="card"><div class="card-header"><h3>Search official filings</h3><span>SEC EDGAR · opens in a new tab</span></div>    <div class="card-body"><p class="heading-note">Search the SEC's public filing database directly. The Research Assistant separately retrieves recent 10-K/10-Q excerpts when you submit a question.</p>
      <form class="search-row" id="edgar-search"><label class="sr-only" for="filing-query">Search filings or disclosures</label><input id="filing-query" type="text" placeholder="Search a topic, form type, or disclosure…" value="${ticker}"><button class="button primary" type="submit">Search EDGAR ↗</button></form>
      <div class="filing-result"><div><strong>Annual reports</strong><small>Form 10-K · business, risks, audited financials</small></div><a href="${secSearchUrl(state.ticker, '10-K')}" target="_blank" rel="noopener noreferrer">Search ↗</a></div>
      <div class="filing-result"><div><strong>Quarterly reports</strong><small>Form 10-Q · interim results and updates</small></div><a href="${secSearchUrl(state.ticker, '10-Q')}" target="_blank" rel="noopener noreferrer">Search ↗</a></div>
      <div class="filing-result"><div><strong>Current reports</strong><small>Form 8-K · material events and announcements</small></div><a href="${secSearchUrl(state.ticker, '8-K')}" target="_blank" rel="noopener noreferrer">Search ↗</a></div>
    </div></section>
    <div class="lower-grid"><section class="card"><div class="card-header"><h3>Source checklist</h3><span>Before adding a claim</span></div><div class="card-body signal-list">
      <div class="signal"><span class="signal-dot">1</span><div><h4>Prefer primary documents</h4><p>Use filings, earnings releases, and official transcripts where possible.</p></div></div>
      <div class="signal"><span class="signal-dot">2</span><div><h4>Record the date and context</h4><p>Capture filing date, reporting period, units, and any restatement.</p></div></div>
      <div class="signal"><span class="signal-dot">3</span><div><h4>Separate fact from interpretation</h4><p>Keep source-backed facts distinct from your thesis and assumptions.</p></div></div>
    </div></section><section class="card"><div class="card-header"><h3>More primary sources</h3><span>External resources</span></div><div class="card-body"><div class="source-list">
      ${sourceRow('SEC', 'SEC EDGAR search', 'Filings and company submissions', secSearchUrl(state.ticker))}
      ${sourceRow('FRED', 'FRED series', 'Federal Reserve Bank of St. Louis', 'https://fred.stlouisfed.org/')}
    </div></div></section></div>`;
}

function renderThesis() {
  const current = state.scenarios[state.ticker] || {};
  return `<div class="page-heading"><div><div class="eyebrow">Structured analysis / ${escapeHtml(state.ticker)}</div><h1>Thesis builder</h1></div><div class="heading-note">Your assumptions · saved in this browser</div></div>
    <section class="thesis-grid">
      ${scenarioCard('bull', 'Bull case', current.bullValue || '', current.bullNotes || '', 'What has to go right?')}
      ${scenarioCard('base', 'Base case', current.baseValue || '', current.baseNotes || '', 'What seems most likely?')}
      ${scenarioCard('bear', 'Bear case', current.bearValue || '', current.bearNotes || '', 'What could break the thesis?')}
    </section>
    <div class="thesis-footer"><p>Scenario values are your assumptions, not forecasts or recommendations.</p><div class="section-actions"><button class="button" data-open-page="overview">Cancel</button><button class="button primary" id="save-scenarios">Save thesis</button></div></div>
    <section class="card" style="margin-top:22px"><div class="card-header"><h3>Thesis quality check</h3><span>Prompts, not advice</span></div><div class="card-body signal-list">
      <div class="signal"><span class="signal-dot">?</span><div><h4>What evidence would change your mind?</h4><p>Name an observable metric or event that would invalidate the thesis.</p></div></div>
      <div class="signal"><span class="signal-dot">?</span><div><h4>Which assumptions carry the most weight?</h4><p>Stress-test the inputs with the largest effect on your scenario range.</p></div></div>
    </div></section>`;
}

function scenarioCard(type, title, value, notes, placeholder) {
  return `<article class="thesis-card ${type}"><h3>${title}</h3><label class="eyebrow" for="${type}-value">Illustrative value / assumption</label><input class="scenario-input" id="${type}-value" type="text" maxlength="80" placeholder="e.g. Revenue growth normalizes" value="${escapeHtml(value)}"><label class="sr-only" for="${type}-notes">${title} rationale</label><textarea class="note-area" id="${type}-notes" placeholder="${placeholder}">${escapeHtml(notes)}</textarea></article>`;
}

function scenarioSummary() {
  const current = state.scenarios[state.ticker] || {};
  return `<div class="scenario-chip bull"><span>Bull case</span><strong>${escapeHtml(current.bullValue || 'Add an assumption')}</strong></div><div class="scenario-chip"><span>Base case</span><strong>${escapeHtml(current.baseValue || 'Add an assumption')}</strong></div><div class="scenario-chip bear"><span>Bear case</span><strong>${escapeHtml(current.bearValue || 'Add an assumption')}</strong></div>`;
}

function renderMacro() {
  const series = [
    ['Federal funds rate', 'Policy rate & monetary conditions', 'https://fred.stlouisfed.org/series/FEDFUNDS'],
    ['10-year Treasury yield', 'Long-term rates & discount-rate context', 'https://fred.stlouisfed.org/series/DGS10'],
    ['Consumer price index', 'Inflation trend (CPIAUCSL)', 'https://fred.stlouisfed.org/series/CPIAUCSL'],
    ['Unemployment rate', 'Labor-market conditions (UNRATE)', 'https://fred.stlouisfed.org/series/UNRATE'],
    ['Real GDP', 'Economic growth (GDPC1)', 'https://fred.stlouisfed.org/series/GDPC1'],
    ['VIX', 'Implied equity-market volatility (VIXCLS)', 'https://fred.stlouisfed.org/series/VIXCLS']
  ];
  return `<div class="page-heading"><div><div class="eyebrow">Economic context / primary data</div><h1>Macro monitor</h1></div><div class="heading-note">No economic values are loaded in this prototype.</div></div>
    <div class="macro-grid">${series.map(([title, detail, href]) => `<article class="macro-card"><h3>${title}</h3><p>${detail}</p><a href="${href}" target="_blank" rel="noopener noreferrer">Open official FRED series ↗</a></article>`).join('')}</div>
    <section class="card" style="margin-top:14px"><div class="card-header"><h3>Research workflow</h3><span>A lightweight starting point</span></div><div class="card-body signal-list">
      <div class="signal"><span class="signal-dot">1</span><div><h4>Choose a relevant series</h4><p>Check units, seasonal adjustment, frequency, and revision history.</p></div></div>
      <div class="signal"><span class="signal-dot">2</span><div><h4>Track the observation date</h4><p>Economic series publish on different schedules and are often revised.</p></div></div>
      <div class="signal"><span class="signal-dot">3</span><div><h4>Connect macro to company drivers</h4><p>Record the mechanism, direction, time horizon, and counter-evidence in your thesis.</p></div></div>
    </div></section>`;
}

function bindPageEvents() {
  const accessTokenInput = document.getElementById('research-access-token');
  if (accessTokenInput) {
    try {
      accessTokenInput.value = sessionStorage.getItem('bossman-research-access-token') || '';
    } catch (error) {
      console.warn('Could not restore the research access token from this tab:', error);
    }
    accessTokenInput.addEventListener('input', () => {
      try {
        sessionStorage.setItem('bossman-research-access-token', accessTokenInput.value);
      } catch (error) {
        console.error('Could not keep the research token in this tab:', error);
        showToast('This browser cannot remember the access token; it will still work until you leave the page.');
      }
    });
  }
  document.querySelectorAll('[data-open-page]').forEach(button => {
    button.addEventListener('click', () => showPage(button.dataset.openPage));
  });
  const notes = document.getElementById('company-notes');
  if (notes) {
    notes.addEventListener('input', () => {
      state.notes[state.ticker] = notes.value;
      const status = document.getElementById('note-status');
      if (status) status.textContent = saveState() ? 'Saved automatically' : 'Not saved';
    });
  }
  const researchForm = document.getElementById('research-form');
  if (researchForm) {
    researchForm.addEventListener('submit', event => {
      event.preventDefault();
      submitResearch();
    });
    document.querySelectorAll('[data-question]').forEach(button => {
      button.addEventListener('click', () => {
        document.getElementById('research-question').value = button.dataset.question;
        submitResearch();
      });
    });
  }
  const search = document.getElementById('edgar-search');
  if (search) {
    search.addEventListener('submit', event => {
      event.preventDefault();
      const query = document.getElementById('filing-query').value.trim();
      if (!query) {
        showToast('Enter a company, form type, or topic to search.');
        return;
      }
      window.open(secSearchUrl(query), '_blank', 'noopener,noreferrer');
    });
  }
  const save = document.getElementById('save-scenarios');
  if (save) save.addEventListener('click', saveThesis);
}

async function loadResearchStatus() {
  try {
    const response = await fetch('/api/research/status');
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || `Request failed (${response.status}).`);
    researchStatus = data;
    const label = document.getElementById('research-status-label');
    const led = document.getElementById('research-status-led');
    label.textContent = data.ready ? 'AI research ready' : 'AI research needs setup';
    led.classList.toggle('ready', data.ready);
    render();
  } catch (error) {
    console.error('Could not load research setup status:', error);
    const label = document.getElementById('research-status-label');
    if (label) label.textContent = 'Research server unavailable';
    showToast('Could not check the research server. Refresh after starting the app server.');
  }
}

async function submitResearch() {
  if (researchBusy) return;
  const question = document.getElementById('research-question').value.trim();
  const ticker = state.ticker;
  const accessToken = document.getElementById('research-access-token').value;
  if (!question) {
    showToast('Enter a question for the research assistant.');
    return;
  }
  if (!accessToken) {
    showToast('Enter the research access token configured on the server.');
    return;
  }
  researchBusy = true;
  delete researchErrors[ticker];
  render();
  try {
    const response = await fetch('/api/research', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${accessToken}`
      },
      body: JSON.stringify({ ticker, question })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.message || `Research request failed (${response.status}).`);
    if (typeof data.answer !== 'string' || !Array.isArray(data.sources)) {
      throw new Error('The research server returned an invalid answer.');
    }
    researchAnswers[ticker] = data;
  } catch (error) {
    console.error(`Research request failed for ${ticker}:`, error);
    researchErrors[ticker] = error.message || 'Research request failed.';
  } finally {
    researchBusy = false;
    render();
  }
}

function saveThesis() {
  const read = id => document.getElementById(id).value.trim();
  state.scenarios[state.ticker] = {
    bullValue: read('bull-value'),
    bullNotes: read('bull-notes'),
    baseValue: read('base-value'),
    baseNotes: read('base-notes'),
    bearValue: read('bear-value'),
    bearNotes: read('bear-notes')
  };
  if (saveState()) {
    showToast('Thesis saved in this browser.');
    render();
  }
}

function secSearchUrl(query, form = '') {
  const terms = [query, form].filter(Boolean).join(' ');
  return `https://www.sec.gov/edgar/search/#/q=${encodeURIComponent(terms)}&dateRange=all`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[character]);
}

function showToast(message) {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.classList.add('visible');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => toast.classList.remove('visible'), 3200);
}
