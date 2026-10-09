/* ======================================================================
   BASAK BOSSMAN ENGINE — CLIENT CONTROLLER & INTERACTIVE TERMINAL
   ====================================================================== */

let currentTicker = 'PLTR';
let currentProfile = 'Dip Hunter';
let activeView = 'single';

const PROFILE_DESCRIPTIONS = {
  'Dip Hunter': 'Turnarounds & deep discounts: rewards 3mo momentum shift, inverted 1y momentum, and deep dip MA200 bonuses.',
  'Alpha Bull': 'Aggressive trend momentum: rides stocks above 50MA & 200MA with high beta amplification and room-to-run RSI.',
  'All-Weather Core': 'Risk-weighted compounders: 46.3% volatility penalty weight, balanced MA proximities, and adaptive gear shifting.',
  'Bear Market Shield': 'Capital preservation: inverted momentum, solvency discipline, strict valuation limits, and protective floors.',
  'Fundamental Bag': 'Deep value discipline: rewards low P/E, PEG < 1.0, and balance sheet solvency with conservative ceilings.'
};

document.addEventListener('DOMContentLoaded', () => {
  initApp();
});

async function initApp() {
  setupEventListeners();
  await loadMarketRegime();
  // Automatically score initial ticker to display full data immediately
  scoreTicker(currentTicker);
}

function setupEventListeners() {
  const tickerInput = document.getElementById('ticker-input');
  if (tickerInput) {
    tickerInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        scoreCurrentTicker();
      }
    });
  }

  const compareInput = document.getElementById('compare-input');
  if (compareInput) {
    compareInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') {
        runCompare();
      }
    });
  }
}

// ── View Switching ────────────────────────────────────────────────────
function switchMainView(viewName) {
  activeView = viewName;

  document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));

  const targetPanel = document.getElementById(`view-${viewName}`);
  const targetTab = document.getElementById(`tab-${viewName}`);

  if (targetPanel) targetPanel.classList.add('active');
  if (targetTab) targetTab.classList.add('active');

  if (viewName === 'scan') {
    runUniverseScan();
  } else if (viewName === 'compare') {
    runCompare();
  }
}

// ── Market Macro Regime ───────────────────────────────────────────────
let _cachedSectorEtfs = {};

async function loadMarketRegime() {
  try {
    const res = await fetch('/api/market-regime');
    if (!res.ok) return;
    const data = await res.json();

    const regimeName = document.getElementById('regime-name');
    const regimeSub = document.getElementById('regime-sub');
    const regimePill = document.getElementById('regime-rec-pill');
    const pulseDot = document.getElementById('regime-pulse');
    const breadthBadge = document.getElementById('regime-breadth-badge');
    const rspBadge = document.getElementById('regime-rsp-badge');
    const vixBadge = document.getElementById('regime-vix-badge');

    if (regimeName && data.regime) {
      regimeName.textContent = `Macro: ${data.regime}`;
      regimeSub.textContent = `SPY $${data.price} (${data.pct_vs_ma200 >= 0 ? '+' : ''}${data.pct_vs_ma200}% vs 200MA) | Updated: ${data.updated_at}`;
      if (pulseDot && data.color) {
        pulseDot.style.backgroundColor = data.color;
        pulseDot.style.boxShadow = `0 0 10px ${data.color}`;
      }
      if (regimePill && data.recommended_profiles && data.recommended_profiles.length > 0) {
        regimePill.textContent = `Optimal: ${data.recommended_profiles.join(' / ')}`;
      }
    }

    // Breadth badge
    if (breadthBadge && data.breadth_50 !== undefined) {
      breadthBadge.textContent = `Breadth: ${data.breadth_50}%`;
      breadthBadge.className = 'macro-metric-badge';
      if (data.breadth_50 < 35) breadthBadge.classList.add('highlight-dip');
      else if (data.breadth_50 < 50) breadthBadge.classList.add('highlight-core');
      else breadthBadge.classList.add('highlight-bull');
    }

    // RSP/SPY badge
    if (rspBadge && data.rsp_spy_diff !== undefined) {
      const sign = data.rsp_spy_diff >= 0 ? '+' : '';
      rspBadge.textContent = `RSP/SPY: ${sign}${data.rsp_spy_diff.toFixed(1)}%`;
      rspBadge.className = 'macro-metric-badge';
      if (data.rsp_outperforming) rspBadge.classList.add('highlight-core');
    }

    // VIX badge
    if (vixBadge && data.vix !== undefined) {
      vixBadge.textContent = `VIX: ${data.vix.toFixed(1)}`;
      vixBadge.className = 'macro-metric-badge';
      if (data.vix >= 55) vixBadge.classList.add('highlight-vix-panic');
      else if (data.vix >= 35) vixBadge.classList.add('highlight-dip');
    }

    // Cache sector ETFs for the radar modal
    if (data.sector_etfs) {
      _cachedSectorEtfs = data.sector_etfs;
    }
  } catch (err) {
    console.warn('Could not load market regime:', err);
  }
}

// ── Sector Radar Modal ────────────────────────────────────────────────
const SECTOR_ETF_NAMES = {
  XLK: 'Technology', XLF: 'Financials', XLV: 'Health Care', XLY: 'Cons. Discret.',
  XLP: 'Cons. Staples', XLE: 'Energy', XLI: 'Industrials', XLU: 'Utilities',
  XLB: 'Materials', XLRE: 'Real Estate', XLC: 'Comm. Services'
};

function toggleSectorModal(event) {
  event.stopPropagation();
  const modal = document.getElementById('sector-radar-modal');
  if (modal.classList.contains('hidden')) {
    renderSectorRadarGrid();
    modal.classList.remove('hidden');
  } else {
    modal.classList.add('hidden');
  }
}

function closeSectorModal(event) {
  const modal = document.getElementById('sector-radar-modal');
  modal.classList.add('hidden');
}

function renderSectorRadarGrid() {
  const grid = document.getElementById('sector-radar-grid');
  if (!grid) return;
  grid.innerHTML = '';

  const etfs = Object.values(_cachedSectorEtfs);
  if (etfs.length === 0) {
    grid.innerHTML = '<p style="color: var(--text-dim); text-align: center; grid-column: 1/-1;">Loading sector data... Score a ticker or visit market regime first.</p>';
    return;
  }

  // Sort: most oversold first (contrarian opportunities)
  etfs.sort((a, b) => a.diff_pct - b.diff_pct);

  etfs.forEach(etf => {
    const isOpp = etf.diff_pct < 0;
    const effPct = isOpp ? Math.min(15, Math.abs(etf.diff_pct)).toFixed(1) : Math.min(15, etf.diff_pct).toFixed(1);
    const tile = document.createElement('div');
    tile.className = `sector-radar-tile ${isOpp ? 'opportunity' : 'overextended'}`;
    tile.innerHTML = `
      <div class="tile-top">
        <div class="tile-sym-group">
          <span class="tile-sym">${etf.symbol}</span>
          <span class="tile-name">${SECTOR_ETF_NAMES[etf.symbol] || ''}</span>
        </div>
        <span class="tile-badge ${isOpp ? 'opportunity' : 'overextended'}">${isOpp ? 'OPPORTUNITY' : 'OVEREXTENDED'}</span>
      </div>
      <div class="tile-prices">
        <span>Price: $${etf.price.toFixed(2)}</span>
        <span>50MA: $${etf.ma50.toFixed(2)}</span>
      </div>
      <div class="tile-diff-row">
        <span class="tile-diff-val ${isOpp ? 'negative' : 'positive'}">${etf.diff_pct >= 0 ? '+' : ''}${etf.diff_pct.toFixed(2)}% vs 50MA</span>
        <span class="tile-impact ${isOpp ? 'bonus' : 'haircut'}">${isOpp ? '+' : '-'}${effPct}% Score Impact</span>
      </div>
    `;
    grid.appendChild(tile);
  });
}

function renderSectorSubRegimeCard(sectorInfo) {
  const badge = document.getElementById('res-sector-badge');
  const etfSym = document.getElementById('res-sector-etf-sym');
  const etfPrice = document.getElementById('res-sector-etf-price');
  const diffVal = document.getElementById('res-sector-diff-val');
  const statusBox = document.getElementById('res-sector-status-box');
  const icon = document.getElementById('res-sector-icon');
  const effectTag = document.getElementById('res-sector-effect-tag');
  const descEl = document.getElementById('res-sector-desc');

  if (!sectorInfo) {
    if (badge) { badge.textContent = 'No Data'; badge.className = 'badge-flag neutral'; }
    if (etfSym) etfSym.textContent = 'N/A';
    if (etfPrice) etfPrice.textContent = '';
    if (diffVal) diffVal.textContent = 'N/A';
    if (effectTag) effectTag.textContent = 'Sector not mapped';
    if (descEl) descEl.textContent = 'No matching GICS sector ETF found. Score impact: neutral.';
    return;
  }

  const isOpp = sectorInfo.status === 'opportunity';
  const etfData = _cachedSectorEtfs[sectorInfo.etf] || {};

  if (badge) {
    badge.textContent = isOpp ? '🎯 Contrarian Opportunity' : '⚠️ Rotation Exhaust Risk';
    badge.className = `badge-flag ${isOpp ? 'success' : 'warning'}`;
  }
  if (etfSym) etfSym.textContent = sectorInfo.etf;
  if (etfPrice) etfPrice.textContent = etfData.price ? `$${etfData.price.toFixed(2)}` : '';
  if (diffVal) {
    const diff = etfData.diff_pct !== undefined ? etfData.diff_pct : 0;
    diffVal.textContent = `${diff >= 0 ? '+' : ''}${diff.toFixed(2)}%`;
    diffVal.style.color = isOpp ? 'var(--emerald)' : 'var(--amber)';
  }
  if (statusBox) {
    statusBox.style.borderLeftColor = isOpp ? 'var(--emerald)' : 'var(--amber)';
  }
  if (icon) icon.textContent = isOpp ? '🎯' : '⚠️';
  if (effectTag) {
    effectTag.textContent = sectorInfo.effect_str;
    effectTag.className = `sector-effect-tag${isOpp ? '' : ' haircut'}`;
  }
  if (descEl) descEl.textContent = sectorInfo.desc;
}

// ── Profile Switching ─────────────────────────────────────────────────
function changeProfile(profileName) {
  currentProfile = profileName;

  // Update tabs UI
  document.querySelectorAll('.profile-tab').forEach(tab => {
    if (tab.dataset.profile === profileName) {
      tab.classList.add('active');
    } else {
      tab.classList.remove('active');
    }
  });

  // Update description
  const descEl = document.getElementById('profile-active-desc');
  if (descEl && PROFILE_DESCRIPTIONS[profileName]) {
    descEl.textContent = PROFILE_DESCRIPTIONS[profileName];
  }

  // Re-score current ticker with new profile
  if (currentTicker) {
    scoreTicker(currentTicker);
  }
}

function selectQuickTicker(ticker) {
  const input = document.getElementById('ticker-input');
  if (input) input.value = ticker;
  scoreTicker(ticker);
}

function scoreCurrentTicker() {
  const input = document.getElementById('ticker-input');
  const val = input ? input.value.trim().toUpperCase() : '';
  if (!val) {
    alert('Please enter a valid ticker symbol.');
    return;
  }
  scoreTicker(val);
}

// ── Core Scoring Query ────────────────────────────────────────────────
async function scoreTicker(ticker) {
  currentTicker = ticker.toUpperCase();

  const loadingEl = document.getElementById('loading-state');
  const resultsEl = document.getElementById('results-display');
  const loadingTicker = document.getElementById('loading-ticker');

  if (loadingTicker) loadingTicker.textContent = currentTicker;
  if (loadingEl) loadingEl.classList.remove('hidden');
  if (resultsEl) resultsEl.style.opacity = '0.35';

  try {
    const url = `/api/score?ticker=${encodeURIComponent(currentTicker)}&profile=${encodeURIComponent(currentProfile)}`;
    const res = await fetch(url);
    const data = await res.json();

    if (data.error) {
      alert(`Error scoring ${currentTicker}: ${data.message}`);
      return;
    }

    renderScoredResults(data);
  } catch (err) {
    alert(`Network error connecting to scoring server: ${err.message}`);
  } finally {
    if (loadingEl) loadingEl.classList.add('hidden');
    if (resultsEl) resultsEl.style.opacity = '1.0';
  }
}

// ── Render Scored Results ─────────────────────────────────────────────
function renderScoredResults(data) {
  // 1. Ticker Hero
  document.getElementById('res-ticker').textContent = data.ticker;
  document.getElementById('res-name').textContent = data.name || data.ticker;
  document.getElementById('res-sector').textContent = data.sector || 'N/A';
  document.getElementById('res-industry').textContent = data.industry || 'General';
  document.getElementById('res-profile').textContent = data.profile;
  document.getElementById('res-price').textContent = `$${data.price.toFixed(2)}`;

  if (data.market_cap) {
    document.getElementById('res-market-cap').textContent = formatMarketCap(data.market_cap);
  }

  const basePill = document.getElementById('res-baseline-pill');
  const baseDivider = document.getElementById('res-baseline-divider');
  if (basePill && baseDivider) {
    if (data.baseline_snapshot) {
      const snap = data.baseline_snapshot;
      const sign = snap.price_change_since_scan >= 0 ? '+' : '';
      basePill.textContent = `Oct 1 Batch Scan: $${snap.baseline_price.toFixed(2)} (${sign}${snap.price_change_since_scan}% live price move)`;
      basePill.style.display = 'inline-block';
      baseDivider.style.display = 'inline-block';
    } else {
      basePill.style.display = 'none';
      baseDivider.style.display = 'none';
    }
  }

  const perf1wEl = document.getElementById('res-perf-1w');
  const perfVal = data.technical_indicators?.perf_1w;
  if (perf1wEl && perfVal !== null && perfVal !== undefined) {
    perf1wEl.textContent = `${perfVal >= 0 ? '+' : ''}${perfVal.toFixed(1)}% (1w)`;
    perf1wEl.className = `perf-badge ${perfVal >= 0 ? '' : 'negative'}`;
  }

  // 52W Range Meter
  const low52 = data.technical_indicators?.fifty_two_low;
  const high52 = data.technical_indicators?.fifty_two_high;
  if (low52 && high52 && high52 > low52) {
    document.getElementById('res-52low').textContent = `52W Low: $${low52.toFixed(2)}`;
    document.getElementById('res-52high').textContent = `52W High: $${high52.toFixed(2)}`;
    const pct = Math.min(100, Math.max(0, ((data.price - low52) / (high52 - low52)) * 100));
    document.getElementById('res-range-fill').style.width = `${pct}%`;
    document.getElementById('res-range-thumb').style.left = `${pct}%`;
  }

  // 2. Score Gauge & Stance
  const scoreVal = data.score;
  document.getElementById('res-score').textContent = scoreVal.toFixed(1);
  document.getElementById('res-percentile-tag').textContent = `Top ${Math.max(1, (100 - data.universe_percentile).toFixed(1))}% Universe`;

  const stanceBadge = document.getElementById('res-stance-badge');
  const stanceText = document.getElementById('res-stance-text');
  if (stanceBadge && stanceText) {
    stanceText.textContent = data.stance;
    stanceBadge.className = `stance-badge ${data.stance_class}`;
  }

  // Circular SVG Gauge Animation
  const gaugeCircle = document.getElementById('gauge-fill-circle');
  if (gaugeCircle) {
    const circumference = 2 * Math.PI * 68; // ~427.25
    const offset = circumference - (scoreVal / 100) * circumference;
    gaugeCircle.style.strokeDashoffset = offset;
    gaugeCircle.style.stroke = data.stance_color || '#10b981';
  }

  // 3. OLS Statistical Forecast
  const fc = data.forecast;
  if (fc && fc.expected_return !== null) {
    const retEl = document.getElementById('res-exp-return');
    retEl.textContent = `${fc.expected_return >= 0 ? '+' : ''}${fc.expected_return.toFixed(1)}%`;
    retEl.style.color = fc.expected_return >= 0 ? '#38bdf8' : '#f43f5e';

    document.getElementById('res-exp-se').textContent = `±${fc.se.toFixed(1)}% SE`;
    document.getElementById('res-ci-range').textContent = `[${fc.conf_lower >= 0 ? '+' : ''}${fc.conf_lower.toFixed(1)}%, ${fc.conf_upper >= 0 ? '+' : ''}${fc.conf_upper.toFixed(1)}%]`;

    // CI Visual Span (map from -50% to +50% range to 0% - 100%)
    const minRange = -50, maxRange = 50, spanRange = maxRange - minRange;
    const leftPct = Math.min(100, Math.max(0, ((fc.conf_lower - minRange) / spanRange) * 100));
    const rightPct = Math.min(100, Math.max(0, ((fc.conf_upper - minRange) / spanRange) * 100));
    const pointPct = Math.min(100, Math.max(0, ((fc.expected_return - minRange) / spanRange) * 100));

    const ciSpan = document.getElementById('res-ci-span');
    const ciPoint = document.getElementById('res-ci-point');
    if (ciSpan) {
      ciSpan.style.left = `${leftPct}%`;
      ciSpan.style.width = `${Math.max(4, rightPct - leftPct)}%`;
    }
    if (ciPoint) {
      ciPoint.style.left = `${pointPct}%`;
    }

    // Win Probability
    document.getElementById('res-prob-val').textContent = `${fc.prob_positive.toFixed(1)}%`;
    document.getElementById('res-prob-fill').style.width = `${fc.prob_positive}%`;
  } else {
    document.getElementById('res-exp-return').textContent = 'N/A';
    document.getElementById('res-ci-range').textContent = '[N/A]';
    document.getElementById('res-prob-val').textContent = '--%';
    document.getElementById('res-prob-fill').style.width = '0%';
  }

  // 4. Balance Sheet & Solvency Flag (Non-Punitive)
  const lev = data.leverage_flag;
  if (lev) {
    document.getElementById('res-de-val').textContent = lev.value_str;
    document.getElementById('res-de-benchmark').textContent = lev.threshold_str;
    document.getElementById('res-de-desc').textContent = lev.description;

    const levBadge = document.getElementById('res-de-badge');
    if (levBadge) {
      levBadge.textContent = lev.label;
      levBadge.className = `badge-flag ${lev.badge_color}`;
    }

    const levStatusBox = document.getElementById('res-de-status-box');
    const levIcon = document.getElementById('res-de-icon');
    if (lev.severity === 'safe') {
      levStatusBox.style.borderLeftColor = 'var(--emerald)';
      levIcon.textContent = '🛡️';
    } else if (lev.severity === 'normal') {
      levStatusBox.style.borderLeftColor = '#38bdf8';
      levIcon.textContent = '⚖️';
    } else if (lev.severity === 'caution') {
      levStatusBox.style.borderLeftColor = 'var(--amber)';
      levIcon.textContent = '⚠️';
    } else {
      levStatusBox.style.borderLeftColor = 'var(--rose)';
      levIcon.textContent = '🚨';
    }
  }

  // 4b. Sector Sub-Regime Card
  renderSectorSubRegimeCard(data.sector_sub_regime);

  // Update sector ETF cache if this score result includes regime data
  if (data.market_regime && data.market_regime.sector_etfs) {
    _cachedSectorEtfs = data.market_regime.sector_etfs;
  }

  // 5. Footnotes & Data Integrity Log
  const fnList = document.getElementById('footnotes-list');
  const fnCountBadge = document.getElementById('fn-count-badge');
  fnList.innerHTML = '';

  if (data.footnotes && data.footnotes.length > 0) {
    fnCountBadge.textContent = `${data.footnotes.length} Footnotes`;
    data.footnotes.forEach(fn => {
      const card = document.createElement('div');
      card.className = 'footnote-card';
      card.innerHTML = `
        <span class="fn-icon">${fn.icon || 'ℹ️'}</span>
        <div class="fn-content">
          <span class="fn-heading">${fn.title}</span>
          <p class="fn-msg">${fn.message}</p>
        </div>
      `;
      fnList.appendChild(card);
    });
  } else {
    fnCountBadge.textContent = '0 Footnotes';
    const cleanCard = document.createElement('div');
    cleanCard.className = 'footnote-card';
    cleanCard.style.borderLeftColor = 'var(--emerald)';
    cleanCard.innerHTML = `
      <span class="fn-icon">✅</span>
      <div class="fn-content">
        <span class="fn-heading">Data Integrity Verified</span>
        <p class="fn-msg">All primary technical, momentum, and fundamental metrics are available. 100% standard baseline weights applied without omissions.</p>
      </div>
    `;
    fnList.appendChild(cleanCard);
  }

  // 6. Factor Attribution Table
  const tableBody = document.getElementById('factors-table-body');
  tableBody.innerHTML = '';

  if (data.factor_breakdown && data.factor_breakdown.length > 0) {
    data.factor_breakdown.forEach(f => {
      const tr = document.createElement('tr');
      const contribPct = Math.min(100, Math.max(0, (f.weighted_score / 0.5) * 100));

      tr.innerHTML = `
        <td class="factor-name-cell">${f.name}</td>
        <td class="factor-raw-cell">${formatRawValue(f.key, f.raw_value)}</td>
        <td><span class="rank-pill">${f.percentile_rank}%</span></td>
        <td><span class="weight-pill">${(f.rebased_weight * 100).toFixed(1)}%</span></td>
        <td>
          <div class="contrib-bar-wrapper">
            <div class="contrib-track">
              <div class="contrib-fill" style="width: ${contribPct}%"></div>
            </div>
            <span class="contrib-val">+${f.weighted_score.toFixed(3)}</span>
          </div>
        </td>
      `;
      tableBody.appendChild(tr);
    });
  }

  // Active Multipliers
  const multContainer = document.getElementById('multipliers-container');
  multContainer.innerHTML = '';

  if (data.multipliers && data.multipliers.length > 0) {
    data.multipliers.forEach(m => {
      const pill = document.createElement('div');
      pill.className = 'multiplier-pill';
      const isNeg = m.effect.startsWith('-');
      pill.innerHTML = `
        <span class="mult-name">${m.name}</span>
        <span class="mult-effect ${isNeg ? 'negative' : ''}">${m.effect}</span>
      `;
      multContainer.appendChild(pill);
    });
  } else {
    multContainer.innerHTML = '<span style="font-size: 11px; color: var(--text-dim);">Neutral multiplier (1.00x). No penalty or bonus triggered.</span>';
  }

  // 7. Key Technical & Fundamental Metrics
  const ind = data.technical_indicators;
  if (ind) {
    document.getElementById('m-ma50').textContent = ind.ma_50 ? `$${ind.ma_50.toFixed(2)}` : 'N/A';
    document.getElementById('m-ma50-diff').textContent = ind.pct_vs_ma50 !== null ? `${ind.pct_vs_ma50 >= 0 ? '+' : ''}${ind.pct_vs_ma50.toFixed(1)}% vs 50MA` : '--';

    document.getElementById('m-ma200').textContent = ind.ma_200 ? `$${ind.ma_200.toFixed(2)}` : 'N/A';
    document.getElementById('m-ma200-diff').textContent = ind.pct_vs_ma200 !== null ? `${ind.pct_vs_ma200 >= 0 ? '+' : ''}${ind.pct_vs_ma200.toFixed(1)}% vs 200MA` : '--';

    const rsiEl = document.getElementById('m-rsi');
    const rsiStance = document.getElementById('m-rsi-stance');
    rsiEl.textContent = ind.rsi !== null ? ind.rsi.toFixed(1) : 'N/A';
    if (ind.rsi > 70) {
      rsiStance.textContent = 'Overbought (>70)';
      rsiStance.style.color = 'var(--rose)';
    } else if (ind.rsi < 30) {
      rsiStance.textContent = 'Oversold (<30)';
      rsiStance.style.color = 'var(--amber)';
    } else {
      rsiStance.textContent = 'Neutral Range';
      rsiStance.style.color = 'var(--text-muted)';
    }

    document.getElementById('m-vol').textContent = ind.annualized_vol !== null ? `${ind.annualized_vol.toFixed(1)}%` : 'N/A';
    document.getElementById('m-beta').textContent = ind.beta !== null ? `${ind.beta.toFixed(2)}` : 'N/A';
    document.getElementById('m-pe').textContent = ind.pe_ratio !== null ? `${ind.pe_ratio.toFixed(1)}x` : 'N/A (Loss)';
    document.getElementById('m-peg').textContent = ind.peg_ratio !== null ? `${ind.peg_ratio.toFixed(2)}` : 'N/A';
    document.getElementById('m-mom1y').textContent = ind.momentum_1y !== null ? `${ind.momentum_1y >= 0 ? '+' : ''}${ind.momentum_1y.toFixed(1)}%` : 'N/A (<1y)';
  }
}

// ── Universe Scanner View ─────────────────────────────────────────────
async function runUniverseScan() {
  const select = document.getElementById('scan-profile-select');
  const prof = select ? select.value : currentProfile;
  const tbody = document.getElementById('scan-table-body');
  if (!tbody) return;

  tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 30px; color: var(--text-muted);">Scanning pre-computed universe leaderboard...</td></tr>';

  try {
    const res = await fetch(`/api/scan?profile=${encodeURIComponent(prof)}&limit=15`);
    const data = await res.json();

    tbody.innerHTML = '';
    if (data.leaderboard && data.leaderboard.length > 0) {
      data.leaderboard.forEach(item => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td class="scan-rank">#${item.rank}</td>
          <td class="scan-ticker" onclick="inspectScannerTicker('${item.ticker}')">${item.ticker}</td>
          <td>${item.sector}</td>
          <td>$${item.price.toFixed(2)}</td>
          <td>${item.momentum_1y >= 0 ? '+' : ''}${item.momentum_1y.toFixed(1)}%</td>
          <td>${item.rsi.toFixed(1)}</td>
          <td>${typeof item.pe_ratio === 'number' ? item.pe_ratio.toFixed(1) + 'x' : item.pe_ratio}</td>
          <td class="scan-score">${item.score.toFixed(1)}</td>
          <td style="color: ${item.expected_return >= 0 ? '#38bdf8' : '#f43f5e'}; font-family: var(--font-mono); font-weight: 700;">
            ${item.expected_return !== null ? (item.expected_return >= 0 ? '+' : '') + item.expected_return.toFixed(1) + '%' : 'N/A'}
          </td>
          <td>
            <button class="btn-inspect" onclick="inspectScannerTicker('${item.ticker}')">Inspect</button>
          </td>
        `;
        tbody.appendChild(tr);
      });
    } else {
      tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 20px;">No scan results available.</td></tr>';
    }
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; padding: 20px; color: var(--rose);">Error loading scanner: ${err.message}</td></tr>`;
  }
}

function inspectScannerTicker(t) {
  switchMainView('single');
  selectQuickTicker(t);
}

// ── Multi-Stock Compare View ──────────────────────────────────────────
async function runCompare() {
  const input = document.getElementById('compare-input');
  const raw = input ? input.value : 'NVDA, AMD, INTC, PLTR';
  const grid = document.getElementById('compare-grid');
  if (!grid) return;

  grid.innerHTML = '<div style="grid-column: 1 / -1; text-align: center; padding: 40px; color: var(--text-muted);">Fetching comparative data & generating scores...</div>';

  try {
    const res = await fetch(`/api/compare?tickers=${encodeURIComponent(raw)}&profile=${encodeURIComponent(currentProfile)}`);
    const data = await res.json();

    grid.innerHTML = '';
    if (data.comparison && data.comparison.length > 0) {
      data.comparison.forEach(item => {
        if (item.error) {
          const card = document.createElement('div');
          card.className = 'glass-panel comp-card';
          card.innerHTML = `<h3 style="color: var(--rose);">${item.ticker}</h3><p style="font-size: 12px; color: var(--text-muted);">${item.error}</p>`;
          grid.appendChild(card);
          return;
        }

        const card = document.createElement('div');
        card.className = 'glass-panel comp-card';
        card.innerHTML = `
          <div class="comp-head">
            <span class="comp-ticker" onclick="inspectScannerTicker('${item.ticker}')" style="cursor: pointer;">${item.ticker}</span>
            <span class="comp-price">$${item.price.toFixed(2)}</span>
          </div>
          <div class="comp-score-box">
            <span style="font-size: 11px; color: var(--text-dim); text-transform: uppercase;">Score (${item.profile})</span>
            <div class="comp-score-val" style="color: ${item.stance_color || '#10b981'};">${item.score.toFixed(1)}</div>
            <div style="font-size: 12px; font-weight: 700; color: ${item.stance_color || '#ffffff'};">${item.stance}</div>
          </div>
          <div class="comp-stats-list">
            <div class="comp-stat-row"><span>Sector</span><strong>${item.sector}</strong></div>
            <div class="comp-stat-row"><span>Exp. 1y Return</span><strong style="color: ${item.forecast?.expected_return >= 0 ? '#38bdf8' : '#f43f5e'}; font-family: var(--font-mono);">${item.forecast?.expected_return !== null ? (item.forecast?.expected_return >= 0 ? '+' : '') + item.forecast?.expected_return.toFixed(1) + '%' : 'N/A'}</strong></div>
            <div class="comp-stat-row"><span>RSI (14)</span><strong style="font-family: var(--font-mono);">${item.technical_indicators?.rsi ? item.technical_indicators.rsi.toFixed(1) : 'N/A'}</strong></div>
            <div class="comp-stat-row"><span>Debt/Equity</span><strong style="font-family: var(--font-mono);">${item.leverage_flag?.value_str}</strong></div>
            <div class="comp-stat-row"><span>P/E Ratio</span><strong style="font-family: var(--font-mono);">${item.technical_indicators?.pe_ratio ? item.technical_indicators.pe_ratio.toFixed(1) + 'x' : 'N/A'}</strong></div>
          </div>
          <button class="btn-inspect" style="width: 100%; margin-top: 6px;" onclick="inspectScannerTicker('${item.ticker}')">Deep Dive Analysis</button>
        `;
        grid.appendChild(card);
      });
    }
  } catch (err) {
    grid.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--rose);">Error running comparison: ${err.message}</div>`;
  }
}

// ── Helpers ───────────────────────────────────────────────────────────
function formatMarketCap(cap) {
  if (!cap) return 'N/A';
  if (cap >= 1e12) return `$${(cap / 1e12).toFixed(2)}T`;
  if (cap >= 1e9) return `$${(cap / 1e9).toFixed(2)}B`;
  if (cap >= 1e6) return `$${(cap / 1e6).toFixed(2)}M`;
  return `$${cap.toLocaleString()}`;
}

function formatRawValue(key, val) {
  if (val === null || val === undefined) return 'N/A';
  if (key.includes('momentum') || key === 'perf_1w') {
    return `${val >= 0 ? '+' : ''}${(val * 100).toFixed(1)}%`;
  }
  if (key === 'annualized_vol' || key === 'risk') {
    return `${(val * 100).toFixed(1)}%`;
  }
  if (key === 'pe_ratio' || key === 'valuation') {
    return `${val.toFixed(1)}x`;
  }
  if (key.includes('proximity')) {
    return val === 0 ? '0.00 (At MA)' : `${val > 0 ? '+' : ''}${val.toFixed(3)}`;
  }
  return typeof val === 'number' ? val.toFixed(2) : val;
}
