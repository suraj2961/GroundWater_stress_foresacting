"""
Groundwater Forecast Dashboard Generator
Builds a single self-contained HTML dashboard (data embedded, no server
needed) from your panel dataset and forecast results.

Run this from D:\research\scripts. Open the output file directly in
your browser afterward - no need to run a local server.
"""

import pandas as pd
import json

DISTRICTS = ['Bengaluru Urban', 'Mysuru', 'Ballari', 'Uttara Kannada',
             'Kalaburagi', 'Vijayapura']

print("=== Loading data ===")
historical = pd.read_csv('../data/processed/panel_dataset.csv', parse_dates=['date'])
forecast = pd.read_csv('../data/processed/multi_district_forecast_24month.csv', parse_dates=['date'])

data = {}
for district in DISTRICTS:
    hist = historical[historical['district'] == district].sort_values('date')
    hist = hist[hist['date'] >= '2018-01-01']
    fc = forecast[forecast['district'] == district].sort_values('date')

    data[district] = {
        'hist_dates': hist['date'].dt.strftime('%Y-%m').tolist(),
        'hist_values': [round(v, 2) for v in hist['GRACE_TWS_anomaly_cm'].tolist()],
        'fc_dates': fc['date'].dt.strftime('%Y-%m').tolist(),
        'fc_values': [round(v, 2) for v in fc['predicted_GRACE_TWS_anomaly_cm'].tolist()],
        'fc_categories': fc['stress_category'].tolist(),
    }

data_json = json.dumps(data)
print(f"Embedded data for {len(DISTRICTS)} districts")

html_template = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Karnataka Groundwater Stress Forecast</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@400;500;600&display=swap');

  :root {
    --bg: #0d2b2e;
    --bg-panel: #123a3d;
    --ochre: #c98a3e;
    --ochre-light: #e0ac6a;
    --text: #f2ede4;
    --text-dim: #a9c2c0;
    --safe: #2ecc71;
    --semi: #f1c40f;
    --critical: #e67e22;
    --over: #e74c3c;
    --border: #23494c;
  }

  * { box-sizing: border-box; }

  body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: 'Inter', sans-serif;
    padding: 0;
  }

  header {
    padding: 48px 32px 24px;
    max-width: 1100px;
    margin: 0 auto;
    border-bottom: 1px solid var(--border);
  }

  .eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--ochre-light);
    margin-bottom: 12px;
  }

  h1 {
    font-family: 'Fraunces', serif;
    font-size: 40px;
    font-weight: 600;
    margin: 0 0 8px;
    line-height: 1.15;
  }

  .subtitle {
    color: var(--text-dim);
    font-size: 15px;
    max-width: 640px;
    line-height: 1.5;
  }

  main {
    max-width: 1100px;
    margin: 0 auto;
    padding: 32px;
  }

  .tabs {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
    margin-bottom: 28px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0;
  }

  .tab {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    background: none;
    border: none;
    color: var(--text-dim);
    padding: 10px 16px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    transition: color 0.15s, border-color 0.15s;
  }

  .tab:hover { color: var(--text); }

  .tab.active {
    color: var(--ochre-light);
    border-bottom-color: var(--ochre);
  }

  .tab:focus-visible {
    outline: 2px solid var(--ochre-light);
    outline-offset: 2px;
  }

  .grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 24px;
    margin-bottom: 24px;
  }

  @media (max-width: 800px) {
    .grid { grid-template-columns: 1fr; }
  }

  .panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 24px;
  }

  .panel h2 {
    font-family: 'Fraunces', serif;
    font-size: 18px;
    font-weight: 600;
    margin: 0 0 16px;
  }

  .core-sample {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .core-row {
    display: grid;
    grid-template-columns: 60px 1fr;
    align-items: center;
    gap: 10px;
  }

  .core-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    text-align: right;
  }

  .core-segment {
    height: 18px;
    border-radius: 2px;
  }

  .stats {
    display: flex;
    flex-direction: column;
    gap: 14px;
  }

  .stat {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    border-bottom: 1px dashed var(--border);
    padding-bottom: 10px;
  }

  .stat-value {
    font-size: 18px;
    font-weight: 600;
  }

  .legend {
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
    margin-top: 16px;
  }

  .legend-item { display: flex; align-items: center; gap: 6px; }
  .legend-swatch { width: 10px; height: 10px; border-radius: 2px; }

  footer {
    max-width: 1100px;
    margin: 0 auto;
    padding: 24px 32px 48px;
    color: var(--text-dim);
    font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
    line-height: 1.6;
  }

  canvas { max-height: 340px; }
</style>
</head>
<body>

<header>
  <div class="eyebrow">Karnataka &middot; GRACE Satellite Data &middot; 2002&ndash;2028</div>
  <h1>Groundwater Stress Forecast</h1>
  <div class="subtitle">
    24-month forward forecast of groundwater storage anomaly across six
    representative districts, derived from GRACE satellite gravimetry,
    ERA5 climate reanalysis, and Landsat vegetation indices.
  </div>
</header>

<main>
  <div class="tabs" role="tablist" id="tabs"></div>

  <div class="grid">
    <div class="panel">
      <h2 id="chart-title">Historical &amp; Forecast</h2>
      <canvas id="mainChart"></canvas>
    </div>
    <div class="panel">
      <h2>Forecast Summary</h2>
      <div class="stats" id="stats"></div>
    </div>
  </div>

  <div class="panel">
    <h2>24-Month Stress Core Sample</h2>
    <div class="core-sample" id="coreSample"></div>
    <div class="legend">
      <div class="legend-item"><span class="legend-swatch" style="background:var(--safe)"></span>Safe</div>
      <div class="legend-item"><span class="legend-swatch" style="background:var(--semi)"></span>Semi-Critical</div>
      <div class="legend-item"><span class="legend-swatch" style="background:var(--critical)"></span>Critical</div>
      <div class="legend-item"><span class="legend-swatch" style="background:var(--over)"></span>Over-Exploited</div>
    </div>
  </div>
</main>

<footer>
  Data: NASA JPL GRACE Tellus Mascon Solutions (RL06 v04) &middot; ECMWF ERA5-Land &middot;
  USGS Landsat 7/8 &middot; WorldPop &middot; Stress categories derived from historical
  per-district GRACE anomaly percentiles (methodological simplification &mdash;
  see limitations section).
</footer>

<script>
const DATA = __DATA_JSON__;
const DISTRICTS = __DISTRICTS_JSON__;
const CATEGORY_COLORS = { 'Safe': '#2ecc71', 'Semi-Critical': '#f1c40f', 'Critical': '#e67e22', 'Over-Exploited': '#e74c3c' };

let currentDistrict = DISTRICTS[0];
let chart = null;

function renderTabs() {
  const tabsEl = document.getElementById('tabs');
  tabsEl.innerHTML = '';
  DISTRICTS.forEach(d => {
    const btn = document.createElement('button');
    btn.className = 'tab' + (d === currentDistrict ? ' active' : '');
    btn.textContent = d;
    btn.setAttribute('role', 'tab');
    btn.onclick = () => { currentDistrict = d; render(); };
    tabsEl.appendChild(btn);
  });
}

function renderChart() {
  const d = DATA[currentDistrict];
  const ctx = document.getElementById('mainChart').getContext('2d');
  const labels = [...d.hist_dates, ...d.fc_dates];
  const histData = [...d.hist_values, ...Array(d.fc_dates.length).fill(null)];
  const fcData = [...Array(d.hist_dates.length - 1).fill(null), d.hist_values[d.hist_values.length-1], ...d.fc_values];

  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: 'Historical', data: histData, borderColor: '#a9c2c0', borderWidth: 1.5, pointRadius: 0, tension: 0.15 },
        { label: 'Forecast', data: fcData, borderColor: '#e0ac6a', borderWidth: 2, borderDash: [5,3], pointRadius: 0, tension: 0.15 }
      ]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#f2ede4', font: { family: 'IBM Plex Mono', size: 11 } } }
      },
      scales: {
        x: { ticks: { color: '#a9c2c0', maxRotation: 45, minRotation: 45, font: {size: 9} }, grid: { color: '#23494c' } },
        y: { ticks: { color: '#a9c2c0' }, grid: { color: '#23494c' }, title: { display: true, text: 'GRACE Anomaly (cm)', color: '#a9c2c0' } }
      }
    }
  });
}

function renderStats() {
  const d = DATA[currentDistrict];
  const counts = {};
  d.fc_categories.forEach(c => counts[c] = (counts[c]||0) + 1);
  const safeMonths = counts['Safe'] || 0;
  const stressedMonths = (counts['Critical']||0) + (counts['Over-Exploited']||0);
  const trend = d.fc_values[d.fc_values.length-1] - d.fc_values[0];

  document.getElementById('stats').innerHTML = `
    <div class="stat"><span>Safe months (of 24)</span><span class="stat-value" style="color:var(--safe)">${safeMonths}</span></div>
    <div class="stat"><span>Critical / Over-Exploited months</span><span class="stat-value" style="color:var(--over)">${stressedMonths}</span></div>
    <div class="stat"><span>Forecast range</span><span class="stat-value">${Math.min(...d.fc_values).toFixed(1)} to ${Math.max(...d.fc_values).toFixed(1)} cm</span></div>
    <div class="stat"><span>Net trend (24mo)</span><span class="stat-value">${trend >= 0 ? '+' : ''}${trend.toFixed(1)} cm</span></div>
  `;
  document.getElementById('chart-title').textContent = currentDistrict + ' \u2014 Historical & Forecast';
}

function renderCoreSample() {
  const d = DATA[currentDistrict];
  const el = document.getElementById('coreSample');
  el.innerHTML = '';
  const row = document.createElement('div');
  row.className = 'core-row';
  const label = document.createElement('div');
  label.className = 'core-label';
  label.textContent = d.fc_dates[0] + ' \u2192';
  const strip = document.createElement('div');
  strip.style.display = 'flex';
  strip.style.gap = '2px';
  d.fc_categories.forEach((cat, i) => {
    const seg = document.createElement('div');
    seg.className = 'core-segment';
    seg.style.flex = '1';
    seg.style.background = CATEGORY_COLORS[cat];
    seg.title = d.fc_dates[i] + ': ' + cat;
    strip.appendChild(seg);
  });
  row.appendChild(label);
  row.appendChild(strip);
  el.appendChild(row);
}

function render() {
  renderTabs();
  renderChart();
  renderStats();
  renderCoreSample();
}

render();
</script>

</body>
</html>
"""

html_output = html_template.replace('__DATA_JSON__', data_json).replace('__DISTRICTS_JSON__', json.dumps(DISTRICTS))

import os
os.makedirs('../outputs', exist_ok=True)
with open('../outputs/dashboard.html', 'w', encoding='utf-8') as f:
    f.write(html_output)

print("\nSaved: ../outputs/dashboard.html")
print("Open this file directly in your browser (double-click it) - no server needed.")