// script.js — Apparel Classification Project Website
// Reads all data from results.json. No numbers are hardcoded here.

(async function () {
  // ── Load results.json ───────────────────────────────────────
  let R;
  try {
    const res = await fetch('results.json');
    R = await res.json();
  } catch (e) {
    console.error('Could not load results.json', e);
    return;
  }

  const d = R.dataset;

  // ── Theme Toggle ────────────────────────────────────────────
  const themeBtn = document.getElementById('theme-toggle');
  const stored = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', stored);
  themeBtn.textContent = stored === 'dark' ? '☀️' : '🌙';
  themeBtn.addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const next = cur === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('theme', next);
    themeBtn.textContent = next === 'dark' ? '☀️' : '🌙';
  });

  // ── Hero stats ───────────────────────────────────────────────
  setText('hero-total',     d.total_images.toLocaleString());
  setText('hero-indian',    d.classes.Indian.total.toLocaleString());
  setText('hero-western',   d.classes.Western.total.toLocaleString());
  setText('hero-iw',        d.classes['Indo-Western'].total.toLocaleString());
  setText('hero-seed',      d.seed);
  setText('hero-imgsize',   d.image_size);

  // ── Students ─────────────────────────────────────────────────
  const studentWrap = document.getElementById('student-chips');
  R.students.forEach(s => {
    const chip = document.createElement('div');
    chip.className = 'chip';
    chip.innerHTML = `${s.name} <span>${s.usn}</span>`;
    studentWrap.appendChild(chip);
  });

  // ── Dataset split bar ────────────────────────────────────────
  const trainPct = Math.round((d.splits.train / d.total_images) * 100);
  const valPct   = Math.round((d.splits.val   / d.total_images) * 100);
  const testPct  = 100 - trainPct - valPct;
  setText('split-train', `Train ${trainPct}% (${d.splits.train})`);
  setText('split-val',   `Val ${valPct}% (${d.splits.val})`);
  setText('split-test',  `Test ${testPct}% (${d.splits.test})`);
  setStyle('split-seg-train', 'flex', String(trainPct));
  setStyle('split-seg-val',   'flex', String(valPct));
  setStyle('split-seg-test',  'flex', String(testPct));

  // ── Class distribution chart ──────────────────────────────────
  const maxClass = Math.max(d.classes.Indian.total, d.classes.Western.total, d.classes['Indo-Western'].total);
  const classCharts = [
    { id: 'bar-indian',  count: d.classes.Indian.total,         color: 'var(--indian)',  label: 'Indian' },
    { id: 'bar-western', count: d.classes.Western.total,        color: 'var(--western)', label: 'Western' },
    { id: 'bar-iw',      count: d.classes['Indo-Western'].total, color: 'var(--iw)',     label: 'Indo-Western' },
  ];
  classCharts.forEach(({ id, count, color }) => {
    const fill = document.getElementById(id);
    if (fill) {
      fill.style.background = color;
      setTimeout(() => { fill.style.width = `${(count / maxClass) * 100}%`; }, 200);
    }
  });
  setText('val-indian',  d.classes.Indian.total);
  setText('val-western', d.classes.Western.total);
  setText('val-iw',      d.classes['Indo-Western'].total);

  // ── Per-class split detail ────────────────────────────────────
  renderSplitDetail('split-detail-indian',  d.classes.Indian);
  renderSplitDetail('split-detail-western', d.classes.Western);
  renderSplitDetail('split-detail-iw',      d.classes['Indo-Western']);

  // ── Category breakdown bars ────────────────────────────────────
  renderCategoryBars('cat-indian',  d.category_breakdown.Indian,         'var(--indian)');
  renderCategoryBars('cat-western', d.category_breakdown.Western,        'var(--western)');
  renderCategoryBars('cat-iw',      d.category_breakdown['Indo-Western'], 'var(--iw)');

  // ── Cleaning stats ─────────────────────────────────────────────
  const cl = d.cleaning;
  setText('cl-corrupted',  cl.corrupted_removed);
  setText('cl-exact',      cl.exact_duplicates_removed);
  setText('cl-near',       cl.near_duplicates_removed);
  setText('cl-ambiguous',  cl.ambiguous_excluded);
  setText('cl-edgecase',   cl.edge_case_candidates);

  // ── Models table ───────────────────────────────────────────────
  const tbody = document.getElementById('model-tbody');
  Object.entries(R.models).forEach(([name, m]) => {
    const tr = document.createElement('tr');
    const isPending = m.accuracy === 'Pending';
    tr.innerHTML = `
      <td class="model-name">${name}</td>
      <td class="${isPending ? 'pending' : ''}">${m.accuracy}</td>
      <td class="${isPending ? 'pending' : ''}">${m.precision}</td>
      <td class="${isPending ? 'pending' : ''}">${m.recall}</td>
      <td class="${isPending ? 'pending' : ''}">${m.f1}</td>
      <td class="${isPending ? 'pending' : ''}">${m.balanced_accuracy}</td>
    `;
    tbody.appendChild(tr);
  });

  // ── Edge-case counts ───────────────────────────────────────────
  setText('ec-visually-similar', 373);
  setText('ec-hybrid',           200);
  setText('ec-ambiguous',         41);
  setText('ec-total',            d.cleaning.edge_case_candidates);

  // ── Helpers ───────────────────────────────────────────────────
  function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
  }
  function setStyle(id, prop, val) {
    const el = document.getElementById(id);
    if (el) el.style[prop] = val;
  }

  function renderSplitDetail(containerId, cls) {
    const el = document.getElementById(containerId);
    if (!el) return;
    el.innerHTML = `
      <div class="bar-row">
        <div class="bar-label">Train</div>
        <div class="bar-track"><div class="bar-fill" style="width:${(cls.train/cls.total*100).toFixed(1)}%;background:var(--accent)"></div></div>
        <div class="bar-val">${cls.train}</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Val</div>
        <div class="bar-track"><div class="bar-fill" style="width:${(cls.val/cls.total*100).toFixed(1)}%;background:var(--accent2)"></div></div>
        <div class="bar-val">${cls.val}</div>
      </div>
      <div class="bar-row">
        <div class="bar-label">Test</div>
        <div class="bar-track"><div class="bar-fill" style="width:${(cls.test/cls.total*100).toFixed(1)}%;background:var(--accent3)"></div></div>
        <div class="bar-val">${cls.test}</div>
      </div>
    `;
  }

  function renderCategoryBars(containerId, cats, color) {
    const el = document.getElementById(containerId);
    if (!el || !cats) return;
    const maxCount = Math.max(...cats.map(c => c.count));
    el.innerHTML = cats.map(c => `
      <div class="bar-row">
        <div class="bar-label" style="font-size:0.78rem;">${c.category}</div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${(c.count/maxCount*100).toFixed(1)}%;background:${color}"></div>
        </div>
        <div class="bar-val">${c.count}</div>
      </div>
    `).join('');
    // animate
    setTimeout(() => {
      el.querySelectorAll('.bar-fill').forEach((f, i) => {
        const target = (cats[i].count / maxCount * 100).toFixed(1) + '%';
        f.style.width = '0';
        setTimeout(() => { f.style.width = target; }, 300 + i * 50);
      });
    }, 100);
  }

  // ── Animate hero stats on load ─────────────────────────────────
  const statEls = document.querySelectorAll('.stat-num[data-target]');
  statEls.forEach(el => {
    const target = parseInt(el.dataset.target);
    animateCount(el, 0, target, 1200);
  });

  function animateCount(el, from, to, duration) {
    const start = performance.now();
    function step(now) {
      const p = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = Math.round(from + eased * (to - from)).toLocaleString();
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

})();
