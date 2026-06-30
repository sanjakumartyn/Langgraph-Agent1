// ─── SalesIntel AI App Logic ────────────────────────────────────

const API_BASE = '';
let currentCompany = '';

// ─── Routing & Templates ──────────────────────────────────────
const appContainer = document.getElementById('app-container');

function navigate(view, data = null) {
  // Clear current view
  appContainer.innerHTML = '';
  
  // Get template content
  const template = document.getElementById(`view-${view}`);
  if (!template) return;
  
  const clone = template.content.cloneNode(true);
  appContainer.appendChild(clone);
  
  // Post-render logic
  if (view === 'internal') {
    loadInternalDashboard();
  } else if (view === 'details' && data) {
    renderCompanyDetails(data);
  } else if (view === 'history') {
    loadHistory();
  }
}

// Initial route
navigate('home');

// ─── Search & Analyze ─────────────────────────────────────────
async function handleAnalyze(e) {
  e.preventDefault();
  const input = document.getElementById('companySearch');
  const query = input.value.trim();
  if (!query) return;
  
  currentCompany = query;
  addToHistory(query);
  navigate('loading');
  
  try {
    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, session_id: 'session-demo' })
    });
    
    const json = await res.json();
    if (json.success && json.data) {
      navigate('details', json.data);
    } else {
      alert('Failed to analyze company data.');
      navigate('home');
    }
  } catch (err) {
    console.error(err);
    alert('Network error while analyzing.');
    navigate('home');
  }
}

// ─── Render Internal Dashboard ────────────────────────────────
async function loadInternalDashboard() {
  try {
    const res = await fetch(`${API_BASE}/api/internal`);
    const json = await res.json();
    if (!json.success) return;
    
    // Stats
    const statsHtml = `
      <div class="stat-card">
        <div class="stat-header">
          <div class="stat-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg></div>
          <div class="stat-label">Active Catalog</div>
        </div>
        <div class="stat-value">${json.stats.total_products}</div>
        <div class="stat-desc">Total Products</div>
      </div>
      <div class="stat-card">
        <div class="stat-header">
          <div class="stat-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg></div>
          <div class="stat-label">Growing</div>
        </div>
        <div class="stat-value">${json.stats.case_studies}</div>
        <div class="stat-desc">Case Studies Delivered</div>
      </div>
      <div class="stat-card">
        <div class="stat-header">
          <div class="stat-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg></div>
          <div class="stat-label">Pipeline</div>
        </div>
        <div class="stat-value">${json.stats.active_opportunities}</div>
        <div class="stat-desc">Active Opportunities</div>
      </div>
      <div class="stat-card">
        <div class="stat-header">
          <div class="stat-icon"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="8" r="7"/><polyline points="8.21 13.89 7 23 12 20 17 23 15.79 13.88"/></svg></div>
          <div class="stat-label">High</div>
        </div>
        <div class="stat-value">${json.stats.success_rate}%</div>
        <div class="stat-desc">Proposal Success Rate</div>
      </div>
    `;
    document.getElementById('internalStats').innerHTML = statsHtml;
    
    // Portfolio
    const portHtml = json.portfolio.map(p => `
      <div class="portfolio-item">
        <div class="portfolio-name">${p.name}</div>
        <div class="portfolio-bar-bg">
          <div class="portfolio-bar-fill" style="width: ${(p.count / json.stats.total_products) * 100}%"></div>
        </div>
        <div class="portfolio-count">${p.count} (${p.percentage}%)</div>
      </div>
    `).join('');
    document.getElementById('portfolioList').innerHTML = portHtml;
    
    // Top Solutions
    const solHtml = json.top_solutions.map(s => `
      <tr>
        <td>${s.name}</td>
        <td><span class="badge" style="background: rgba(245, 183, 0, 0.15); color: var(--accent); padding: 4px 10px; border-radius: 12px; font-size: 12px;">${s.opportunities}</span></td>
        <td class="text-right text-success">$${s.value.toLocaleString()}</td>
      </tr>
    `).join('');
    document.getElementById('topSolutionsBody').innerHTML = solHtml;
    
  } catch(e) {
    console.error(e);
  }
}

// ─── Render Company Details ───────────────────────────────────
function renderCompanyDetails(data) {
  // Title & Fit
  document.getElementById('detCompanyTitle').textContent = data.company || 'Company Analysis';
  
  const fit = parseInt(data.strategic_fit) || 0;
  document.getElementById('stratFitValue').textContent = fit + '%';
  document.getElementById('stratFitGauge').style.background = `conic-gradient(var(--accent) ${fit * 3.6}deg, var(--bg-input) 0deg)`;
  
  // Overview
  document.getElementById('detOverview').innerHTML = `<p>${data.overview}</p>`;
  
  // Needs
  const needsHtml = (data.needs_prediction || []).map(n => `<li>${n}</li>`).join('');
  document.getElementById('detNeeds').innerHTML = needsHtml;
  
  // Solutions Table
  const solHtml = (data.solution_mapping || []).map(sm => `
    <tr>
      <td style="font-weight: 500">${sm.requirement}</td>
      <td class="text-accent-2">${sm.solution}</td>
      <td>
        <div class="match-bar-container">
          <div class="match-bar-bg"><div class="match-bar-fill" style="width: ${sm.match}%"></div></div>
          <span style="font-size: 12px; font-weight: 600">${sm.match}%</span>
        </div>
      </td>
      <td class="text-muted">${sm.value || 'TBD'}</td>
      <td class="action-arrow">›</td>
    </tr>
  `).join('');
  document.getElementById('detSolutions').innerHTML = solHtml;
  
  // Meeting Prep Lists
  const prep = data.meeting_prep || {};
  document.getElementById('prepPriorities').innerHTML = (prep.priorities || []).map(p => `<li>${p}</li>`).join('');
  document.getElementById('prepGrowth').innerHTML = (prep.growth_initiatives || []).map(p => `<li>${p}</li>`).join('');
  document.getElementById('prepRisks').innerHTML = (prep.risks || []).map(p => `<li>${p}</li>`).join('');
  document.getElementById('prepSignals').innerHTML = (prep.buying_signals || []).map(p => `<li>${p}</li>`).join('');
  document.getElementById('prepObjections').innerHTML = (prep.objections || []).map(p => `<li>${p}</li>`).join('');
  document.getElementById('prepStakeholders').innerHTML = (prep.stakeholders || []).map(p => `<li>${p}</li>`).join('');
  
  initDealCoachBindings();
}

// ─── Tabs logic ───────────────────────────────────────────────
window.switchTab = function(btn, targetId) {
  const container = btn.closest('.tabs-container');
  // Update buttons
  container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  // Update content
  container.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById(targetId).classList.add('active');
};

window.showTab = function(containerId, targetId = null) {
  const container = document.getElementById(containerId);
  if (!container) return;
  // scroll to container
  container.scrollIntoView({ behavior: 'smooth' });
  
  if (targetId) {
    const btn = container.querySelector(`button[onclick*="'${targetId}'"]`);
    if (btn) {
      window.switchTab(btn, targetId);
    }
  }
};

// ─── Deal Coach Chat & Proposals ──────────────────────────────
async function handleChatSend() {
  const input = document.getElementById('coachChatInput');
  const msg = input.value.trim();
  if (!msg) return;
  
  input.value = '';
  const messagesDiv = document.getElementById('coachChatMessages');
  
  // Append user msg
  const userDiv = document.createElement('div');
  userDiv.className = 'msg user';
  userDiv.style = "background: rgba(245, 183, 0, 0.15); margin-left: auto; border-radius: 8px 8px 0px 8px; color: var(--text-main); border: 1px solid rgba(245, 183, 0, 0.3);";
  userDiv.textContent = msg;
  messagesDiv.appendChild(userDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  
  // Append typing indicator
  const typingDiv = document.createElement('div');
  typingDiv.className = 'msg assistant';
  typingDiv.textContent = 'Typing...';
  messagesDiv.appendChild(typingDiv);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
  
  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: msg, company: currentCompany })
    });
    const json = await res.json();
    if (json.success) {
      typingDiv.innerHTML = json.response;
    } else {
      typingDiv.textContent = 'Error: Could not connect to Deal Coach.';
    }
  } catch (err) {
    typingDiv.textContent = 'Error: Network failure.';
  }
}

async function handleGenerateProposal() {
  const btn = document.getElementById('generateProposalBtn');
  const status = document.getElementById('docStatus');
  const docList = document.getElementById('docList');
  
  btn.disabled = true;
  btn.textContent = 'Generating...';
  status.textContent = 'Synthesizing data into a personalized proposal...';
  
  try {
    const res = await fetch(`${API_BASE}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ company: currentCompany })
    });
    const json = await res.json();
    if (json.success) {
      status.textContent = 'Documents generated successfully.';
      const docItem = document.createElement('div');
      docItem.className = 'doc-item';
      // Render a clickable link for the PDF
      docItem.innerHTML = `
        <a href="${json.url}" target="_blank" style="color: inherit; text-decoration: none; font-weight: 500;">
          ${json.filename}
        </a>
        <div class="doc-formats">
          <a href="${json.url}" target="_blank" style="background: rgba(245, 183, 0, 0.15); color: var(--accent); padding: 4px 8px; border-radius: 4px; font-size: 11px; text-decoration: none; cursor: pointer;">
            DOWNLOAD PDF
          </a>
        </div>
      `;
      docList.appendChild(docItem);
    } else {
      status.textContent = 'Failed to generate proposal: ' + (json.error || 'Unknown error');
    }
  } catch (err) {
    status.textContent = 'Network error while generating.';
  } finally {
    btn.disabled = false;
    btn.textContent = 'Generate Proposal';
  }
}

function initDealCoachBindings() {
  const sendBtn = document.getElementById('coachChatSend');
  const input = document.getElementById('coachChatInput');
  const genBtn = document.getElementById('generateProposalBtn');
  
  if (sendBtn && input) {
    sendBtn.addEventListener('click', handleChatSend);
    input.addEventListener('keypress', (e) => {
      if (e.key === 'Enter') handleChatSend();
    });
  }
  if (genBtn) {
    genBtn.addEventListener('click', handleGenerateProposal);
  }
}

// ─── History Logic ─────────────────────────────────

function loadHistory() {
  const historyList = document.getElementById('historyList');
  if (!historyList) return;
  const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
  if (history.length === 0) {
    historyList.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted); padding: 24px;">No history available.</td></tr>';
    return;
  }
  // Show latest first
  historyList.innerHTML = history.slice().reverse().map(item => `
    <tr>
      <td style="font-weight: 500">${item.company}</td>
      <td class="text-muted">${item.date}</td>
      <td><button class="btn btn-sm btn-primary" onclick="reAnalyze('${item.company}')">Re-Analyze</button></td>
    </tr>
  `).join('');
}

function addToHistory(company) {
  let history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
  // Avoid immediate duplicates
  if (history.length === 0 || history[history.length - 1].company !== company) {
    history.push({ company, date: new Date().toLocaleString() });
    localStorage.setItem('searchHistory', JSON.stringify(history));
  }
}

window.reAnalyze = function(company) {
  const input = document.getElementById('companySearch');
  if (input) input.value = company;
  
  currentCompany = company;
  navigate('loading');
  fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: company, session_id: 'session-demo' })
  }).then(res => res.json()).then(json => {
      if (json.success && json.data) {
          navigate('details', json.data);
      } else {
          alert("Error re-analyzing");
          navigate('home');
      }
  }).catch(e => {
      alert("Network error");
      navigate('home');
  });
};

