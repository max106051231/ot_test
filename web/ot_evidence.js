/**
 * 監控戰情室：Annex A 全控制項 · 證據鏈流程 · 傳統 vs Semi-Shield 比較
 */
(function () {
    let evidenceDashboard = null;
    let controlIndex = {};

    function tierClass(tier) {
        if (tier === 'auto_syslog') return 'tier-auto';
        if (tier === 'auto_import' || tier === 'assisted') return 'tier-auto-import';
        if (tier === 'manual_only') return 'tier-auto-import';
        return 'tier-auto-import';
    }

    function statusDot(st) {
        if (st === 'fail') return 'fail';
        if (st === 'review' || st === 'warn') return 'review';
        if (st === 'auto') return 'pass';
        if (st === 'manual') return 'manual';
        if (st === 'assisted' || st === 'n/a') return 'pass';
        return 'pass';
    }

    function escapeHtml(s) {
        return String(s ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;');
    }

    function flowDetailText(s, maxLen) {
        const t = String(s ?? '').trim();
        if (t.length <= maxLen) return t;
        return t.slice(0, maxLen - 1) + '…';
    }

    function renderFlowSvg(chain, side) {
        if (!chain || !chain.length) return '';
        const n = chain.length;
        const stepW = n >= 9 ? 122 : (n >= 7 ? 128 : 138);
        const stepH = 58;
        const gap = n >= 9 ? 12 : (n >= 7 ? 16 : 22);
        const pad = 10;
        const w = pad * 2 + n * stepW + Math.max(0, n - 1) * gap;
        const h = stepH + pad * 2;
        const fill = side === 'semi' ? 'rgba(56,189,248,0.15)' : 'rgba(148,163,184,0.12)';
        const stroke = side === 'semi' ? '#38bdf8' : '#94a3b8';
        const markerId = `ev-arr-${side}-${n}`;
        let svg = `<svg viewBox="0 0 ${w} ${h}" class="ev-flow-svg ev-flow-h" xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">`;
        svg += `<defs><marker id="${markerId}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="${stroke}"/></marker></defs>`;
        chain.forEach((step, i) => {
            const x = pad + i * (stepW + gap);
            const y = pad;
            const cy = y + stepH / 2;
            svg += `<rect x="${x}" y="${y}" width="${stepW}" height="${stepH}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1"/>`;
            svg += `<text x="${x + 8}" y="${y + 20}" class="ev-flow-title">${escapeHtml(flowDetailText(step.label, n >= 9 ? 16 : 14))}</text>`;
            svg += `<text x="${x + 8}" y="${y + 38}" class="ev-flow-sub">${escapeHtml(flowDetailText(step.detail || step.kind || '', n >= 9 ? 26 : 22))}</text>`;
            if (i < n - 1) {
                const x1 = x + stepW;
                const x2 = x + stepW + gap;
                svg += `<line x1="${x1}" y1="${cy}" x2="${x2}" y2="${cy}" stroke="${stroke}" stroke-width="1.5" marker-end="url(#${markerId})"/>`;
            }
        });
        svg += '</svg>';
        return svg;
    }

    function renderControlGrid(controls) {
        const grid = document.getElementById('controlGrid');
        if (!grid) return;
        controlIndex = {};
        let html = '';
        controls.forEach((c) => {
            const key = c.control_key;
            controlIndex[key] = c;
            const dot = statusDot(c.metric_status);
            let val = c.metric_text != null ? c.metric_text : null;
            if (val == null) {
                const mi = Number(c.manual_import_count || 0);
                if (mi > 0) val = `人工 ${mi}`;
                else if (c.automation_tier === 'auto_import' || c.automation_tier === 'manual_only' || c.automation_tier === 'assisted') {
                    val = 'AI 匯入';
                } else val = '—';
            }
            html += `<div class="summary-card control-card ${tierClass(c.automation_tier)}" data-key="${escapeHtml(key)}" onclick="showDetails('${escapeHtml(key)}')">
                <div class="card-header">
                    <span class="card-title">${escapeHtml(c.annex_a_id)} ${escapeHtml(c.title || '')}</span>
                    <div class="status-dot ${dot}"></div>
                </div>
                <div class="big-value card-metric">${escapeHtml(String(val))}</div>
                <div class="card-foot">${escapeHtml(c.automation_label || '')}</div>
            </div>`;
        });
        grid.innerHTML = html;
    }

    function syncComplianceDb(controls) {
        if (typeof complianceDb === 'undefined') return;
        controls.forEach((c) => {
            const key = c.control_key;
            const bundle = (typeof globalControlBundles !== 'undefined' && globalControlBundles[c.control_key]) || {};
            const prev = complianceDb[key] || {};
            complianceDb[key] = {
                title: `${c.annex_a_id} ${c.title || ''}`,
                meta: [
                    ['Annex A', c.annex_a_id],
                    ['自動化', c.automation_label],
                    ['領域', c.domain_label || c.domain],
                    ['metric', c.metric_text || '—'],
                    ['evidence', c.evidence_id || '—'],
                ],
                log: c.log || bundle.log || prev.log || (c.automation_tier === 'auto_syslog'
                    ? '（無 syslog 事件摘要）'
                    : '（AI 全自動匯入稽核資料 · 等待匯入批次或已解析摘要）'),
                aiAnalysis: prev.aiAnalysis || '尚無 AI 診斷數據，請點擊「執行地端診斷分析」。',
                evidenceId: c.evidence_id || '',
                metricSummary: bundle.metric_summary || '',
                evidenceChain: c,
                ragCitations: prev.ragCitations || [],
            };
        });
    }

    function renderCompareTab(comparison, summary) {
        const host = document.getElementById('comparePanelBody');
        if (!host || !comparison) return;
        const table = (comparison.comparison_table || []).map((row) =>
            `<tr><td>${escapeHtml(row.aspect)}</td><td>${escapeHtml(row.semi)}</td><td>${escapeHtml(row.traditional)}</td></tr>`
        ).join('');
        const semiPts = (comparison.semi_shield?.points || []).map((p) => `<li>${escapeHtml(p)}</li>`).join('');
        const tradPts = (comparison.traditional?.points || []).map((p) => `<li>${escapeHtml(p)}</li>`).join('');
        host.innerHTML = `
            <div class="compare-summary-bar">
                <span>Annex A 共 <strong>${summary?.annex_total ?? 93}</strong> 項</span>
                <span class="pill tier-auto">Syslog 自動 ${summary?.auto_syslog ?? 0}</span>
                <span class="pill tier-auto-import">AI 全自動匯入 ${summary?.auto_import ?? summary?.manual_only ?? 0}</span>
            </div>
            <div class="compare-columns">
                <div class="compare-col semi">
                    <h3>${escapeHtml(comparison.semi_shield?.title || 'Semi-Shield')}</h3>
                    <ul>${semiPts}</ul>
                </div>
                <div class="compare-col trad">
                    <h3>${escapeHtml(comparison.traditional?.title || '傳統')}</h3>
                    <ul>${tradPts}</ul>
                </div>
            </div>
            <h3 class="compare-table-title">逐項對照</h3>
            <table class="compare-table">
                <thead><tr><th>面向</th><th>Semi-Shield</th><th>傳統稽核</th></tr></thead>
                <tbody>${table}</tbody>
            </table>`;
    }

    function renderModalEvidenceChain(c) {
        const box = document.getElementById('modalEvidenceFlow');
        if (!box || !c) return;
        const srcHtml = (c.sources || []).map((s) =>
            `<span class="src-chip">${escapeHtml(s.type)} · ${escapeHtml(s.origin)}</span>`
        ).join('') || '<span class="src-chip muted">無自動來源</span>';
        box.innerHTML = `
            <h3 class="ev-section-title">證據來源</h3>
            <div class="src-list">${srcHtml}</div>
            <div class="ev-flow-row">
                <div class="ev-flow-col"><h4>Semi-Shield 證據鏈</h4>${renderFlowSvg(c.semi_shield_chain, 'semi')}</div>
                <div class="ev-flow-col"><h4>傳統證據鏈</h4>${renderFlowSvg(c.traditional_chain, 'trad')}</div>
            </div>`;
        box.style.display = 'block';
    }

    window.renderModalEvidenceChainForKey = function (key) {
        const c = controlIndex[key] || (complianceDb[key] && complianceDb[key].evidenceChain);
        renderModalEvidenceChain(c);
        const btn = document.getElementById('aiBtn');
        const btnText = document.getElementById('aiBtnText');
        if (btn && btnText) {
            if (c && c.automation_tier !== 'auto_syslog' && c.automation_tier !== 'auto_import') {
                btn.disabled = true;
                btnText.innerText = '此控制項尚無自動證據管線';
            } else {
                btn.disabled = false;
                btnText.innerText = c.automation_tier === 'auto_import'
                    ? '執行 AI 匯入資料診斷'
                    : '執行地端診斷分析';
            }
        }
    };

    function applyEvidenceDashboard(dash) {
        evidenceDashboard = dash;
        if (!dash || !dash.ok) return;
        renderControlGrid(dash.controls || []);
        syncComplianceDb(dash.controls || []);
        populateManualAnnexSelect(dash.controls || []);
        renderCompareTab(dash.comparison, dash.summary);
        const sub = document.getElementById('warSubtitle');
        if (sub) {
            sub.textContent = '證據鏈導向 · Annex A 全控制項 · No Evidence, No Compliance Claim';
        }
    }

    function setManualImportStatus(msg, kind) {
        const el = document.getElementById('manualImportStatus');
        if (!el) return;
        el.textContent = msg || '';
        el.className = 'manual-import-status' + (kind ? ' ' + kind : '');
    }

    function populateManualAnnexSelect(controls) {
        const sel = document.getElementById('manualAnnexSelect');
        if (!sel || !controls.length) return;
        const prev = sel.value;
        const importCtrls = controls.filter((c) =>
            c.automation_tier === 'auto_import' || c.automation_tier === 'assisted' || c.automation_tier === 'manual_only'
        );
        const list = importCtrls.length ? importCtrls : controls;
        sel.innerHTML = list.map((c) =>
            `<option value="${escapeHtml(c.annex_a_id)}">${escapeHtml(c.annex_a_id)} ${escapeHtml(c.title || '')}</option>`
        ).join('');
        if (prev && [...sel.options].some((o) => o.value === prev)) sel.value = prev;
    }

    async function loadManualImportRecent() {
        const ul = document.getElementById('manualImportRecent');
        if (!ul) return;
        try {
            const res = await fetch('/api/monitor/manual_imports?limit=12');
            const data = await res.json();
            const rows = data.imports || [];
            if (!rows.length) {
                ul.innerHTML = '<li class="mi-empty">尚無紀錄<br><span style="opacity:0.75;font-size:11px">可提交表單或按「模擬匯入」產生示範資料</span></li>';
                return;
            }
            ul.innerHTML = rows.map((r) => {
                const tag = r.simulated
                    ? '<span class="mi-tag sim">SIM</span>'
                    : '<span class="mi-tag">MANUAL</span>';
                const preview = String(r.content || '').split('\n')[0].slice(0, 72);
                return `<li>${tag}<strong>${escapeHtml(r.annex_a_id)}</strong> ${escapeHtml(r.title || '')} · ${escapeHtml(r.imported_at || '')}<br><span style="opacity:0.85">${escapeHtml(preview)}</span></li>`;
            }).join('');
        } catch (e) {
            ul.innerHTML = '<li class="mi-empty" style="color:#fca5a5">無法載入匯入紀錄</li>';
        }
    }

    async function refreshAfterManualImport(message) {
        setManualImportStatus(message, 'ok');
        await loadManualImportRecent();
        await loadEvidenceDashboard();
        if (typeof fetchBackendData === 'function') await fetchBackendData();
    }

    window.submitManualImport = async function (ev) {
        if (ev && ev.preventDefault) ev.preventDefault();
        const annex = document.getElementById('manualAnnexSelect')?.value;
        const title = document.getElementById('manualImportTitle')?.value || '';
        const content = document.getElementById('manualImportBody')?.value || '';
        if (!annex || !content.trim()) {
            setManualImportStatus('請選擇控制項並填寫證據原文。', 'err');
            return false;
        }
        setManualImportStatus('匯入中…', '');
        try {
            const res = await fetch('/api/monitor/manual_import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    annex_a_id: annex,
                    title,
                    content,
                    source: 'warroom_form',
                }),
            });
            const data = await res.json();
            if (!res.ok || !data.ok) {
                setManualImportStatus(data.error || '匯入失敗', 'err');
                return false;
            }
            document.getElementById('manualImportBody').value = '';
            await refreshAfterManualImport(`已匯入 ${annex}（${data.record?.import_id || 'ok'}）`);
        } catch (e) {
            setManualImportStatus('匯入失敗：' + e.message, 'err');
        }
        return false;
    };

    async function simulateManualImport() {
        const btn = document.getElementById('btnSimManualImport');
        if (btn) btn.disabled = true;
        setManualImportStatus('模擬批次匯入中…', '');
        try {
            const res = await fetch('/api/monitor/manual_import/simulate', { method: 'POST' });
            const data = await res.json();
            if (!res.ok || !data.ok) {
                setManualImportStatus(data.error || '模擬失敗', 'err');
                return;
            }
            await refreshAfterManualImport(data.message || '模擬匯入完成');
        } catch (e) {
            setManualImportStatus('模擬失敗：' + e.message, 'err');
        } finally {
            if (btn) btn.disabled = false;
        }
    }

    async function loadEvidenceDashboard() {
        try {
            const topoSite = sessionStorage.getItem('ot_topo_site') || 'fab-a';
            const res = await fetch('/api/monitor/evidence_dashboard?site=' + encodeURIComponent(topoSite));
            const dash = await res.json();
            applyEvidenceDashboard(dash);
        } catch (e) {
            console.warn('evidence_dashboard load failed', e);
        }
    }

    window.initEvidenceWarroom = function () {
        document.querySelectorAll('.war-tab').forEach((btn) => {
            btn.addEventListener('click', () => {
                const tab = btn.dataset.tab;
                document.querySelectorAll('.war-tab').forEach((b) => b.classList.toggle('active', b.dataset.tab === tab));
                document.getElementById('panelWarroom')?.classList.toggle('panel-hidden', tab !== 'warroom');
                document.getElementById('panelManual')?.classList.toggle('panel-hidden', tab !== 'manual');
                document.getElementById('panelCompare')?.classList.toggle('panel-hidden', tab !== 'compare');
                if (tab === 'manual') loadManualImportRecent();
            });
        });
        document.getElementById('btnSimManualImport')?.addEventListener('click', simulateManualImport);
        loadEvidenceDashboard();
        loadManualImportRecent();
        const origFetch = window.fetchBackendData;
        if (typeof origFetch === 'function') {
            window.fetchBackendData = async function () {
                await origFetch();
                await loadEvidenceDashboard();
            };
        }
    };

})();
