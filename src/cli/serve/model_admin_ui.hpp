#pragma once

#include <string_view>

namespace gufo::server {

inline constexpr std::string_view kModelAdminUiHtml = R"GUFO_UI(<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MBufo Router & Multi-Model Control Plane (:8741)</title>
  <style>
    :root {
      --bg: #0b0e14;
      --panel: #131822;
      --panel-alt: #1a2130;
      --border: #263044;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --green: #10b981;
      --amber: #f59e0b;
      --red: #ef4444;
      --purple: #8b5cf6;
      --cyan: #06b6d4;
      --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans TC", sans-serif;
      --mono: "JetBrains Mono", "Fira Code", Consolas, monospace;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 14px;
      line-height: 1.5;
      min-height: 100vh;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 14px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      position: sticky;
      top: 0;
      z-index: 20;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-logo {
      width: 34px;
      height: 34px;
      border-radius: 8px;
      background: linear-gradient(135deg, var(--accent), var(--purple));
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 800;
      font-size: 16px;
      color: #fff;
      box-shadow: 0 2px 10px rgba(59,130,246,0.35);
    }
    .brand h1 {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 0.2px;
    }
    .brand-sub {
      font-size: 12px;
      color: var(--muted);
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
      background: rgba(16, 185, 129, 0.12);
      color: var(--green);
      border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .status-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 8px currentColor;
    }
    .container {
      max-width: 1480px;
      margin: 0 auto;
      padding: 20px 24px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 18px 20px;
    }
    .card-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
      gap: 12px;
      flex-wrap: wrap;
    }
    .card-title {
      font-size: 15px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 2px 8px;
      border-radius: 6px;
      font-size: 11px;
      font-weight: 600;
      font-family: var(--mono);
      background: var(--panel-alt);
      color: var(--muted);
      border: 1px solid var(--border);
    }
    .badge-primary { background: rgba(59,130,246,0.15); color: #60a5fa; border-color: rgba(59,130,246,0.35); }
    .badge-green { background: rgba(16,185,129,0.15); color: #34d399; border-color: rgba(16,185,129,0.35); }
    .badge-amber { background: rgba(245,158,11,0.15); color: #fbbf24; border-color: rgba(245,158,11,0.35); }
    .badge-purple { background: rgba(139,92,246,0.15); color: #a78bfa; border-color: rgba(139,92,246,0.35); }
    .badge-cyan { background: rgba(6,182,212,0.15); color: #22d3ee; border-color: rgba(6,182,212,0.35); }
    .badge-red { background: rgba(239,68,68,0.15); color: #f87171; border-color: rgba(239,68,68,0.35); }

    /* LM Studio style Segmented Memory Bar */
    .mem-summary {
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      margin-bottom: 8px;
      font-size: 13px;
      flex-wrap: wrap;
      gap: 8px;
    }
    .mem-bar-outer {
      width: 100%;
      height: 24px;
      background: #090c10;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      display: flex;
      position: relative;
    }
    .mem-seg {
      height: 100%;
      transition: width 0.3s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 11px;
      font-family: var(--mono);
      font-weight: 600;
      color: #fff;
      overflow: hidden;
      white-space: nowrap;
      border-right: 1px solid rgba(0,0,0,0.4);
    }
    .mem-seg-preview {
      background: repeating-linear-gradient(
        45deg,
        rgba(245, 158, 11, 0.45),
        rgba(245, 158, 11, 0.45) 8px,
        rgba(245, 158, 11, 0.2) 8px,
        rgba(245, 158, 11, 0.2) 16px
      );
      border: 1px dashed var(--amber);
      color: #fde68a;
    }
    .mem-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 10px;
      font-size: 12px;
      color: var(--muted);
    }
    .mem-legend-item {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .mem-swatch {
      width: 10px;
      height: 10px;
      border-radius: 3px;
    }
    .gate-banner {
      margin-top: 10px;
      padding: 8px 12px;
      border-radius: 8px;
      font-size: 12px;
      display: none;
      align-items: center;
      justify-content: space-between;
      background: rgba(245, 158, 11, 0.1);
      border: 1px solid rgba(245, 158, 11, 0.35);
      color: #fcd34d;
    }

    /* Main Grid */
    .main-grid {
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 20px;
    }
    @media (max-width: 1120px) {
      .main-grid { grid-template-columns: 1fr; }
    }

    /* Active Models */
    .loaded-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .model-card {
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 10px;
      transition: border-color 0.2s;
    }
    .model-card.is-default {
      border-color: rgba(59, 130, 246, 0.55);
      box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.2);
    }
    .model-top {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .model-id {
      font-size: 15px;
      font-weight: 700;
      font-family: var(--mono);
      color: #fff;
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .model-path {
      font-size: 11px;
      font-family: var(--mono);
      color: var(--muted);
      word-break: break-all;
      margin-top: 2px;
    }
    .model-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      font-size: 12px;
      color: var(--muted);
    }
    .slots-bar {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
      background: rgba(0,0,0,0.25);
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.05);
    }
    .slot-pill {
      font-family: var(--mono);
      font-size: 11px;
      padding: 2px 7px;
      border-radius: 4px;
      background: #1f2937;
      color: #9ca3af;
      border: 1px solid #374151;
    }
    .slot-pill.busy {
      background: rgba(16, 185, 129, 0.2);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.5);
      animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.6; }
    }
    .model-actions {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }

    /* Buttons & Inputs */
    button, .btn {
      background: var(--panel-alt);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 6px 12px;
      border-radius: 7px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s;
    }
    button:hover { background: #242f44; border-color: #3b4b6b; }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .btn-primary { background: var(--accent); border-color: var(--accent); color: #fff; }
    .btn-primary:hover { background: var(--accent-hover); border-color: var(--accent-hover); }
    .btn-danger { background: rgba(239,68,68,0.12); border-color: rgba(239,68,68,0.35); color: #f87171; }
    .btn-danger:hover { background: rgba(239,68,68,0.24); }
    .btn-sm { padding: 4px 9px; font-size: 11px; }

    input[type="text"], input[type="number"], select, textarea {
      background: #0d1117;
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 7px 10px;
      font-size: 13px;
      font-family: var(--font);
      outline: none;
    }
    input:focus, select:focus, textarea:focus {
      border-color: var(--accent);
    }

    /* Catalog */
    .catalog-controls {
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .catalog-controls input { flex: 1; min-width: 180px; }
    .family-pills {
      display: flex;
      gap: 6px;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .family-pill {
      padding: 3px 10px;
      border-radius: 999px;
      font-size: 11px;
      cursor: pointer;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      color: var(--muted);
    }
    .family-pill.active {
      background: rgba(59,130,246,0.2);
      border-color: var(--accent);
      color: #60a5fa;
      font-weight: 600;
    }
    .catalog-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      max-height: 540px;
      overflow-y: auto;
      padding-right: 4px;
    }
    .cat-item {
      background: var(--panel-alt);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 14px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      transition: border-color 0.15s;
    }
    .cat-item:hover { border-color: #3b4b6b; }
    .cat-info { min-width: 0; flex: 1; }
    .cat-name {
      font-weight: 600;
      font-size: 13px;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }
    .cat-rel {
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    /* Playground */
    .playground-grid {
      display: grid;
      grid-template-columns: 300px 1fr;
      gap: 16px;
    }
    @media (max-width: 900px) {
      .playground-grid { grid-template-columns: 1fr; }
    }
    .pg-sidebar {
      display: flex;
      flex-direction: column;
      gap: 10px;
      background: var(--panel-alt);
      padding: 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
    }
    .field-label {
      font-size: 11px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.4px;
      margin-bottom: 4px;
      display: block;
    }
    .pg-chat {
      display: flex;
      flex-direction: column;
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 10px;
      height: 360px;
      overflow: hidden;
    }
    .pg-messages {
      flex: 1;
      padding: 14px;
      overflow-y: auto;
      display: flex;
      flex-direction: column;
      gap: 10px;
      font-size: 13px;
    }
    .msg {
      padding: 10px 12px;
      border-radius: 8px;
      max-width: 88%;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .msg-user {
      align-self: flex-end;
      background: rgba(59,130,246,0.2);
      border: 1px solid rgba(59,130,246,0.4);
    }
    .msg-assistant {
      align-self: flex-start;
      background: var(--panel-alt);
      border: 1px solid var(--border);
      font-family: var(--mono);
      font-size: 12px;
    }
    .pg-input-bar {
      display: flex;
      gap: 8px;
      padding: 10px;
      background: var(--panel);
      border-top: 1px solid var(--border);
    }
    .pg-input-bar input { flex: 1; }
    .pg-metrics {
      display: flex;
      gap: 14px;
      font-family: var(--mono);
      font-size: 11px;
      color: var(--muted);
      margin-top: 8px;
      flex-wrap: wrap;
    }

    /* Modal */
    .modal-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.7);
      backdrop-filter: blur(3px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 50;
      padding: 16px;
    }
    .modal-backdrop.open { display: flex; }
    .modal {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      width: 100%;
      max-width: 620px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      box-shadow: 0 20px 50px rgba(0,0,0,0.6);
    }
    .form-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }
    .form-full { grid-column: 1 / -1; }
    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      cursor: pointer;
    }

    /* Loading Banner */
    .loading-banner {
      background: rgba(59, 130, 246, 0.14);
      border: 1px solid rgba(59, 130, 246, 0.45);
      border-radius: 10px;
      padding: 12px 16px;
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid rgba(96,165,250,0.3);
      border-top-color: #60a5fa;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
    .toast {
      position: fixed;
      bottom: 20px;
      right: 20px;
      background: #1e293b;
      border: 1px solid #475569;
      color: #fff;
      padding: 10px 16px;
      border-radius: 8px;
      font-size: 13px;
      z-index: 100;
      display: none;
      box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }
  </style>
</head>
<body>
  <header>
    <div class="brand">
      <div class="brand-logo">G</div>
      <div>
        <h1>MBufo Multi-Model Router & Control Plane</h1>
        <div class="brand-sub">Unified Port :8741 · OpenAI /v1/chat/completions · Anthropic /v1/messages · Dynamic Hot-Load & LRU Eviction</div>
      </div>
    </div>
    <div class="header-actions">
      <span id="conn-pill" class="status-pill"><span class="status-dot"></span> Online (:8741)</span>
      <button onclick="refreshAll(true)">↻ Refresh</button>
    </div>
  </header>

  <div class="container">
    <!-- Loading In-Progress Banner -->
    <div id="loading-banner" class="loading-banner">
      <div style="display:flex;align-items:center;gap:10px;">
        <div class="spinner"></div>
        <div>
          <strong id="loading-title">Loading model...</strong>
          <div id="loading-sub" style="font-size:12px;color:var(--muted);font-family:var(--mono);"></div>
        </div>
      </div>
      <span id="loading-elapsed" class="badge badge-primary">0.0s</span>
    </div>

    <!-- LM Studio Style Unified Memory / VRAM Gate Bar -->
    <div class="card">
      <div class="card-header" style="margin-bottom:8px;">
        <div class="card-title">
          <span>Unified Memory & VRAM Gate (Strix Halo UMA / HIP)</span>
          <span id="mem-rss-badge" class="badge">gufo RSS: -- GiB</span>
        </div>
        <div id="mem-numbers" style="font-family:var(--mono);font-size:12px;color:var(--muted);">
          Loading memory telemetry...
        </div>
      </div>
      <div class="mem-bar-outer" id="mem-bar"></div>
      <div class="mem-legend" id="mem-legend"></div>
      <div class="gate-banner" id="gate-banner"></div>
    </div>

    <!-- Active Loaded Models + Model Catalog -->
    <div class="main-grid">
      <!-- Left: Loaded Models & Live Session Slots -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <span>Loaded Models & Live Slots</span>
            <span id="loaded-count" class="badge badge-green">0 loaded</span>
          </div>
          <div style="font-size:12px;color:var(--muted);">
            Requests route by <code style="color:#93c5fd;">"model"</code> field or fallback to Primary
          </div>
        </div>
        <div id="loaded-list" class="loaded-list"></div>
      </div>

      <!-- Right: Local GGUF Catalog -->
      <div class="card">
        <div class="card-header">
          <div class="card-title">
            <span>Model Catalog</span>
            <span id="catalog-count" class="badge">0 GGUFs</span>
          </div>
          <span style="font-size:11px;font-family:var(--mono);color:var(--muted);">/mnt/nvme-data/gufo-models</span>
        </div>
        <div class="catalog-controls">
          <input type="text" id="cat-search" placeholder="Search model name, quant (e.g. Q4_K, 27B, gemma, coder)..." oninput="renderCatalog()" />
        </div>
        <div class="family-pills" id="family-pills"></div>
        <div class="catalog-list" id="catalog-list"></div>
      </div>
    </div>

    <!-- Quick Chat & Router Verification Playground -->
    <div class="card">
      <div class="card-header">
        <div class="card-title">
          <span>Router & Multi-Model Playground</span>
          <span class="badge badge-purple">Live Stream Test</span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <button class="btn-sm" onclick="clearChat()">Clear Output</button>
        </div>
      </div>
      <div class="playground-grid">
        <div class="pg-sidebar">
          <div>
            <label class="field-label">Target Model ("model" field)</label>
            <select id="pg-model" style="width:100%;"></select>
          </div>
          <div>
            <label class="field-label">API Endpoint Format</label>
            <select id="pg-endpoint" style="width:100%;">
              <option value="/v1/chat/completions">OpenAI (/v1/chat/completions)</option>
              <option value="/v1/messages">Anthropic (/v1/messages)</option>
            </select>
          </div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;">
            <div>
              <label class="field-label">Max Tokens</label>
              <input type="number" id="pg-max-tokens" value="256" style="width:100%;" />
            </div>
            <div>
              <label class="field-label">Temperature</label>
              <input type="number" id="pg-temp" value="0.6" step="0.1" style="width:100%;" />
            </div>
          </div>
          <div>
            <label class="field-label">cURL / API Routing Snippet</label>
            <pre id="pg-curl" style="background:#090c10;border:1px solid var(--border);padding:8px;border-radius:6px;font-family:var(--mono);font-size:10px;color:#93c5fd;overflow-x:auto;white-space:pre-wrap;"></pre>
          </div>
        </div>
        <div>
          <div class="pg-chat">
            <div class="pg-messages" id="pg-messages">
              <div style="color:var(--muted);font-size:12px;">Select any loaded model (or Default Route) and send a message to verify simultaneous multi-model serving on :8741.</div>
            </div>
            <div class="pg-input-bar">
              <input type="text" id="pg-input" placeholder="Ask something (e.g. '你是誰？用一句話回答')..." onkeydown="if(event.key==='Enter') sendPlayground()" />
              <button class="btn-primary" id="pg-send-btn" onclick="sendPlayground()">Send ↵</button>
            </div>
          </div>
          <div class="pg-metrics" id="pg-metrics">
            <span>Routed Model: <strong id="m-model">-</strong></span>
            <span>TTFT: <strong id="m-ttft">-</strong></span>
            <span>Decode Speed: <strong id="m-tps">-</strong></span>
            <span>Tokens: <strong id="m-tokens">-</strong></span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Configure & Load Modal -->
  <div class="modal-backdrop" id="load-modal">
    <div class="modal">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <h3 style="font-size:16px;">Configure & Hot-Load Model into :8741</h3>
        <button class="btn-sm" onclick="closeLoadModal()">✕</button>
      </div>
      <div class="form-grid">
        <div>
          <label class="field-label">Model ID (Router Key)</label>
          <input type="text" id="lm-id" style="width:100%;" />
        </div>
        <div>
          <label class="field-label">Aliases (comma-separated)</label>
          <input type="text" id="lm-aliases" placeholder="e.g. coder, fast" style="width:100%;" />
        </div>
        <div class="form-full">
          <label class="field-label">GGUF Path</label>
          <input type="text" id="lm-path" style="width:100%;font-family:var(--mono);font-size:12px;" />
        </div>
        <div>
          <label class="field-label">Context Length (-c)</label>
          <select id="lm-ctx" style="width:100%;" onchange="updateModalEstimate()">
            <option value="8192">8,192 (8K - Low RAM)</option>
            <option value="16384">16,384 (16K)</option>
            <option value="32768" selected>32,768 (32K - Balanced)</option>
            <option value="65536">65,536 (64K)</option>
            <option value="131072">131,072 (128K)</option>
            <option value="262144">262,144 (256K - Full)</option>
          </select>
        </div>
        <div>
          <label class="field-label">Parallel Sessions (--sessions)</label>
          <select id="lm-sessions" style="width:100%;" onchange="updateModalEstimate()">
            <option value="1" selected>1 Session</option>
            <option value="2">2 Sessions</option>
            <option value="4">4 Sessions</option>
          </select>
        </div>
        <div>
          <label class="field-label">KV Cache Quantization</label>
          <select id="lm-kv" style="width:100%;">
            <option value="q8" selected>q8 (Recommended)</option>
            <option value="fp16">fp16</option>
            <option value="fp8">fp8</option>
          </select>
        </div>
        <div>
          <label class="field-label">Speculative / MTP Mode</label>
          <select id="lm-spec" style="width:100%;">
            <option value="off" selected>off (Safe for all architectures)</option>
            <option value="mtp">mtp (Multi-Token Prediction)</option>
            <option value="dflash2">dflash2 (Block Diffusion Draft)</option>
            <option value="dspark">dspark (Draft Spark)</option>
          </select>
        </div>
        <div class="form-full">
          <label class="field-label">Draft / MTP Sidecar Path (Optional)</label>
          <select id="lm-draft" style="width:100%;font-family:var(--mono);font-size:12px;"></select>
        </div>
        <div class="form-full">
          <label class="field-label">Vision mmproj Sidecar Path (Optional)</label>
          <select id="lm-mmproj" style="width:100%;font-family:var(--mono);font-size:12px;"></select>
        </div>
        <div class="form-full" style="display:flex;gap:18px;flex-wrap:wrap;padding-top:4px;">
          <label class="checkbox-row">
            <input type="checkbox" id="lm-default" /> Set as Primary Default Model
          </label>
          <label class="checkbox-row">
            <input type="checkbox" id="lm-evict" checked /> Allow LRU Auto-Eviction if RAM Full
          </label>
          <label class="checkbox-row">
            <input type="checkbox" id="lm-yarn" /> Enable YaRN Scaling
          </label>
        </div>
      </div>
      <div id="lm-estimate" style="background:#090c10;border:1px solid var(--border);padding:10px 12px;border-radius:8px;font-size:12px;font-family:var(--mono);"></div>
      <div style="display:flex;justify-content:flex-end;gap:10px;">
        <button onclick="closeLoadModal()">Cancel</button>
        <button class="btn-primary" id="lm-submit" onclick="submitLoadModal()">Load Model Now</button>
      </div>
    </div>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    const MODEL_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#06b6d4', '#ec4899', '#eab308'];
    let stateModels = { models: [], memory: {}, loading: { active: false } };
    let stateCatalog = { models: [], mmproj_files: [], draft_files: [] };
    let selectedFamily = 'ALL';
    let previewModel = null;

    function showToast(msg, isError = false) {
      const t = document.getElementById('toast');
      t.textContent = msg;
      t.style.borderColor = isError ? 'var(--red)' : '#475569';
      t.style.display = 'block';
      clearTimeout(window._toastTimer);
      window._toastTimer = setTimeout(() => { t.style.display = 'none'; }, 4200);
    }

    function fmtGiB(mib) {
      if (!mib && mib !== 0) return '0.0 GiB';
      return (mib / 1024).toFixed(1) + ' GiB';
    }

    function fmtDuration(ms) {
      if (ms < 1000) return ms + 'ms';
      const s = Math.floor(ms / 1000);
      if (s < 60) return s + 's';
      const m = Math.floor(s / 60);
      if (m < 60) return m + 'm ' + (s % 60) + 's';
      const h = Math.floor(m / 60);
      return h + 'h ' + (m % 60) + 'm';
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/admin/models');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        stateModels = await res.json();
        document.getElementById('conn-pill').style.color = 'var(--green)';
        renderLoadingBanner();
        renderMemoryBar();
        renderLoadedModels();
        updatePlaygroundModelSelect();
      } catch (e) {
        document.getElementById('conn-pill').style.color = 'var(--red)';
      }
    }

    async function fetchCatalog() {
      try {
        const res = await fetch('/admin/catalog');
        if (!res.ok) return;
        stateCatalog = await res.json();
        renderFamilyPills();
        renderCatalog();
      } catch (e) {}
    }

    async function refreshAll(manual = false) {
      await Promise.all([fetchStatus(), fetchCatalog()]);
      if (manual) showToast('Dashboard state refreshed');
    }

    function renderLoadingBanner() {
      const b = document.getElementById('loading-banner');
      const ld = stateModels.loading;
      if (ld && ld.active) {
        b.style.display = 'flex';
        document.getElementById('loading-title').textContent = 'Loading model "' + ld.id + '" into GPU/UMA memory...';
        document.getElementById('loading-sub').textContent = ld.path;
        document.getElementById('loading-elapsed').textContent = (ld.elapsed_ms / 1000).toFixed(1) + 's';
      } else {
        b.style.display = 'none';
      }
    }

    function renderMemoryBar() {
      const mem = stateModels.memory || {};
      const totalMib = mem.host_total_mib || mem.device_total_mib || 126976;
      const availMib = mem.host_available_mib || mem.device_free_mib || 0;
      const rssMib = mem.process_rss_mib || 0;

      document.getElementById('mem-rss-badge').textContent = 'gufo RSS: ' + fmtGiB(rssMib);
      document.getElementById('mem-numbers').textContent =
        'Available: ' + fmtGiB(availMib) + ' / Total Unified RAM: ' + fmtGiB(totalMib) +
        ' (Safety Reserve: ' + fmtGiB(mem.safety_margin_mib || 2048) + ')';

      const bar = document.getElementById('mem-bar');
      const legend = document.getElementById('mem-legend');
      bar.innerHTML = '';
      legend.innerHTML = '';

      let modelsSumMib = 0;
      const models = stateModels.models || [];
      models.forEach(m => {
        modelsSumMib += (m.file_size_mib || m.estimated_resident_mib || 2048);
      });

      const usedTotalMib = Math.max(0, totalMib - availMib);
      const scaleDenom = Math.max(totalMib, modelsSumMib + availMib);

      models.forEach((m, idx) => {
        const est = m.file_size_mib || m.estimated_resident_mib || 2048;
        const pct = Math.max(1.5, Math.min(100, (est / scaleDenom) * 100));
        const color = MODEL_COLORS[idx % MODEL_COLORS.length];

        const seg = document.createElement('div');
        seg.className = 'mem-seg';
        seg.style.width = pct + '%';
        seg.style.background = color;
        seg.title = m.id + ': ~' + fmtGiB(est);
        if (pct > 6) seg.textContent = m.id + ' (' + fmtGiB(est) + ')';
        bar.appendChild(seg);

        const item = document.createElement('div');
        item.className = 'mem-legend-item';
        item.innerHTML = `<span class="mem-swatch" style="background:${color}"></span><strong>${m.id}</strong> (~${fmtGiB(est)})`;
        legend.appendChild(item);
      });

      const otherSystemMib = Math.max(0, usedTotalMib - modelsSumMib);
      if (otherSystemMib > 512) {
        const pct = Math.max(1, (otherSystemMib / scaleDenom) * 100);
        const seg = document.createElement('div');
        seg.className = 'mem-seg';
        seg.style.width = pct + '%';
        seg.style.background = '#334155';
        seg.title = 'OS / KV Buffers / Other: ' + fmtGiB(otherSystemMib);
        if (pct > 7) seg.textContent = 'System & KV (' + fmtGiB(otherSystemMib) + ')';
        bar.appendChild(seg);

        const item = document.createElement('div');
        item.className = 'mem-legend-item';
        item.innerHTML = `<span class="mem-swatch" style="background:#334155"></span><span>System & KV Pools (${fmtGiB(otherSystemMib)})</span>`;
        legend.appendChild(item);
      }

      const gateBanner = document.getElementById('gate-banner');
      if (previewModel) {
        const needMib = previewModel.estMib;
        const pct = Math.min((availMib / scaleDenom) * 100, (needMib / scaleDenom) * 100);
        if (pct > 0.5) {
          const seg = document.createElement('div');
          seg.className = 'mem-seg mem-seg-preview';
          seg.style.width = pct + '%';
          seg.textContent = '+' + fmtGiB(needMib);
          bar.appendChild(seg);
        }
        const safeAvail = Math.max(0, availMib - (mem.safety_margin_mib || 2048));
        gateBanner.style.display = 'flex';
        if (needMib <= safeAvail) {
          gateBanner.style.borderColor = 'rgba(16,185,129,0.4)';
          gateBanner.style.background = 'rgba(16,185,129,0.1)';
          gateBanner.style.color = '#6ee7b7';
          gateBanner.innerHTML = `<span>✓ <strong>${previewModel.name}</strong> (~${fmtGiB(needMib)}) fits alongside currently loaded models without eviction!</span><span>Free after load: ~${fmtGiB(safeAvail - needMib)}</span>`;
        } else {
          gateBanner.style.borderColor = 'rgba(245,158,11,0.45)';
          gateBanner.style.background = 'rgba(245,158,11,0.12)';
          gateBanner.style.color = '#fcd34d';
          gateBanner.innerHTML = `<span>⚠ <strong>${previewModel.name}</strong> needs ~${fmtGiB(needMib)} (Safe free: ${fmtGiB(safeAvail)}). Loading will trigger LRU eviction of idle models if allow_evict=true.</span>`;
        }
      } else {
        gateBanner.style.display = 'none';
      }

      const freeItem = document.createElement('div');
      freeItem.className = 'mem-legend-item';
      freeItem.innerHTML = `<span class="mem-swatch" style="background:#090c10;border:1px solid #475569"></span><span>Free Available (${fmtGiB(availMib)})</span>`;
      legend.appendChild(freeItem);
    }

    function renderLoadedModels() {
      const container = document.getElementById('loaded-list');
      const models = stateModels.models || [];
      document.getElementById('loaded-count').textContent = models.length + ' loaded';

      if (models.length === 0) {
        container.innerHTML = `<div style="padding:24px;text-align:center;color:var(--muted);">No models currently loaded. Choose one from the Catalog to load.</div>`;
        return;
      }

      container.innerHTML = '';
      models.forEach((m, idx) => {
        const card = document.createElement('div');
        card.className = 'model-card' + (m.is_default ? ' is-default' : '');

        const badges = [];
        if (m.is_default) badges.push(`<span class="badge badge-primary">★ PRIMARY DEFAULT</span>`);
        else if (m.lru_rank > 0) badges.push(`<span class="badge">LRU #${m.lru_rank}</span>`);
        if (m.ready) badges.push(`<span class="badge badge-green">READY</span>`);
        if (m.supports_images) badges.push(`<span class="badge badge-purple">VISION</span>`);
        const specStr = typeof m.speculative === 'string' ? m.speculative : (m.speculative ? 'mtp' : 'off');
        if (specStr && specStr !== 'off') badges.push(`<span class="badge badge-cyan">SPEC: ${specStr.toUpperCase()}</span>`);
        badges.push(`<span class="badge">CTX: ${(m.context_length / 1024).toFixed(0)}K</span>`);
        badges.push(`<span class="badge">~${fmtGiB(m.file_size_mib || m.estimated_resident_mib)}</span>`);

        const slotsHtml = (m.sessions || []).map(s => {
          const cls = s.busy ? 'slot-pill busy' : 'slot-pill';
          const tok = s.cached_tokens ? ` · ${s.cached_tokens} tok` : '';
          return `<span class="${cls}">Slot #${s.index}: ${s.busy ? 'BUSY' : 'IDLE'}${tok}</span>`;
        }).join('');

        card.innerHTML = `
          <div class="model-top">
            <div>
              <div class="model-id">
                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${MODEL_COLORS[idx % MODEL_COLORS.length]}"></span>
                <span>${m.id}</span>
                ${badges.join('')}
              </div>
              <div class="model-path">${m.model_path}</div>
            </div>
            <div class="model-actions">
              <button class="btn-sm" onclick="selectInPlayground('${m.id}')">💬 Test</button>
              ${!m.is_default ? `<button class="btn-sm" onclick="setPrimaryModel('${m.id}')">★ Set Primary</button>` : ''}
              <button class="btn-sm btn-danger" onclick="ejectModel('${m.id}', ${m.busy})">⏏ Eject</button>
            </div>
          </div>
          <div class="slots-bar">
            <span style="font-size:11px;color:var(--muted);margin-right:4px;">Sessions (${m.active_sessions}/${m.total_sessions} busy):</span>
            ${slotsHtml}
            <span style="margin-left:auto;font-size:11px;color:var(--muted);">Idle: ${fmtDuration(m.idle_ms || 0)}</span>
          </div>
        `;
        container.appendChild(card);
      });
    }

    function renderFamilyPills() {
      const pills = document.getElementById('family-pills');
      const models = stateCatalog.models || [];
      const counts = { ALL: models.length };
      models.forEach(m => { counts[m.family] = (counts[m.family] || 0) + 1; });

      pills.innerHTML = '';
      Object.keys(counts).forEach(fam => {
        const el = document.createElement('span');
        el.className = 'family-pill' + (selectedFamily === fam ? ' active' : '');
        el.textContent = `${fam} (${counts[fam]})`;
        el.onclick = () => { selectedFamily = fam; renderFamilyPills(); renderCatalog(); };
        pills.appendChild(el);
      });
    }

    function renderCatalog() {
      const list = document.getElementById('catalog-list');
      const q = (document.getElementById('cat-search').value || '').toLowerCase().trim();
      const loadedPaths = new Set((stateModels.models || []).map(m => m.model_path));
      const loadedIds = new Set((stateModels.models || []).map(m => m.id));

      const filtered = (stateCatalog.models || []).filter(m => {
        if (selectedFamily !== 'ALL' && m.family !== selectedFamily) return false;
        if (!q) return true;
        return m.id_suggestion.toLowerCase().includes(q) ||
               m.relative_path.toLowerCase().includes(q) ||
               m.family.toLowerCase().includes(q);
      });

      document.getElementById('catalog-count').textContent = filtered.length + ' GGUFs';
      list.innerHTML = '';

      filtered.forEach(m => {
        const isLoaded = loadedPaths.has(m.path) || loadedIds.has(m.id_suggestion);
        const item = document.createElement('div');
        item.className = 'cat-item';
        item.onmouseenter = () => {
          previewModel = { name: m.id_suggestion, estMib: m.size_mib + 1536 };
          renderMemoryBar();
        };
        item.onmouseleave = () => {
          previewModel = null;
          renderMemoryBar();
        };

        const badges = [`<span class="badge">${fmtGiB(m.size_mib)}</span>`];
        if (m.shards > 1) badges.push(`<span class="badge">${m.shards} shards</span>`);
        if (m.suggested_mmproj) badges.push(`<span class="badge badge-purple" title="${m.suggested_mmproj}">+mmproj</span>`);
        if (m.suggested_draft) badges.push(`<span class="badge badge-cyan" title="${m.suggested_draft}">+MTP/Draft</span>`);
        if (isLoaded) badges.push(`<span class="badge badge-green">LOADED</span>`);

        item.innerHTML = `
          <div class="cat-info">
            <div class="cat-name">
              <span>${m.id_suggestion}</span>
              ${badges.join('')}
            </div>
            <div class="cat-rel" title="${m.path}">${m.relative_path}</div>
          </div>
          <div style="display:flex;gap:6px;flex-shrink:0;">
            ${isLoaded
              ? `<button class="btn-sm" onclick="selectInPlayground('${m.id_suggestion}')">Use</button>`
              : `<button class="btn-sm" onclick='openLoadModal(${JSON.stringify(m)})'>⚙ Configure</button>
                 <button class="btn-sm btn-primary" onclick='quickLoadModel(${JSON.stringify(m)})'>⚡ Load</button>`
            }
          </div>
        `;
        list.appendChild(item);
      });
    }

    function openLoadModal(m) {
      window._modalModel = m;
      document.getElementById('lm-id').value = m.id_suggestion;
      document.getElementById('lm-aliases').value = '';
      document.getElementById('lm-path').value = m.path;
      document.getElementById('lm-ctx').value = m.size_mib > 30000 ? '32768' : '32768';
      document.getElementById('lm-sessions').value = '1';
      document.getElementById('lm-kv').value = 'q8';
      document.getElementById('lm-spec').value = 'off';
      document.getElementById('lm-default').checked = false;
      document.getElementById('lm-evict').checked = true;
      document.getElementById('lm-yarn').checked = false;

      const draftSel = document.getElementById('lm-draft');
      draftSel.innerHTML = `<option value="">(None)</option>`;
      (stateCatalog.draft_files || []).forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p.replace('/mnt/nvme-data/gufo-models/', '');
        if (m.suggested_draft === p) opt.selected = true;
        draftSel.appendChild(opt);
      });

      const mmSel = document.getElementById('lm-mmproj');
      mmSel.innerHTML = `<option value="">(None - Text Only)</option>`;
      (stateCatalog.mmproj_files || []).forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p.replace('/mnt/nvme-data/gufo-models/', '');
        if (m.suggested_mmproj === p) opt.selected = true;
        mmSel.appendChild(opt);
      });

      updateModalEstimate();
      document.getElementById('load-modal').classList.add('open');
    }

    function closeLoadModal() {
      document.getElementById('load-modal').classList.remove('open');
    }

    function updateModalEstimate() {
      const m = window._modalModel || { size_mib: 4096 };
      const ctx = parseInt(document.getElementById('lm-ctx').value, 10) || 32768;
      const sess = parseInt(document.getElementById('lm-sessions').value, 10) || 1;
      const kvMib = Math.round((ctx / 32768) * sess * 768);
      const totalEst = m.size_mib + kvMib + 512;
      const avail = (stateModels.memory && stateModels.memory.host_available_mib) || 0;
      document.getElementById('lm-estimate').innerHTML =
        `Weights: <strong>${fmtGiB(m.size_mib)}</strong> + Est. KV/Buffers (${sess}×${(ctx/1024).toFixed(0)}K): <strong>~${fmtGiB(kvMib + 512)}</strong> → Total: <strong>~${fmtGiB(totalEst)}</strong> (Current Available: ${fmtGiB(avail)})`;
    }

    async function quickLoadModel(m) {
      const payload = {
        id: m.id_suggestion,
        path: m.path,
        context_length: 32768,
        sessions: 1,
        kv_quant: 'q8',
        speculative: 'off',
        mmproj_path: m.suggested_mmproj || '',
        allow_evict: true,
        set_default: false
      };
      await doLoadRequest(payload);
    }

    async function submitLoadModal() {
      const aliasesRaw = document.getElementById('lm-aliases').value.trim();
      const aliases = aliasesRaw ? aliasesRaw.split(',').map(s => s.trim()).filter(Boolean) : [];
      const payload = {
        id: document.getElementById('lm-id').value.trim(),
        aliases: aliases,
        path: document.getElementById('lm-path').value.trim(),
        context_length: parseInt(document.getElementById('lm-ctx').value, 10),
        sessions: parseInt(document.getElementById('lm-sessions').value, 10),
        kv_quant: document.getElementById('lm-kv').value,
        speculative: document.getElementById('lm-spec').value,
        draft_model_path: document.getElementById('lm-draft').value,
        mmproj_path: document.getElementById('lm-mmproj').value,
        enable_yarn: document.getElementById('lm-yarn').checked,
        allow_evict: document.getElementById('lm-evict').checked,
        set_default: document.getElementById('lm-default').checked
      };
      closeLoadModal();
      await doLoadRequest(payload);
    }

    async function doLoadRequest(payload) {
      showToast('Loading "' + payload.id + '" into :8741...');
      const pollTimer = setInterval(fetchStatus, 1000);
      try {
        const res = await fetch('/admin/load', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        clearInterval(pollTimer);
        await fetchStatus();
        renderCatalog();
        if (!res.ok || data.error) {
          showToast('Load failed: ' + (data.error ? data.error.message : res.statusText), true);
        } else {
          const ev = (data.evicted && data.evicted.length) ? ` (Evicted: ${data.evicted.join(', ')})` : '';
          showToast(`Loaded "${payload.id}" in ${(data.load_ms / 1000).toFixed(1)}s!${ev}`);
        }
      } catch (e) {
        clearInterval(pollTimer);
        showToast('Network error loading model: ' + e.message, true);
      }
    }

    async function ejectModel(id, isBusy) {
      const force = isBusy ? confirm(`Model "${id}" is currently BUSY. Force eject anyway?`) : false;
      if (isBusy && !force) return;
      try {
        const res = await fetch('/admin/eject', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id, force })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Eject failed: ' + (data.error ? data.error.message : res.statusText), true);
        } else {
          showToast(`Ejected "${id}" and freed GPU/UMA memory`);
          await fetchStatus();
          renderCatalog();
        }
      } catch (e) {
        showToast('Error ejecting model: ' + e.message, true);
      }
    }

    async function setPrimaryModel(id) {
      try {
        const res = await fetch('/admin/default', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Failed to set default: ' + (data.error ? data.error.message : res.statusText), true);
        } else {
          showToast(`"${id}" is now the Primary Default model on :8741`);
          await fetchStatus();
        }
      } catch (e) {
        showToast('Error: ' + e.message, true);
      }
    }

    function updatePlaygroundModelSelect() {
      const sel = document.getElementById('pg-model');
      const prev = sel.value;
      const models = stateModels.models || [];
      const defModel = models.find(m => m.is_default);

      sel.innerHTML = '';
      const defOpt = document.createElement('option');
      defOpt.value = '';
      defOpt.textContent = `(Default Route → ${defModel ? defModel.id : 'Primary'})`;
      sel.appendChild(defOpt);

      models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.id + (m.is_default ? ' (Primary)' : '');
        sel.appendChild(opt);
      });

      if ([...sel.options].some(o => o.value === prev)) {
        sel.value = prev;
      }
      updateCurlSnippet();
    }

    function selectInPlayground(id) {
      const sel = document.getElementById('pg-model');
      sel.value = id;
      updateCurlSnippet();
      document.getElementById('pg-input').focus();
    }

    function updateCurlSnippet() {
      const model = document.getElementById('pg-model').value || (stateModels.default_model || 'default');
      const ep = document.getElementById('pg-endpoint').value;
      const host = window.location.origin;
      document.getElementById('pg-curl').textContent =
`curl ${host}${ep} \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello!"}]}'`;
    }

    document.getElementById('pg-model').addEventListener('change', updateCurlSnippet);
    document.getElementById('pg-endpoint').addEventListener('change', updateCurlSnippet);

    function clearChat() {
      document.getElementById('pg-messages').innerHTML = '';
    }

    async function sendPlayground() {
      const input = document.getElementById('pg-input');
      const prompt = input.value.trim();
      if (!prompt) return;
      input.value = '';

      const model = document.getElementById('pg-model').value;
      const ep = document.getElementById('pg-endpoint').value;
      const maxTokens = parseInt(document.getElementById('pg-max-tokens').value, 10) || 256;
      const temperature = parseFloat(document.getElementById('pg-temp').value) || 0.6;

      const msgs = document.getElementById('pg-messages');
      const userDiv = document.createElement('div');
      userDiv.className = 'msg msg-user';
      userDiv.textContent = `[To: ${model || 'Default Route'}] ${prompt}`;
      msgs.appendChild(userDiv);

      const botDiv = document.createElement('div');
      botDiv.className = 'msg msg-assistant';
      botDiv.textContent = '...';
      msgs.appendChild(botDiv);
      msgs.scrollTop = msgs.scrollHeight;

      const t0 = performance.now();
      let tFirst = null;
      let outText = '';
      let outTokens = 0;

      document.getElementById('m-model').textContent = model || '(Default)';
      document.getElementById('m-ttft').textContent = '...';
      document.getElementById('m-tps').textContent = '...';
      document.getElementById('m-tokens').textContent = '...';

      try {
        const body = {
          messages: [{ role: 'user', content: prompt }],
          max_tokens: maxTokens,
          temperature: temperature,
          stream: true
        };
        if (model) body.model = model;

        const res = await fetch(ep, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });

        if (!res.ok) {
          const errText = await res.text();
          botDiv.textContent = 'Error HTTP ' + res.status + ': ' + errText;
          return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          const lines = buf.split('\n');
          buf = lines.pop();

          for (const line of lines) {
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (!payload || payload === '[DONE]') continue;
            try {
              const ev = JSON.parse(payload);
              let delta = '';
              if (ev.choices && ev.choices[0] && ev.choices[0].delta) {
                delta = ev.choices[0].delta.content || ev.choices[0].delta.reasoning_content || '';
              } else if (ev.type === 'content_block_delta' && ev.delta) {
                delta = ev.delta.text || ev.delta.thinking || '';
              }
              if (delta) {
                if (!tFirst) {
                  tFirst = performance.now();
                  document.getElementById('m-ttft').textContent = ((tFirst - t0) / 1000).toFixed(2) + 's';
                }
                outText += delta;
                outTokens++;
                botDiv.textContent = outText;
                msgs.scrollTop = msgs.scrollHeight;
              }
            } catch (_) {}
          }
        }

        const tEnd = performance.now();
        const decodeSec = tFirst ? Math.max(0.001, (tEnd - tFirst) / 1000) : ((tEnd - t0) / 1000);
        document.getElementById('m-tps').textContent = (outTokens / decodeSec).toFixed(1) + ' tok/s';
        document.getElementById('m-tokens').textContent = outTokens + ' chunks';
        fetchStatus();
      } catch (e) {
        botDiv.textContent = 'Request failed: ' + e.message;
      }
    }

    refreshAll();
    setInterval(fetchStatus, 3000);
  </script>
</body>
</html>
)GUFO_UI";

}  // namespace gufo::server
