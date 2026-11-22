#pragma once

#include <string_view>

namespace mbufo::server {

inline constexpr std::string_view kModelAdminUiHtml = R"MBUFO_UI(<!DOCTYPE html>
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
    .cap-pill {
      display: inline-flex;
      align-items: center;
      gap: 3px;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 10px;
      font-weight: 700;
      font-family: var(--mono);
      letter-spacing: 0.2px;
      flex-shrink: 0;
      border: 1px solid transparent;
    }
    .cap-chat   { background: rgba(59,130,246,0.18); color: #93c5fd; border-color: rgba(59,130,246,0.45); }
    .cap-decide { background: rgba(99,102,241,0.18); color: #a5b4fc; border-color: rgba(99,102,241,0.45); }
    .cap-embed  { background: rgba(16,185,129,0.18); color: #34d399; border-color: rgba(16,185,129,0.45); }
    .cap-rerank { background: rgba(245,158,11,0.18); color: #fbbf24; border-color: rgba(245,158,11,0.45); }
    .cap-vision { background: rgba(168,85,247,0.20); color: #d8b4fe; border-color: rgba(168,85,247,0.48); }
    .cap-audio  { background: rgba(236,72,153,0.20); color: #f472b6; border-color: rgba(236,72,153,0.48); }
    .cap-tools  { background: rgba(14,165,233,0.15); color: #7dd3fc; border-color: rgba(14,165,233,0.38); }
    .cap-spec   { background: rgba(6,182,212,0.18); color: #22d3ee; border-color: rgba(6,182,212,0.42); }

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
      padding: 0 12px;
      border-radius: 8px;
      font-size: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      height: 36px;
      min-height: 36px;
      max-height: 36px;
      white-space: nowrap;
      overflow: hidden;
      background: rgba(148, 163, 184, 0.06);
      border: 1px solid var(--border);
      color: var(--muted);
    }
    .gate-banner > span:first-child {
      flex: 1;
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .gate-banner > span:last-child {
      flex-shrink: 0;
      white-space: nowrap;
      font-family: var(--mono);
    }
    .mem-card-sticky {
      position: sticky;
      top: 63px;
      z-index: 15;
      box-shadow: 0 8px 24px rgba(0, 0, 0, 0.45);
    }

    /* Main Grid — strictly locked column widths (no content-driven width jumping) */
    .main-grid {
      display: grid;
      grid-template-columns: minmax(0, 46%) minmax(0, 54%);
      gap: 20px;
      align-items: stretch;
      width: 100%;
    }
    .main-grid > .card {
      min-width: 0;
      max-width: 100%;
      width: 100%;
      overflow: hidden;
    }
    @media (max-width: 1120px) {
      .main-grid { grid-template-columns: minmax(0, 1fr); }
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
    .filter-panel {
      background: #0c1017;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 12px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .filter-row {
      display: flex;
      align-items: flex-start;
      gap: 10px;
    }
    .filter-row + .filter-row {
      padding-top: 9px;
      border-top: 1px dashed rgba(255, 255, 255, 0.08);
    }
    .filter-dim-label {
      font-size: 10px;
      font-weight: 700;
      font-family: var(--mono);
      text-transform: uppercase;
      letter-spacing: 0.6px;
      color: #64748b;
      min-width: 78px;
      padding-top: 5px;
      flex-shrink: 0;
      user-select: none;
    }
    /* Dimension 1: Capability & Modality Segmented Tabs */
    .cap-tabs {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      flex: 1;
    }
    .cap-tab {
      padding: 5px 11px;
      border-radius: 7px;
      font-size: 12px;
      font-weight: 600;
      cursor: pointer;
      background: #151c28;
      border: 1px solid #263248;
      color: #cbd5e1;
      display: inline-flex;
      align-items: center;
      gap: 5px;
      transition: all 0.15s ease;
    }
    .cap-tab:hover {
      border-color: #475569;
      background: #1e293b;
      color: #fff;
    }
    .cap-tab .cap-cnt {
      font-family: var(--mono);
      font-size: 10px;
      padding: 1px 5px;
      border-radius: 4px;
      background: rgba(0, 0, 0, 0.35);
      color: #94a3b8;
    }
    .cap-tab.active[data-cap="ALL"]       { background: rgba(59,130,246,0.22); border-color: #3b82f6; color: #93c5fd; box-shadow: 0 0 12px rgba(59,130,246,0.2); }
    .cap-tab.active[data-cap="CHAT"]      { background: rgba(59,130,246,0.25); border-color: #3b82f6; color: #bfdbfe; box-shadow: 0 0 12px rgba(59,130,246,0.22); }
    .cap-tab.active[data-cap="DECIDE"]    { background: rgba(99,102,241,0.25); border-color: #6366f1; color: #c7d2fe; box-shadow: 0 0 12px rgba(99,102,241,0.22); }
    .cap-tab.active[data-cap="EMBEDDING"] { background: rgba(16,185,129,0.22); border-color: #10b981; color: #6ee7b7; box-shadow: 0 0 12px rgba(16,185,129,0.2); }
    .cap-tab.active[data-cap="RERANKER"]  { background: rgba(245,158,11,0.22); border-color: #f59e0b; color: #fde68a; box-shadow: 0 0 12px rgba(245,158,11,0.2); }
    .cap-tab.active[data-cap="VISION"]    { background: rgba(168,85,247,0.25); border-color: #a855f7; color: #e9d5ff; box-shadow: 0 0 12px rgba(168,85,247,0.22); }
    .cap-tab.active[data-cap="AUDIO"]     { background: rgba(236,72,153,0.25); border-color: #ec4899; color: #fbcfe8; box-shadow: 0 0 12px rgba(236,72,153,0.22); }
    .cap-tab.active[data-cap="TOOLS"]     { background: rgba(14,165,233,0.22); border-color: #0ea5e9; color: #bae6fd; box-shadow: 0 0 12px rgba(14,165,233,0.2); }
    /* Device (NPU/GPU) switch pills — same row language as capability pills. */
    .cap-tab.active[data-dev="ALL"] { background: rgba(59,130,246,0.22); border-color: #3b82f6; color: #93c5fd; box-shadow: 0 0 12px rgba(59,130,246,0.2); }
    .cap-tab.active[data-dev="GPU"] { background: rgba(59,130,246,0.25); border-color: #3b82f6; color: #bfdbfe; box-shadow: 0 0 12px rgba(59,130,246,0.25); }
    .cap-tab.active[data-dev="NPU"] { background: rgba(245,158,11,0.25); border-color: #f59e0b; color: #fde68a; box-shadow: 0 0 12px rgba(245,158,11,0.25); }
    .cap-tab.active .cap-cnt {
      background: rgba(255, 255, 255, 0.16);
      color: #fff;
    }

    /* Dimension 2: Model Family / Directory Compact Monospace Chips */
    .family-pills {
      display: flex;
      gap: 5px;
      flex-wrap: wrap;
      flex: 1;
    }
    .family-pill {
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-family: var(--mono);
      cursor: pointer;
      background: transparent;
      border: 1px solid #222c3c;
      color: #64748b;
      transition: all 0.12s ease;
    }
    .family-pill:hover {
      border-color: #3b4b6b;
      color: #cbd5e1;
      background: rgba(255, 255, 255, 0.03);
    }
    .family-pill.active {
      background: #1e293b;
      border-color: #64748b;
      color: #f8fafc;
      font-weight: 600;
    }
    .catalog-list {
      display: flex;
      flex-direction: column;
      gap: 8px;
      height: 540px;
      min-height: 540px;
      max-height: 540px;
      overflow-y: scroll;
      scrollbar-gutter: stable;
      padding-right: 4px;
      width: 100%;
      min-width: 0;
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
      width: 100%;
      min-width: 0;
      flex-shrink: 0;
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
      flex-wrap: nowrap;
      overflow: hidden;
      white-space: nowrap;
    }
    .cat-name > span:first-child {
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      min-width: 0;
    }
    .cat-name > .badge {
      flex-shrink: 0;
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
      <span id="version-pill" class="badge" style="font-family:var(--mono);cursor:pointer;padding:5px 9px;" onclick="checkServerUpdate(true)" title="Click to check GitHub (dardeaw/MBufo) for updates">checking version...</span>
      <span id="conn-pill" class="status-pill"><span class="status-dot"></span> Online (:8741)</span>
      <button class="btn-primary" id="btn-download-models" onclick="openDownloadModal()">⬇ Download Models</button>
      <button onclick="refreshAll(true)">↻ Refresh</button>
    </div>
  </header>

  <div class="container">
    <!-- GitHub Self-Update Banner -->
    <div id="update-banner" style="display:none;background:rgba(59,130,246,0.12);border:1px solid rgba(59,130,246,0.45);border-radius:10px;padding:12px 16px;flex-direction:column;gap:8px;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
          <span id="update-phase-badge" class="badge badge-amber">UPDATE AVAILABLE</span>
          <strong id="update-title" style="font-size:13px;">New commit available on GitHub (dardeaw/MBufo)</strong>
          <span id="update-detail" style="font-size:12px;color:var(--muted);font-family:var(--mono);"></span>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <button class="btn-sm" onclick="checkServerUpdate(true)">↻ Re-check</button>
          <button class="btn-sm btn-primary" id="btn-apply-update" onclick="applyServerUpdate()">🚀 Update & Restart Server</button>
        </div>
      </div>
      <pre id="update-log-tail" style="display:none;margin:0;background:#090c10;border:1px solid var(--border);border-radius:6px;padding:8px 10px;font-family:var(--mono);font-size:11px;color:#cbd5e1;max-height:130px;overflow-y:auto;white-space:pre-wrap;"></pre>
    </div>

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
    <div class="card mem-card-sticky">
      <div class="card-header" style="margin-bottom:8px;">
        <div class="card-title">
          <span>Unified Memory & VRAM Gate (Strix Halo UMA / HIP)</span>
          <span id="mem-rss-badge" class="badge">mbufo RSS: -- GiB</span>
        </div>
        <div id="mem-numbers" style="font-family:var(--mono);font-size:12px;color:var(--muted);">
          Loading memory telemetry...
        </div>
      </div>
      <div class="mem-bar-outer" id="mem-bar"></div>
      <div class="mem-legend" id="mem-legend"></div>
      <div class="gate-banner" id="gate-banner">
        <span>Hover over any model in the Catalog to preview memory fit & LRU eviction impact</span>
        <span>Safe Free Pool: -- GiB</span>
      </div>
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
          <span id="catalog-roots" style="font-size:11px;font-family:var(--mono);color:var(--muted);"></span>
        </div>
        <div class="catalog-controls">
          <input type="text" id="cat-search" placeholder="Search model name, quant, capability (e.g. bge, rerank, vision, 27B)..." oninput="renderCatalog()" />
        </div>
        <div class="filter-panel">
          <div class="filter-row">
            <div class="filter-dim-label">CAPABILITY</div>
            <div class="cap-tabs" id="cap-pills"></div>
          </div>
          <div class="filter-row">
            <div class="filter-dim-label">DEVICE</div>
            <div class="cap-tabs" id="dev-pills"></div>
          </div>

        </div>
        <div class="catalog-list" id="catalog-list"></div>
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
          <select id="lm-kv" style="width:100%;" title="q4/k8v4/k4v8 use packed 4-bit key/value grids and apply server-wide (process-global)">
            <option value="q4" selected>q4 K4V4 (Recommended)</option>
            <option value="q8">q8</option>
            <option value="k8v4">k8v4 (K8V4)</option>
            <option value="k4v8">k4v8 (K4V8)</option>
            <option value="fp16">fp16</option>
          </select>
        </div>
        <div>
          <label class="field-label">Compute Device</label>
          <select id="lm-device" style="width:100%;" onchange="onModalDeviceChange()" title="Dual-capable models default to NPU; switch between NPU and GPU here.">
            <option value="npu">&#x26A1; NPU (XDNA2)</option>
            <option value="gpu" selected>&#x1F3AE; GPU (HIP)</option>
          </select>
        </div>
        <div id="lm-npu-quant-wrap" style="display:none;">
          <label class="field-label">NPU Quantization</label>
          <select id="lm-npu-quant" style="width:100%;">
            <option value="q8" selected>Q8 (bfp16ebs8 - High Accuracy)</option>
            <option value="q4">Q4 (q4nx - Low Memory)</option>
            <option value="bf16">BF16 (Native)</option>
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

  <!-- Download Models Modal (Option B: Local Mapping + Server-Side HF Repo Browser) -->
  <div class="modal-backdrop" id="download-modal">
    <div class="modal" style="width:min(940px,96vw);max-height:88vh;display:flex;flex-direction:column;gap:12px;">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
        <div>
          <h3 style="font-size:16px;">Server-Side HuggingFace Model Downloader</h3>
          <div id="dl-source-note" style="font-size:11px;color:var(--muted);margin-top:2px;font-family:var(--mono);">Local Cache: /admin/model-mapping</div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
          <button class="btn-sm" onclick="syncModelMapping()">↻ Sync Mapping from GH</button>
          <button class="btn-sm" onclick="closeDownloadModal()">✕</button>
        </div>
      </div>

      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;background:#090c10;border:1px solid var(--border);padding:8px 10px;border-radius:8px;">
        <span style="font-size:12px;color:var(--muted);">Target Root:</span>
        <code id="dl-models-root" style="font-size:12px;color:#93c5fd;">/mnt/nvme-data/mbufo-models</code>
        <span id="dl-disk-free" class="badge badge-green" style="margin-left:auto;">Free: --</span>
        <input type="password" id="dl-hf-token" placeholder="Optional HF_TOKEN (for gated repos)" style="width:230px;padding:4px 8px;font-size:12px;" />
      </div>

      <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
        <input type="text" id="dl-custom-repo" placeholder="Custom HF Repo (e.g. unsloth/Qwen3.8-27B-GGUF or org/model)" style="flex:1;min-width:240px;font-family:var(--mono);font-size:12px;" onkeydown="if(event.key==='Enter') browseCustomRepo()" />
        <button class="btn-primary" onclick="browseCustomRepo()">📂 Browse Repo Files</button>
      </div>

      <div id="dl-jobs-box" style="display:none;flex-direction:column;gap:6px;background:#0b1220;border:1px solid rgba(59,130,246,0.35);border-radius:8px;padding:8px 10px;max-height:170px;overflow-y:auto;"></div>

      <!-- Mode 1: Cached Model Mapping List -->
      <div id="dl-catalog-view" style="display:flex;flex-direction:column;gap:8px;min-height:0;flex:1;">
        <div style="display:flex;gap:8px;flex-wrap:wrap;align-items:center;">
          <input type="text" id="dl-search" placeholder="Filter models by name, task, arch, quant..." style="flex:1;min-width:200px;" oninput="renderDownloadList()" />
          <select id="dl-task" onchange="renderDownloadList()">
            <option value="ALL">All Tasks</option>
            <option value="chat">Chat LLM</option>
            <option value="embedding">Embedding</option>
            <option value="reranking">Reranking</option>
            <option value="asr">ASR (Speech-to-Text)</option>
            <option value="tts">TTS (Text-to-Speech)</option>
            <option value="image_generation">Image Generation</option>
            <option value="video_generation">Video Generation</option>
            <option value="decide">Decide (D3)</option>
          </select>
        </div>
        <div id="dl-list" style="overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding-right:4px;max-height:46vh;"></div>
      </div>

      <!-- Mode 2: Option B HF Repo File & Quantization Picker -->
      <div id="dl-browser-view" style="display:none;flex-direction:column;gap:8px;min-height:0;flex:1;">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;background:#0d1117;border:1px solid var(--border);padding:8px 10px;border-radius:8px;">
          <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">
            <button class="btn-sm" onclick="backToModelList()">← Back to Models</button>
            <strong id="dl-browse-repo" style="font-family:var(--mono);font-size:13px;color:#93c5fd;"></strong>
          </div>
          <div style="display:flex;align-items:center;gap:6px;">
            <span style="font-size:12px;color:var(--muted);">Subdir:</span>
            <input type="text" id="dl-browse-subdir" style="width:190px;padding:4px 8px;font-family:var(--mono);font-size:12px;" />
          </div>
        </div>
        <div id="dl-browse-files" style="overflow-y:auto;display:flex;flex-direction:column;gap:6px;padding-right:4px;max-height:40vh;"></div>
        <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;border-top:1px solid var(--border);padding-top:8px;">
          <span id="dl-browse-selected-info" style="font-size:12px;font-family:var(--mono);color:var(--muted);">Select quantizations / files above</span>
          <button class="btn-primary" id="btn-start-dl" onclick="startSelectedDownload()">⬇ Download Selected to Server</button>
        </div>
      </div>

      <div style="display:flex;justify-content:flex-end;">
        <button onclick="closeDownloadModal()">Close</button>
      </div>
    </div>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    const MODEL_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#06b6d4', '#ec4899', '#eab308'];
    let stateModels = { models: [], memory: {}, loading: { active: false } };
    let stateCatalog = { models: [], mmproj_files: [], draft_files: [] };
    let stateMapping = { models: [] };
    let stateUpdate = null;
    let dlPollTimer = null;
    let updPollTimer = null;
    let lastCompletedDlIds = new Set();
    window._browseRepo = '';
    window._browseItems = [];
    let mappingSource = 'none';
    let selectedCap = 'ALL';
    let selectedDevice = 'ALL';
    let previewModel = null;
    window._catalogFiltered = [];

    function globToRegex(pat) {
      const esc = pat.replace(/[.+^${}()|[\]\\]/g, '\\$&').replace(/\*/g, '.*').replace(/\?/g, '.');
      return new RegExp('^' + esc + '$', 'i');
    }

    function findMappingForCatalog(m) {
      const list = (stateMapping && stateMapping.models) || [];
      const id = (m.id_suggestion || m.id || '').toLowerCase();
      const rel = (m.relative_path || m.path || '').toLowerCase();
      for (const entry of list) {
        if (entry.id && (id === entry.id.toLowerCase() || id.startsWith(entry.id.toLowerCase()))) return entry;
        const pats = (entry.match && entry.match.filename_patterns) || [];
        for (const pat of pats) {
          const rx = globToRegex(pat);
          if (rx.test(m.name || '') || rx.test(rel) || rx.test(id)) return entry;
        }
      }
      return null;
    }

    function mappingSupportsGpu(entry) {
      return !!(entry && (entry.supported_backends || []).some(b => b.startsWith('gpu')));
    }

    function mappingSupportsNpu(entry) {
      return !!(entry && (entry.supported_backends || []).some(b => b.startsWith('npu')));
    }

    function isDualCapable(m) {
      const entry = m._mapping || findMappingForCatalog(m);
      if (entry) return mappingSupportsGpu(entry) && mappingSupportsNpu(entry);
      return !!m.npu_capable && !m.sidecar;
    }

    function defaultDeviceFor(m) {
      const entry = m._mapping || findMappingForCatalog(m);
      if (m.sidecar || (m.backend || '').toLowerCase() === 'npu') return 'npu';
      if (entry) {
        const hasGpu = mappingSupportsGpu(entry);
        const hasNpu = mappingSupportsNpu(entry);
        if (hasGpu && hasNpu) return 'npu';  // dual-capable defaults to NPU
        if (hasNpu && !hasGpu) return 'npu';
      }
      if (m.npu_capable) return 'npu';
      return 'gpu';
    }

    async function fetchModelMapping() {
      try {
        const res = await fetch('/admin/model-mapping', { cache: 'no-store' });
        if (res.ok) {
          stateMapping = await res.json();
          mappingSource = 'Local Cache (/admin/model-mapping)';
          return true;
        }
      } catch (_) {}
      return false;
    }

    async function syncModelMapping() {
      showToast('Syncing model_mapping.json from GitHub...');
      try {
        const res = await fetch('/admin/model-mapping/sync', { method: 'POST' });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Sync failed: ' + (data.error ? data.error.message : res.statusText), true);
          return;
        }
        stateMapping = data;
        mappingSource = 'Synced from GitHub (cached locally)';
        document.getElementById('dl-source-note').textContent =
          `Local Cache: /admin/model-mapping (${mappingSource}, ${(stateMapping.models || []).length} models)`;
        renderDevicePills();
        renderCapPills();
        renderCatalog();
        renderDownloadList();
        showToast(`Synced ${(stateMapping.models || []).length} models from GitHub!`);
      } catch (e) {
        showToast('Sync error: ' + e.message, true);
      }
    }

    const GPU_COLOR = '#3b82f6';
    const NPU_COLOR = '#f59e0b';

    function backendOf(m) {
      const b = (m && m.backend ? String(m.backend) : 'gpu').toLowerCase();
      return b === 'npu' ? 'NPU' : 'GPU';
    }

    function renderBackendBadge(m) {
      const be = backendOf(m);
      if (be === 'NPU') return `<span class="badge badge-amber" title="Runs on NPU (XDNA2 / flm sidecar)">⚡ NPU</span>`;
      return `<span class="badge badge-primary" title="Runs on GPU (HIP in-process)">🎮 GPU</span>`;
    }

    function supportsDevice(m, dev) {
      const entry = m._mapping || findMappingForCatalog(m);
      if (dev === 'NPU') {
        if (backendOf(m) === 'NPU' || m.npu_capable) return true;
        return mappingSupportsNpu(entry);
      }
      if (dev === 'GPU') {
        if (backendOf(m) === 'GPU') return true;
        return mappingSupportsGpu(entry);
      }
      return true;
    }

    function renderDevicePills() {
      const devContainer = document.getElementById('dev-pills');
      if (!devContainer) return;
      const models = stateCatalog.models || [];
      const counts = { ALL: models.length, GPU: 0, NPU: 0 };
      models.forEach(m => {
        if (supportsDevice(m, 'GPU')) counts.GPU += 1;
        if (supportsDevice(m, 'NPU')) counts.NPU += 1;
      });
      const defs = [
        { key: 'ALL', label: 'All Devices' },
        { key: 'GPU', label: '🎮 GPU' },
        { key: 'NPU', label: '⚡ NPU' }
      ];
      devContainer.innerHTML = '';
      defs.forEach(dd => {
        if (dd.key !== 'ALL' && !counts[dd.key]) return;
        const el = document.createElement('span');
        el.className = 'cap-tab' + (selectedDevice === dd.key ? ' active' : '');
        el.setAttribute('data-dev', dd.key);
        el.innerHTML = `<span>${dd.label}</span><span class="cap-cnt">${counts[dd.key]}</span>`;
        el.onclick = () => {
          selectedDevice = dd.key;
          renderDevicePills();
          renderCapPills();
          renderCatalog();
        };
        devContainer.appendChild(el);
      });
    }

    function renderCapabilityBadges(caps) {
      const list = Array.isArray(caps) ? caps : [];
      return list.map(c => {
        if (c === 'CHAT')      return `<span class="cap-pill cap-chat" title="Chat / Reasoning LLM (/v1/chat/completions)">💬 CHAT</span>`;
        if (c === 'DECIDE')    return `<span class="cap-pill cap-decide" title="D3 Fast Decision Engine (/v1/systemone)">⚡ DECIDE</span>`;
        if (c === 'EMBEDDING') return `<span class="cap-pill cap-embed" title="Vector Embeddings (/v1/embeddings)">🧬 EMBEDDING</span>`;
        if (c === 'RERANKER')  return `<span class="cap-pill cap-rerank" title="Cross-Encoder / Relevance Reranker (/v1/reranking)">🎯 RERANKER</span>`;
        if (c === 'VISION')    return `<span class="cap-pill cap-vision" title="Multimodal Vision (Image Input)">👁 VISION</span>`;
        if (c === 'IMAGE')     return `<span class="cap-pill cap-vision" title="Image Generation">🎨 IMAGE</span>`;
        if (c === 'AUDIO' || c === 'SPEECH') return `<span class="cap-pill cap-audio" title="Multimodal Audio">🎧 AUDIO</span>`;
        if (c === 'TOOLS')     return `<span class="cap-pill cap-tools" title="Function & Tool Calling">🛠 TOOLS</span>`;
        if (c === 'MTP' || c === 'DFLASH2' || c === 'DSPARK')
          return `<span class="cap-pill cap-spec" title="Speculative Decoding (${c})">⚡ ${c}</span>`;
        return `<span class="cap-pill">${c}</span>`;
      }).join('');
    }

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

    let isServerOnline = true;

    function setConnStatus(mode, text) {
      const pill = document.getElementById('conn-pill');
      if (mode === 'online') {
        pill.style.color = 'var(--green)';
        pill.style.background = 'rgba(16, 185, 129, 0.12)';
        pill.style.borderColor = 'rgba(16, 185, 129, 0.3)';
      } else if (mode === 'starting') {
        pill.style.color = 'var(--amber)';
        pill.style.background = 'rgba(245, 158, 11, 0.14)';
        pill.style.borderColor = 'rgba(245, 158, 11, 0.38)';
      } else {
        pill.style.color = 'var(--red)';
        pill.style.background = 'rgba(239, 68, 68, 0.15)';
        pill.style.borderColor = 'rgba(239, 68, 68, 0.42)';
      }
      pill.innerHTML = `<span class="status-dot"></span> ${text}`;
    }

    async function fetchStatus() {
      try {
        const res = await fetch('/admin/models', { cache: 'no-store' });
        if (res.status === 503) {
          isServerOnline = false;
          setConnStatus('starting', 'Starting (:8741)');
          const b = document.getElementById('loading-banner');
          b.style.display = 'flex';
          document.getElementById('loading-title').textContent = 'Starting mbufo service & loading primary model into GPU/UMA...';
          document.getElementById('loading-sub').textContent = 'Admin API will activate automatically once startup model finishes loading';
          document.getElementById('loading-elapsed').textContent = 'BOOTING';
          return false;
        }
        if (!res.ok) throw new Error('HTTP ' + res.status);
        stateModels = await res.json();
        isServerOnline = true;
        setConnStatus('online', 'Online (:8741)');
        renderLoadingBanner();
        renderMemoryBar();
        renderLoadedModels();
        return true;
      } catch (e) {
        isServerOnline = false;
        setConnStatus('offline', 'Offline (:8741 Stopped)');
        stateModels = { models: [], memory: {}, loading: { active: false } };
        const b = document.getElementById('loading-banner');
        b.style.display = 'flex';
        document.getElementById('loading-title').textContent = 'mbufo.service (:8741) is currently stopped or unreachable';
        document.getElementById('loading-sub').textContent = 'Another benchmark/process may have stopped mbufo.service. Dashboard will reconnect automatically when :8741 restarts.';
        document.getElementById('loading-elapsed').textContent = 'OFFLINE';
        document.getElementById('mem-rss-badge').textContent = 'mbufo RSS: OFFLINE';
        document.getElementById('mem-numbers').textContent = 'Service :8741 unreachable — cached telemetry cleared';
        document.getElementById('mem-bar').innerHTML = '';
        document.getElementById('mem-legend').innerHTML = '';
        document.getElementById('loaded-count').textContent = '0 loaded (offline)';
        document.getElementById('loaded-list').innerHTML =
          `<div style="padding:24px;text-align:center;color:var(--red);border:1px dashed rgba(239,68,68,0.4);border-radius:8px;">mbufo.service (:8741) is offline. Stale model cards cleared.</div>`;
        return false;
      }
    }

    async function fetchCatalog() {
      try {
        const res = await fetch('/admin/catalog', { cache: 'no-store' });
        if (!res.ok) return false;
        stateCatalog = await res.json();
        renderDevicePills();
        renderCapPills();
        renderCatalog();
        const rootsEl = document.getElementById('catalog-roots');
        if (rootsEl) {
          const roots = stateCatalog.scan_roots || [];
          rootsEl.textContent = roots.length ? roots.join(' · ') : '(default scan roots)';
          rootsEl.title = 'Directories scanned for the catalog (server $MBUFO_MODELS_DIR plus loaded-model ancestors)';
        }
        return true;
      } catch (e) {
        return false;
      }
    }

    async function refreshAll(manual = false) {
      const mappingPromise = fetchModelMapping().then(ok => {
        if (ok && stateCatalog && stateCatalog.models) {
          renderDevicePills();
          renderCapPills();
          renderCatalog();
        }
        return ok;
      });
      const [okStatus] = await Promise.all([
        fetchStatus(),
        fetchCatalog(),
        fetchDownloads(false),
        checkServerUpdate(manual)
      ]);
      if (manual) {
        await mappingPromise;
        if (okStatus) showToast(`Refreshed (mapping: ${mappingSource})`);
        else showToast('Cannot reach :8741 — mbufo.service is currently stopped', true);
      }
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

      document.getElementById('mem-rss-badge').textContent = 'mbufo RSS: ' + fmtGiB(rssMib);
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
        const be = backendOf(m);
        const color = be === 'NPU' ? NPU_COLOR : GPU_COLOR;

        const seg = document.createElement('div');
        seg.className = 'mem-seg';
        seg.style.width = pct + '%';
        seg.style.background = color;
        seg.title = be + ' · ' + m.id + ': ~' + fmtGiB(est);
        if (pct > 6) seg.textContent = be + ' · ' + m.id + ' (' + fmtGiB(est) + ')';
        bar.appendChild(seg);

        const item = document.createElement('div');
        item.className = 'mem-legend-item';
        item.innerHTML = `<span class="mem-swatch" style="background:${color}"></span><strong>${be} · ${m.id}</strong> (~${fmtGiB(est)})`;
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
      const safeAvail = Math.max(0, availMib - (mem.safety_margin_mib || 2048));
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
        if (needMib <= safeAvail) {
          const msg = `✓ ${previewModel.name} (~${fmtGiB(needMib)}) fits alongside currently loaded models without eviction!`;
          gateBanner.title = msg;
          gateBanner.style.borderColor = 'rgba(16,185,129,0.4)';
          gateBanner.style.background = 'rgba(16,185,129,0.1)';
          gateBanner.style.color = '#6ee7b7';
          gateBanner.innerHTML = `<span>✓ <strong>${previewModel.name}</strong> (~${fmtGiB(needMib)}) fits alongside currently loaded models without eviction!</span><span>Free after load: ~${fmtGiB(safeAvail - needMib)}</span>`;
        } else {
          const msg = `⚠ ${previewModel.name} needs ~${fmtGiB(needMib)} (Safe free: ${fmtGiB(safeAvail)}). Loading will trigger LRU eviction of idle models if allow_evict=true.`;
          gateBanner.title = msg;
          gateBanner.style.borderColor = 'rgba(245,158,11,0.45)';
          gateBanner.style.background = 'rgba(245,158,11,0.12)';
          gateBanner.style.color = '#fcd34d';
          gateBanner.innerHTML = `<span>⚠ <strong>${previewModel.name}</strong> needs ~${fmtGiB(needMib)} (Safe free: ${fmtGiB(safeAvail)}). Loading will trigger LRU eviction of idle models if allow_evict=true.</span><span>Shortfall: ~${fmtGiB(needMib - safeAvail)}</span>`;
        }
      } else {
        gateBanner.title = '';
        gateBanner.style.borderColor = 'var(--border)';
        gateBanner.style.background = 'rgba(148, 163, 184, 0.06)';
        gateBanner.style.color = 'var(--muted)';
        gateBanner.innerHTML = `<span>Hover over any model in the Catalog to preview memory fit & LRU eviction impact</span><span>Safe Free Pool: ${fmtGiB(safeAvail)}</span>`;
      }

      const freeItem = document.createElement('div');
      freeItem.className = 'mem-legend-item';
      freeItem.innerHTML = `<span class="mem-swatch" style="background:#090c10;border:1px solid #475569"></span><span>Free Available (${fmtGiB(availMib)})</span>`;
      legend.appendChild(freeItem);
    }

    function renderLoadedModels() {
      const container = document.getElementById('loaded-list');
      const allModels = stateModels.models || [];
      const models = allModels;
      const nGpu = allModels.filter(m => backendOf(m) === 'GPU').length;
      const nNpu = allModels.filter(m => backendOf(m) === 'NPU').length;
      document.getElementById('loaded-count').textContent =
        allModels.length + ' loaded (🎮 ' + nGpu + ' GPU · ⚡ ' + nNpu + ' NPU)';

      if (models.length === 0) {
        container.innerHTML = `<div style="padding:24px;text-align:center;color:var(--muted);">No models currently loaded. Choose one from the Catalog to load.</div>`;
        return;
      }

      container.innerHTML = '';
      models.forEach((m, idx) => {
        const card = document.createElement('div');
        const be = backendOf(m);
        card.className = 'model-card' + (m.is_default ? ' is-default' : '');
        card.style.borderLeft = '3px solid ' + (be === 'NPU' ? NPU_COLOR : GPU_COLOR);

        const badges = [];
        badges.push(renderBackendBadge(m));
        if (m.sidecar) badges.push(`<span class="badge badge-green">RUNNING</span>`);
        if (m.is_default) badges.push(`<span class="badge badge-primary">★ PRIMARY DEFAULT</span>`);
        else if (m.lru_rank > 0 && !m.sidecar) badges.push(`<span class="badge">LRU #${m.lru_rank}</span>`);
        if (m.ready && !m.sidecar) badges.push(`<span class="badge badge-green">READY</span>`);
        badges.push(renderCapabilityBadges(m.capabilities || []));
        if (!m.sidecar) badges.push(`<span class="badge">CTX: ${(m.context_length / 1024).toFixed(0)}K</span>`);
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
                <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${be === 'NPU' ? NPU_COLOR : GPU_COLOR}"></span>
                <span>${m.id}</span>
                ${badges.join('')}
              </div>
              <div class="model-path">${m.model_path}</div>
            </div>
          <div class="model-actions">
            ${m.sidecar
              ? `<button class="btn-sm btn-danger" onclick="ejectModel('${m.id}', false)">⏏ Eject</button>`
              : `${!m.is_default ? `<button class="btn-sm" onclick="setPrimaryModel('${m.id}')">★ Set Primary</button>` : ''}
                  <button class="btn-sm btn-danger" onclick="ejectModel('${m.id}', ${m.busy})">⏏ Eject</button>`}
          </div>
        </div>
        ${m.sidecar
          ? `<div class="slots-bar">
               <span style="font-size:11px;color:var(--muted);">Sidecar process${m.pid > 0 ? ' · pid ' + m.pid : ''}${m.external ? ' (externally started, adopted)' : ''} · POST ${m.endpoint || ''}</span>
             </div>`
          : `<div class="slots-bar">
          <span style="font-size:11px;color:var(--muted);margin-right:4px;">Sessions (${m.active_sessions}/${m.total_sessions} busy):</span>
          ${slotsHtml}
          <span style="margin-left:auto;font-size:11px;color:var(--muted);">Idle: ${fmtDuration(m.idle_ms || 0)}</span>
        </div>`}
        `;
        container.appendChild(card);
      });
    }

    function renderCapPills() {
      const capContainer = document.getElementById('cap-pills');
      const models = stateCatalog.models || [];
      const capDefs = [
        { key: 'ALL',       label: 'All Capabilities', match: () => true },
        { key: 'CHAT',      label: '💬 Chat LLM',     match: m => (m.capabilities || []).includes('CHAT') },
        { key: 'DECIDE',    label: '⚡ Decide (D3)',  match: m => (m.capabilities || []).includes('DECIDE') },
        { key: 'EMBEDDING', label: '🧬 Embedding',     match: m => (m.capabilities || []).includes('EMBEDDING') },
        { key: 'RERANKER',  label: '🎯 Reranker',      match: m => (m.capabilities || []).includes('RERANKER') },
        { key: 'VISION',    label: '👁 Vision',        match: m => (m.capabilities || []).includes('VISION') },
        { key: 'IMAGE',     label: '🎨 Image',         match: m => (m.capabilities || []).includes('IMAGE') },
        { key: 'AUDIO',     label: '🎧 Audio',         match: m => (m.capabilities || []).includes('AUDIO') },
        { key: 'TOOLS',     label: '🛠 Tools',         match: m => (m.capabilities || []).includes('TOOLS') }
      ];
      if (capContainer) {
        capContainer.innerHTML = '';
        capDefs.forEach(cd => {
          const cnt = models.filter(cd.match).length;
          if (cd.key !== 'ALL' && cnt === 0) return;
          const el = document.createElement('span');
          el.className = 'cap-tab' + (selectedCap === cd.key ? ' active' : '');
          el.setAttribute('data-cap', cd.key);
          el.innerHTML = `<span>${cd.label}</span><span class="cap-cnt">${cnt}</span>`;
          el.onclick = () => {
            selectedCap = cd.key;
            renderCapPills();
            renderCatalog();
          };
          capContainer.appendChild(el);
        });
      }
    }

    function renderCatalog() {
      const list = document.getElementById('catalog-list');
      const q = (document.getElementById('cat-search').value || '').toLowerCase().trim();
      const loadedPaths = new Set((stateModels.models || []).map(m => m.model_path));
      const loadedIds = new Set((stateModels.models || []).map(m => m.id));

      const enriched = (stateCatalog.models || []).map(m => {
        const entry = findMappingForCatalog(m);
        return Object.assign({}, m, { _mapping: entry });
      });

      const filtered = enriched.filter(m => {
        if (selectedCap !== 'ALL' && !(m.capabilities || []).includes(selectedCap)) return false;
        if (selectedDevice !== 'ALL' && !supportsDevice(m, selectedDevice)) return false;
        if (!q) return true;
        const capsStr = (m.capabilities || []).join(' ').toLowerCase();
        return (m.id_suggestion || '').toLowerCase().includes(q) ||
               (m.relative_path || '').toLowerCase().includes(q) ||
               capsStr.includes(q);
      });

      window._catalogFiltered = filtered;
      document.getElementById('catalog-count').textContent = filtered.length + ' models';
      list.innerHTML = '';

      filtered.forEach((m, idx) => {
        const isLoaded = loadedPaths.has(m.path) || loadedIds.has(m.id_suggestion);
        const dual = isDualCapable(m);
        const item = document.createElement('div');
        item.className = 'cat-item';
        item.onmouseenter = () => {
          previewModel = { name: m.id_suggestion, estMib: (m.size_mib || 0) + 512 };
          renderMemoryBar();
        };
        item.onmouseleave = () => {
          previewModel = null;
          renderMemoryBar();
        };

        const badges = [];
        if (dual) {
          badges.push(`<span class="badge badge-amber" title="Default: NPU (switchable to GPU in Configure)">⚡ NPU (Default)</span>`);
          badges.push(`<span class="badge badge-primary" title="Also supports GPU HIP">🎮 GPU</span>`);
        } else {
          badges.push(renderBackendBadge(m));
        }
        badges.push(renderCapabilityBadges(m.capabilities || []));
        badges.push(`<span class="badge">${fmtGiB(m.size_mib)}</span>`);
        if (m.shards > 1) badges.push(`<span class="badge">${m.shards} shards</span>`);
        if (isLoaded) badges.push(`<span class="badge badge-green">LOADED</span>`);
        if (m.sidecar) {
          badges.push(`<span class="badge">sidecar :${m.default_port}</span>`);
          if (m.running) badges.push(`<span class="badge badge-green">RUNNING</span>`);
        }

        const isActionable = m.actionable !== false;
        const actionsHtml = !isActionable
          ? `<span class="badge" style="color:var(--muted);" title="Verified on C++ runtime; HTTP serving endpoint not wired yet">🧪 Runtime Only (No HTTP)</span>`
          : m.sidecar
            ? (m.running
              ? `<button class="btn-sm" onclick="openLoadModal(${idx})">⚙ Configure</button>
                 <button class="btn-sm btn-danger" onclick="ejectModel('${m.id_suggestion}', false)">⏏ Eject</button>`
              : `<button class="btn-sm" onclick="openLoadModal(${idx})">⚙ Configure</button>
                 <button class="btn-sm btn-primary" onclick="quickLoadModel(${idx})">⚡ Load</button>`)
            : (isLoaded
              ? ''
              : `<button class="btn-sm" onclick="openLoadModal(${idx})">⚙ Configure</button>
                 <button class="btn-sm btn-primary" onclick="quickLoadModel(${idx})">⚡ Load</button>`);

        item.innerHTML = `
          <div class="cat-info">
            <div class="cat-name">
              <span>${m.id_suggestion}</span>
              ${badges.join('')}
            </div>
            <div class="cat-rel" title="${m.path}">${m.relative_path}</div>
            ${(!m.actionable && m.note) ? `<div class="cat-rel" style="color:var(--muted);font-style:italic;">${m.note}</div>` : ''}
            ${m.sidecar ? `<div class="cat-rel" style="color:var(--muted);">POST ${m.endpoint} · ${backendOf(m)} sidecar</div>` : ''}
          </div>
          <div style="display:flex;gap:6px;flex-shrink:0;">
            ${actionsHtml}
          </div>
        `;
        list.appendChild(item);
      });
    }

    function onModalDeviceChange() {
      const dev = document.getElementById('lm-device').value;
      document.getElementById('lm-npu-quant-wrap').style.display = (dev === 'npu') ? 'block' : 'none';
    }

    function openLoadModal(idx) {
      const m = (window._catalogFiltered && window._catalogFiltered[idx]) || null;
      if (!m) return;
      window._modalModel = m;
      document.getElementById('lm-id').value = m.id_suggestion;
      document.getElementById('lm-aliases').value = '';
      document.getElementById('lm-path').value = m.path || '';
      const isEnc = (m.task === 'embedding' || m.task === 'reranker');
      document.getElementById('lm-ctx').value = isEnc ? '8192' : '32768';
      document.getElementById('lm-sessions').value = '1';
      document.getElementById('lm-kv').value = isEnc ? 'fp16' : 'q4';

      const dual = isDualCapable(m);
      const npuOk = !!(m.npu_capable || m.sidecar || (m._mapping && mappingSupportsNpu(m._mapping)) || backendOf(m) === 'NPU');
      const gpuOk = !m.sidecar && (!m._mapping || mappingSupportsGpu(m._mapping) || backendOf(m) === 'GPU');
      const devSel = document.getElementById('lm-device');
      devSel.value = defaultDeviceFor(m);
      devSel.disabled = !(dual || (npuOk && gpuOk));
      devSel.title = (dual || (npuOk && gpuOk))
        ? 'Dual-capable: defaults to NPU, switchable between NPU and GPU'
        : (npuOk ? 'NPU model' : 'GPU model');
      onModalDeviceChange();

      document.getElementById('lm-spec').value = 'off';
      document.getElementById('lm-default').checked = false;
      document.getElementById('lm-evict').checked = true;
      document.getElementById('lm-yarn').checked = false;

      const draftSel = document.getElementById('lm-draft');
      draftSel.innerHTML = `<option value="">(None)</option>`;
      (stateCatalog.draft_files || []).forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p.replace(/^.*\/mbufo-models\//, '');
        if (m.suggested_draft === p) opt.selected = true;
        draftSel.appendChild(opt);
      });

      const mmSel = document.getElementById('lm-mmproj');
      mmSel.innerHTML = `<option value="">(None - Text Only)</option>`;
      (stateCatalog.mmproj_files || []).forEach(p => {
        const opt = document.createElement('option');
        opt.value = p;
        opt.textContent = p.replace(/^.*\/mbufo-models\//, '');
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

    async function quickLoadModel(idx) {
      const m = (window._catalogFiltered && window._catalogFiltered[idx]) || null;
      if (!m) return;
      const isEnc = (m.task === 'embedding' || m.task === 'reranker');
      const dev = defaultDeviceFor(m);
      const payload = {
        id: m.id_suggestion,
        path: m.path,
        context_length: isEnc ? 0 : 32768,
        sessions: 1,
        kv_quant: isEnc ? 'fp16' : 'q4',
        device: dev,
        npu_quant: dev === 'npu' ? 'q8' : '',
        speculative: 'off',
        mmproj_path: (!isEnc && m.suggested_mmproj) ? m.suggested_mmproj : '',
        allow_evict: true,
        set_default: false
      };
      await doLoadRequest(payload);
    }

    async function submitLoadModal() {
      const aliasesRaw = document.getElementById('lm-aliases').value.trim();
      const aliases = aliasesRaw ? aliasesRaw.split(',').map(s => s.trim()).filter(Boolean) : [];
      const dev = document.getElementById('lm-device').value;
      const payload = {
        id: document.getElementById('lm-id').value.trim(),
        aliases: aliases,
        path: document.getElementById('lm-path').value.trim(),
        context_length: parseInt(document.getElementById('lm-ctx').value, 10),
        sessions: parseInt(document.getElementById('lm-sessions').value, 10),
        kv_quant: document.getElementById('lm-kv').value,
        device: dev,
        npu_quant: dev === 'npu' ? document.getElementById('lm-npu-quant').value : '',
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

    function fmtBytes(bytes) {
      if (!bytes || bytes <= 0) return '0 B';
      const gib = bytes / (1024 * 1024 * 1024);
      if (gib >= 1) return gib.toFixed(2) + ' GiB';
      const mib = bytes / (1024 * 1024);
      if (mib >= 1) return mib.toFixed(1) + ' MiB';
      return (bytes / 1024).toFixed(0) + ' KiB';
    }

    function escHtml(s) {
      return String(s || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    async function openDownloadModal() {
      document.getElementById('download-modal').classList.add('open');
      if (!stateMapping.models || !stateMapping.models.length) {
        await fetchModelMapping();
      }
      document.getElementById('dl-source-note').textContent =
        `Local Cache: /admin/model-mapping (${mappingSource}, ${(stateMapping.models || []).length} models)`;
      renderDownloadList();
      await fetchDownloads(true);
    }

    function closeDownloadModal() {
      document.getElementById('download-modal').classList.remove('open');
    }

    function backToModelList() {
      document.getElementById('dl-browser-view').style.display = 'none';
      document.getElementById('dl-catalog-view').style.display = 'flex';
    }

    function renderDownloadList() {
      const box = document.getElementById('dl-list');
      const q = (document.getElementById('dl-search').value || '').toLowerCase().trim();
      const task = document.getElementById('dl-task').value;
      const list = (stateMapping.models || []).filter(m => {
        if (task !== 'ALL' && m.task !== task) return false;
        if (!q) return true;
        const quants = JSON.stringify(m.quant_formats || {}).toLowerCase();
        return (m.id || '').toLowerCase().includes(q) ||
               (m.name || '').toLowerCase().includes(q) ||
               (m.engine || '').toLowerCase().includes(q) ||
               (m.task || '').toLowerCase().includes(q) ||
               quants.includes(q);
      });
      if (!list.length) {
        box.innerHTML = `<div style="padding:20px;text-align:center;color:var(--muted);">No matching models in local model_mapping.json</div>`;
        return;
      }
      box.innerHTML = list.map((m, i) => {
        const backs = (m.supported_backends || []).map(b => {
          const isNpu = b.startsWith('npu');
          return `<span class="badge ${isNpu ? 'badge-amber' : 'badge-primary'}">${escHtml(b)}</span>`;
        }).join('');
        const recQ = (m.quant_formats && m.quant_formats.recommended) ? m.quant_formats.recommended : (m.format || '');
        const hf = m.huggingface || {};
        const repos = (hf.repos && hf.repos.length) ? hf.repos : (hf.repo ? [hf.repo] : []);
        const defSubdir = m.default_subdir || m.id || '';
        const repoBtns = repos.length
          ? repos.map((rp, ridx) => `<button class="btn-sm ${ridx === 0 ? 'btn-primary' : ''}" onclick="browseHfRepo('${escHtml(rp)}', '${escHtml(defSubdir)}')" title="Browse files & quants on server">📂 ${escHtml(repos.length > 1 ? rp.split('/')[0] : 'Browse & Download')}</button>`).join('')
          : `<span class="badge" title="FLM sidecar managed automatically or enter custom HF repo above">FLM Sidecar / Custom Repo</span>`;
        return `
          <div style="background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:10px 12px;display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap;">
            <div style="min-width:0;flex:1;">
              <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-weight:600;font-size:13px;">
                <span>${escHtml(m.name || m.id)}</span>
                <span class="badge">${escHtml(m.task || '')}</span>
                <span class="badge">${escHtml(m.engine || '')}</span>
                ${recQ ? `<span class="badge badge-green">${escHtml(recQ)}</span>` : ''}
                ${backs}
              </div>
              <div style="font-size:11px;color:var(--muted);font-family:var(--mono);margin-top:3px;">
                id: ${escHtml(m.id)} · repo: ${escHtml(repos[0] || 'n/a')} · dir: ${escHtml(defSubdir)}
              </div>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;flex-shrink:0;">
              ${repoBtns}
            </div>
          </div>
        `;
      }).join('');
    }

    function browseCustomRepo() {
      const rp = (document.getElementById('dl-custom-repo').value || '').trim();
      if (!rp || !rp.includes('/')) {
        showToast('Enter a valid HuggingFace repo slug (org/model)', true);
        return;
      }
      const sub = rp.split('/')[1].toLowerCase().replace(/-gguf$/i, '');
      browseHfRepo(rp, sub);
    }

    async function browseHfRepo(repo, defaultSubdir) {
      document.getElementById('dl-catalog-view').style.display = 'none';
      document.getElementById('dl-browser-view').style.display = 'flex';
      document.getElementById('dl-browse-repo').textContent = repo;
      document.getElementById('dl-browse-subdir').value = defaultSubdir || repo.split('/')[1].toLowerCase();
      document.getElementById('dl-browse-files').innerHTML =
        `<div style="padding:24px;text-align:center;color:var(--muted);">Querying HuggingFace tree API for <code>${escHtml(repo)}</code>...</div>`;
      document.getElementById('dl-browse-selected-info').textContent = 'Loading files...';
      window._browseRepo = repo;
      window._browseItems = [];

      try {
        const hdrs = {};
        const tok = (document.getElementById('dl-hf-token').value || '').trim();
        if (tok) hdrs['X-HF-Token'] = tok;
        const res = await fetch('/admin/hf/files?repo=' + encodeURIComponent(repo) + '&subdir=' + encodeURIComponent(document.getElementById('dl-browse-subdir').value), {
          headers: hdrs,
          cache: 'no-store'
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          document.getElementById('dl-browse-files').innerHTML =
            `<div style="padding:18px;color:var(--red);background:rgba(239,68,68,0.1);border-radius:8px;">Failed: ${escHtml(data.error ? data.error.message : res.statusText)}</div>`;
          document.getElementById('dl-browse-selected-info').textContent = 'Error';
          return;
        }
        if (data.models_dir) document.getElementById('dl-models-root').textContent = data.models_dir;
        if (data.disk_free_bytes) document.getElementById('dl-disk-free').textContent = 'Free: ' + fmtBytes(data.disk_free_bytes);

        const files = data.files || [];
        window._browseItems = files;
        if (!files.length) {
          document.getElementById('dl-browse-files').innerHTML =
            `<div style="padding:18px;text-align:center;color:var(--muted);">No .gguf or .safetensors weight files found in this repo.</div>`;
          updateSelectedDownloadSummary();
          return;
        }

        document.getElementById('dl-browse-files').innerHTML = files.map((f, idx) => {
          const roleBadge = f.role === 'mmproj'
            ? `<span class="badge badge-purple">VISION mmproj</span>`
            : (f.role === 'draft' ? `<span class="badge badge-amber">DRAFT / MTP</span>`
            : (f.full_repo ? `<span class="badge badge-green">FULL MODEL</span>` : `<span class="badge">WEIGHTS</span>`));
          const quantBadge = `<span class="badge badge-primary">${escHtml(f.quant || 'GGUF')}</span>`;
          const shardInfo = f.shards > 1 ? `<span class="badge">${f.shards} shards</span>` : '';
          const diskBadge = f.on_disk ? `<span class="badge badge-green">✅ On Disk</span>` : '';
          return `
            <label style="display:flex;align-items:center;gap:10px;background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:8px 10px;cursor:pointer;">
              <input type="checkbox" class="dl-file-cb" data-idx="${idx}" onchange="updateSelectedDownloadSummary()" ${f.full_repo && !f.on_disk ? 'checked' : ''} />
              <div style="flex:1;min-width:0;">
                <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:12px;">
                  ${quantBadge}
                  ${roleBadge}
                  ${shardInfo}
                  ${diskBadge}
                </div>
                <div style="font-family:var(--mono);font-size:11px;color:#e2e8f0;margin-top:3px;word-break:break-all;">${escHtml(f.name)}</div>
              </div>
              <div style="font-family:var(--mono);font-size:12px;font-weight:700;color:#93c5fd;flex-shrink:0;">
                ${fmtBytes(f.size_bytes)}
              </div>
            </label>
          `;
        }).join('');
        updateSelectedDownloadSummary();
      } catch (e) {
        document.getElementById('dl-browse-files').innerHTML =
          `<div style="padding:18px;color:var(--red);">Network error: ${escHtml(e.message)}</div>`;
      }
    }

    function updateSelectedDownloadSummary() {
      const cbs = [...document.querySelectorAll('.dl-file-cb:checked')];
      let totalBytes = 0;
      let fileCount = 0;
      cbs.forEach(cb => {
        const it = window._browseItems[parseInt(cb.dataset.idx, 10)];
        if (it) {
          totalBytes += it.size_bytes || 0;
          fileCount += (it.files || []).length;
        }
      });
      document.getElementById('dl-browse-selected-info').textContent =
        cbs.length
          ? `Selected ${cbs.length} item(s) (${fileCount} file(s)) · Est. Size: ${fmtBytes(totalBytes)}`
          : 'Select quantizations / files above';
      document.getElementById('btn-start-dl').disabled = (cbs.length === 0);
    }

    async function startSelectedDownload() {
      const cbs = [...document.querySelectorAll('.dl-file-cb:checked')];
      if (!cbs.length) return;
      const files = [];
      let fullRepo = false;
      let totalBytes = 0;
      cbs.forEach(cb => {
        const it = window._browseItems[parseInt(cb.dataset.idx, 10)];
        if (!it) return;
        totalBytes += it.size_bytes || 0;
        if (it.full_repo) fullRepo = true;
        else (it.files || []).forEach(f => files.push(f));
      });
      const payload = {
        repo: window._browseRepo,
        target_subdir: document.getElementById('dl-browse-subdir').value.trim(),
        full_repo: fullRepo,
        files: files,
        total_bytes: totalBytes,
        hf_token: (document.getElementById('dl-hf-token').value || '').trim()
      };
      try {
        const res = await fetch('/admin/download', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Download rejected: ' + (data.error ? data.error.message : res.statusText), true);
          return;
        }
        showToast(`Started server download (${payload.repo} -> ${payload.target_subdir})`);
        await fetchDownloads(true);
      } catch (e) {
        showToast('Failed starting download: ' + e.message, true);
      }
    }

    async function fetchDownloads(ensurePoll = false) {
      try {
        const res = await fetch('/admin/downloads', { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        if (data.models_dir) {
          const rEl = document.getElementById('dl-models-root');
          if (rEl) rEl.textContent = data.models_dir;
        }
        if (data.disk_free_bytes) {
          const dfEl = document.getElementById('dl-disk-free');
          if (dfEl) dfEl.textContent = 'Free: ' + fmtBytes(data.disk_free_bytes);
        }
        const jobs = data.jobs || [];
        const runningCount = jobs.filter(j => j.status === 'running').length;
        const btn = document.getElementById('btn-download-models');
        if (btn) {
          btn.textContent = runningCount > 0 ? `⬇ Downloading (${runningCount})` : '⬇ Download Models';
        }
        const box = document.getElementById('dl-jobs-box');
        if (box) {
          if (!jobs.length) {
            box.style.display = 'none';
          } else {
            box.style.display = 'flex';
            box.innerHTML = jobs.slice(0, 5).map(j => {
              const exp = j.expected_bytes || 0;
              const got = j.downloaded_bytes || 0;
              const pct = exp > 0 ? Math.min(100, Math.round((got / exp) * 100)) : (j.status === 'completed' ? 100 : 0);
              const stCls = j.status === 'completed' ? 'badge-green'
                          : (j.status === 'running' ? 'badge-primary' : 'badge-red');
              return `
                <div style="font-size:12px;border-bottom:1px solid rgba(255,255,255,0.06);padding-bottom:6px;">
                  <div style="display:flex;justify-content:space-between;align-items:center;gap:8px;flex-wrap:wrap;">
                    <div>
                      <strong>${escHtml(j.repo)}</strong> → <code>${escHtml(j.target_subdir)}</code>
                      <span class="badge ${stCls}">${escHtml(j.status.toUpperCase())}</span>
                      <span style="font-family:var(--mono);color:var(--muted);">${fmtBytes(got)}${exp > 0 ? ' / ' + fmtBytes(exp) + ' (' + pct + '%)' : ''}</span>
                    </div>
                    ${j.status === 'running' ? `<button class="btn-sm btn-danger" onclick="cancelDownloadJob('${escHtml(j.id)}')">✕ Cancel</button>` : ''}
                  </div>
                  ${exp > 0 ? `<div style="height:5px;background:#1e293b;border-radius:3px;overflow:hidden;margin-top:4px;"><div style="height:100%;width:${pct}%;background:${j.status === 'completed' ? 'var(--green)' : 'var(--primary)'};"></div></div>` : ''}
                  ${j.log_tail ? `<div style="font-family:var(--mono);font-size:10px;color:var(--muted);margin-top:3px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">${escHtml(j.log_tail.split('\n').pop())}</div>` : ''}
                </div>
              `;
            }).join('');
          }
        }
        let newlyDone = false;
        jobs.forEach(j => {
          if (j.status === 'completed' && !lastCompletedDlIds.has(j.id)) {
            lastCompletedDlIds.add(j.id);
            newlyDone = true;
          }
        });
        if (newlyDone) {
          fetchCatalog();
          showToast('Server download completed — Catalog refreshed!');
        }
        if ((runningCount > 0 || ensurePoll) && runningCount > 0) {
          if (!dlPollTimer) dlPollTimer = setInterval(() => fetchDownloads(false), 2000);
        } else if (runningCount === 0 && dlPollTimer) {
          clearInterval(dlPollTimer);
          dlPollTimer = null;
        }
      } catch (_) {}
    }

    async function cancelDownloadJob(id) {
      try {
        await fetch('/admin/download/cancel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ id })
        });
        showToast('Cancelled download ' + id);
        await fetchDownloads(false);
      } catch (e) {
        showToast('Cancel failed: ' + e.message, true);
      }
    }

    // ---- Real-Time GitHub Version Check & Self-Update ----
    async function checkServerUpdate(manual = false) {
      try {
        const res = await fetch('/admin/update/check' + (manual ? '?force=1' : ''), { cache: 'no-store' });
        if (!res.ok) return;
        const data = await res.json();
        stateUpdate = data;
        const pill = document.getElementById('version-pill');
        if (pill) {
          const shortSha = data.running_short || 'unknown';
          const tag = data.running_tag || 'dev';
          if (data.update_available) {
            pill.className = 'badge badge-amber';
            pill.textContent = `${tag} · ${shortSha} (Update Ready!)`;
          } else {
            pill.className = 'badge badge-green';
            pill.textContent = `${tag} · ${shortSha} (Up to date)`;
          }
          pill.title = `Built: ${data.build_time_utc || ''} | GitHub main: ${data.remote_short || '?'}`;
        }
        renderUpdateBanner(data, data.status || {});
        if (manual) {
          if (data.check_error) showToast('Update check warning: ' + data.check_error, true);
          else if (data.update_available) showToast(`New GitHub commit ${data.remote_short}: ${data.remote_message}`);
          else showToast(`Server is up to date (${data.running_short})`);
        }
      } catch (_) {}
    }

    function renderUpdateBanner(checkData, st) {
      const banner = document.getElementById('update-banner');
      if (!banner) return;
      const phase = (st && st.phase) ? st.phase : 'idle';
      const active = !!(st && st.active);
      const show = active || phase === 'failed' || (checkData && checkData.update_available);
      banner.style.display = show ? 'flex' : 'none';
      if (!show) return;

      const badge = document.getElementById('update-phase-badge');
      const title = document.getElementById('update-title');
      const detail = document.getElementById('update-detail');
      const btn = document.getElementById('btn-apply-update');
      const logEl = document.getElementById('update-log-tail');

      if (active) {
        badge.className = 'badge badge-primary';
        badge.textContent = phase.toUpperCase();
        title.textContent = st.message || ('Updating server: ' + phase);
        detail.textContent = `target=${(st.target_sha || '').slice(0, 8)}`;
        btn.disabled = true;
        btn.textContent = 'Updating...';
      } else if (phase === 'failed') {
        badge.className = 'badge badge-red';
        badge.textContent = 'UPDATE FAILED';
        title.textContent = st.error || st.message || 'Self-update failed';
        detail.textContent = 'Running server remained online.';
        btn.disabled = false;
        btn.textContent = '🚀 Retry Update & Restart';
      } else if (checkData && checkData.update_available) {
        badge.className = 'badge badge-amber';
        badge.textContent = 'UPDATE AVAILABLE';
        title.textContent = `New commit on GitHub (${checkData.remote_short}): ${checkData.remote_message || ''}`;
        detail.textContent = `running ${checkData.running_short} → remote ${checkData.remote_short}`;
        btn.disabled = false;
        btn.textContent = '🚀 Update & Restart Server';
      }
      if (st && st.log_tail) {
        logEl.style.display = 'block';
        logEl.textContent = st.log_tail;
      } else {
        logEl.style.display = 'none';
      }
    }

    async function applyServerUpdate() {
      if (!confirm('Pull latest main from GitHub, rebuild mbufo, and restart :8741? Active inference requests will briefly reconnect (~30s).')) {
        return;
      }
      try {
        const res = await fetch('/admin/update/apply', { method: 'POST' });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Update failed to start: ' + (data.error ? data.error.message : res.statusText), true);
          return;
        }
        const beforeSha = data.running_sha;
        showToast('Self-update started (fetching & building)...');
        if (updPollTimer) clearInterval(updPollTimer);
        updPollTimer = setInterval(async () => {
          try {
            const sr = await fetch('/admin/update/status', { cache: 'no-store' });
            if (sr.ok) {
              const st = await sr.json();
              renderUpdateBanner(stateUpdate, st);
              if (st.phase === 'completed' || (st.running_sha && beforeSha && st.running_sha !== beforeSha)) {
                clearInterval(updPollTimer);
                updPollTimer = null;
                showToast('Server updated & restarted! Reloading dashboard...');
                setTimeout(() => window.location.reload(), 1000);
              } else if (st.phase === 'failed') {
                clearInterval(updPollTimer);
                updPollTimer = null;
                showToast('Self-update failed (previous binary preserved)', true);
              }
            }
          } catch (_) {
            // Server is restarting; keep polling until :8741 is back
          }
        }, 2000);
      } catch (e) {
        showToast('Error starting update: ' + e.message, true);
      }
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


    refreshAll();
    setInterval(fetchStatus, 3000);
  </script>
</body>
</html>
)MBUFO_UI";

}  // namespace mbufo::server
