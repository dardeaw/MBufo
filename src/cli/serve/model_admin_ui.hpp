#pragma once

#include <string_view>

namespace mbufo::server {

inline constexpr std::string_view kModelAdminUiHtml = R"MBUFO_UI(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MBufo Router & Multi-Model Control Plane (:8741)</title>
  <style>
    :root {
      color-scheme: dark;
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
    html {
      height: 100%;
      overflow: hidden;
    }
    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--font);
      font-size: 14px;
      line-height: 1.5;
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      padding: 10px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-shrink: 0;
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
      width: 100%;
      max-width: 1480px;
      margin: 0 auto;
      padding: 12px 24px 0;
      display: flex;
      flex-direction: column;
      gap: 12px;
      flex: 1 1 0%;
      min-height: 0;
      overflow: hidden;
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
      flex-shrink: 0;
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
      margin-top: 8px;
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
      margin-top: 8px;
      padding: 0 12px;
      border-radius: 8px;
      font-size: 12px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      height: 32px;
      min-height: 32px;
      max-height: 32px;
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
      flex-shrink: 0;
      padding: 12px 18px;
      box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    }

    /* Main Grid — viewport-docked 2-column layout (no outer double scrollbar) */
    .main-grid {
      display: grid;
      grid-template-columns: minmax(0, 46%) minmax(0, 54%);
      grid-template-rows: minmax(0, 1fr);
      gap: 16px;
      align-items: stretch;
      width: 100%;
      flex: 1 1 0%;
      min-height: 0;
    }
    .main-grid > .card {
      display: flex;
      flex-direction: column;
      min-width: 0;
      min-height: 0;
      max-width: 100%;
      width: 100%;
      height: 100%;
      overflow: hidden;
      padding: 14px 18px 0;
      border-bottom: none;
      border-bottom-left-radius: 0;
      border-bottom-right-radius: 0;
    }
    .main-grid > .card > .card-header {
      margin-bottom: 10px;
    }

    /* Active Models */
    .loaded-list {
      display: flex;
      flex-direction: column;
      gap: 12px;
      flex: 1 1 0%;
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      scrollbar-gutter: stable;
      padding-right: 4px;
      padding-bottom: 16px;
    }
    .model-card {
      flex-shrink: 0;
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
      flex-wrap: nowrap;
      white-space: nowrap;
      overflow: hidden;
      background: rgba(0,0,0,0.25);
      padding: 6px 10px;
      border-radius: 6px;
      border: 1px solid rgba(255,255,255,0.05);
    }
    .slot-pill {
      font-family: var(--mono);
      font-size: 11px;
      padding: 2px 7px;
      flex: 0 0 auto;
      min-width: 96px;
      text-align: center;
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
      margin-bottom: 8px;
      flex-wrap: wrap;
      flex-shrink: 0;
    }
    .catalog-controls input { flex: 1; min-width: 180px; }
    .filter-panel {
      background: #0c1017;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 12px;
      margin-bottom: 10px;
      display: flex;
      flex-direction: column;
      gap: 8px;
      flex-shrink: 0;
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
      flex: 1 1 0%;
      min-height: 0;
      overflow-y: auto;
      overflow-x: hidden;
      scrollbar-gutter: stable;
      padding-right: 4px;
      padding-bottom: 16px;
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
      max-height: 92vh;
      overflow-y: auto;
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

    /* Update & Loading Banners */
    #update-banner { flex-shrink: 0; }
    .loading-banner {
      background: rgba(59, 130, 246, 0.14);
      border: 1px solid rgba(59, 130, 246, 0.45);
      border-radius: 10px;
      padding: 12px 16px;
      display: none;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-shrink: 0;
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
    .settings-modal {
      max-width: 880px;
      width: min(880px, 96vw);
    }
    .tab-bar {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--border);
      padding-bottom: 10px;
    }
    .tab-btn {
      padding: 7px 12px;
      font-size: 12px;
      font-weight: 600;
      border-radius: 7px;
      border: 1px solid var(--border);
      background: #0d1117;
      color: var(--muted);
      cursor: pointer;
    }
    .tab-btn.active {
      background: rgba(59,130,246,0.18);
      border-color: rgba(59,130,246,0.55);
      color: #93c5fd;
    }
    .tab-pane { display: none; flex-direction: column; gap: 12px; }
    .tab-pane.active { display: flex; }
    .settings-section {
      background: #0d1117;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    body.auth-pending header,
    body.auth-pending .container {
      visibility: hidden;
    }
    .cache-table-wrap {
      max-height: 300px;
      overflow: auto;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #090c10;
    }
    .cache-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      font-family: var(--mono);
    }
    .cache-table th, .cache-table td {
      padding: 7px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.06);
      text-align: left;
      white-space: nowrap;
    }
    .cache-table th {
      position: sticky;
      top: 0;
      background: #111827;
      color: var(--muted);
      z-index: 2;
    }
    .auth-overlay {
      position: fixed;
      inset: 0;
      background: rgba(6, 9, 14, 0.94);
      backdrop-filter: blur(8px);
      display: none;
      align-items: center;
      justify-content: center;
      z-index: 200;
      padding: 16px;
    }
    .auth-overlay.open { display: flex; }
    .auth-card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      width: min(420px, 95vw);
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      box-shadow: 0 24px 60px rgba(0,0,0,0.75);
    }

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

    /* Subtle dark scrollbars for internal containers */
    .loaded-list, .catalog-list, .modal, #dl-list, #dl-browse-files, #dl-jobs-box, #update-log-tail {
      scrollbar-width: thin;
      scrollbar-color: #2d3a52 transparent;
    }
    .loaded-list::-webkit-scrollbar, .catalog-list::-webkit-scrollbar, .modal::-webkit-scrollbar,
    #dl-list::-webkit-scrollbar, #dl-browse-files::-webkit-scrollbar, #dl-jobs-box::-webkit-scrollbar, #update-log-tail::-webkit-scrollbar {
      width: 7px;
      height: 7px;
    }
    .loaded-list::-webkit-scrollbar-track, .catalog-list::-webkit-scrollbar-track, .modal::-webkit-scrollbar-track,
    #dl-list::-webkit-scrollbar-track, #dl-browse-files::-webkit-scrollbar-track, #dl-jobs-box::-webkit-scrollbar-track, #update-log-tail::-webkit-scrollbar-track {
      background: transparent;
    }
    .loaded-list::-webkit-scrollbar-thumb, .catalog-list::-webkit-scrollbar-thumb, .modal::-webkit-scrollbar-thumb,
    #dl-list::-webkit-scrollbar-thumb, #dl-browse-files::-webkit-scrollbar-thumb, #dl-jobs-box::-webkit-scrollbar-thumb, #update-log-tail::-webkit-scrollbar-thumb {
      background: #2d3a52;
      border-radius: 999px;
    }
    .loaded-list::-webkit-scrollbar-thumb:hover, .catalog-list::-webkit-scrollbar-thumb:hover, .modal::-webkit-scrollbar-thumb:hover,
    #dl-list::-webkit-scrollbar-thumb:hover, #dl-browse-files::-webkit-scrollbar-thumb:hover, #dl-jobs-box::-webkit-scrollbar-thumb:hover, #update-log-tail::-webkit-scrollbar-thumb:hover {
      background: #475569;
    }

    /* Responsive fallback for narrow or ultra-short viewports */
    @media (max-width: 1120px), (max-height: 460px) {
      html, body {
        height: auto;
        min-height: 100vh;
        overflow-y: auto;
      }
      header {
        position: sticky;
        top: 0;
      }
      .container {
        flex: none;
        overflow: visible;
        padding: 16px 20px 24px;
      }
      .main-grid {
        grid-template-columns: minmax(0, 1fr);
        grid-template-rows: auto;
        flex: none;
      }
      .main-grid > .card {
        height: auto;
        border-bottom: 1px solid var(--border);
        border-bottom-left-radius: 12px;
        border-bottom-right-radius: 12px;
        padding-bottom: 16px;
      }
      .loaded-list, .catalog-list {
        flex: none;
        max-height: 60vh;
        padding-bottom: 4px;
      }
    }
  </style>
</head>
<body class="auth-pending">
  <header>
    <div class="brand">
      <div class="brand-logo">G</div>
      <div>
        <h1>MBufo Multi-Model Router & Control Plane</h1>
        <div class="brand-sub">Unified Port :<span id="brand-port">8741</span> · OpenAI /v1/chat/completions · Anthropic /v1/messages · Dynamic Hot-Load & LRU Eviction</div>
      </div>
    </div>
    <div class="header-actions">
      <span id="version-pill" class="badge" style="font-family:var(--mono);cursor:pointer;padding:5px 9px;" onclick="checkServerUpdate(true)" title="Click to check GitHub (dardeaw/MBufo) release tags for updates">checking version...</span>
      <span id="conn-pill" class="status-pill"><span class="status-dot"></span> Online (:8741)</span>
      <button id="btn-settings" onclick="openSettingsModal()" title="Settings: Startup Auto-Load Cards, Router Port, Password Auth, Disk Cache">⚙ Settings</button>
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
          <strong id="update-title" style="font-size:13px;">New release tag available on GitHub (dardeaw/MBufo)</strong>
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
        <h3 id="lm-title" style="font-size:16px;">Configure & Hot-Load Model into :8741</h3>
        <button class="btn-sm" onclick="closeLoadModal()">✕</button>
      </div>
      <div class="form-grid">
        <div id="lm-id-wrap">
          <label class="field-label">Model ID (Router Key)</label>
          <input type="text" id="lm-id" style="width:100%;" />
        </div>
        <div id="lm-aliases-wrap">
          <label class="field-label">Aliases (comma-separated)</label>
          <input type="text" id="lm-aliases" placeholder="e.g. coder, fast" style="width:100%;" />
        </div>
        <div class="form-full" id="lm-path-wrap">
          <label class="field-label">GGUF Path</label>
          <input type="text" id="lm-path" style="width:100%;font-family:var(--mono);font-size:12px;" />
        </div>
        <div class="form-full" id="lm-sidecar-info" style="display:none;background:#090c10;border:1px solid var(--border);padding:8px 10px;border-radius:8px;font-size:12px;font-family:var(--mono);color:var(--muted);"></div>
        <div id="lm-ctx-wrap">
          <label class="field-label">Context Length (-c)</label>
          <select id="lm-ctx" style="width:100%;" onchange="onModalCtxChange()">
            <option value="8192">8,192 (8K - Low RAM)</option>
            <option value="16384">16,384 (16K)</option>
            <option value="32768" selected>32,768 (32K - Balanced)</option>
            <option value="65536">65,536 (64K)</option>
            <option value="131072">131,072 (128K)</option>
            <option value="262144">262,144 (256K - Full)</option>
          </select>
        </div>
        <div id="lm-sessions-wrap">
          <label class="field-label">Parallel Sessions (--sessions)</label>
          <select id="lm-sessions" style="width:100%;" onchange="updateModalEstimate()">
            <option value="1" selected>1 Session</option>
            <option value="2">2 Sessions</option>
            <option value="4">4 Sessions</option>
          </select>
        </div>
        <div id="lm-kv-wrap">
          <label class="field-label">KV Cache Quantization</label>
          <select id="lm-kv" style="width:100%;" title="q4/k8v4/k4v8 use packed 4-bit key/value grids and apply server-wide (process-global)">
            <option value="q4" selected>q4 K4V4 (Recommended)</option>
            <option value="q8">q8</option>
            <option value="k8v4">k8v4 (K8V4)</option>
            <option value="k4v8">k4v8 (K4V8)</option>
            <option value="fp16">fp16</option>
          </select>
        </div>
        <div id="lm-device-wrap">
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
        <div class="form-full" id="lm-npu-note" style="display:none;background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.35);color:#fbbf24;padding:8px 10px;border-radius:8px;font-size:12px;">
          &#x26A1; NPU (XDNA2) executes requests serially via NpuDeviceLock + AdmissionQueue (1 active execution at a time — no GPU Parallel Sessions or KV Cache Quantization).
        </div>
        <div id="lm-spec-wrap" style="display:none;">
          <label class="field-label">Speculative / MTP Mode</label>
          <select id="lm-spec" style="width:100%;" onchange="onModalSpecChange()">
            <option value="off" selected>off (Disabled)</option>
          </select>
        </div>
        <div class="form-full" id="lm-draft-wrap" style="display:none;">
          <label class="field-label">Draft / MTP Sidecar Path (Auto-Selected)</label>
          <select id="lm-draft" style="width:100%;font-family:var(--mono);font-size:12px;" onchange="updateModalEstimate()"></select>
        </div>
        <div class="form-full" id="lm-mmproj-wrap" style="display:none;">
          <label class="field-label">Vision mmproj Sidecar Path (Auto-Selected)</label>
          <select id="lm-mmproj" style="width:100%;font-family:var(--mono);font-size:12px;" onchange="updateModalEstimate()"></select>
        </div>
        <div class="form-full" id="lm-image-wrap" style="display:none;background:#0d1117;border:1px solid var(--border);border-radius:8px;padding:10px 12px;">
          <div style="font-size:12px;font-weight:600;color:#93c5fd;margin-bottom:8px;">🎨 Image Generation Sidecar Parameters (ROCm HIP / diffusers)</div>
          <div class="form-grid">
            <div>
              <label class="field-label">Precision (dtype)</label>
              <select id="lm-img-dtype" style="width:100%;" onchange="updateModalEstimate()">
                <option value="bf16" selected>bf16 (Recommended)</option>
                <option value="fp16">fp16</option>
              </select>
            </div>
            <div>
              <label class="field-label">Default Inference Steps</label>
              <input type="number" id="lm-img-steps" min="1" max="50" value="4" style="width:100%;" />
            </div>
            <div>
              <label class="field-label">Default Image Size</label>
              <select id="lm-img-size" style="width:100%;">
                <option value="512x512">512x512 (Fast)</option>
                <option value="768x768">768x768 (Balanced)</option>
                <option value="1024x1024" selected>1024x1024 (Full Quality)</option>
                <option value="1024x768">1024x768 (Landscape)</option>
                <option value="768x1024">768x1024 (Portrait)</option>
              </select>
            </div>
            <div>
              <label class="field-label">CPU Offload Strategy</label>
              <select id="lm-img-offload" style="width:100%;" onchange="updateModalEstimate()">
                <option value="none" selected>none (Keep full pipeline in UMA/GPU VRAM)</option>
                <option value="model">model (Offload idle components between stages)</option>
                <option value="sequential">sequential (Lowest VRAM / slowest)</option>
              </select>
            </div>
          </div>
        </div>
        <div class="form-full" id="lm-checks-wrap" style="display:flex;gap:18px;flex-wrap:wrap;padding-top:4px;">
          <label class="checkbox-row" id="lm-default-wrap">
            <input type="checkbox" id="lm-default" /> Set as Primary Default Model
          </label>
          <label class="checkbox-row" id="lm-evict-wrap">
            <input type="checkbox" id="lm-evict" checked /> Allow LRU Auto-Eviction if RAM Full
          </label>
          <label class="checkbox-row" id="lm-yarn-wrap" style="display:none;">
            <input type="checkbox" id="lm-yarn" onchange="onModalYarnChange()" /> Enable YaRN Scaling (&gt;256K)
          </label>
          <label class="checkbox-row" id="lm-disk-cache-wrap" style="display:none;">
            <input type="checkbox" id="lm-disk-cache" checked /> Enable Prompt KV Disk Cache
          </label>
          <label class="checkbox-row" id="lm-autoload-wrap">
            <input type="checkbox" id="lm-autoload" /> Auto-Load on Startup (persist across reboot)
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
          <div id="dl-source-note" style="font-size:11px;color:var(--muted);margin-top:2px;font-family:var(--mono);">Local model_mapping.json (bundled with server release)</div>
        </div>
        <div style="display:flex;gap:8px;align-items:center;">
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

  <!-- Settings Modal -->
  <div class="modal-backdrop" id="settings-modal">
    <div class="modal settings-modal">
      <div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">
        <div>
          <h3 style="font-size:16px;">Server Settings & Storage Management</h3>
          <div style="font-size:11px;color:var(--muted);margin-top:2px;">Persistent across reboots via server_config.json · Live Auth & Disk Cache Controls</div>
        </div>
        <button class="btn-sm" onclick="closeSettingsModal()">✕</button>
      </div>

      <div class="tab-bar">
        <button class="tab-btn active" id="set-btn-default" onclick="switchSettingsTab('default')">★ Default Model</button>
        <button class="tab-btn" id="set-btn-port" onclick="switchSettingsTab('port')">🔌 Router Port</button>
        <button class="tab-btn" id="set-btn-auth" onclick="switchSettingsTab('auth')">🔐 Passwords & Auth</button>
        <button class="tab-btn" id="set-btn-cache" onclick="switchSettingsTab('cache')">💾 Disk Cache</button>
      </div>

      <!-- Tab 1: Startup Auto-Load Model Cards -->
      <div class="tab-pane active" id="set-tab-default">
        <div class="settings-section">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap;">
            <div>
              <strong id="set-startup-count" style="font-size:13px;">Startup Auto-Load Models (0)</strong>
              <div style="font-size:11px;color:var(--muted);margin-top:2px;">Models configured to load automatically on server boot or restart. Pin models using "📌 Auto-Load" on loaded cards or check the option during load.</div>
            </div>
            <button class="btn-sm" onclick="confirmAndRestartServer()">↻ Restart Server</button>
          </div>
          <div id="set-startup-cards" style="display:flex;flex-direction:column;gap:8px;"></div>
        </div>
      </div>

      <!-- Tab 2: Router Port -->
      <div class="tab-pane" id="set-tab-port">
        <div class="settings-section">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:12px;font-family:var(--mono);">
            <div>Running Port: <strong id="set-port-running" style="color:var(--green);">:8741</strong></div>
            <div>Configured Boot Port: <strong id="set-port-configured" style="color:#60a5fa;">:8741</strong></div>
            <div style="grid-column:1 / -1;color:var(--muted);">Config File: <span id="set-port-config-path">--</span></div>
          </div>
          <div id="set-port-restart-note" style="display:none;background:rgba(245,158,11,0.12);border:1px solid rgba(245,158,11,0.35);color:#fbbf24;padding:8px 10px;border-radius:8px;font-size:12px;">
            Configured port differs from running port — click Restart Server to bind the new port.
          </div>
          <div style="display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap;">
            <div>
              <label class="field-label">New Router Port (1-65535)</label>
              <input type="number" id="set-port" min="1" max="65535" value="8741" style="width:160px;font-family:var(--mono);" />
            </div>
            <button onclick="savePortSetting(false)">💾 Save Port Only</button>
            <button class="btn-primary" onclick="savePortSetting(true)">🚀 Save Port & Restart Server</button>
          </div>
          <div id="set-restart-status" style="display:none;background:#090c10;border:1px solid var(--border);padding:10px 12px;border-radius:8px;font-size:12px;font-family:var(--mono);"></div>
        </div>
      </div>

      <!-- Tab 3: Passwords & Auth -->
      <div class="tab-pane" id="set-tab-auth">
        <div class="settings-section">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="font-size:13px;">Layer 1 · Web UI & Control Plane (/admin/*)</strong>
            <span id="set-web-pwd-badge" class="badge">Not Set</span>
          </div>
          <label class="checkbox-row">
            <input type="checkbox" id="set-web-auth-enabled" /> Require password login to access Web UI & /admin/* APIs
          </label>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <input type="password" id="set-web-password" placeholder="Enter new Web UI password (blank = keep existing)" style="flex:1;min-width:220px;" />
            <button class="btn-sm btn-danger" onclick="clearWebPassword()">Clear Password</button>
          </div>
        </div>

        <div class="settings-section">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <strong style="font-size:13px;">Layer 2 · Inference API Endpoints (/v1/*)</strong>
            <span id="set-api-key-badge" class="badge">Not Set</span>
          </div>
          <label class="checkbox-row">
            <input type="checkbox" id="set-api-auth-enabled" /> Require Authorization: Bearer &lt;key&gt; or x-api-key on /v1/*
          </label>
          <label class="checkbox-row">
            <input type="checkbox" id="set-api-share-web" onchange="syncAuthFieldState()" /> Share Web UI password as the API key
          </label>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <input type="text" id="set-api-key" placeholder="Enter or generate dedicated API key (blank = keep existing)" style="flex:1;min-width:220px;font-family:var(--mono);font-size:12px;" />
            <button class="btn-sm" onclick="generateRandomApiKey()">🎲 Generate</button>
            <button class="btn-sm" onclick="copyApiKeyInput()">📋 Copy</button>
            <button class="btn-sm btn-danger" onclick="clearApiKeySetting()">Clear Key</button>
          </div>
          <div id="set-cli-key-note" style="display:none;font-size:11px;color:#fbbf24;">Note: Server was launched with CLI --api-key (also accepted on /v1/*).</div>
        </div>

        <div style="display:flex;justify-content:flex-end;gap:8px;">
          <button class="btn-primary" onclick="saveAuthSettings()">💾 Save Security Settings (Immediate)</button>
        </div>
      </div>

      <!-- Tab 4: Disk Cache -->
      <div class="tab-pane" id="set-tab-cache">
        <div class="settings-section">
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:12px;font-family:var(--mono);">
            <div>Dir: <strong id="cache-summary-dir" style="color:#93c5fd;">--</strong></div>
            <div>Store Mode: <strong id="cache-summary-mode">--</strong></div>
            <div>Entries: <strong id="cache-summary-counts">--</strong></div>
            <div>Managed Usage: <strong id="cache-summary-usage">--</strong></div>
            <div style="grid-column:1 / -1;color:var(--muted);">Active Compatibility: <span id="cache-summary-compat">--</span></div>
          </div>
          <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap;">
            <input type="text" id="cache-search" placeholder="Filter by checkpoint ID, compat, purpose..." oninput="renderDiskCacheTable()" style="flex:1;min-width:180px;" />
            <label class="checkbox-row" style="font-size:12px;">
              <input type="checkbox" id="cache-select-all" onchange="toggleSelectAllCache()" /> Select Visible
            </label>
            <button class="btn-sm btn-danger" id="btn-cache-del-sel" onclick="deleteSelectedCache()">🗑 Delete Selected</button>
            <button class="btn-sm" id="btn-cache-del-incompat" onclick="deleteIncompatibleCache()">🧹 Clean Incompatible</button>
            <button class="btn-sm btn-danger" id="btn-cache-del-all" onclick="deleteAllCache()">🔥 Delete All</button>
            <button class="btn-sm" onclick="loadDiskCache(true)">↻ Refresh</button>
          </div>
          <div class="cache-table-wrap">
            <table class="cache-table">
              <thead>
                <tr>
                  <th></th>
                  <th>Checkpoint ID</th>
                  <th>Purpose / Compat</th>
                  <th>Tokens</th>
                  <th>Chunks</th>
                  <th>Private / Manifest / Refs</th>
                  <th>Last Modified</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody id="cache-tbody">
                <tr><td colspan="8" style="text-align:center;color:var(--muted);padding:16px;">Open Disk Cache tab to inspect checkpoints...</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>

      <div style="display:flex;justify-content:flex-end;">
        <button onclick="closeSettingsModal()">Close</button>
      </div>
    </div>
  </div>

  <!-- Web UI Auth Overlay -->
  <div class="auth-overlay" id="auth-overlay">
    <div class="auth-card">
      <div>
        <h3 style="font-size:18px;margin-bottom:4px;">🔐 MBufo Control Plane</h3>
        <div id="auth-lock-subtitle" style="font-size:12px;color:var(--muted);">Password authentication is enabled for /admin/*. Enter Web UI password to unlock.</div>
      </div>
      <input type="password" id="auth-login-password" placeholder="Web UI password..." style="width:100%;" onkeydown="if(event.key==='Enter')submitLogin()" />
      <div id="auth-login-error" style="display:none;font-size:12px;color:var(--red);"></div>
      <button class="btn-primary" id="auth-login-btn" style="width:100%;padding:9px;" onclick="submitLogin()">Unlock Dashboard</button>
    </div>
  </div>

  <div class="toast" id="toast"></div>

  <script>
    const MODEL_COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#06b6d4', '#ec4899', '#eab308'];
    let stateModels = { models: [], memory: {}, loading: { active: false } };
    let uiBootSha = null;
    let stateCatalog = { models: [], mmproj_files: [], draft_files: [] };
    let stateMapping = { models: [] };
    let stateUpdate = null;
    let stateSettings = null;
    let stateDiskCache = null;
    let authLocked = false;
    let restartingServer = false;
    let statusPollTimer = null;
    window._modalMode = 'load';
    window._reloadTargetId = '';
    window._settingsTab = 'default';
    window._selectedCacheIds = new Set();
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

    function livePort() {
      return (stateSettings && stateSettings.running_port) || window.location.port || '8741';
    }

    function updatePortDisplays() {
      const p = livePort();
      const bp = document.getElementById('brand-port');
      if (bp) bp.textContent = p;
    }

    function stopBackgroundPollers() {
      if (statusPollTimer) { clearInterval(statusPollTimer); statusPollTimer = null; }
      if (dlPollTimer) { clearInterval(dlPollTimer); dlPollTimer = null; }
    }

    function handleWebAuthRequired(msg) {
      if (restartingServer) return;
      document.body.classList.add('auth-pending');
      if (!authLocked) {
        authLocked = true;
        stopBackgroundPollers();
        const ov = document.getElementById('auth-overlay');
        if (ov) ov.classList.add('open');
        const sub = document.getElementById('auth-lock-subtitle');
        if (sub && msg) sub.textContent = msg;
        const errEl = document.getElementById('auth-login-error');
        if (errEl) errEl.style.display = 'none';
        const inp = document.getElementById('auth-login-password');
        if (inp) { inp.value = ''; setTimeout(() => inp.focus(), 50); }
      }
    }

    (function installAdminAuthFetchHook() {
      const rawFetch = window.fetch.bind(window);
      window.fetch = async function(input, init) {
        const opts = Object.assign({ credentials: 'same-origin' }, init || {});
        const res = await rawFetch(input, opts);
        const url = (typeof input === 'string') ? input : (input && input.url ? input.url : '');
        if (res.status === 401 && url.startsWith('/admin') && !url.startsWith('/admin/auth/')) {
          handleWebAuthRequired('Authentication required or session expired. Enter Web UI password to continue.');
        }
        return res;
      };
    })();

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
      if (m.sidecar) return backendOf(m) === 'NPU' ? 'npu' : 'gpu';
      if ((m.backend || '').toLowerCase() === 'npu') return 'npu';
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
      await fetchModelMapping();
      renderDownloadList();
      showToast(`Loaded local model_mapping.json (${(stateMapping.models || []).length} models)`);
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
      if (m.sidecar) return backendOf(m) === dev;
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
      if (authLocked || restartingServer) return false;
      try {
        const res = await fetch('/admin/models', { cache: 'no-store' });
        if (res.status === 401) return false;
        if (res.status === 503) {
          isServerOnline = false;
          setConnStatus('starting', `Starting (:${livePort()})`);
          const b = document.getElementById('loading-banner');
          b.style.display = 'flex';
          document.getElementById('loading-title').textContent = 'Starting mbufo service & loading primary model into GPU/UMA...';
          document.getElementById('loading-sub').textContent = 'Admin API will activate automatically once startup model finishes loading';
          document.getElementById('loading-elapsed').textContent = 'BOOTING';
          return false;
        }
        if (!res.ok) throw new Error('HTTP ' + res.status);
        stateModels = await res.json();
        if (stateModels.build_sha) {
          if (uiBootSha === null) {
            uiBootSha = stateModels.build_sha;
          } else if (stateModels.build_sha !== uiBootSha) {
            showToast('Server updated — reloading Admin UI...');
            setTimeout(() => window.location.reload(), 600);
            return true;
          }
        }
        isServerOnline = true;
        setConnStatus('online', `Online (:${livePort()})`);
        renderLoadingBanner();
        renderMemoryBar();
        renderLoadedModels();
        return true;
      } catch (e) {
        if (authLocked || restartingServer) return false;
        isServerOnline = false;
        setConnStatus('offline', `Offline (:${livePort()} Stopped)`);
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
        checkServerUpdate(manual),
        loadSettingsSummary()
      ]);
      if (manual && !authLocked) {
        await mappingPromise;
        if (okStatus) showToast(`Refreshed (mapping: ${mappingSource})`);
        else showToast(`Cannot reach :${livePort()} — mbufo.service is currently stopped`, true);
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

    function findStartupEntry(m) {
      const list = (stateSettings && stateSettings.startup_models) || [];
      if (!m) return null;
      const mp = m.model_path || m.path || '';
      const mid = m.id || m.id_suggestion || '';
      return list.find(x => (mid && x.id === mid) || (!m.sidecar && mp && x.path === mp)) || null;
    }

    function buildStartupRecordFromLoaded(loaded) {
      return {
        configured: true,
        primary: !!loaded.is_default,
        sidecar: !!loaded.sidecar,
        task: loaded.task || '',
        id: loaded.id,
        aliases: Array.isArray(loaded.aliases) ? loaded.aliases : [],
        path: loaded.model_path || loaded.path || '',
        device: (loaded.backend || '').toLowerCase() === 'npu' ? 'npu' : 'gpu',
        max_context: loaded.context_length || 0,
        session_count: loaded.total_sessions || 1,
        kv_cache_quant: loaded.kv_cache_quant || 'q8',
        npu_quant: loaded.npu_quant || '',
        speculative: loaded.speculative_mode || 'off',
        draft_model_path: loaded.draft_model_path || '',
        vision_model_path: loaded.vision_model_path || '',
        enable_yarn: !!loaded.enable_yarn,
        enable_disk_cache: loaded.enable_disk_cache !== false,
        image_dtype: loaded.image_dtype || '',
        image_steps: loaded.image_steps || 0,
        image_size: loaded.image_size || '',
        image_cpu_offload: loaded.image_cpu_offload || ''
      };
    }

    async function toggleLoadedAutoLoad(id, add) {
      const loaded = (stateModels.models || []).find(x => x.id === id);
      if (!loaded) {
        showToast('Loaded model "' + id + '" not found', true);
        return;
      }
      const existing = findStartupEntry(loaded);
      const body = add
        ? { upsert_startup_model: buildStartupRecordFromLoaded(loaded) }
        : { remove_startup_model: existing ? existing.id : loaded.id };
      try {
        const res = await fetch('/admin/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body)
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Failed to update startup auto-load: ' + (data.error ? data.error.message : res.statusText), true);
          return;
        }
        stateSettings = Object.assign({}, stateSettings || {}, data);
        renderLoadedModels();
        if (document.getElementById('settings-modal').classList.contains('open')) {
          renderStartupModelCards();
        }
        showToast(add ? `Added "${id}" to startup auto-load list` : `Removed "${id}" from startup auto-load list`);
      } catch (e) {
        showToast('Error: ' + e.message, true);
      }
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
        const isPinned = !!findStartupEntry(m);
        card.className = 'model-card' + (m.is_default ? ' is-default' : '');
        card.style.borderLeft = '3px solid ' + (be === 'NPU' ? NPU_COLOR : GPU_COLOR);

        const badges = [];
        badges.push(renderBackendBadge(m));
        if (m.sidecar) badges.push(`<span class="badge badge-green">RUNNING</span>`);
        if (m.is_default) badges.push(`<span class="badge badge-primary">★ PRIMARY DEFAULT</span>`);
        else if (m.lru_rank > 0 && !m.sidecar) badges.push(`<span class="badge">LRU #${m.lru_rank}</span>`);
        if (isPinned) badges.push(`<span class="badge badge-amber" title="Loads automatically on server boot or restart">📌 AUTO-LOAD</span>`);
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
            <button class="btn-sm ${isPinned ? 'badge-amber' : ''}" onclick="toggleLoadedAutoLoad(decodeURIComponent('${encodeURIComponent(m.id)}'), ${isPinned ? 'false' : 'true'})" title="${isPinned ? 'Remove from startup auto-load list' : 'Add to startup auto-load list'}">${isPinned ? '📌 Unpin' : '📌 Auto-Load'}</button>
            <button class="btn-sm" onclick="openReloadModal(decodeURIComponent('${encodeURIComponent(m.id)}'))" title="Reconfigure parameters & reload this model">⚙ Configure</button>
            ${m.sidecar
              ? `<button class="btn-sm btn-danger" onclick="ejectModel(decodeURIComponent('${encodeURIComponent(m.id)}'), false)">⏏ Eject</button>`
              : `${(!m.is_default && m.task !== 'embedding' && m.task !== 'reranker') ? `<button class="btn-sm" onclick="setPrimaryModel(decodeURIComponent('${encodeURIComponent(m.id)}'))">★ Set Primary</button>` : ''}
                  <button class="btn-sm btn-danger" onclick="ejectModel(decodeURIComponent('${encodeURIComponent(m.id)}'), ${m.busy})">⏏ Eject</button>`}
          </div>
        </div>
        ${m.sidecar
          ? `<div class="slots-bar">
               <span style="font-size:11px;color:var(--muted);">${m.task === 'image_generation'
                 ? `ROCm diffusers sidecar · dtype: ${escHtml(m.image_dtype || 'bf16')} · steps: ${m.image_steps || 4} · size: ${escHtml(m.image_size || '1024x1024')} · offload: ${escHtml(m.image_cpu_offload || 'none')}`
                 : 'Sidecar process'}${m.pid > 0 ? ' · pid ' + m.pid : ''}${m.external ? ' (externally started, adopted)' : ''} · POST ${escHtml(m.endpoint || '')}</span>
             </div>`
          : be === 'NPU'
            ? `<div class="slots-bar">
                 <span style="font-size:11px;color:#fbbf24;">⚡ XDNA2 NPU2 · Serialized queue (NpuScheduler)</span>
                 <span style="margin-left:auto;font-size:11px;color:var(--muted);">Idle: ${fmtDuration(m.idle_ms || 0)}</span>
               </div>`
            : `<div class="slots-bar">${slotsHtml}</div>`}
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
        if (m.sidecar) {
          badges.push(renderBackendBadge(m));
        } else if (dual) {
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
                 <button class="btn-sm btn-danger" onclick="ejectModel(decodeURIComponent('${encodeURIComponent(m.id_suggestion)}'), false)">⏏ Eject</button>`
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

    const SPEC_LABELS = {
      off: 'off (Disabled)',
      mtp: 'mtp (Multi-Token Prediction)',
      dflash2: 'dflash2 (Block Diffusion Draft)',
      dspark: 'dspark (Draft Spark)'
    };

    function populateCtxOptions(supportsYarn, defaultVal) {
      const ctxSel = document.getElementById('lm-ctx');
      const opts = [
        ['8192', '8,192 (8K - Low RAM)'],
        ['16384', '16,384 (16K)'],
        ['32768', '32,768 (32K - Balanced)'],
        ['65536', '65,536 (64K)'],
        ['131072', '131,072 (128K)'],
        ['262144', '262,144 (256K - Full)']
      ];
      if (supportsYarn) {
        opts.push(['524288', '524,288 (512K - YaRN)']);
        opts.push(['1048576', '1,048,576 (1M - YaRN)']);
      }
      ctxSel.innerHTML = '';
      opts.forEach(([v, label]) => {
        const o = document.createElement('option');
        o.value = v;
        o.textContent = label;
        ctxSel.appendChild(o);
      });
      ctxSel.value = String(defaultVal || 32768);
    }

    function onModalCtxChange() {
      const ctx = parseInt(document.getElementById('lm-ctx').value, 10) || 32768;
      if (ctx > 262144 && document.getElementById('lm-yarn-wrap').style.display !== 'none') {
        document.getElementById('lm-yarn').checked = true;
      }
      updateModalEstimate();
    }

    function onModalYarnChange() {
      const yarnChecked = document.getElementById('lm-yarn').checked;
      const ctxSel = document.getElementById('lm-ctx');
      if (!yarnChecked && (parseInt(ctxSel.value, 10) || 0) > 262144) {
        ctxSel.value = '262144';
      }
      updateModalEstimate();
    }

    function onModalSpecChange() {
      const m = window._modalModel || {};
      const specWrap = document.getElementById('lm-spec-wrap');
      const draftWrap = document.getElementById('lm-draft-wrap');
      const specSel = document.getElementById('lm-spec');
      const draftSel = document.getElementById('lm-draft');
      if (specWrap.style.display === 'none' || specSel.value === 'off') {
        draftWrap.style.display = 'none';
      } else {
        draftWrap.style.display = 'block';
        if (!draftSel.value) {
          const compatDrafts = Array.isArray(m.compatible_drafts) ? m.compatible_drafts : [];
          draftSel.value = m.suggested_draft || compatDrafts[0] || '';
        }
      }
      updateModalEstimate();
    }

    function syncModalVisibility(resetSelections) {
      const m = window._modalModel;
      if (!m) return;
      const dev = document.getElementById('lm-device').value;
      const isSidecar = !!m.sidecar;
      const isImage = (m.task === 'image_generation');
      const isEnc = (m.task === 'embedding' || m.task === 'reranker');
      const isChatTask = (!m.task || m.task === 'chat');
      const isGpuChat = (!isSidecar && dev === 'gpu' && isChatTask);

      const imgWrap = document.getElementById('lm-image-wrap');
      // Sidecar (ROCm HIP diffusers) params only: in-process NPU models
      // use the NPU quant selector below, not the sidecar torch dtype.
      const isNpuInProcess =
        !isSidecar && typeof backendOf === 'function' &&
        backendOf(m) === 'NPU';
      if (imgWrap)
        imgWrap.style.display = isImage && !isNpuInProcess ? 'block' : 'none';

      document.getElementById('lm-id').readOnly = isSidecar;
      document.getElementById('lm-aliases-wrap').style.display = isSidecar ? 'none' : 'block';
      document.getElementById('lm-path-wrap').style.display = isSidecar ? 'none' : 'block';

      const scInfo = document.getElementById('lm-sidecar-info');
      if (isSidecar) {
        scInfo.style.display = 'block';
        scInfo.textContent = `Sidecar Endpoint: POST ${m.endpoint || ''} · Port :${m.default_port || ''}${m.launch_cmd ? ' · ' + m.launch_cmd : ''}`;
      } else {
        scInfo.style.display = 'none';
      }

      document.getElementById('lm-ctx-wrap').style.display = (dev === 'gpu' && !isSidecar) ? 'block' : 'none';

      const canParallel = isGpuChat && (m.supports_parallel_sessions !== false);
      document.getElementById('lm-sessions-wrap').style.display = canParallel ? 'block' : 'none';
      if (!canParallel) document.getElementById('lm-sessions').value = '1';

      document.getElementById('lm-kv-wrap').style.display = isGpuChat ? 'block' : 'none';
      if (!isGpuChat) document.getElementById('lm-kv').value = 'fp16';

      const isEmbGemma = (m.id_suggestion || '').toLowerCase().includes('embeddinggemma');
      const npuInProcess =
        !isSidecar && typeof backendOf === 'function' &&
        backendOf(m) === 'NPU';
      document.getElementById('lm-npu-quant-wrap').style.display =
        (dev === 'npu' && !isSidecar && (isEmbGemma || npuInProcess)) ? 'block' : 'none';
      document.getElementById('lm-npu-note').style.display = (dev === 'npu') ? 'block' : 'none';

      // Speculative & Draft Path: auto-select default when supported; hide when unsupported.
      const compatDrafts = Array.isArray(m.compatible_drafts) ? m.compatible_drafts : [];
      const supportedSpecs = Array.isArray(m.supported_speculative) ? m.supported_speculative : [];
      const canSpec = isGpuChat && !!m.supports_speculative && supportedSpecs.length > 0 && compatDrafts.length > 0;
      const specWrap = document.getElementById('lm-spec-wrap');
      const draftWrap = document.getElementById('lm-draft-wrap');
      const specSel = document.getElementById('lm-spec');
      const draftSel = document.getElementById('lm-draft');
      if (!canSpec) {
        specWrap.style.display = 'none';
        draftWrap.style.display = 'none';
        specSel.innerHTML = '<option value="off" selected>off (Disabled)</option>';
        specSel.value = 'off';
        draftSel.innerHTML = '<option value="">(None)</option>';
        draftSel.value = '';
      } else {
        specWrap.style.display = 'block';
        if (resetSelections || specSel.options.length <= 1) {
          specSel.innerHTML = '<option value="off">off (Disabled)</option>';
          supportedSpecs.forEach(sp => {
            const opt = document.createElement('option');
            opt.value = sp;
            opt.textContent = SPEC_LABELS[sp] || sp;
            specSel.appendChild(opt);
          });
          draftSel.innerHTML = '<option value="">(None)</option>';
          compatDrafts.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p;
            opt.textContent = p.replace(/^.*\/mbufo-models\//, '');
            draftSel.appendChild(opt);
          });
          const defSpec = m.suggested_speculative || supportedSpecs[0] || 'off';
          specSel.value = defSpec;
          draftSel.value = (defSpec !== 'off') ? (m.suggested_draft || compatDrafts[0] || '') : '';
        }
        draftWrap.style.display = (specSel.value !== 'off') ? 'block' : 'none';
      }

      // Vision mmproj: auto-select default when supported; hide when unsupported.
      const compatMm = Array.isArray(m.compatible_mmprojs) ? m.compatible_mmprojs : [];
      const canMm = isGpuChat && !!m.supports_mmproj && compatMm.length > 0;
      const mmWrap = document.getElementById('lm-mmproj-wrap');
      const mmSel = document.getElementById('lm-mmproj');
      if (!canMm) {
        mmWrap.style.display = 'none';
        mmSel.innerHTML = '<option value="">(None - Text Only)</option>';
        mmSel.value = '';
      } else {
        mmWrap.style.display = 'block';
        if (resetSelections || mmSel.options.length <= 1) {
          mmSel.innerHTML = '<option value="">(None - Text Only)</option>';
          compatMm.forEach(p => {
            const opt = document.createElement('option');
            opt.value = p;
            opt.textContent = p.replace(/^.*\/mbufo-models\//, '');
            mmSel.appendChild(opt);
          });
          mmSel.value = m.suggested_mmproj || compatMm[0] || '';
        }
      }

      // Checkboxes: Primary Default (GPU Chat only), LRU Eviction (non-sidecar), YaRN (Qwen3.8-Flash-Next GPU only).
      const isStartupEdit = (window._modalMode === 'startup_edit');
      const canDefault = isGpuChat;
      const canEvict = !isSidecar && !isStartupEdit;
      const canYarn = isGpuChat && !!m.supports_yarn;
      const canAutoLoad = !isStartupEdit;
      document.getElementById('lm-default-wrap').style.display = canDefault ? 'flex' : 'none';
      if (!canDefault) document.getElementById('lm-default').checked = false;
      document.getElementById('lm-evict-wrap').style.display = canEvict ? 'flex' : 'none';
      document.getElementById('lm-yarn-wrap').style.display = canYarn ? 'flex' : 'none';
      if (!canYarn) document.getElementById('lm-yarn').checked = false;
      const diskWrap = document.getElementById('lm-disk-cache-wrap');
      if (diskWrap) diskWrap.style.display = isGpuChat ? 'flex' : 'none';
      const autoWrap = document.getElementById('lm-autoload-wrap');
      if (autoWrap) autoWrap.style.display = canAutoLoad ? 'flex' : 'none';
      document.getElementById('lm-checks-wrap').style.display = (canDefault || canEvict || canYarn || isGpuChat || canAutoLoad) ? 'flex' : 'none';

      updateModalEstimate();
    }

    function onModalDeviceChange() {
      syncModalVisibility(true);
    }

    function ensureSelectValue(selId, val, labelTransform) {
      const sel = document.getElementById(selId);
      if (!sel || val === undefined || val === null || val === '') return;
      const sval = String(val);
      const exists = Array.from(sel.options).some(o => o.value === sval);
      if (!exists) {
        const opt = document.createElement('option');
        opt.value = sval;
        opt.textContent = labelTransform ? labelTransform(sval) : sval;
        sel.appendChild(opt);
      }
      sel.value = sval;
    }

    function openLoadModal(idx) {
      const m = (window._catalogFiltered && window._catalogFiltered[idx]) || null;
      if (!m) return;
      window._modalMode = 'load';
      window._reloadTargetId = '';
      window._modalModel = m;
      document.getElementById('lm-title').textContent = `Configure & Hot-Load Model into :${livePort()}`;
      document.getElementById('lm-submit').textContent = 'Load Model Now';
      document.getElementById('lm-id').value = m.id_suggestion;
      document.getElementById('lm-aliases').value = '';
      document.getElementById('lm-path').value = m.path || '';
      const isEnc = (m.task === 'embedding' || m.task === 'reranker');
      populateCtxOptions(!!m.supports_yarn, isEnc ? 8192 : 32768);
      document.getElementById('lm-sessions').value = '1';
      document.getElementById('lm-kv').value = isEnc ? 'fp16' : 'q4';

      const devSel = document.getElementById('lm-device');
      if (m.sidecar) {
        const scDual = isDualCapable(m);
        devSel.value = defaultDeviceFor(m);
        devSel.disabled = !scDual;
        devSel.title = scDual
          ? 'Sidecar launch path runs on GPU (HIP); NPU runtime verified in model mapping but no NPU sidecar launcher is wired yet'
          : (backendOf(m) === 'NPU' ? 'NPU sidecar' : 'GPU sidecar');
      } else {
        const dual = isDualCapable(m);
        const npuOk = !!(m.npu_capable || (m._mapping && mappingSupportsNpu(m._mapping)) || backendOf(m) === 'NPU');
        const gpuOk = !m._mapping || mappingSupportsGpu(m._mapping) || backendOf(m) === 'GPU';
        devSel.value = defaultDeviceFor(m);
        devSel.disabled = !(dual || (npuOk && gpuOk));
        devSel.title = (dual || (npuOk && gpuOk))
          ? 'Dual-capable: defaults to NPU, switchable between NPU and GPU'
          : (npuOk ? 'NPU model' : 'GPU model');
      }

      document.getElementById('lm-img-dtype').value = m.image_dtype || 'bf16';
      document.getElementById('lm-img-steps').value = m.image_steps || (m.id_suggestion && m.id_suggestion.includes('z-image') ? 9 : 4);
      document.getElementById('lm-img-size').value = m.image_size || '1024x1024';
      document.getElementById('lm-img-offload').value = m.image_cpu_offload || 'none';

      document.getElementById('lm-default').checked = false;
      document.getElementById('lm-evict').checked = true;
      document.getElementById('lm-yarn').checked = false;
      document.getElementById('lm-disk-cache').checked = true;
      document.getElementById('lm-autoload').checked = !!findStartupEntry(m);

      syncModalVisibility(true);
      document.getElementById('load-modal').classList.add('open');
    }

    function openReloadModal(id) {
      const loaded = (stateModels.models || []).find(x => x.id === id);
      if (!loaded) {
        showToast('Loaded model "' + id + '" not found', true);
        return;
      }
      const catList = (stateCatalog && stateCatalog.models) || [];
      const loadedPath = loaded.model_path || loaded.path || '';
      const cat = catList.find(c =>
        (loadedPath && c.path && c.path === loadedPath) ||
        c.id === loaded.id ||
        c.id_suggestion === loaded.id
      ) || null;

      const merged = Object.assign({}, cat || {}, {
        id_suggestion: loaded.id,
        path: loadedPath || (cat && cat.path) || '',
        backend: loaded.backend || (cat && cat.backend) || 'gpu',
        sidecar: !!loaded.sidecar,
        task: loaded.task || (cat && cat.task) || 'chat',
        size_mib: loaded.file_size_mib || (cat && cat.size_mib) || 4096,
        default_port: loaded.sidecar_port || (cat && cat.default_port) || 0,
        endpoint: loaded.endpoint || (cat && cat.endpoint) || '',
        supports_yarn: (cat && cat.supports_yarn) || !!loaded.enable_yarn || ((loaded.context_length || 0) > 262144),
        supports_speculative: (cat && cat.supports_speculative) || !!(loaded.speculative_mode && loaded.speculative_mode !== 'off'),
        supported_speculative: (cat && Array.isArray(cat.supported_speculative) && cat.supported_speculative.length)
          ? cat.supported_speculative
          : ((loaded.speculative_mode && loaded.speculative_mode !== 'off') ? [loaded.speculative_mode] : []),
        compatible_drafts: (cat && Array.isArray(cat.compatible_drafts) && cat.compatible_drafts.length)
          ? cat.compatible_drafts
          : (loaded.draft_model_path ? [loaded.draft_model_path] : []),
        supports_mmproj: (cat && cat.supports_mmproj) || !!loaded.vision_model_path,
        compatible_mmprojs: (cat && Array.isArray(cat.compatible_mmprojs) && cat.compatible_mmprojs.length)
          ? cat.compatible_mmprojs
          : (loaded.vision_model_path ? [loaded.vision_model_path] : []),
        npu_capable: (cat && cat.npu_capable) || (loaded.backend === 'npu')
      });
      if (!merged._mapping) merged._mapping = findMappingForCatalog(merged);

      window._modalMode = 'reload';
      window._reloadTargetId = loaded.id;
      window._modalModel = merged;
      document.getElementById('lm-title').textContent = `Reconfigure & Reload "${loaded.id}" on :${livePort()}`;
      document.getElementById('lm-submit').textContent = 'Confirm & Reload Model';

      document.getElementById('lm-id').value = loaded.id;
      document.getElementById('lm-aliases').value = Array.isArray(loaded.aliases) ? loaded.aliases.join(', ') : '';
      document.getElementById('lm-path').value = merged.path;

      const devSel = document.getElementById('lm-device');
      const curDev = (loaded.backend || '').toLowerCase() === 'npu' ? 'npu' : 'gpu';
      if (merged.sidecar) {
        devSel.value = curDev;
        devSel.disabled = true;
      } else {
        const dual = isDualCapable(merged);
        const npuOk = !!(merged.npu_capable || (merged._mapping && mappingSupportsNpu(merged._mapping)) || curDev === 'npu');
        const gpuOk = !merged._mapping || mappingSupportsGpu(merged._mapping) || curDev === 'gpu';
        devSel.value = curDev;
        devSel.disabled = !(dual || (npuOk && gpuOk));
      }

      const initYarn = !!loaded.enable_yarn || ((loaded.context_length || 0) > 262144);
      populateCtxOptions(merged.supports_yarn || initYarn, loaded.context_length || 32768);

      document.getElementById('lm-img-dtype').value = loaded.image_dtype || 'bf16';
      document.getElementById('lm-img-steps').value = loaded.image_steps || (loaded.id.includes('z-image') ? 9 : 4);
      document.getElementById('lm-img-size').value = loaded.image_size || '1024x1024';
      document.getElementById('lm-img-offload').value = loaded.image_cpu_offload || 'none';

      // Build option lists first, then apply exact live values.
      syncModalVisibility(true);

      ensureSelectValue('lm-ctx', loaded.context_length || 32768, v => Number(v).toLocaleString());
      ensureSelectValue('lm-sessions', loaded.total_sessions || 1, v => v + ' Sessions');
      ensureSelectValue('lm-kv', loaded.kv_cache_quant || 'q8');
      if (loaded.npu_quant) ensureSelectValue('lm-npu-quant', loaded.npu_quant);
      if (loaded.speculative_mode && document.getElementById('lm-spec-wrap').style.display !== 'none') {
        ensureSelectValue('lm-spec', loaded.speculative_mode);
      }
      if (loaded.draft_model_path && document.getElementById('lm-draft').options.length > 0) {
        ensureSelectValue('lm-draft', loaded.draft_model_path, p => p.replace(/^.*\/mbufo-models\//, ''));
      }
      if (loaded.vision_model_path && document.getElementById('lm-mmproj-wrap').style.display !== 'none') {
        ensureSelectValue('lm-mmproj', loaded.vision_model_path, p => p.replace(/^.*\/mbufo-models\//, ''));
      }

      document.getElementById('lm-default').checked = !!loaded.is_default;
      document.getElementById('lm-evict').checked = loaded.allow_evict !== false;
      document.getElementById('lm-yarn').checked = !!loaded.enable_yarn;
      document.getElementById('lm-disk-cache').checked = loaded.enable_disk_cache !== false;
      document.getElementById('lm-autoload').checked = !!findStartupEntry(loaded);

      syncModalVisibility(false);
      document.getElementById('load-modal').classList.add('open');
    }

    function openStartupEditModal(id) {
      const items = (stateSettings && stateSettings.startup_models) || [];
      const item = items.find(x => x.id === id);
      if (!item) {
        showToast('Startup model "' + id + '" not found', true);
        return;
      }
      const catList = (stateCatalog && stateCatalog.models) || [];
      const cat = catList.find(c =>
        (!item.sidecar && item.path && c.path && c.path === item.path) ||
        c.id === item.id ||
        c.id_suggestion === item.id
      ) || null;

      const merged = Object.assign({}, cat || {}, {
        id_suggestion: item.id,
        path: item.path || (cat && cat.path) || '',
        backend: item.device || (cat && cat.backend) || 'gpu',
        sidecar: !!(item.sidecar || (cat && cat.sidecar)),
        task: item.task || (cat && cat.task) || 'chat',
        size_mib: (cat && cat.size_mib) || 4096,
        default_port: (cat && cat.default_port) || 0,
        endpoint: (cat && cat.endpoint) || '',
        supports_yarn: (cat && cat.supports_yarn) || !!item.enable_yarn || ((item.max_context || 0) > 262144),
        supports_speculative: (cat && cat.supports_speculative) || !!(item.speculative && item.speculative !== 'off'),
        supported_speculative: (cat && Array.isArray(cat.supported_speculative) && cat.supported_speculative.length)
          ? cat.supported_speculative
          : ((item.speculative && item.speculative !== 'off') ? [item.speculative] : []),
        compatible_drafts: (cat && Array.isArray(cat.compatible_drafts) && cat.compatible_drafts.length)
          ? cat.compatible_drafts
          : (item.draft_model_path ? [item.draft_model_path] : []),
        supports_mmproj: (cat && cat.supports_mmproj) || !!item.vision_model_path,
        compatible_mmprojs: (cat && Array.isArray(cat.compatible_mmprojs) && cat.compatible_mmprojs.length)
          ? cat.compatible_mmprojs
          : (item.vision_model_path ? [item.vision_model_path] : []),
        npu_capable: (cat && cat.npu_capable) || (item.device === 'npu'),
        _startup_primary: !!item.primary
      });
      if (!merged._mapping) merged._mapping = findMappingForCatalog(merged);

      window._modalMode = 'startup_edit';
      window._reloadTargetId = item.id;
      window._modalModel = merged;
      document.getElementById('lm-title').textContent = `Configure Startup Auto-Load "${item.id}"`;
      document.getElementById('lm-submit').textContent = '💾 Save Startup Config';

      document.getElementById('lm-id').value = item.id;
      document.getElementById('lm-aliases').value = Array.isArray(item.aliases) ? item.aliases.join(', ') : '';
      document.getElementById('lm-path').value = merged.path;

      const devSel = document.getElementById('lm-device');
      const curDev = (item.device || '').toLowerCase() === 'npu' ? 'npu' : 'gpu';
      if (merged.sidecar) {
        devSel.value = curDev;
        devSel.disabled = true;
      } else {
        const dual = isDualCapable(merged);
        const npuOk = !!(merged.npu_capable || (merged._mapping && mappingSupportsNpu(merged._mapping)) || curDev === 'npu');
        const gpuOk = !merged._mapping || mappingSupportsGpu(merged._mapping) || curDev === 'gpu';
        devSel.value = curDev;
        devSel.disabled = !(dual || (npuOk && gpuOk));
      }

      const initYarn = !!item.enable_yarn || ((item.max_context || 0) > 262144);
      populateCtxOptions(merged.supports_yarn || initYarn, item.max_context || 32768);

      document.getElementById('lm-img-dtype').value = item.image_dtype || 'bf16';
      document.getElementById('lm-img-steps').value = item.image_steps || (item.id.includes('z-image') ? 9 : 4);
      document.getElementById('lm-img-size').value = item.image_size || '1024x1024';
      document.getElementById('lm-img-offload').value = item.image_cpu_offload || 'none';

      syncModalVisibility(true);

      ensureSelectValue('lm-ctx', item.max_context || 32768, v => Number(v).toLocaleString());
      ensureSelectValue('lm-sessions', item.session_count || 1, v => v + ' Sessions');
      ensureSelectValue('lm-kv', item.kv_cache_quant || 'q8');
      if (item.npu_quant) ensureSelectValue('lm-npu-quant', item.npu_quant);
      if (item.speculative && document.getElementById('lm-spec-wrap').style.display !== 'none') {
        ensureSelectValue('lm-spec', item.speculative);
      }
      if (item.draft_model_path && document.getElementById('lm-draft').options.length > 0) {
        ensureSelectValue('lm-draft', item.draft_model_path, p => p.replace(/^.*\/mbufo-models\//, ''));
      }
      if (item.vision_model_path && document.getElementById('lm-mmproj-wrap').style.display !== 'none') {
        ensureSelectValue('lm-mmproj', item.vision_model_path, p => p.replace(/^.*\/mbufo-models\//, ''));
      }

      document.getElementById('lm-default').checked = !!item.primary;
      document.getElementById('lm-yarn').checked = !!item.enable_yarn;
      document.getElementById('lm-disk-cache').checked = item.enable_disk_cache !== false;

      syncModalVisibility(false);
      document.getElementById('load-modal').classList.add('open');
    }

    function closeLoadModal() {
      document.getElementById('load-modal').classList.remove('open');
    }

    function updateModalEstimate() {
      const m = window._modalModel || { size_mib: 4096 };
      const dev = document.getElementById('lm-device').value;
      const isSidecar = !!m.sidecar;
      const isEnc = (m.task === 'embedding' || m.task === 'reranker');
      const avail = (stateModels.memory && stateModels.memory.host_available_mib) || 0;
      const el = document.getElementById('lm-estimate');
      if (isSidecar) {
        if (m.task === 'image_generation') {
          const offload = document.getElementById('lm-img-offload') ? document.getElementById('lm-img-offload').value : 'none';
          const dtype = document.getElementById('lm-img-dtype') ? document.getElementById('lm-img-dtype').value : 'bf16';
          el.innerHTML =
            `ROCm Image Sidecar (diffusers): <strong>~${fmtGiB(m.size_mib)}</strong> on disk · dtype: ${dtype} · offload: ${offload} · port :${m.default_port || '?'} · Available: ${fmtGiB(avail)}`;
        } else {
          el.innerHTML =
            `Sidecar Process: <strong>~${fmtGiB(m.size_mib)}</strong> (managed on port :${m.default_port || '?'} · Serialized execution · Current Available: ${fmtGiB(avail)})`;
        }
        return;
      }
      if (dev === 'npu') {
        el.innerHTML =
          `NPU (XDNA2) Weights: <strong>~${fmtGiB(m.size_mib)}</strong> · Serialized 1-at-a-time queue (Current Available: ${fmtGiB(avail)})`;
        return;
      }
      if (isEnc) {
        el.innerHTML =
          `GPU Encoder Weights: <strong>${fmtGiB(m.size_mib)}</strong> + Buffers: <strong>~0.2 GiB</strong> → Total: <strong>~${fmtGiB(m.size_mib + 256)}</strong> (Current Available: ${fmtGiB(avail)})`;
        return;
      }
      const ctx = parseInt(document.getElementById('lm-ctx').value, 10) || 32768;
      const sess = parseInt(document.getElementById('lm-sessions').value, 10) || 1;
      const kvMib = Math.round((ctx / 32768) * sess * 768);
      const totalEst = m.size_mib + kvMib + 512;
      const extras = [];
      const specVal = document.getElementById('lm-spec').value;
      if (document.getElementById('lm-spec-wrap').style.display !== 'none' && specVal !== 'off' && document.getElementById('lm-draft').value) {
        extras.push(`+${specVal}`);
      }
      if (document.getElementById('lm-mmproj-wrap').style.display !== 'none' && document.getElementById('lm-mmproj').value) {
        extras.push('+mmproj');
      }
      const extraTxt = extras.length ? ` [${extras.join(', ')}]` : '';
      el.innerHTML =
        `Weights: <strong>${fmtGiB(m.size_mib)}</strong> + Est. KV/Buffers (${sess}×${(ctx/1024).toFixed(0)}K): <strong>~${fmtGiB(kvMib + 512)}</strong>${extraTxt} → Total: <strong>~${fmtGiB(totalEst)}</strong> (Current Available: ${fmtGiB(avail)})`;
    }

    async function quickLoadModel(idx) {
      const m = (window._catalogFiltered && window._catalogFiltered[idx]) || null;
      if (!m) return;
      const dev = defaultDeviceFor(m);
      if (m.sidecar) {
        const scPayload = { id: m.id_suggestion, device: dev };
        if (m.task === 'image_generation') {
          scPayload.image_dtype = m.image_dtype || 'bf16';
          scPayload.image_steps = m.image_steps || (m.id_suggestion && m.id_suggestion.includes('z-image') ? 9 : 4);
          scPayload.image_size = m.image_size || '1024x1024';
          scPayload.image_cpu_offload = m.image_cpu_offload || 'none';
        }
        await doLoadRequest(scPayload);
        return;
      }
      const isEnc = (m.task === 'embedding' || m.task === 'reranker');
      const isGpuChat = (dev === 'gpu' && !isEnc && (!m.task || m.task === 'chat'));
      const useSpec = (isGpuChat && m.supports_speculative && m.suggested_speculative && m.suggested_draft);
      const payload = {
        id: m.id_suggestion,
        path: m.path,
        context_length: (dev === 'npu' || isEnc) ? 0 : 32768,
        sessions: 1,
        kv_quant: isGpuChat ? 'q4' : 'fp16',
        device: dev,
        npu_quant: dev === 'npu' ? 'q8' : '',
        speculative: useSpec ? m.suggested_speculative : 'off',
        draft_model_path: useSpec ? m.suggested_draft : '',
        mmproj_path: (isGpuChat && m.supports_mmproj && m.suggested_mmproj) ? m.suggested_mmproj : '',
        enable_yarn: false,
        allow_evict: true,
        set_default: false
      };
      await doLoadRequest(payload);
    }

    async function syncModalAutoLoad(rec, wantAutoLoad, prevTargetId) {
      const existing = findStartupEntry({ id: prevTargetId || rec.id, path: rec.path, sidecar: rec.sidecar });
      if (wantAutoLoad) {
        const res = await fetch('/admin/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ upsert_startup_model: rec })
        });
        if (res.ok) {
          stateSettings = Object.assign({}, stateSettings || {}, await res.json());
          renderLoadedModels();
          if (document.getElementById('settings-modal').classList.contains('open')) renderStartupModelCards();
        }
      } else if (existing) {
        const res = await fetch('/admin/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ remove_startup_model: existing.id })
        });
        if (res.ok) {
          stateSettings = Object.assign({}, stateSettings || {}, await res.json());
          renderLoadedModels();
          if (document.getElementById('settings-modal').classList.contains('open')) renderStartupModelCards();
        }
      }
    }

    async function submitLoadModal() {
      const m = window._modalModel || {};
      const dev = document.getElementById('lm-device').value;
      const mode = window._modalMode || 'load';
      const isReload = (mode === 'reload');
      const isStartupEdit = (mode === 'startup_edit');
      const wantAutoLoad = document.getElementById('lm-autoload').checked;

      if (m.sidecar) {
        const id = document.getElementById('lm-id').value.trim();
        const scPayload = { id, device: dev };
        const scRecord = {
          configured: true,
          primary: false,
          sidecar: true,
          task: m.task || '',
          id,
          aliases: [],
          path: m.path || '',
          device: dev
        };
        if (m.task === 'image_generation') {
          scPayload.image_dtype = document.getElementById('lm-img-dtype').value;
          scPayload.image_steps = Math.max(1, parseInt(document.getElementById('lm-img-steps').value, 10) || 4);
          scPayload.image_size = document.getElementById('lm-img-size').value;
          scPayload.image_cpu_offload = document.getElementById('lm-img-offload').value;
          scRecord.image_dtype = scPayload.image_dtype;
          scRecord.image_steps = scPayload.image_steps;
          scRecord.image_size = scPayload.image_size;
          scRecord.image_cpu_offload = scPayload.image_cpu_offload;
        }
        if (isStartupEdit) {
          scRecord.target_id = window._reloadTargetId;
          closeLoadModal();
          await syncModalAutoLoad(scRecord, true, window._reloadTargetId);
          showToast(`Updated startup auto-load configuration for "${id}"`);
        } else if (isReload) {
          if (!confirm(`Confirm reload of sidecar "${window._reloadTargetId}" with updated settings?`)) return;
          scPayload.target_id = window._reloadTargetId;
          closeLoadModal();
          if (await doReloadRequest(scPayload)) {
            await syncModalAutoLoad(scRecord, wantAutoLoad, window._reloadTargetId);
          }
        } else {
          closeLoadModal();
          if (await doLoadRequest(scPayload)) {
            if (wantAutoLoad) await syncModalAutoLoad(scRecord, true, id);
          }
        }
        return;
      }

      const isEnc = (m.task === 'embedding' || m.task === 'reranker');
      const isGpuChat = (dev === 'gpu' && !isEnc && (!m.task || m.task === 'chat'));
      const aliasesRaw = document.getElementById('lm-aliases').value.trim();
      const aliases = aliasesRaw ? aliasesRaw.split(',').map(s => s.trim()).filter(Boolean) : [];
      const specVisible = document.getElementById('lm-spec-wrap').style.display !== 'none';
      const specVal = (isGpuChat && specVisible) ? document.getElementById('lm-spec').value : 'off';
      const draftVal = (specVal !== 'off' && document.getElementById('lm-draft-wrap').style.display !== 'none')
        ? document.getElementById('lm-draft').value
        : '';
      const mmVisible = document.getElementById('lm-mmproj-wrap').style.display !== 'none';
      const mmVal = (isGpuChat && mmVisible) ? document.getElementById('lm-mmproj').value : '';
      const yarnVisible = document.getElementById('lm-yarn-wrap').style.display !== 'none';
      const isDefChecked = (isGpuChat && document.getElementById('lm-default-wrap').style.display !== 'none')
        ? document.getElementById('lm-default').checked
        : false;
      const payload = {
        id: document.getElementById('lm-id').value.trim(),
        aliases: aliases,
        path: document.getElementById('lm-path').value.trim(),
        context_length: (dev === 'gpu') ? parseInt(document.getElementById('lm-ctx').value, 10) : 0,
        sessions: (isGpuChat && document.getElementById('lm-sessions-wrap').style.display !== 'none')
          ? parseInt(document.getElementById('lm-sessions').value, 10)
          : 1,
        kv_quant: (isGpuChat && document.getElementById('lm-kv-wrap').style.display !== 'none')
          ? document.getElementById('lm-kv').value
          : 'fp16',
        device: dev,
        npu_quant: (dev === 'npu' && document.getElementById('lm-npu-quant-wrap').style.display !== 'none') ? document.getElementById('lm-npu-quant').value : '',
        speculative: specVal,
        draft_model_path: draftVal,
        mmproj_path: mmVal,
        vision_model_path: mmVal,
        enable_yarn: (isGpuChat && yarnVisible) ? document.getElementById('lm-yarn').checked : false,
        enable_disk_cache: isGpuChat ? document.getElementById('lm-disk-cache').checked : false,
        allow_evict: document.getElementById('lm-evict').checked,
        set_default: isDefChecked
      };
      const startupRec = {
        configured: true,
        primary: isStartupEdit ? (isGpuChat ? isDefChecked : !!m._startup_primary) : isDefChecked,
        sidecar: false,
        task: m.task || '',
        id: payload.id,
        aliases: payload.aliases,
        path: payload.path,
        max_context: payload.context_length,
        session_count: payload.sessions,
        kv_cache_quant: payload.kv_quant,
        device: payload.device,
        npu_quant: payload.npu_quant,
        speculative: payload.speculative,
        draft_model_path: payload.draft_model_path,
        vision_model_path: payload.vision_model_path,
        enable_yarn: payload.enable_yarn,
        enable_disk_cache: payload.enable_disk_cache
      };
      if (isStartupEdit) {
        startupRec.target_id = window._reloadTargetId;
        closeLoadModal();
        await syncModalAutoLoad(startupRec, true, window._reloadTargetId);
        showToast(`Updated startup auto-load configuration for "${payload.id}"`);
      } else if (isReload) {
        if (!confirm(`Confirm reload of "${window._reloadTargetId}" with new settings? Previous Primary status is preserved.`)) return;
        payload.target_id = window._reloadTargetId;
        closeLoadModal();
        if (await doReloadRequest(payload)) {
          await syncModalAutoLoad(startupRec, wantAutoLoad, window._reloadTargetId);
        }
      } else {
        closeLoadModal();
        if (await doLoadRequest(payload)) {
          if (wantAutoLoad) await syncModalAutoLoad(startupRec, true, payload.id);
        }
      }
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
        `Local Repository: /admin/model-mapping (${(stateMapping.models || []).length} models bundled with server version)`;
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
              ${m.note ? `<div style="font-size:11px;color:#93c5fd;margin-top:2px;">${escHtml(m.note)}</div>` : ''}
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
          const rtag = data.remote_tag || '';
          if (data.update_available) {
            pill.className = 'badge badge-amber';
            pill.textContent = `${tag} → ${rtag || data.remote_short || 'new'} (Update!)`;
          } else {
            pill.className = 'badge badge-green';
            pill.textContent = `${tag} · ${shortSha}`;
          }
          pill.title = `Running: ${tag} (${shortSha}) | GitHub Latest Tag: ${rtag || 'none'} (${data.build_time_utc || ''})`;
        }
        renderUpdateBanner(data, data.status || {});
        if (manual) {
          if (data.check_error) showToast('Update check warning: ' + data.check_error, true);
          else if (data.update_available) showToast(`New release tag available: ${data.remote_tag || data.remote_short}`);
          else showToast(`Server is up to date (${data.running_tag || data.running_short})`);
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
        badge.textContent = 'RELEASE UPDATE';
        const rTag = checkData.remote_tag || checkData.remote_short || 'latest';
        const lTag = checkData.running_tag || checkData.running_short || 'current';
        title.textContent = `New GitHub release tag ${rTag} is available (includes latest code & model_mapping.json)`;
        detail.textContent = `running ${lTag} (${checkData.running_short || ''}) → latest tag ${rTag}${checkData.remote_short ? ' (' + checkData.remote_short + ')' : ''}`;
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

    async function doReloadRequest(payload) {
      showToast(`Reloading "${payload.target_id}" on :${livePort()}...`);
      const pollTimer = setInterval(fetchStatus, 1000);
      try {
        const res = await fetch('/admin/reload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        clearInterval(pollTimer);
        await fetchStatus();
        renderCatalog();
        if (!res.ok || data.error) {
          const msg = data.error ? data.error.message : res.statusText;
          showToast('Reload failed' + (data.rolled_back ? ' (rolled back)' : '') + ': ' + msg, true);
          return false;
        } else {
          showToast(`Reloaded "${data.id || payload.id}" cleanly!`);
          return true;
        }
      } catch (e) {
        clearInterval(pollTimer);
        showToast('Reload request error: ' + e.message, true);
        return false;
      }
    }

    async function doLoadRequest(payload) {
      showToast(`Loading "${payload.id}" into :${livePort()}...`);
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
          return false;
        } else {
          const ev = (data.evicted && data.evicted.length) ? ` (Evicted: ${data.evicted.join(', ')})` : '';
          showToast(`Loaded "${payload.id}" in ${(data.load_ms / 1000).toFixed(1)}s!${ev}`);
          return true;
        }
      } catch (e) {
        clearInterval(pollTimer);
        showToast('Network error loading model: ' + e.message, true);
        return false;
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
          await Promise.all([fetchStatus(), fetchCatalog()]);
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


    // ---------------- Settings, Auth & Disk Cache Management ----------------
    async function loadSettingsSummary() {
      if (authLocked || restartingServer) return null;
      try {
        const res = await fetch('/admin/settings', { cache: 'no-store' });
        if (!res.ok) return null;
        stateSettings = await res.json();
        updatePortDisplays();
        renderLoadedModels();
        if (document.getElementById('settings-modal').classList.contains('open')) {
          renderStartupModelCards();
        }
        return stateSettings;
      } catch (_) {
        return null;
      }
    }

    function switchSettingsTab(tab) {
      window._settingsTab = tab;
      ['default', 'port', 'auth', 'cache'].forEach(k => {
        const btn = document.getElementById('set-btn-' + k);
        const pane = document.getElementById('set-tab-' + k);
        if (btn) btn.classList.toggle('active', k === tab);
        if (pane) pane.classList.toggle('active', k === tab);
      });
      if (tab === 'cache') {
        loadDiskCache(false);
      }
    }

    async function openSettingsModal(initialTab) {
      document.getElementById('settings-modal').classList.add('open');
      switchSettingsTab(initialTab || window._settingsTab || 'default');
      const s = await loadSettingsSummary();
      if (s) populateSettingsForms(s);
    }

    function closeSettingsModal() {
      document.getElementById('settings-modal').classList.remove('open');
    }

    function populateSettingsForms(s) {
      renderStartupModelCards();

      document.getElementById('set-port-running').textContent = ':' + (s.running_port || 8741);
      document.getElementById('set-port-configured').textContent = ':' + (s.configured_port || s.running_port || 8741);
      document.getElementById('set-port-config-path').textContent = s.config_path || '';
      document.getElementById('set-port').value = s.configured_port || s.running_port || 8741;
      document.getElementById('set-port-restart-note').style.display = s.restart_required ? 'block' : 'none';
      document.getElementById('set-restart-status').style.display = 'none';

      document.getElementById('set-web-auth-enabled').checked = !!s.web_auth_enabled;
      const webBadge = document.getElementById('set-web-pwd-badge');
      webBadge.textContent = s.web_password_set ? 'Password Set' : 'Not Set';
      webBadge.className = 'badge ' + (s.web_password_set ? 'badge-green' : '');
      document.getElementById('set-web-password').value = '';

      document.getElementById('set-api-auth-enabled').checked = !!s.api_auth_enabled;
      document.getElementById('set-api-share-web').checked = !!s.api_auth_share_web_password;
      const apiBadge = document.getElementById('set-api-key-badge');
      apiBadge.textContent = s.api_key_set ? 'Key Set' : (s.cli_api_key_active ? 'CLI Key Active' : 'Not Set');
      apiBadge.className = 'badge ' + ((s.api_key_set || s.cli_api_key_active) ? 'badge-green' : '');
      document.getElementById('set-api-key').value = '';
      document.getElementById('set-cli-key-note').style.display = s.cli_api_key_active ? 'block' : 'none';
      syncAuthFieldState();
    }

    function renderStartupModelCards() {
      const box = document.getElementById('set-startup-cards');
      const countEl = document.getElementById('set-startup-count');
      if (!box) return;
      const items = (stateSettings && Array.isArray(stateSettings.startup_models)) ? stateSettings.startup_models : [];
      if (countEl) countEl.textContent = `Startup Auto-Load Models (${items.length})`;
      if (!items.length) {
        box.innerHTML = `<div style="padding:20px;text-align:center;color:var(--muted);border:1px dashed var(--border);border-radius:8px;font-size:12px;">No startup auto-load models configured. Use "📌 Auto-Load" on any loaded model card to add it here.</div>`;
        return;
      }
      const loadedList = (stateModels && stateModels.models) || [];
      box.innerHTML = items.map(m => {
        const dev = (m.device || 'gpu').toLowerCase() === 'npu' ? 'NPU' : 'GPU';
        const isLoaded = loadedList.some(lm => lm.id === m.id || (!m.sidecar && m.path && (lm.model_path === m.path || lm.path === m.path)));
        const badges = [];
        badges.push(dev === 'NPU' ? '<span class="badge badge-amber">⚡ NPU</span>' : '<span class="badge badge-primary">🎮 GPU</span>');
        if (m.primary) badges.push('<span class="badge badge-primary">★ PRIMARY</span>');
        if (isLoaded) badges.push('<span class="badge badge-green">LOADED</span>');
        if (m.sidecar) {
          badges.push('<span class="badge">SIDECAR</span>');
        } else if (dev === 'NPU') {
          if (m.npu_quant) badges.push(`<span class="badge">${escHtml(m.npu_quant.toUpperCase())}</span>`);
        } else {
          if (m.max_context > 0) badges.push(`<span class="badge">CTX: ${Math.round(m.max_context / 1024)}K</span>`);
          if (m.session_count > 1) badges.push(`<span class="badge">${m.session_count} Sessions</span>`);
          if (m.kv_cache_quant) badges.push(`<span class="badge">KV: ${escHtml(m.kv_cache_quant)}</span>`);
          if (m.speculative && m.speculative !== 'off') badges.push(`<span class="badge">${escHtml(m.speculative.toUpperCase())}</span>`);
          if (m.vision_model_path) badges.push('<span class="badge">VISION</span>');
          if (m.enable_yarn) badges.push('<span class="badge">YARN</span>');
          if (m.enable_disk_cache) badges.push('<span class="badge">DISK CACHE</span>');
        }
        if (m.task === 'image_generation' || m.image_dtype) {
          if (m.image_dtype) badges.push(`<span class="badge">${escHtml(m.image_dtype)}</span>`);
          if (m.image_steps) badges.push(`<span class="badge">${m.image_steps} steps</span>`);
          if (m.image_size) badges.push(`<span class="badge">${escHtml(m.image_size)}</span>`);
          if (m.image_cpu_offload) badges.push(`<span class="badge">offload: ${escHtml(m.image_cpu_offload)}</span>`);
        }
        const canPrimary = !m.sidecar && !m.primary && (!m.task || m.task === 'chat');
        const subText = m.path || (m.sidecar ? `${m.task || 'sidecar'} managed process` : '');
        return `
          <div class="model-card${m.primary ? ' is-default' : ''}" style="border-left:3px solid ${dev === 'NPU' ? NPU_COLOR : GPU_COLOR};">
            <div class="model-top">
              <div>
                <div class="model-id">
                  <span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${dev === 'NPU' ? NPU_COLOR : GPU_COLOR}"></span>
                  <span>${escHtml(m.id)}</span>
                  ${badges.join('')}
                </div>
                <div class="model-path">${escHtml(subText)}</div>
              </div>
              <div class="model-actions">
                ${canPrimary ? `<button class="btn-sm" onclick="setStartupPrimary(decodeURIComponent('${encodeURIComponent(m.id)}'))" title="Set as primary default model">★ Primary</button>` : ''}
                <button class="btn-sm" onclick="openStartupEditModal(decodeURIComponent('${encodeURIComponent(m.id)}'))" title="Configure startup auto-load parameters">⚙ Configure</button>
                <button class="btn-sm btn-danger" onclick="removeStartupModel(decodeURIComponent('${encodeURIComponent(m.id)}'))" title="Remove from startup auto-load list">✕ Remove</button>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    async function setStartupPrimary(id) {
      try {
        const res = await fetch('/admin/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ set_primary_startup_model: id })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Failed to set primary default model: ' + (data.error ? data.error.message : res.statusText), true);
          return;
        }
        stateSettings = Object.assign({}, stateSettings || {}, data);
        renderStartupModelCards();
        showToast(`Set "${id}" as startup primary default model`);
      } catch (e) {
        showToast('Error: ' + e.message, true);
      }
    }

    async function removeStartupModel(id) {
      if (!confirm(`Remove "${id}" from startup auto-load list?`)) return;
      try {
        const res = await fetch('/admin/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ remove_startup_model: id })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Failed to remove model: ' + (data.error ? data.error.message : res.statusText), true);
          return;
        }
        stateSettings = Object.assign({}, stateSettings || {}, data);
        renderStartupModelCards();
        renderLoadedModels();
        showToast(`Removed "${id}" from startup auto-load list`);
      } catch (e) {
        showToast('Error: ' + e.message, true);
      }
    }

    async function confirmAndRestartServer() {
      const targetPort = (stateSettings && stateSettings.configured_port) || livePort();
      if (!confirm(`Restart MBufo server (:${targetPort}) now to apply startup model changes? Active connections will briefly disconnect.`)) return;
      await triggerServerRestart(targetPort);
    }

    async function savePortSetting(andRestart) {
      const port = parseInt(document.getElementById('set-port').value, 10);
      if (!port || port < 1 || port > 65535) {
        showToast('Invalid port (must be 1-65535)', true);
        return;
      }
      if (andRestart && !confirm(`Save router port :${port} and restart mbufo server now? Active connections will reconnect.`)) {
        return;
      }
      try {
        const res = await fetch('/admin/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ port })
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Port save failed: ' + (data.error ? data.error.message : res.statusText), true);
          return;
        }
        stateSettings = Object.assign({}, stateSettings || {}, data);
        populateSettingsForms(stateSettings);
        showToast(`Configured port :${port} saved`);
        if (andRestart) {
          await triggerServerRestart(port);
        }
      } catch (e) {
        showToast('Port save error: ' + e.message, true);
      }
    }

    async function triggerServerRestart(targetPort) {
      const stBox = document.getElementById('set-restart-status');
      const newUrl = `${location.protocol}//${location.hostname}:${targetPort}/admin`;
      const healthUrl = `${location.protocol}//${location.hostname}:${targetPort}/health`;
      restartingServer = true;
      stopBackgroundPollers();
      if (stBox) {
        stBox.style.display = 'block';
        stBox.innerHTML = `Restarting mbufo server on port <strong>:${targetPort}</strong>... Waiting for /health... (<a href="${newUrl}" style="color:#60a5fa;">${newUrl}</a>)`;
      }
      try {
        await fetch('/admin/server/restart', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
      } catch (_) {}
      setTimeout(() => {
        let tries = 0;
        const timer = setInterval(async () => {
          tries += 1;
          try {
            await fetch(healthUrl, { mode: 'no-cors', cache: 'no-store' });
            clearInterval(timer);
            showToast(`Server ready on :${targetPort}! Redirecting...`);
            window.location.href = newUrl;
          } catch (_) {
            if (tries >= 60) {
              clearInterval(timer);
              restartingServer = false;
              if (stBox) stBox.innerHTML += '<br/><span style="color:var(--red);">Timed out waiting — click link above manually.</span>';
            }
          }
        }, 1200);
      }, 1500);
    }

    function syncAuthFieldState() {
      const share = document.getElementById('set-api-share-web').checked;
      document.getElementById('set-api-key').disabled = share;
    }

    function generateRandomApiKey() {
      const bytes = new Uint8Array(24);
      if (window.crypto && window.crypto.getRandomValues) {
        window.crypto.getRandomValues(bytes);
      } else {
        for (let i = 0; i < bytes.length; i++) bytes[i] = Math.floor(Math.random() * 256);
      }
      const hex = Array.from(bytes).map(b => b.toString(16).padStart(2, '0')).join('');
      document.getElementById('set-api-share-web').checked = false;
      syncAuthFieldState();
      document.getElementById('set-api-key').value = 'mb-' + hex;
      showToast('Generated new API key — remember to copy & Save Security Settings!');
    }

    async function copyApiKeyInput() {
      const val = document.getElementById('set-api-key').value.trim();
      if (!val) { showToast('Nothing in API Key box to copy', true); return; }
      try { await navigator.clipboard.writeText(val); showToast('Copied API key'); }
      catch (_) { showToast('Copy blocked by browser', true); }
    }

    async function saveAuthSettings() {
      const webEnabled = document.getElementById('set-web-auth-enabled').checked;
      const webPwd = document.getElementById('set-web-password').value;
      const apiEnabled = document.getElementById('set-api-auth-enabled').checked;
      const apiShare = document.getElementById('set-api-share-web').checked;
      const apiKey = document.getElementById('set-api-key').value.trim();

      if (webEnabled && !webPwd && !(stateSettings && stateSettings.web_password_set)) {
        showToast('Enter a Web UI password before enabling Layer 1 Web UI Auth', true);
        return;
      }
      if (apiEnabled && !apiShare && !apiKey && !(stateSettings && (stateSettings.api_key_set || stateSettings.cli_api_key_active))) {
        showToast('Enter/generate an API key (or enable Share Web Password) before enabling Layer 2 API Auth', true);
        return;
      }
      if (apiEnabled && apiShare && !webPwd && !(stateSettings && stateSettings.web_password_set)) {
        showToast('Layer 2 is set to share Web UI password, so please enter a Web UI password above', true);
        return;
      }

      const payload = {
        web_auth_enabled: webEnabled,
        api_auth_enabled: apiEnabled,
        api_auth_share_web_password: apiShare
      };
      if (webPwd) payload.web_password = webPwd;
      if (!apiShare && apiKey) payload.api_key = apiKey;

      try {
        const res = await fetch('/admin/settings', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Auth save failed: ' + (data.error ? data.error.message : res.statusText), true);
          return;
        }
        // Keep current operator session signed in seamlessly if password just got set/updated
        if (webEnabled && webPwd) {
          await fetch('/admin/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ password: webPwd })
          });
        }
        const s = await loadSettingsSummary();
        if (s) populateSettingsForms(s);
        showToast('Security & Auth settings updated cleanly!');
      } catch (e) {
        showToast('Auth save error: ' + e.message, true);
      }
    }

    async function clearWebPassword() {
      if (!confirm('Clear Web UI password and disable Layer 1 Web UI Auth?')) return;
      const res = await fetch('/admin/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clear_web_password: true, web_auth_enabled: false })
      });
      if (res.ok) {
        const s = await loadSettingsSummary();
        if (s) populateSettingsForms(s);
        showToast('Cleared Web UI password');
      }
    }

    async function clearApiKeySetting() {
      if (!confirm('Clear dedicated API key? (You can still leave Layer 2 off or share Web UI password.)')) return;
      const res = await fetch('/admin/settings', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clear_api_key: true, api_auth_enabled: false })
      });
      if (res.ok) {
        const s = await loadSettingsSummary();
        if (s) populateSettingsForms(s);
        showToast('Cleared dedicated API key');
      }
    }

    // --- Disk Cache Tab ---
    async function loadDiskCache(showBanner) {
      try {
        const res = await fetch('/admin/disk-cache', { cache: 'no-store' });
        if (!res.ok) return;
        stateDiskCache = await res.json();
        renderDiskCacheSummary();
        renderDiskCacheTable();
        if (showBanner) showToast(`Disk cache refreshed (${stateDiskCache.checkpoint_count || 0} checkpoints)`);
      } catch (e) {
        showToast('Failed to query disk cache: ' + e.message, true);
      }
    }

    function renderDiskCacheSummary() {
      const c = stateDiskCache || {};
      document.getElementById('cache-summary-dir').textContent = c.directory || '(disabled)';
      document.getElementById('cache-summary-mode').textContent = c.enabled
        ? (c.live ? 'LIVE (DiskCatalog::Retire)' : 'OFFLINE (Direct Manifest Cleanup)')
        : 'DISABLED';
      document.getElementById('cache-summary-counts').textContent =
        `${c.checkpoint_count || 0} total (${c.compatible_count || 0} compat / ${c.incompatible_count || 0} incompat)`;
      document.getElementById('cache-summary-usage').textContent =
        `${fmtBytes(c.managed_bytes || 0)} / ${fmtBytes(c.capacity_bytes || 0)} (priv ${fmtBytes(c.sum_private_bytes || 0)} + chunks ${fmtBytes(c.sum_chunk_referenced_bytes || 0)})`;
      document.getElementById('cache-summary-compat').textContent = c.active_compatibility || '(no active runner loaded)';
      const btnIncompat = document.getElementById('btn-cache-del-incompat');
      if (btnIncompat) {
        btnIncompat.textContent = `🧹 Clean Incompatible (${c.incompatible_count || 0})`;
        btnIncompat.disabled = !(c.incompatible_count > 0);
      }
    }

    function filteredCacheEntries() {
      const entries = (stateDiskCache && stateDiskCache.entries) || [];
      const q = (document.getElementById('cache-search').value || '').toLowerCase().trim();
      return entries.filter(e => {
        if (!q) return true;
        return String(e.checkpoint_id).includes(q) ||
               (e.compatibility_hex || '').toLowerCase().includes(q) ||
               (e.purpose || '').toLowerCase().includes(q) ||
               (e.manifest_file || '').toLowerCase().includes(q);
      });
    }

    function toggleSelectAllCache() {
      const checked = document.getElementById('cache-select-all').checked;
      filteredCacheEntries().forEach(e => {
        if (checked) window._selectedCacheIds.add(String(e.checkpoint_id));
        else window._selectedCacheIds.delete(String(e.checkpoint_id));
      });
      renderDiskCacheTable();
    }

    function toggleCacheRow(idStr, checked) {
      if (checked) window._selectedCacheIds.add(idStr);
      else window._selectedCacheIds.delete(idStr);
    }

    function renderDiskCacheTable() {
      const tbody = document.getElementById('cache-tbody');
      const rows = filteredCacheEntries();
      if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;color:var(--muted);padding:18px;">No disk cache checkpoints found.</td></tr>`;
        return;
      }
      tbody.innerHTML = rows.map(e => {
        const idStr = String(e.checkpoint_id);
        const chk = window._selectedCacheIds.has(idStr) ? 'checked' : '';
        const dt = e.mtime_epoch ? new Date(e.mtime_epoch * 1000).toLocaleString() : '--';
        const compatBadge = e.invalid
          ? `<span class="badge badge-red">INVALID</span>`
          : (e.compatible
              ? `<span class="badge badge-green">COMPAT</span>`
              : `<span class="badge badge-amber">INCOMPAT</span>`);
        return `
          <tr>
            <td><input type="checkbox" ${chk} onchange="toggleCacheRow('${idStr}', this.checked)" /></td>
            <td title="lineage ${e.lineage_id} · ${escHtml(e.manifest_file || '')}">#${idStr}</td>
            <td>${compatBadge} ${escHtml(e.purpose || '')} <span style="color:var(--muted);">${escHtml((e.compatibility_hex || '').slice(0, 12))}</span></td>
            <td>${Number(e.token_count || 0).toLocaleString()}</td>
            <td>${e.chunk_count || 0}</td>
            <td>${fmtBytes(e.private_bytes || 0)} / ${fmtBytes(e.manifest_bytes || 0)} / ${fmtBytes(e.chunk_bytes || 0)}</td>
            <td style="color:var(--muted);">${escHtml(dt)}</td>
            <td><button class="btn-sm btn-danger" onclick="deleteCacheIds(['${idStr}'])">🗑 Delete</button></td>
          </tr>
        `;
      }).join('');
    }

    async function doCacheDelete(payload, confirmMsg) {
      if (!confirm(confirmMsg)) return;
      try {
        const res = await fetch('/admin/disk-cache/delete', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        if (!res.ok || data.error) {
          showToast('Cache delete failed: ' + (data.error ? data.error.message : res.statusText), true);
          return;
        }
        window._selectedCacheIds.clear();
        document.getElementById('cache-select-all').checked = false;
        showToast(`Deleted ${data.deleted} checkpoint(s) (pinned skipped: ${data.skipped_pinned}, busy skipped: ${data.skipped_busy})`);
        await loadDiskCache(false);
      } catch (e) {
        showToast('Cache delete error: ' + e.message, true);
      }
    }

    function deleteCacheIds(idStrings) {
      const ids = idStrings.map(x => Number(x)).filter(n => Number.isFinite(n));
      if (!ids.length) return;
      doCacheDelete({ ids }, `Delete ${ids.length} disk cache checkpoint(s)?`);
    }

    function deleteSelectedCache() {
      const ids = Array.from(window._selectedCacheIds);
      if (!ids.length) { showToast('No checkpoints selected', true); return; }
      deleteCacheIds(ids);
    }

    function deleteIncompatibleCache() {
      doCacheDelete({ delete_incompatible: true }, 'Delete all incompatible disk cache checkpoints?');
    }

    function deleteAllCache() {
      doCacheDelete({ delete_all: true }, 'PURGE ALL disk cache checkpoints? Unpinned checkpoints and unreferenced chunks will be removed.');
    }

    // --- Auth Overlay & Boot ---
    async function submitLogin() {
      const pwd = document.getElementById('auth-login-password').value;
      const errEl = document.getElementById('auth-login-error');
      const btn = document.getElementById('auth-login-btn');
      errEl.style.display = 'none';
      btn.disabled = true;
      try {
        const res = await fetch('/admin/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ password: pwd })
        });
        const data = await res.json();
        if (!res.ok || data.error || !data.authenticated) {
          errEl.textContent = (data.error && data.error.message) ? data.error.message : 'Invalid password';
          errEl.style.display = 'block';
          btn.disabled = false;
          return;
        }
        authLocked = false;
        document.getElementById('auth-overlay').classList.remove('open');
        document.body.classList.remove('auth-pending');
        btn.disabled = false;
        startDashboardBoot();
      } catch (e) {
        errEl.textContent = 'Login request error: ' + e.message;
        errEl.style.display = 'block';
        btn.disabled = false;
      }
    }

    function startDashboardBoot() {
      document.body.classList.remove('auth-pending');
      refreshAll();
      if (statusPollTimer) clearInterval(statusPollTimer);
      statusPollTimer = setInterval(fetchStatus, 3000);
    }

    async function checkAuthAndBoot() {
      try {
        const res = await fetch('/admin/auth/status', { cache: 'no-store' });
        if (res.ok) {
          const st = await res.json();
          if (st.web_auth_enabled && !st.authenticated) {
            handleWebAuthRequired('Password authentication is enabled. Enter Web UI password to unlock.');
            return;
          }
        }
      } catch (_) {}
      startDashboardBoot();
    }

    checkAuthAndBoot();
  </script>
</body>
</html>
)MBUFO_UI";

}  // namespace mbufo::server
