#!/usr/bin/env python3
"""Generate a self-contained browser review desk for Stage 6 CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage 6 Error Review</title>
  <style>
    :root {
      --ink: #172028;
      --muted: #66717a;
      --paper: #f4f2ed;
      --surface: #fffefb;
      --line: #cbc8bf;
      --line-dark: #969188;
      --red: #b63b31;
      --red-soft: #f6e5e1;
      --green: #26704b;
      --green-soft: #e1efe7;
      --amber: #a86814;
      --amber-soft: #f4ead6;
      --blue: #245c7b;
      --blue-soft: #e0edf4;
      --focus: #111820;
      --sidebar: 360px;
    }

    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: "IBM Plex Sans", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      font-size: 14px;
      letter-spacing: 0;
      overflow: hidden;
    }

    button, input, select, textarea { font: inherit; letter-spacing: 0; }
    button { cursor: pointer; }
    button:focus-visible, input:focus-visible, select:focus-visible, textarea:focus-visible {
      outline: 2px solid var(--focus);
      outline-offset: 2px;
    }

    .app {
      height: 100%;
      display: grid;
      grid-template-rows: 58px minmax(0, 1fr);
    }

    .topbar {
      display: grid;
      grid-template-columns: minmax(260px, 1fr) auto;
      align-items: center;
      gap: 18px;
      padding: 0 18px;
      background: var(--ink);
      color: white;
      border-bottom: 1px solid #000;
    }

    .brand { display: flex; align-items: baseline; gap: 12px; min-width: 0; }
    .brand h1 {
      margin: 0;
      font-family: "IBM Plex Serif", "Noto Serif CJK SC", SimSun, serif;
      font-size: 19px;
      font-weight: 700;
      white-space: nowrap;
    }
    .brand span { color: #bfc7cc; font-size: 12px; white-space: nowrap; }

    .top-actions { display: flex; align-items: center; gap: 8px; }
    .status-chip {
      min-width: 116px;
      padding: 7px 10px;
      border: 1px solid #59636a;
      border-radius: 4px;
      color: #dce3e7;
      text-align: center;
      font-variant-numeric: tabular-nums;
    }

    .action {
      min-height: 34px;
      border: 1px solid #7c858b;
      border-radius: 4px;
      padding: 7px 11px;
      background: transparent;
      color: white;
    }
    .action:hover { background: #2b353c; }
    .action.primary { background: #f5f4ef; color: var(--ink); border-color: #f5f4ef; }
    .action.primary:hover { background: white; }

    .workspace { min-height: 0; display: grid; grid-template-columns: var(--sidebar) minmax(0, 1fr); }
    .sidebar {
      min-height: 0;
      display: grid;
      grid-template-rows: auto auto minmax(0, 1fr);
      background: #e8e5de;
      border-right: 1px solid var(--line-dark);
    }

    .tabs { display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid var(--line-dark); }
    .tab {
      min-height: 48px;
      border: 0;
      border-right: 1px solid var(--line);
      background: transparent;
      color: var(--muted);
      font-weight: 700;
    }
    .tab:last-child { border-right: 0; }
    .tab.active { color: var(--ink); background: var(--surface); box-shadow: inset 0 -3px 0 var(--red); }

    .filters {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .filters .wide { grid-column: 1 / -1; }
    input, select, textarea {
      width: 100%;
      color: var(--ink);
      background: var(--surface);
      border: 1px solid var(--line-dark);
      border-radius: 4px;
    }
    input, select { height: 36px; padding: 0 9px; }
    textarea { min-height: 78px; padding: 9px; resize: vertical; line-height: 1.5; }

    .sample-list { overflow: auto; padding: 6px; }
    .sample-item {
      width: 100%;
      min-height: 62px;
      display: grid;
      grid-template-columns: 5px minmax(0, 1fr) auto;
      gap: 9px;
      align-items: stretch;
      margin: 0 0 4px;
      padding: 0;
      border: 1px solid transparent;
      border-radius: 4px;
      background: transparent;
      text-align: left;
      color: var(--ink);
    }
    .sample-item:hover { border-color: var(--line-dark); background: #f1efe9; }
    .sample-item.active { border-color: var(--ink); background: var(--surface); }
    .sample-marker { background: var(--line-dark); border-radius: 3px 0 0 3px; }
    .sample-item.reviewed .sample-marker { background: var(--green); }
    .sample-copy { min-width: 0; padding: 9px 0; }
    .sample-id { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 700; }
    .sample-preview { margin-top: 5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }
    .sample-meta { padding: 9px 8px 0 0; color: var(--muted); font-size: 11px; text-transform: uppercase; }

    .main {
      min-width: 0;
      min-height: 0;
      overflow: auto;
      background: var(--surface);
    }
    .empty { height: 100%; display: grid; place-items: center; color: var(--muted); }
    .review { min-height: 100%; display: grid; grid-template-rows: auto auto auto 1fr; }

    .record-head {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      align-items: center;
      gap: 12px;
      padding: 15px 22px;
      border-bottom: 1px solid var(--line);
    }
    .record-title { min-width: 0; }
    .record-title h2 { margin: 0 0 5px; font: 700 18px/1.2 "IBM Plex Serif", "Noto Serif CJK SC", SimSun, serif; }
    .record-subtitle { display: flex; flex-wrap: wrap; gap: 6px 12px; color: var(--muted); font-size: 12px; }
    .record-nav { display: grid; grid-template-columns: 38px 38px; gap: 6px; }
    .icon-button {
      width: 38px; height: 36px; border: 1px solid var(--line-dark); border-radius: 4px;
      background: var(--surface); color: var(--ink); font-size: 18px;
    }
    .icon-button:hover { background: var(--paper); }
    .icon-button:disabled { opacity: .35; cursor: default; }

    .image-stage {
      height: clamp(180px, 34vh, 390px);
      display: grid;
      place-items: center;
      padding: 18px 24px;
      overflow: hidden;
      background-color: #d9d7d1;
      background-image: linear-gradient(45deg, #cfcdc6 25%, transparent 25%),
        linear-gradient(-45deg, #cfcdc6 25%, transparent 25%),
        linear-gradient(45deg, transparent 75%, #cfcdc6 75%),
        linear-gradient(-45deg, transparent 75%, #cfcdc6 75%);
      background-size: 20px 20px;
      background-position: 0 0, 0 10px, 10px -10px, -10px 0;
      border-bottom: 1px solid var(--line-dark);
    }
    .image-stage img {
      display: block;
      max-width: 100%;
      max-height: 100%;
      min-height: 64px;
      object-fit: contain;
      image-rendering: auto;
      background: white;
      border: 1px solid #77736c;
      box-shadow: 0 5px 18px rgba(23, 32, 40, .18);
    }
    .image-error { display: none; padding: 18px; border: 1px solid var(--red); color: var(--red); background: var(--red-soft); }

    .comparison {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      border-bottom: 1px solid var(--line-dark);
    }
    .text-panel { min-width: 0; padding: 14px 18px 16px; border-right: 1px solid var(--line); }
    .text-panel:last-child { border-right: 0; }
    .text-panel header { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 9px; }
    .text-panel strong { font-size: 11px; text-transform: uppercase; color: var(--muted); }
    .distance { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
    .ocr-text {
      min-height: 56px;
      font-family: "Noto Sans Arabic", "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
      font-size: 20px;
      line-height: 1.65;
      overflow-wrap: anywhere;
      unicode-bidi: plaintext;
      direction: auto;
    }
    .text-panel.gt { background: #f3f1eb; }
    .text-panel.b1 { background: #f8eee9; }
    .text-panel.m3 { background: #eaf3ed; }

    .decision-area { padding: 18px 22px 26px; }
    .decision-area h3 { margin: 0 0 12px; font-size: 13px; }
    .decision-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; }
    .choice {
      min-height: 56px;
      padding: 8px 10px;
      border: 1px solid var(--line-dark);
      border-radius: 4px;
      background: var(--surface);
      color: var(--ink);
      font-weight: 700;
      line-height: 1.25;
    }
    .choice:hover { background: var(--paper); }
    .choice.selected { color: white; border-color: var(--ink); background: var(--ink); }
    .choice[data-tone="green"].selected { background: var(--green); border-color: var(--green); }
    .choice[data-tone="red"].selected { background: var(--red); border-color: var(--red); }
    .choice[data-tone="amber"].selected { background: var(--amber); border-color: var(--amber); }
    .choice[data-tone="blue"].selected { background: var(--blue); border-color: var(--blue); }
    .notes-label { display: block; margin: 16px 0 7px; color: var(--muted); font-size: 12px; font-weight: 700; }
    .decision-footer { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-top: 10px; }
    .save-state { color: var(--green); font-size: 12px; }
    .clear-button { border: 0; background: transparent; color: var(--red); padding: 7px 0; }

    .toast {
      position: fixed;
      right: 18px;
      bottom: 18px;
      max-width: min(420px, calc(100vw - 36px));
      padding: 11px 14px;
      border-radius: 4px;
      color: white;
      background: var(--ink);
      box-shadow: 0 8px 28px rgba(0,0,0,.25);
      opacity: 0;
      transform: translateY(8px);
      pointer-events: none;
      transition: opacity .16s ease, transform .16s ease;
    }
    .toast.show { opacity: 1; transform: translateY(0); }

    @media (max-width: 900px) {
      body { overflow: auto; }
      .app { height: auto; min-height: 100%; grid-template-rows: auto auto; }
      .topbar { grid-template-columns: 1fr; padding: 12px; }
      .top-actions { flex-wrap: wrap; }
      .workspace { grid-template-columns: 1fr; }
      .sidebar { height: 46vh; border-right: 0; border-bottom: 1px solid var(--line-dark); }
      .comparison { grid-template-columns: 1fr; }
      .text-panel { border-right: 0; border-bottom: 1px solid var(--line); }
      .decision-grid { grid-template-columns: 1fr 1fr; }
      .brand span { display: none; }
    }

    @media (max-width: 520px) {
      .decision-grid { grid-template-columns: 1fr; }
      .record-head { padding: 12px; }
      .decision-area { padding: 14px 12px 22px; }
      .image-stage { padding: 12px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="brand">
        <h1>Stage 6 Error Review</h1>
        <span>B1 Epoch 20 vs SOAR-SVTR Epoch 34</span>
      </div>
      <div class="top-actions">
        <div class="status-chip" id="progress">0 / 0</div>
        <button class="action" id="importButton" title="导入之前导出的审核 CSV">↑ 导入</button>
        <input type="file" id="importFile" accept=".csv,text/csv" hidden>
        <button class="action primary" id="exportButton" title="导出当前标签和备注">↓ 导出当前 CSV</button>
      </div>
    </header>

    <div class="workspace">
      <aside class="sidebar">
        <div class="tabs">
          <button class="tab active" data-dataset="ed1">ED=1 审核 <span id="ed1Count"></span></button>
          <button class="tab" data-dataset="ug">UG Recovered <span id="ugCount"></span></button>
        </div>
        <div class="filters">
          <input class="wide" id="search" type="search" placeholder="搜索 ID、GT 或预测">
          <select id="languageFilter" title="语言筛选">
            <option value="all">全部语言</option>
            <option value="zh">中文</option>
            <option value="ug">维吾尔语</option>
            <option value="kk">哈萨克语</option>
          </select>
          <select id="statusFilter" title="审核状态筛选">
            <option value="all">全部状态</option>
            <option value="unreviewed">未审核</option>
            <option value="reviewed">已审核</option>
          </select>
          <select class="wide" id="migrationFilter" title="迁移类型筛选">
            <option value="all">全部迁移类型</option>
            <option value="WW">WW · 两者都错</option>
            <option value="CW">CW · B1 对、M3 错</option>
          </select>
        </div>
        <div class="sample-list" id="sampleList"></div>
      </aside>

      <main class="main" id="main">
        <div class="empty">没有符合当前筛选条件的样本</div>
      </main>
    </div>
  </div>
  <div class="toast" id="toast"></div>

  <script>
    const payload = __DATA__;
    const STORAGE_KEY = `soar-svtr-stage6-review-v1-${payload.source_sha256.ed1.slice(0, 12)}-${payload.source_sha256.ug.slice(0, 12)}`;
    const definitions = {
      ed1: {
        title: 'ED=1 错误来源',
        decisionField: 'manual_decision',
        notesField: 'manual_notes',
        filename: 'ed1_manual_review.csv',
        choices: [
          ['genuine_ocr_error', '真实 OCR 错误', 'green'],
          ['gt_annotation_error', 'GT 标注错误', 'red'],
          ['normalization_issue', '规范化问题', 'blue'],
          ['ambiguous_image', '图像不可判', 'amber']
        ]
      },
      ug: {
        title: '维语恢复类型',
        decisionField: 'manual_category',
        notesField: 'manual_notes',
        filename: 'recovered_ug_review.csv',
        choices: [
          ['order_related', '顺序相关', 'blue'],
          ['joining_form', '连写形态', 'green'],
          ['confusable_character', '形近字符', 'amber'],
          ['other', '其他', 'red']
        ]
      }
    };

    let activeDataset = 'ed1';
    let activeKey = null;
    let filtered = [];
    let saveTimer = null;

    function recordKey(dataset, row) {
      return `${dataset}:${row.language}:${row.sample_id}`;
    }

    function loadState() {
      let saved = {};
      try { saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || '{}'); } catch (_) {}
      for (const dataset of ['ed1', 'ug']) {
        const def = definitions[dataset];
        for (const row of payload[dataset]) {
          const prior = saved[recordKey(dataset, row)];
          if (!prior) continue;
          row[def.decisionField] = prior.decision || '';
          row[def.notesField] = prior.notes || '';
        }
      }
    }

    function persist() {
      const state = {};
      for (const dataset of ['ed1', 'ug']) {
        const def = definitions[dataset];
        for (const row of payload[dataset]) {
          state[recordKey(dataset, row)] = {
            decision: row[def.decisionField] || '',
            notes: row[def.notesField] || ''
          };
        }
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
      const label = document.getElementById('saveState');
      if (label) label.textContent = '已自动保存';
      updateProgress();
      renderList();
    }

    function schedulePersist() {
      const label = document.getElementById('saveState');
      if (label) label.textContent = '保存中';
      clearTimeout(saveTimer);
      saveTimer = setTimeout(persist, 180);
    }

    function isReviewed(row, dataset = activeDataset) {
      return Boolean((row[definitions[dataset].decisionField] || '').trim());
    }

    function applyFilters() {
      const query = document.getElementById('search').value.trim().toLocaleLowerCase();
      const language = document.getElementById('languageFilter').value;
      const status = document.getElementById('statusFilter').value;
      const migration = document.getElementById('migrationFilter').value;
      filtered = payload[activeDataset].filter(row => {
        if (language !== 'all' && row.language !== language) return false;
        if (migration !== 'all' && row.migration !== migration) return false;
        if (status === 'reviewed' && !isReviewed(row)) return false;
        if (status === 'unreviewed' && isReviewed(row)) return false;
        if (query) {
          const haystack = [row.sample_id, row.gt, row.pred_b1, row.pred_m3].join('\n').toLocaleLowerCase();
          if (!haystack.includes(query)) return false;
        }
        return true;
      });
      if (!filtered.some(row => recordKey(activeDataset, row) === activeKey)) {
        activeKey = filtered.length ? recordKey(activeDataset, filtered[0]) : null;
      }
      renderList();
      renderMain();
      updateProgress();
    }

    function renderList() {
      const list = document.getElementById('sampleList');
      list.innerHTML = '';
      for (const row of filtered) {
        const key = recordKey(activeDataset, row);
        const button = document.createElement('button');
        button.className = `sample-item ${key === activeKey ? 'active' : ''} ${isReviewed(row) ? 'reviewed' : ''}`;
        button.innerHTML = `
          <span class="sample-marker"></span>
          <span class="sample-copy">
            <span class="sample-id">${escapeHtml(row.sample_id)}</span>
            <span class="sample-preview" dir="auto">${escapeHtml(row.gt)}</span>
          </span>
          <span class="sample-meta">${escapeHtml(row.language)} · ${escapeHtml(row.migration)}</span>`;
        button.addEventListener('click', () => {
          activeKey = key;
          renderList();
          renderMain();
        });
        list.appendChild(button);
      }
    }

    function activeRecord() {
      return payload[activeDataset].find(row => recordKey(activeDataset, row) === activeKey) || null;
    }

    function renderMain() {
      const main = document.getElementById('main');
      const row = activeRecord();
      if (!row) {
        main.innerHTML = '<div class="empty">没有符合当前筛选条件的样本</div>';
        return;
      }
      const def = definitions[activeDataset];
      const index = filtered.findIndex(item => recordKey(activeDataset, item) === activeKey);
      const choices = def.choices.map(([value, label, tone]) => `
        <button class="choice ${row[def.decisionField] === value ? 'selected' : ''}"
          data-value="${value}" data-tone="${tone}">${label}</button>`).join('');
      main.innerHTML = `
        <article class="review">
          <header class="record-head">
            <div class="record-title">
              <h2>${escapeHtml(row.sample_id)}</h2>
              <div class="record-subtitle">
                <span>${escapeHtml(row.language.toUpperCase())}</span>
                <span>${escapeHtml(row.migration)}</span>
                <span>B1 ED ${escapeHtml(row.ed_b1)}</span>
                <span>M3 ED ${escapeHtml(row.ed_m3)}</span>
                <span>${index + 1} / ${filtered.length}</span>
              </div>
            </div>
            <nav class="record-nav">
              <button class="icon-button" id="previous" title="上一条" ${index <= 0 ? 'disabled' : ''}>←</button>
              <button class="icon-button" id="next" title="下一条" ${index >= filtered.length - 1 ? 'disabled' : ''}>→</button>
            </nav>
          </header>
          <div class="image-stage">
            <img src="${escapeAttribute(row.review_image_src)}" alt="${escapeAttribute(row.sample_id)}" id="sampleImage">
            <div class="image-error" id="imageError">图片无法加载：${escapeHtml(row.review_image_src)}</div>
          </div>
          <section class="comparison">
            ${textPanel('GT', row.gt, '', 'gt')}
            ${textPanel('B1', row.pred_b1, `ED ${row.ed_b1}`, 'b1')}
            ${textPanel('M3', row.pred_m3, `ED ${row.ed_m3}`, 'm3')}
          </section>
          <section class="decision-area">
            <h3>${def.title}</h3>
            <div class="decision-grid">${choices}</div>
            <label class="notes-label" for="notes">审核备注</label>
            <textarea id="notes" placeholder="记录图像证据、具体错字或判断理由">${escapeHtml(row[def.notesField] || '')}</textarea>
            <div class="decision-footer">
              <span class="save-state" id="saveState">已自动保存</span>
              <button class="clear-button" id="clearDecision">清除此条审核</button>
            </div>
          </section>
        </article>`;

      document.getElementById('previous').addEventListener('click', () => navigate(-1));
      document.getElementById('next').addEventListener('click', () => navigate(1));
      document.getElementById('sampleImage').addEventListener('error', event => {
        event.currentTarget.style.display = 'none';
        document.getElementById('imageError').style.display = 'block';
      });
      document.querySelectorAll('.choice').forEach(button => {
        button.addEventListener('click', () => {
          row[def.decisionField] = button.dataset.value;
          persist();
          if (document.getElementById('statusFilter').value === 'unreviewed') {
            applyFilters();
            return;
          }
          renderMain();
          const currentIndex = filtered.findIndex(item => recordKey(activeDataset, item) === activeKey);
          if (currentIndex < filtered.length - 1) {
            setTimeout(() => navigate(1), 80);
          }
        });
      });
      document.getElementById('notes').addEventListener('input', event => {
        row[def.notesField] = event.target.value;
        schedulePersist();
      });
      document.getElementById('clearDecision').addEventListener('click', () => {
        row[def.decisionField] = '';
        row[def.notesField] = '';
        persist();
        applyFilters();
      });
    }

    function textPanel(label, value, distance, kind) {
      return `<div class="text-panel ${kind}">
        <header><strong>${label}</strong><span class="distance">${distance}</span></header>
        <div class="ocr-text" dir="auto">${escapeHtml(value)}</div>
      </div>`;
    }

    function navigate(delta) {
      const index = filtered.findIndex(row => recordKey(activeDataset, row) === activeKey);
      const target = filtered[index + delta];
      if (!target) return;
      activeKey = recordKey(activeDataset, target);
      renderList();
      renderMain();
      document.getElementById('main').scrollTop = 0;
      document.querySelector('.sample-item.active')?.scrollIntoView({block: 'nearest'});
    }

    function updateProgress() {
      const rows = payload[activeDataset];
      const reviewed = rows.filter(row => isReviewed(row)).length;
      document.getElementById('progress').textContent = `${reviewed} / ${rows.length}`;
      document.getElementById('ed1Count').textContent = `(${payload.ed1.filter(row => isReviewed(row, 'ed1')).length}/${payload.ed1.length})`;
      document.getElementById('ugCount').textContent = `(${payload.ug.filter(row => isReviewed(row, 'ug')).length}/${payload.ug.length})`;
    }

    function csvEscape(value) {
      const text = value == null ? '' : String(value);
      return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
    }

    function exportCsv() {
      persist();
      const rows = payload[activeDataset];
      const columns = payload.columns[activeDataset];
      const lines = [columns.map(csvEscape).join(',')];
      for (const row of rows) lines.push(columns.map(column => csvEscape(row[column] || '')).join(','));
      const blob = new Blob(['\ufeff' + lines.join('\r\n') + '\r\n'], {type: 'text/csv;charset=utf-8'});
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = definitions[activeDataset].filename;
      link.click();
      URL.revokeObjectURL(url);
      toast(`已导出 ${link.download}`);
    }

    function parseCsv(text) {
      const rows = [];
      let row = [], field = '', quoted = false;
      for (let index = 0; index < text.length; index++) {
        const char = text[index];
        if (quoted) {
          if (char === '"' && text[index + 1] === '"') { field += '"'; index++; }
          else if (char === '"') quoted = false;
          else field += char;
        } else if (char === '"') quoted = true;
        else if (char === ',') { row.push(field); field = ''; }
        else if (char === '\n') { row.push(field.replace(/\r$/, '')); rows.push(row); row = []; field = ''; }
        else field += char;
      }
      if (field || row.length) { row.push(field); rows.push(row); }
      const header = rows.shift().map(value => value.replace(/^\ufeff/, ''));
      return rows.filter(values => values.some(Boolean)).map(values => Object.fromEntries(header.map((key, i) => [key, values[i] || ''])));
    }

    function importCsv(file) {
      const reader = new FileReader();
      reader.onload = () => {
        try {
          const incoming = parseCsv(reader.result);
          const dataset = incoming[0]?.manual_category !== undefined ? 'ug' : 'ed1';
          const def = definitions[dataset];
          const index = new Map(payload[dataset].map(row => [recordKey(dataset, row), row]));
          let merged = 0;
          for (const row of incoming) {
            const target = index.get(recordKey(dataset, row));
            if (!target) continue;
            target[def.decisionField] = row[def.decisionField] || '';
            target[def.notesField] = row[def.notesField] || '';
            merged++;
          }
          if (merged !== payload[dataset].length) throw new Error(`样本数不匹配：${merged}/${payload[dataset].length}`);
          activeDataset = dataset;
          document.querySelectorAll('.tab').forEach(tab => tab.classList.toggle('active', tab.dataset.dataset === dataset));
          configureDatasetFilters();
          persist();
          applyFilters();
          toast(`已导入 ${merged} 条审核记录`);
        } catch (error) { toast(`导入失败：${error.message}`); }
      };
      reader.readAsText(file, 'utf-8');
    }

    function configureDatasetFilters() {
      const language = document.getElementById('languageFilter');
      const migration = document.getElementById('migrationFilter');
      if (activeDataset === 'ug') {
        language.value = 'ug';
        language.disabled = true;
        migration.value = 'all';
        migration.disabled = true;
      } else {
        language.disabled = false;
        migration.disabled = false;
      }
    }

    function toast(message) {
      const element = document.getElementById('toast');
      element.textContent = message;
      element.classList.add('show');
      setTimeout(() => element.classList.remove('show'), 2200);
    }

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
    }
    function escapeAttribute(value) { return escapeHtml(value); }

    document.querySelectorAll('.tab').forEach(tab => tab.addEventListener('click', () => {
      activeDataset = tab.dataset.dataset;
      activeKey = null;
      document.querySelectorAll('.tab').forEach(item => item.classList.toggle('active', item === tab));
      configureDatasetFilters();
      applyFilters();
    }));
    ['search', 'languageFilter', 'statusFilter', 'migrationFilter'].forEach(id => {
      document.getElementById(id).addEventListener(id === 'search' ? 'input' : 'change', applyFilters);
    });
    document.getElementById('exportButton').addEventListener('click', exportCsv);
    document.getElementById('importButton').addEventListener('click', () => document.getElementById('importFile').click());
    document.getElementById('importFile').addEventListener('change', event => {
      if (event.target.files[0]) importCsv(event.target.files[0]);
      event.target.value = '';
    });
    document.addEventListener('keydown', event => {
      if (event.target.matches('input, textarea, select')) return;
      if (event.key === 'ArrowLeft') navigate(-1);
      if (event.key === 'ArrowRight') navigate(1);
      const choice = definitions[activeDataset].choices[Number(event.key) - 1];
      if (choice) document.querySelector(`.choice[data-value="${choice[0]}"]`)?.click();
    });
    window.addEventListener('beforeunload', () => {
      clearTimeout(saveTimer);
      persist();
    });

    loadState();
    configureDatasetFilters();
    applyFilters();
  </script>
</body>
</html>
'''


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {path}")
        return list(reader.fieldnames), rows


def attach_image_sources(root: Path, output: Path, rows: list[dict[str, str]]) -> None:
    target_root = root / "01_data_preparation/real_line_dataset_eval_reviewed"
    for row in rows:
        image = target_root / row["image_path"]
        if not image.is_file():
            raise FileNotFoundError(image)
        row["review_image_src"] = Path(os.path.relpath(image, output.parent)).as_posix()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage6-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    stage6 = (args.stage6_dir or root / "04_model_training/eval_reports/stage6_error_migration_v1").resolve()
    output = (args.output or stage6 / "stage6_manual_review.html").resolve()
    sources = {
        "ed1": stage6 / "ed1_manual_review.csv",
        "ug": stage6 / "recovered_ug_review.csv",
    }
    columns: dict[str, list[str]] = {}
    rows: dict[str, list[dict[str, str]]] = {}
    hashes: dict[str, str] = {}
    for name, path in sources.items():
        columns[name], rows[name] = read_csv(path)
        attach_image_sources(root, output, rows[name])
        hashes[name] = sha256(path)
    if len(rows["ed1"]) != 64 or len(rows["ug"]) != 6:
        raise ValueError(f"Unexpected review membership: ED1={len(rows['ed1'])}, UG={len(rows['ug'])}")
    payload = {
        "protocol": "stage6_error_migration_v1",
        "columns": columns,
        "source_sha256": hashes,
        "ed1": rows["ed1"],
        "ug": rows["ug"],
    }
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "STAGE6_REVIEW_HTML_READY",
                "output": str(output),
                "ed1_rows": len(rows["ed1"]),
                "ug_recovered_rows": len(rows["ug"]),
                "source_sha256": hashes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
