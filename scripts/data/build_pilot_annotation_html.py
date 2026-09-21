import argparse
import csv
import html
import json
from pathlib import Path


KEEP_FIELDS = [
    "candidate_id",
    "language",
    "split",
    "source_id",
    "source_image",
    "image_path_resolved",
    "crop_path_rel",
    "overlay_path_rel",
    "text",
    "text_length",
    "bbox",
    "bbox_aspect",
    "review_status",
    "review_note",
]


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            clean = {field: row.get(field, "") for field in KEEP_FIELDS}
            clean["review_status"] = clean["review_status"] or "pending"
            rows.append(clean)
        return rows


def default_paths():
    script_path = Path(__file__).resolve()
    svtr_root = script_path.parents[2]
    real_lines_dir = svtr_root / "01_data_preparation" / "real_lines"
    input_csv = real_lines_dir / "pilot_annotation_450.csv"
    output_html = real_lines_dir / "pilot_review" / "annotate.html"
    return input_csv, output_html


def build_html(rows):
    data_json = json.dumps(rows, ensure_ascii=False)
    escaped_data = data_json.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Real Line Pilot Annotation</title>
  <style>
    :root {{
      --ink: #17201a;
      --muted: #64706a;
      --line: #d7ded8;
      --paper: #f7f5ef;
      --panel: #ffffff;
      --pass: #177245;
      --recrop: #b15b00;
      --drop: #a52323;
      --pending: #5d6670;
      --accent: #1f5f8b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--paper);
      font-family: Georgia, "Times New Roman", "Microsoft YaHei", serif;
      letter-spacing: 0;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247, 245, 239, .96);
      border-bottom: 1px solid var(--line);
      padding: 14px 22px 12px;
      backdrop-filter: blur(8px);
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 24px;
      font-weight: 700;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: 1fr auto auto auto;
      gap: 10px;
      align-items: center;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .badge {{
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 4px 7px;
      background: var(--panel);
    }}
    select, input, button, textarea {{
      font: inherit;
      letter-spacing: 0;
    }}
    select, input {{
      height: 34px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--panel);
      color: var(--ink);
      padding: 0 8px;
    }}
    button {{
      border: 1px solid var(--line);
      border-radius: 4px;
      background: var(--panel);
      color: var(--ink);
      min-height: 34px;
      padding: 7px 11px;
      cursor: pointer;
    }}
    button:hover {{ border-color: var(--ink); }}
    .export {{
      color: white;
      border-color: var(--accent);
      background: var(--accent);
    }}
    main {{
      max-width: 1380px;
      margin: 0 auto;
      padding: 18px 20px 36px;
    }}
    .item {{
      display: grid;
      grid-template-columns: minmax(360px, 1.05fr) minmax(280px, .95fr);
      gap: 14px;
      margin-bottom: 16px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      overflow: hidden;
    }}
    .image-pane {{
      padding: 12px;
      border-right: 1px solid var(--line);
    }}
    .crop-wrap {{
      min-height: 96px;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #ece9df;
      border: 1px solid #e3ded2;
      border-radius: 4px;
      padding: 8px;
    }}
    .crop-wrap img {{
      max-width: 100%;
      max-height: 150px;
      display: block;
    }}
    details {{
      margin-top: 10px;
    }}
    summary {{
      cursor: pointer;
      color: var(--muted);
      font-size: 13px;
    }}
    .overlay {{
      margin-top: 8px;
      max-width: 100%;
      max-height: 260px;
      background: #eee;
      border: 1px solid var(--line);
    }}
    .meta-pane {{
      padding: 12px 14px;
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      gap: 10px;
    }}
    .topline {{
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
    }}
    .id {{
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      color: var(--muted);
    }}
    .lang {{
      min-width: 34px;
      text-align: center;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 3px 6px;
      font-weight: 700;
      text-transform: uppercase;
      background: #f4f2ea;
    }}
    .status {{
      border-radius: 4px;
      padding: 3px 7px;
      color: white;
      font-size: 12px;
      background: var(--pending);
    }}
    .status.pass {{ background: var(--pass); }}
    .status.recrop {{ background: var(--recrop); }}
    .status.drop {{ background: var(--drop); }}
    .label {{
      min-height: 52px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 4px;
      background: #fbfaf6;
      font-size: 20px;
      line-height: 1.4;
      overflow-wrap: anywhere;
    }}
    .small {{
      font-size: 12px;
      color: var(--muted);
      line-height: 1.5;
      font-family: Consolas, "Courier New", monospace;
    }}
    .actions {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }}
    .actions button {{
      font-weight: 700;
    }}
    .actions .pass.active, .actions .pass:hover {{
      border-color: var(--pass);
      color: white;
      background: var(--pass);
    }}
    .actions .recrop.active, .actions .recrop:hover {{
      border-color: var(--recrop);
      color: white;
      background: var(--recrop);
    }}
    .actions .drop.active, .actions .drop:hover {{
      border-color: var(--drop);
      color: white;
      background: var(--drop);
    }}
    textarea {{
      width: 100%;
      min-height: 54px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 8px;
      background: #fbfaf6;
    }}
    .empty {{
      border: 1px dashed var(--line);
      padding: 40px;
      text-align: center;
      color: var(--muted);
      background: var(--panel);
    }}
    @media (max-width: 900px) {{
      .toolbar {{ grid-template-columns: 1fr 1fr; }}
      .item {{ grid-template-columns: 1fr; }}
      .image-pane {{ border-right: 0; border-bottom: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Real Meme Line Pilot Annotation</h1>
    <div class="toolbar">
      <div class="stats" id="stats"></div>
      <select id="langFilter">
        <option value="all">all languages</option>
        <option value="zh">zh</option>
        <option value="ug">ug</option>
        <option value="kk">kk</option>
      </select>
      <select id="statusFilter">
        <option value="all">all statuses</option>
        <option value="pending">pending</option>
        <option value="pass">pass</option>
        <option value="recrop">recrop</option>
        <option value="drop">drop</option>
      </select>
      <button class="export" id="exportBtn">Export CSV</button>
    </div>
  </header>
  <main id="list"></main>
  <script>
    const rows = {escaped_data};
    const storeKey = "svtrv2-real-line-annotation-train-v3";
    let state = loadState();

    function loadState() {{
      try {{
        return JSON.parse(localStorage.getItem(storeKey) || "{{}}");
      }} catch (err) {{
        return {{}};
      }}
    }}

    function saveState() {{
      localStorage.setItem(storeKey, JSON.stringify(state));
    }}

    function merged(row) {{
      const saved = state[row.candidate_id] || {{}};
      return {{
        ...row,
        review_status: saved.review_status || row.review_status || "pending",
        review_note: saved.review_note ?? row.review_note ?? ""
      }};
    }}

    function setStatus(id, status, button) {{
      state[id] = {{ ...(state[id] || {{}}), review_status: status }};
      saveState();
      const item = button ? button.closest(".item") : null;
      if (item) {{
        item.dataset.status = status;
        const badge = item.querySelector(".status");
        if (badge) {{
          badge.className = `status ${{status}}`;
          badge.textContent = status;
        }}
        item.querySelectorAll(".actions button").forEach(btn => btn.classList.remove("active"));
        button.classList.add("active");
        const activeStatus = document.getElementById("statusFilter").value;
        if (activeStatus !== "all" && activeStatus !== status) {{
          item.remove();
        }}
      }} else {{
        render();
      }}
      updateStats();
    }}

    function setNote(id, note) {{
      state[id] = {{ ...(state[id] || {{}}), review_note: note }};
      saveState();
      updateStats();
    }}

    function escapeHtml(value) {{
      return String(value ?? "").replace(/[&<>"']/g, ch => ({{
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
      }}[ch]));
    }}

    function dirFor(lang) {{
      return lang === "ug" ? "rtl" : "ltr";
    }}

    function filteredRows() {{
      const lang = document.getElementById("langFilter").value;
      const status = document.getElementById("statusFilter").value;
      return rows.map(merged).filter(row => {{
        return (lang === "all" || row.language === lang) &&
               (status === "all" || row.review_status === status);
      }});
    }}

    function updateStats() {{
      const counts = {{pending: 0, pass: 0, recrop: 0, drop: 0}};
      const langCounts = {{}};
      rows.map(merged).forEach(row => {{
        counts[row.review_status] = (counts[row.review_status] || 0) + 1;
        langCounts[row.language] = (langCounts[row.language] || 0) + 1;
      }});
      document.getElementById("stats").innerHTML = `
        <span class="badge">total: ${{rows.length}}</span>
        <span class="badge">pending: ${{counts.pending || 0}}</span>
        <span class="badge">pass: ${{counts.pass || 0}}</span>
        <span class="badge">recrop: ${{counts.recrop || 0}}</span>
        <span class="badge">drop: ${{counts.drop || 0}}</span>
        <span class="badge">zh: ${{langCounts.zh || 0}}</span>
        <span class="badge">ug: ${{langCounts.ug || 0}}</span>
        <span class="badge">kk: ${{langCounts.kk || 0}}</span>
      `;
    }}

    function render() {{
      updateStats();
      const list = document.getElementById("list");
      const visible = filteredRows();
      if (!visible.length) {{
        list.innerHTML = `<div class="empty">No samples match the current filters.</div>`;
        return;
      }}
      list.innerHTML = visible.map(row => `
        <article class="item" data-id="${{escapeHtml(row.candidate_id)}}" data-status="${{escapeHtml(row.review_status)}}">
          <section class="image-pane">
            <div class="crop-wrap">
              <img src="${{escapeHtml(row.crop_path_rel)}}" alt="${{escapeHtml(row.candidate_id)}} crop">
            </div>
            <details>
              <summary>show source overlay</summary>
              <img class="overlay" src="${{escapeHtml(row.overlay_path_rel)}}" alt="${{escapeHtml(row.candidate_id)}} overlay">
            </details>
          </section>
          <section class="meta-pane">
            <div class="topline">
              <span class="lang">${{escapeHtml(row.language)}}</span>
              <span class="status ${{escapeHtml(row.review_status)}}">${{escapeHtml(row.review_status)}}</span>
              <span class="id">${{escapeHtml(row.candidate_id)}}</span>
            </div>
            <div class="label" dir="${{dirFor(row.language)}}">${{escapeHtml(row.text)}}</div>
            <div class="small">
              source_id=${{escapeHtml(row.source_id)}}<br>
              split=${{escapeHtml(row.split)}} | length=${{escapeHtml(row.text_length)}} | aspect=${{escapeHtml(row.bbox_aspect)}}<br>
              bbox=${{escapeHtml(row.bbox)}}
            </div>
            <div>
              <div class="actions">
                <button class="pass ${{row.review_status === "pass" ? "active" : ""}}" onclick="setStatus('${{escapeHtml(row.candidate_id)}}','pass', this)">Pass</button>
                <button class="recrop ${{row.review_status === "recrop" ? "active" : ""}}" onclick="setStatus('${{escapeHtml(row.candidate_id)}}','recrop', this)">Recrop</button>
                <button class="drop ${{row.review_status === "drop" ? "active" : ""}}" onclick="setStatus('${{escapeHtml(row.candidate_id)}}','drop', this)">Drop</button>
              </div>
              <textarea placeholder="review note" oninput="setNote('${{escapeHtml(row.candidate_id)}}', this.value)">${{escapeHtml(row.review_note)}}</textarea>
            </div>
          </section>
        </article>
      `).join("");
    }}

    function csvEscape(value) {{
      const text = String(value ?? "");
      if (/[",\\n\\r\\t]/.test(text)) {{
        return '"' + text.replace(/"/g, '""') + '"';
      }}
      return text;
    }}

    function exportCsv() {{
      const fields = {json.dumps(KEEP_FIELDS, ensure_ascii=False)};
      const lines = [fields.join(",")];
      rows.map(merged).forEach(row => {{
        lines.push(fields.map(field => csvEscape(row[field])).join(","));
      }});
      const blob = new Blob([lines.join("\\n") + "\\n"], {{type: "text/csv;charset=utf-8"}});
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "pilot_annotation_450.reviewed.csv";
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }}

    document.getElementById("langFilter").addEventListener("change", render);
    document.getElementById("statusFilter").addEventListener("change", render);
    document.getElementById("exportBtn").addEventListener("click", exportCsv);
    render();
  </script>
</body>
</html>
"""


def main():
    default_input, default_output = default_paths()
    parser = argparse.ArgumentParser(description="Build a static HTML annotation page for real-line pilot review.")
    parser.add_argument("--input-csv", type=Path, default=default_input)
    parser.add_argument("--output-html", type=Path, default=default_output)
    args = parser.parse_args()

    rows = read_rows(args.input_csv)
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(build_html(rows), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_html": str(args.output_html)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()



