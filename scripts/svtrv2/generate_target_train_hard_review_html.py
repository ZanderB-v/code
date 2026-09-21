#!/usr/bin/env python3
"""Generate an offline review console for target-train HEM candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Target Train Hard Sample Review</title>
  <style>
    :root{--ink:#18211d;--muted:#68716c;--line:#cdd3cf;--paper:#f5f6f3;--panel:#fff;--green:#1f6b49;--amber:#a64b14;--red:#9e2f2f;--blue:#285b87;--violet:#6b4c8b}
    *{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Segoe UI","Noto Sans CJK SC",sans-serif;letter-spacing:0}
    button,input,select,textarea{font:inherit}.shell{min-height:100vh;display:grid;grid-template-rows:auto auto 1fr auto}
    header{display:flex;align-items:center;gap:22px;padding:15px 22px;background:#14231d;color:#fff;border-bottom:3px solid #c7d256}
    h1{font-family:Georgia,"Noto Serif CJK SC",serif;font-size:20px;margin:0;font-weight:700}.protocol{font-size:12px;color:#bdc9c2}
    .progress{margin-left:auto;display:flex;align-items:center;gap:10px;min-width:280px}.bar{height:8px;flex:1;background:#435149;overflow:hidden}.bar i{display:block;height:100%;background:#c7d256}.counter{font-variant-numeric:tabular-nums;font-size:13px}
    .toolbar{display:flex;align-items:center;gap:10px;padding:10px 22px;background:#e8ebe7;border-bottom:1px solid var(--line);flex-wrap:wrap}
    .seg{display:flex;border:1px solid #9aa49e;background:#fff}.seg button{border:0;border-right:1px solid #c5cbc7;padding:7px 11px;background:#fff;cursor:pointer}.seg button:last-child{border-right:0}.seg button.active{background:#243b31;color:#fff}
    .toolbar select,.toolbar input{height:34px;border:1px solid #9aa49e;background:#fff;padding:0 9px}.toolbar .spacer{flex:1}.cmd{height:34px;border:1px solid #647169;background:#fff;padding:0 12px;cursor:pointer}.cmd.primary{background:#285b87;color:#fff;border-color:#285b87}
    main{display:grid;grid-template-columns:minmax(0,1.45fr) minmax(350px,.55fr);min-height:0}
    .visual{padding:24px;border-right:1px solid var(--line);display:flex;flex-direction:column;gap:16px;min-width:0}.image-stage{height:clamp(280px,50vh,600px);display:grid;place-items:center;background:#dfe3df;border:1px solid #aeb7b1;position:relative;overflow:auto}.image-stage img{max-width:96%;max-height:92%;image-rendering:auto;box-shadow:0 8px 24px #18211d25;background:#fff}.empty{color:var(--red);font-weight:700}
    .identity{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.badge{border:1px solid #aab3ad;background:#fff;padding:4px 8px;font-size:12px}.badge.ed1{border-color:#cc9c3c}.badge.ed2{border-color:#ba6969}.sample-id{font-family:Consolas,monospace;font-size:12px;color:var(--muted);overflow-wrap:anywhere}
    .texts{display:grid;grid-template-columns:1fr 1fr;gap:12px}.text-panel{background:#fff;border:1px solid var(--line);padding:13px;min-height:110px}.text-panel label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;margin-bottom:8px}.line-text{font-family:"Noto Sans Arabic","Microsoft YaHei",sans-serif;font-size:22px;line-height:1.55;overflow-wrap:anywhere;unicode-bidi:plaintext}
    aside{background:#fff;padding:20px;overflow:auto}.meta{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--line);border-left:1px solid var(--line);margin-bottom:18px}.meta div{padding:9px;border-right:1px solid var(--line);border-bottom:1px solid var(--line)}.meta small{display:block;color:var(--muted);font-size:10px;text-transform:uppercase}.meta strong{font-size:14px;font-variant-numeric:tabular-nums}
    fieldset{border:0;padding:0;margin:0}legend{font-family:Georgia,"Noto Serif CJK SC",serif;font-weight:700;margin-bottom:10px}.choice{display:grid;grid-template-columns:18px 1fr;gap:9px;align-items:start;padding:10px;border:1px solid var(--line);border-bottom:0;cursor:pointer}.choice:last-of-type{border-bottom:1px solid var(--line)}.choice:has(input:checked){background:#edf5ef;border-left:4px solid var(--green)}.choice input{margin-top:3px}.choice b{display:block;font-size:14px}.choice span{font-size:11px;color:var(--muted)}
    textarea{width:100%;min-height:80px;margin-top:14px;border:1px solid #9aa49e;padding:9px;resize:vertical}.risk{margin-top:15px;padding:10px;background:#f4eee5;border-left:4px solid var(--amber);font-size:12px;overflow-wrap:anywhere}.risk.none{background:#edf4ef;border-color:var(--green)}
    footer{display:flex;align-items:center;gap:10px;padding:10px 22px;border-top:1px solid var(--line);background:#e8ebe7}.nav{width:38px;height:34px;border:1px solid #647169;background:#fff;font-size:20px;cursor:pointer}.footer-status{font-size:12px;color:var(--muted)}
    #importFile{display:none}@media(max-width:900px){main{grid-template-columns:1fr}.visual{border-right:0}.texts{grid-template-columns:1fr}.image-stage{height:36vh}aside{border-top:1px solid var(--line)}}
  </style>
</head>
<body><div class="shell">
  <header><div><h1>Target Train Hard Sample Review</h1><div class="protocol" id="protocol"></div></div><div class="progress"><div class="bar"><i id="bar"></i></div><span class="counter" id="counter"></span></div></header>
  <div class="toolbar">
    <div class="seg" id="langSeg"><button data-value="all" class="active">全部语言</button><button data-value="zh">ZH</button><button data-value="ug">UG</button><button data-value="kk">KK</button></div>
    <div class="seg" id="edSeg"><button data-value="all" class="active">ED 1+2</button><button data-value="1">ED 1</button><button data-value="2">ED 2</button></div>
    <select id="status"><option value="all">全部状态</option><option value="pending">未审核</option><option value="done">已审核</option></select>
    <input id="search" placeholder="样本 ID / 文本">
    <span class="spacer"></span>
    <button class="cmd" id="importBtn">导入 CSV</button><input type="file" id="importFile" accept=".csv">
    <button class="cmd primary" id="exportBtn">导出 CSV</button>
  </div>
  <main>
    <section class="visual">
      <div class="identity"><span class="badge" id="language"></span><span class="badge" id="ed"></span><span class="sample-id" id="sampleId"></span></div>
      <div class="image-stage" id="imageStage"><img id="image" alt="review crop"></div>
      <div class="texts"><div class="text-panel"><label>Ground Truth</label><div class="line-text" id="gt"></div></div><div class="text-panel"><label>M3 Prediction</label><div class="line-text" id="pred"></div></div></div>
    </section>
    <aside>
      <div class="meta"><div><small>Confidence</small><strong id="confidence"></strong></div><div><small>Image</small><strong id="dimensions"></strong></div><div><small>Contrast</small><strong id="contrast"></strong></div><div><small>Blur var</small><strong id="blur"></strong></div></div>
      <fieldset><legend>人工结论</legend>
        <label class="choice"><input type="radio" name="decision" value="genuine_ocr_error"><span><b>真实 OCR 错误</b><span>标签与图像一致，模型确实识别错误</span></span></label>
        <label class="choice"><input type="radio" name="decision" value="gt_annotation_error"><span><b>GT 标注错误</b><span>图像文字与标签不一致</span></span></label>
        <label class="choice"><input type="radio" name="decision" value="normalization_issue"><span><b>规范化问题</b><span>Unicode、空格或等价字符造成差异</span></span></label>
        <label class="choice"><input type="radio" name="decision" value="ambiguous_image"><span><b>图像不可判定</b><span>裁剪、模糊、遮挡或文字本身无法辨认</span></span></label>
      </fieldset>
      <textarea id="notes" placeholder="审核备注（可选）"></textarea>
      <div class="risk" id="risk"></div>
    </aside>
  </main>
  <footer><button class="nav" id="prev" title="上一条">←</button><button class="nav" id="next" title="下一条">→</button><span class="footer-status" id="footerStatus"></span></footer>
</div>
<script>
const payload=__DATA__;
const fields=payload.fields, rows=payload.rows;
const key=`target-train-hard-review-${payload.source_sha256.slice(0,16)}`;
let state={lang:'all',ed:'all',status:'all',search:'',index:0};
const saved=JSON.parse(localStorage.getItem(key)||'{}');
rows.forEach(r=>{const x=saved[r.sample_id];if(x){r.manual_decision=x.manual_decision||'';r.manual_notes=x.manual_notes||''}});
const $=id=>document.getElementById(id);
function esc(v){return String(v??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]))}
function filtered(){const q=state.search.toLowerCase();return rows.filter(r=>(state.lang==='all'||r.language===state.lang)&&(state.ed==='all'||String(r.ed)===state.ed)&&(state.status==='all'||(state.status==='done'&&r.manual_decision)||(state.status==='pending'&&!r.manual_decision))&&(!q||`${r.sample_id} ${r.gt_text} ${r.prediction}`.toLowerCase().includes(q)))}
function persist(){const x={};rows.forEach(r=>{if(r.manual_decision||r.manual_notes)x[r.sample_id]={manual_decision:r.manual_decision,manual_notes:r.manual_notes}});localStorage.setItem(key,JSON.stringify(x))}
function current(){const list=filtered();if(!list.length)return [list,null];state.index=Math.max(0,Math.min(state.index,list.length-1));return [list,list[state.index]]}
function render(){const [list,r]=current(),done=rows.filter(x=>x.manual_decision).length;$('bar').style.width=`${rows.length?100*done/rows.length:0}%`;$('counter').textContent=`${done} / ${rows.length}`;$('protocol').textContent=`${payload.protocol_id} · sample ${payload.source_sha256.slice(0,12)}`;
 if(!r){$('imageStage').innerHTML='<div class="empty">当前筛选没有样本</div>';$('footerStatus').textContent='0 条';return}
 if(!$('image')){$('imageStage').innerHTML='<img id="image" alt="review crop">'}
 $('language').textContent=r.language.toUpperCase();$('ed').textContent=`ED ${r.ed}`;$('ed').className=`badge ed${r.ed}`;$('sampleId').textContent=r.sample_id;$('image').src=r.review_image_src;$('gt').textContent=r.gt_text;$('pred').textContent=r.prediction;$('confidence').textContent=Number(r.confidence).toFixed(4);$('dimensions').textContent=`${r.width} × ${r.height}`;$('contrast').textContent=Number(r.contrast_span).toFixed(1);$('blur').textContent=Number(r.blur_variance).toFixed(1);
 document.querySelectorAll('input[name=decision]').forEach(x=>x.checked=x.value===r.manual_decision);$('notes').value=r.manual_notes||'';const risk=r.risk_flags||'';$('risk').className=`risk ${risk?'':'none'}`;$('risk').innerHTML=risk?`<b>自动风险：</b> ${esc(risk)}`:'自动过滤通过，仍需人工确认标签与图像一致';$('footerStatus').textContent=`筛选结果 ${state.index+1} / ${list.length} · ${r.image_path}`}
function move(delta){const list=filtered();if(list.length){state.index=Math.max(0,Math.min(state.index+delta,list.length-1));render()}}
document.querySelectorAll('#langSeg button').forEach(b=>b.onclick=()=>{state.lang=b.dataset.value;state.index=0;document.querySelectorAll('#langSeg button').forEach(x=>x.classList.toggle('active',x===b));render()});
document.querySelectorAll('#edSeg button').forEach(b=>b.onclick=()=>{state.ed=b.dataset.value;state.index=0;document.querySelectorAll('#edSeg button').forEach(x=>x.classList.toggle('active',x===b));render()});
$('status').onchange=e=>{state.status=e.target.value;state.index=0;render()};$('search').oninput=e=>{state.search=e.target.value;state.index=0;render()};$('prev').onclick=()=>move(-1);$('next').onclick=()=>move(1);
document.querySelectorAll('input[name=decision]').forEach(x=>x.onchange=()=>{const [list,r]=current();if(r){r.manual_decision=x.value;persist();render();if(state.status!=='done')setTimeout(()=>move(1),100)}});$('notes').oninput=e=>{const [list,r]=current();if(r){r.manual_notes=e.target.value;persist()}};
function quote(v){v=String(v??'');return /[",\r\n]/.test(v)?`"${v.replace(/"/g,'""')}"`:v}
$('exportBtn').onclick=()=>{const csv='\ufeff'+[fields.join(','),...rows.map(r=>fields.map(f=>quote(r[f])).join(','))].join('\r\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download='train_hard_candidate_review.csv';a.click();URL.revokeObjectURL(a.href)};
$('importBtn').onclick=()=>$('importFile').click();$('importFile').onchange=async e=>{const text=await e.target.files[0].text();const parsed=parseCsv(text.replace(/^\ufeff/,''));const map=new Map(parsed.slice(1).map(r=>[r[0],r]));const di=fields.indexOf('manual_decision'),ni=fields.indexOf('manual_notes');rows.forEach(r=>{const x=map.get(r.sample_id);if(x){r.manual_decision=x[di]||'';r.manual_notes=x[ni]||''}});persist();render()};
function parseCsv(s){let out=[],row=[],cell='',q=false;for(let i=0;i<s.length;i++){const c=s[i];if(q){if(c==='"'&&s[i+1]==='"'){cell+='"';i++}else if(c==='"')q=false;else cell+=c}else if(c==='"')q=true;else if(c===','){row.push(cell);cell=''}else if(c==='\n'){row.push(cell.replace(/\r$/,''));out.push(row);row=[];cell=''}else cell+=c}if(cell||row.length){row.push(cell);out.push(row)}return out}
document.addEventListener('keydown',e=>{if(e.target.matches('textarea,input'))return;if(e.key==='ArrowLeft')move(-1);if(e.key==='ArrowRight')move(1)});render();
</script></body></html>'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--review-csv", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    audit_dir = args.audit_dir.resolve()
    review_csv = (args.review_csv or audit_dir / "train_hard_candidate_review.csv").resolve()
    output = (args.output or audit_dir / "train_hard_candidate_review.html").resolve()
    with review_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Missing CSV header: {review_csv}")
        fields = list(reader.fieldnames)
        rows = list(reader)
    image_root = root / "01_data_preparation/real_line_dataset_eval_reviewed"
    for row in rows:
        image = image_root / row["image_path"]
        if not image.is_file():
            raise FileNotFoundError(image)
        row["review_image_src"] = Path(os.path.relpath(image, output.parent)).as_posix()
    selection_path = audit_dir / "review_selection_manifest.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8-sig"))
    if len(rows) != selection["review_rows"]:
        raise ValueError("Review CSV row count no longer matches the frozen selection")
    payload = {
        "protocol_id": selection["protocol_id"],
        "source_sha256": sha256(review_csv),
        "fields": fields,
        "rows": rows,
    }
    encoded = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    output.write_text(HTML.replace("__DATA__", encoded), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "TARGET_TRAIN_HARD_REVIEW_HTML_READY",
                "output": str(output),
                "rows": len(rows),
                "source_sha256": payload["source_sha256"],
                "test_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
