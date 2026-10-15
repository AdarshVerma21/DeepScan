from __future__ import annotations

import html
from pathlib import Path
from typing import Any

from .config import REPORT_DIR


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _list(items: list[Any]) -> str:
    return "".join(f"<li>{_esc(item)}</li>" for item in items)


def _summary_html(markdown: str) -> str:
    output: list[str] = []
    in_list = False
    for line in markdown.splitlines():
        if line.startswith("## "):
            if in_list:
                output.append("</ul>")
                in_list = False
            output.append(f"<h3>{_esc(line[3:])}</h3>")
        elif line.startswith("- "):
            if not in_list:
                output.append("<ul>")
                in_list = True
            output.append(f"<li>{_esc(line[2:])}</li>")
        elif line.strip():
            if in_list:
                output.append("</ul>")
                in_list = False
            output.append(f"<p>{_esc(line)}</p>")
    if in_list:
        output.append("</ul>")
    return "".join(output)


def save_report(report_id: str, payload: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(exist_ok=True)
    verdict = str(payload.get("verdict", "Inconclusive"))
    color = "#fb7185" if verdict == "Likely Deepfake" else "#34d399" if verdict == "Likely Real" else "#fbbf24"
    visualizations = payload.get("visualizations", {})
    original = payload.get("original_image", "")
    heatmap = visualizations.get("heatmap", (payload.get("heatmaps") or [""])[0])
    chips = "".join(f"<span class='chip'>{_esc(item.get('label', ''))}</span>" for item in payload.get("reason_chips", []))
    evidence = "".join(
        f"<article class='evidence'><div class='row'><b>{_esc(item.get('signal'))}</b><span>{round(float(item.get('score', 0)) * 100)}% · {round(float(item.get('weight', 0)) * 100)}% weight</span></div><div class='bar'><i style='width:{max(0, min(100, float(item.get('score', 0)) * 100))}%;background:{color}'></i></div><p>{_esc(item.get('finding', ''))}</p></article>"
        for item in payload.get("evidence", [])
    )
    gallery = "".join(
        f"<figure><img src='{_esc(source)}'><figcaption>{_esc(payload.get('captions', {}).get(name, ''))}</figcaption></figure>"
        for name, source in visualizations.items()
        if name in {"face_crop", "ela", "fft"} and source
    )
    limitations = payload.get("input_warnings") or payload.get("limitations") or []
    frames = "".join(
        f"<figure><img src='{_esc(frame.get('image'))}'><figcaption>{float(frame.get('timestamp', 0)):.2f}s · {round(float(frame.get('score', 0)) * 100)}% suspicious</figcaption></figure>"
        for frame in payload.get("top_suspicious_frames", [])
    )
    video_section = f"<section class='card'><p class='eyebrow'>VIDEO ANALYSIS</p><h2>Temporal evidence</h2><div class='gallery'>{frames}</div></section>" if payload.get("type") == "video" else ""
    provider = "Deterministic evidence summary" if payload.get("summary_is_fallback") else "Groq-generated explanation"
    document = f"""<!doctype html>
<html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>DeepScan | Analysis report</title>
<style>
:root{{--bg:#080d18;--panel:#121d30;--elevated:#18253a;--line:#263752;--text:#f1f5f9;--muted:#94a3b8}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:16px Inter,system-ui,sans-serif;line-height:1.6}}main{{max-width:1180px;margin:auto;padding:28px 18px 80px}}.brand{{color:#67e8f9;letter-spacing:.22em;font-weight:900;font-size:13px}}.card{{background:var(--panel);border:1px solid var(--line);border-radius:22px;padding:26px;margin:18px 0;box-shadow:0 15px 40px #02061744}}h1{{font-size:clamp(32px,6vw,58px);line-height:1.05;margin:8px 0}}h2{{font-size:28px;margin:5px 0 18px}}h3{{font-size:20px;margin:24px 0 7px}}.eyebrow{{color:#67e8f9;font-size:11px;font-weight:800;letter-spacing:.18em;text-transform:uppercase}}.hero{{display:flex;justify-content:space-between;align-items:center;gap:24px}}.verdict{{color:{color};font-size:clamp(28px,5vw,52px);font-weight:900;line-height:1.1}}.score{{color:{color};font-size:48px;font-weight:900;white-space:nowrap}}.sub{{color:var(--muted);font-size:14px}}.chips{{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0}}.chip{{border:1px solid #38506d;border-radius:999px;padding:6px 11px;color:#cbd5e1;font-size:13px}}.summary{{font-size:18px;line-height:1.8;color:#cbd5e1}}.summary ul{{padding-left:22px}}.summary li{{margin:6px 0}}.summary h3{{color:#f1f5f9}}.images,.gallery{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}.gallery figure{{margin:0}}img{{max-width:100%;width:100%;border-radius:14px;background:#020617;display:block}}figcaption{{color:var(--muted);font-size:13px;margin-top:8px}}.row{{display:flex;justify-content:space-between;gap:14px;font-size:14px}}.bar{{height:9px;background:#24334c;border-radius:8px;overflow:hidden;margin-top:8px}}.bar i{{display:block;height:100%}}.evidence{{margin:18px 0}}.evidence p{{color:var(--muted);font-size:14px}}li{{margin:7px 0;color:#cbd5e1}}@media(max-width:700px){{main{{padding:20px 12px 60px}}.card{{padding:20px;border-radius:18px}}.hero,.images,.gallery{{display:block}}.hero .score{{margin-top:18px}}figure{{margin:16px 0}}}}
</style></head><body><main>
<p class='brand'>DEEPSCAN · ANALYSIS REPORT</p><h1>Evidence review</h1><p class='sub'>Generated for { _esc(payload.get('type', 'media')) } · Report ID { _esc(report_id) }</p>
<section class='card hero'><div><p class='eyebrow'>Detection verdict</p><div class='verdict'>{_esc(verdict)}</div><p class='sub'>{_esc(payload.get('confidence'))} confidence · {_esc(payload.get('type'))} analysis · {_esc(payload.get('processing_time'))}s</p><p class='sub'>Fake-likelihood is a heuristic index, not a calibrated probability. Inconclusive means evidence was insufficient or materially contradictory.</p></div><div class='score'>{_esc(payload.get('fake_score'))}<small>/100</small></div></section>
<section class='card'><p class='eyebrow'>Evidence synthesis</p><h2>Analysis Explanation</h2><p class='sub'>{provider}</p><div class='chips'>{chips}</div><div class='summary'>{_summary_html(str(payload.get('summary_markdown', '')))}</div></section>
<section class='card'><p class='eyebrow'>Visual evidence</p><h2>What the detector looked at</h2>{'<p style="color:#fbbf24"><b>No face detected.</b> This result is unreliable for a person-focused detector.</p>' if payload.get('no_face_detected') else ''}<div class='images'><figure><img src='{_esc(original)}'><figcaption>Original media</figcaption></figure><figure><img src='{_esc(heatmap)}'><figcaption>Suspicion heatmap · blue looks natural, red is most suspicious</figcaption></figure></div><div class='gallery'>{gallery}</div></section>
<section class='card'><p class='eyebrow'>Signal review</p><h2>Evidence breakdown</h2>{evidence}</section>
{video_section}
<section class='card'><p class='eyebrow'>Interpretation</p><h2>Limitations</h2><ul>{_list(limitations)}</ul><b style='color:#fbbf24'>Probabilistic result, not proof.</b></section>
</main></body></html>"""
    path = REPORT_DIR / f"{report_id}.html"
    path.write_text(document, encoding="utf-8")
    return path
