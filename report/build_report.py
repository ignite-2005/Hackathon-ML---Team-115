"""Combine everyone's report sections into one HTML file and one PDF (Person D).

Usage (repo root):  python report/build_report.py
Output:             report/PathSnap_Team115_Report.html and .pdf

Each section is a Markdown file in report/ (owner in brackets). A missing section is
replaced by a clearly marked placeholder, so the PDF can be built at any time and the
gaps are visible. Images are referenced relative to report/ (e.g. ../results/x.png).
The PDF is printed with headless Microsoft Edge or Google Chrome (no extra tools needed).
"""
import base64
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from datetime import date

import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = "PathSnap: Distributed Map-Matching and Route Reconstruction"
TEAM = "Team 115 · UE24CS352A Machine Learning · PES University"

# (section title, candidate files in order of preference, owner)
SECTIONS = [
    ("2.2 Dataset Understanding and EDA",
     ["section_2_dataset_eda.md", "data_preprocessing_features.md"], "Person A"),
    ("3. Model Experimentation and Comparison", ["section_3_model_comparison.md"], "Person B"),
    ("4. Bayesian Hyperparameter Optimisation", ["section_4_bayesian_optimisation.md"], "Person C"),
    ("5. HMM and Viterbi Route Reconstruction", ["section_5_hmm_viterbi.md"], "Person C"),
    ("6. Geographical Visualisation Using Folium", ["section_6_folium_maps.md"], "Person A"),
    ("7. API, UI and Deployment", ["section_7_api_ui_deployment.md"], "Person D"),
]

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }
h1 { font-size: 19pt; border-bottom: 2px solid #15616d; padding-bottom: 4px; margin-top: 0; }
h2 { font-size: 14pt; color: #15616d; margin-top: 18px; }
h3 { font-size: 12pt; margin-top: 14px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #bbb; padding: 4px 6px; vertical-align: top; }
th { background: #eef3f4; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f3f3f3; padding: 0 2px; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 8px; font-size: 8.5pt; white-space: pre-wrap; page-break-inside: avoid; }
pre code { background: none; }
img { max-width: 100%; display: block; margin: 8px auto; border: 1px solid #ddd; page-break-inside: avoid; }
.section { page-break-before: always; }
.cover { text-align: center; padding-top: 70mm; }
.cover h1 { border: none; font-size: 24pt; }
.missing { border: 2px dashed #c0392b; color: #c0392b; padding: 14px; }
"""


def inline_images(html, base_dir):
    """Embed local images as data URIs so the HTML/PDF is self-contained."""
    def repl(m):
        src = m.group(2)
        if src.startswith(("http://", "https://", "data:")):
            return m.group(0)
        path = os.path.normpath(os.path.join(base_dir, src))
        if not os.path.exists(path):
            print(f"  warning: image not found: {src}")
            return m.group(0)
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f'{m.group(1)}data:{mime};base64,{data}{m.group(3)}'
    return re.sub(r'(<img[^>]*src=")([^"]+)(")', repl, html)


def find_browser():
    for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              shutil.which("msedge") or "", shutil.which("google-chrome") or "",
              shutil.which("chromium") or ""]:
        if p and os.path.exists(p):
            return p
    return None


def main():
    md = markdown.Markdown(extensions=["tables", "fenced_code"])
    parts = [f'<div class="cover"><h1>{TITLE}</h1><p>{TEAM}</p>'
             f'<p>Technical report · {date.today():%d %B %Y}</p></div>']
    for title, files, owner in SECTIONS:
        path = next((os.path.join(HERE, f) for f in files
                     if os.path.exists(os.path.join(HERE, f))), None)
        if path is None:
            print(f"MISSING  {title}  ({owner}: add report/{files[0]})")
            body = (f"<h1>{title}</h1><div class='missing'><b>Not written yet</b>: "
                    f"{owner} adds <code>report/{files[0]}</code>.</div>")
        else:
            print(f"ok       {title}  <- {os.path.basename(path)}")
            with open(path, encoding="utf-8") as f:
                body = md.reset().convert(f.read())
        parts.append(f'<div class="section">{body}</div>')

    html = (f"<!doctype html><html><head><meta charset='utf-8'><title>{TITLE}</title>"
            f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")
    html = inline_images(html, HERE)
    out_html = os.path.join(HERE, "PathSnap_Team115_Report.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"saved {os.path.relpath(out_html)}")

    browser = find_browser()
    if browser is None:
        print("No Edge/Chrome found: open the HTML in a browser and use Print -> Save as PDF.")
        return 1
    out_pdf = os.path.join(HERE, "PathSnap_Team115_Report.pdf")
    subprocess.run([browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={out_pdf}", "file:///" + out_html.replace("\\", "/")],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
    print(f"saved {os.path.relpath(out_pdf)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
