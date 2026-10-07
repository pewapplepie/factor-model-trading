"""Minimal Markdown -> self-contained HTML converter for the research paper.
Supports: # headers, paragraphs, **bold**, *italic*, `code`, pipe tables, bullet/numbered lists,
blockquote boxes (> ...), images ![cap](path) embedded as base64, --- rules, inline HTML passthrough."""
import re, sys, base64, html, pathlib

src_path = pathlib.Path(sys.argv[1]); out_path = pathlib.Path(sys.argv[2])
base = src_path.parent
lines = src_path.read_text(encoding='utf-8').split('\n')

def inline(s):
    # protect inline code
    codes = []
    def _c(m):
        codes.append(m.group(1)); return f'\x00{len(codes)-1}\x00'
    s = re.sub(r'`([^`]+)`', _c, s)
    s = s.replace(r'\|', '&#124;')
    s = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', s)
    s = re.sub(r'(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])', r'<em>\1</em>', s)
    s = re.sub(r'\x00(\d+)\x00', lambda m: f'<code>{html.escape(codes[int(m.group(1))])}</code>', s)
    return s

def img_tag(alt, path):
    p = (base / path)
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{data}" alt="{html.escape(alt)}"><figcaption>{inline(alt)}</figcaption></figure>'

out = []; i = 0; n = len(lines)
para = []
def flush_para():
    global para
    if para:
        out.append('<p>' + inline(' '.join(para)) + '</p>'); para = []

while i < n:
    ln = lines[i]
    st = ln.strip()
    if not st:
        flush_para(); i += 1; continue
    m = re.match(r'^(#{1,6})\s+(.*)$', st)
    if m:
        flush_para(); lvl = len(m.group(1)); txt = m.group(2)
        anchor = re.sub(r'[^a-z0-9]+', '-', re.sub(r'<[^>]+>', '', txt).lower()).strip('-')
        out.append(f'<h{lvl} id="{anchor}">{inline(txt)}</h{lvl}>'); i += 1; continue
    if st == '---':
        flush_para(); out.append('<hr>'); i += 1; continue
    if st == '↓':
        flush_para(); out.append('<div class="arrow">↓</div>'); i += 1; continue
    m = re.match(r'^!\[(.*?)\]\((.*?)\)$', st)
    if m:
        flush_para(); out.append(img_tag(m.group(1), m.group(2))); i += 1; continue
    if st.startswith('|'):
        flush_para(); rows = []
        while i < n and lines[i].strip().startswith('|'):
            rows.append(lines[i].strip()); i += 1
        cells = [[c.strip() for c in re.split(r'(?<!\\)\|', r)[1:-1]] for r in rows]
        # drop separator rows
        body = [c for c in cells if not all(re.fullmatch(r':?-{2,}:?', x or '---') for x in c)]
        if not body: continue
        # table caption: first row where only the first cell is non-empty and it is bold
        cap = None
        if len(body) > 1 and body[0][0].startswith('**') and all(x == '' for x in body[0][1:]):
            cap = body[0][0].strip('*'); body = body[1:]
        hdr = body[0]; rest = body[1:]
        t = ['<table>']
        if cap: t.append(f'<caption>{inline(cap)}</caption>')
        t.append('<thead><tr>' + ''.join(f'<th>{inline(c.strip("*"))}</th>' for c in hdr) + '</tr></thead><tbody>')
        for r in rest:
            r = r + [''] * (len(hdr) - len(r))
            cls = ' class="hl"' if any(c.startswith('**') for c in r) else ''
            t.append(f'<tr{cls}>' + ''.join(f'<td>{inline(c)}</td>' for c in r) + '</tr>')
        t.append('</tbody></table>'); out.append('\n'.join(t)); continue
    if st.startswith('>'):
        flush_para(); q = []
        while i < n and lines[i].strip().startswith('>'):
            q.append(lines[i].strip()[1:].strip()); i += 1
        out.append('<div class="box">' + ''.join(f'<p>{inline(p)}</p>' for p in ' '.join(q).split('  ') if p.strip()) + '</div>'); continue
    m = re.match(r'^(\d+)\.\s+(.*)$', st)
    if m:
        flush_para(); items = []
        while i < n and re.match(r'^\d+\.\s+', lines[i].strip()):
            items.append(re.sub(r'^\d+\.\s+', '', lines[i].strip())); i += 1
        out.append('<ol>' + ''.join(f'<li>{inline(x)}</li>' for x in items) + '</ol>'); continue
    if st.startswith('- '):
        flush_para(); items = []
        while i < n and lines[i].strip().startswith('- '):
            items.append(lines[i].strip()[2:]); i += 1
        out.append('<ul>' + ''.join(f'<li>{inline(x)}</li>' for x in items) + '</ul>'); continue
    para.append(st); i += 1
flush_para()

title = re.sub(r'<[^>]+>', '', out[0]) if out and out[0].startswith('<h1') else 'Research note'
css = """
:root{--ink:#1b1f24;--muted:#555b63;--line:#d9dde3;--box:#f4f6f9;--accent:#1d4e89;--hl:#eef4fb;--bg:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 Georgia,"Times New Roman",serif}
main{max-width:860px;margin:0 auto;padding:40px 16px 80px}
h1{font-size:1.55em;line-height:1.25;margin:1.6em 0 .5em;color:var(--accent);font-family:Helvetica,Arial,sans-serif}
h1:first-child{font-size:2em;margin-top:0}
h2{font-size:1.2em;margin:1.5em 0 .4em;font-family:Helvetica,Arial,sans-serif}
h3{font-size:1.02em;margin:1.2em 0 .3em;font-family:Helvetica,Arial,sans-serif;color:var(--accent)}
p{margin:.55em 0;text-align:left}
hr{border:0;border-top:1px solid var(--line);margin:1.8em 0}
code{font:0.88em/1.4 Menlo,Consolas,monospace;background:var(--box);padding:1px 4px;border-radius:3px}
table{border-collapse:collapse;width:100%;margin:1em 0 1.3em;font:13.5px/1.4 Helvetica,Arial,sans-serif;page-break-inside:avoid}
caption{text-align:left;font-weight:700;padding:0 0 6px;color:var(--ink)}
th,td{border:1px solid var(--line);padding:5px 8px;vertical-align:top;text-align:left}
th{background:var(--box);font-weight:700}
tr.hl td{background:var(--hl)}
.box{background:var(--box);border-left:4px solid var(--accent);padding:10px 16px;margin:1.1em 0;font-size:.95em;page-break-inside:avoid}
.box p{margin:.4em 0}
.arrow{text-align:center;color:var(--muted);font-size:1.3em;margin:-.2em 0}
figure{margin:1.3em 0;page-break-inside:avoid}
figure img{width:100%;height:auto;border:1px solid var(--line)}
figcaption{font:13px/1.45 Helvetica,Arial,sans-serif;color:var(--muted);margin-top:6px}
ul,ol{padding-left:1.4em}
li{margin:.25em 0}
sub,sup{font-size:.72em}
@media print{body{font-size:11.5pt}main{max-width:none;padding:0}h1{page-break-after:avoid}h2,h3{page-break-after:avoid}figure img{max-height:9in;width:auto;max-width:100%}}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ink:#e6e8eb;--muted:#a3a9b1;--line:#3a4049;--box:#1f242b;--accent:#8ab4f8;--hl:#1b2a3d;--bg:#121418}}
:root[data-theme="dark"]{--ink:#e6e8eb;--muted:#a3a9b1;--line:#3a4049;--box:#1f242b;--accent:#8ab4f8;--hl:#1b2a3d;--bg:#121418}
"""
doc = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>{css}</style></head><body><main>' + '\n'.join(out) + '</main></body></html>'
out_path.write_text(doc, encoding='utf-8')
print(out_path, len(doc), 'chars;', doc.count('<figure>'), 'figures;', doc.count('<table>'), 'tables;', doc.count('class="box"'), 'boxes')
