import json
import csv
import re
from pathlib import Path
from html import escape
from datetime import datetime

class Reporter:
    def __init__(self, out_dir: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        
    def save_results(self, pt_filename: str, text_pt: str, text_trans: str, all_volume_alignments: dict, 
                     secondary_matches: list = None, used_corpora: list = None,
                     title_left: str = None, title_right: str = None):
        """Save results in JSONL, CSV, HTML, and SVG formats."""
        if secondary_matches is None: secondary_matches = []
        if used_corpora is None: used_corpora = ["All (Default)"]
        
        primary_matches = all_volume_alignments.get(1, {}).get('matches', [])
        
        self._write_jsonl(pt_filename, primary_matches)
        self._write_csv(pt_filename, primary_matches)
        self._write_html(pt_filename, text_pt, text_trans, all_volume_alignments, secondary_matches, used_corpora)
        self._write_svg(pt_filename, text_pt, all_volume_alignments, title_left=title_left, title_right=title_right)

    def _write_jsonl(self, pt_filename, matches):
        p = self.out_dir / "results.jsonl"
        with open(p, 'a', encoding='utf-8') as f:
            for m in matches:
                rec = {
                    "pt_file": pt_filename,
                    "score": m['score'],
                    "corpus_doc": m['corpus_doc']['title'],
                    "corpus_id": m['corpus_doc']['doc_id'],
                    "query_chunk": m['query_chunk'],
                    "source_chunk": m['corpus_doc']['text']
                }
                f.write(json.dumps(rec) + "\n")

    def _write_csv(self, pt_filename, matches):
        p = self.out_dir / "results.csv"
        file_exists = p.exists()
        with open(p, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["pt_file", "score", "article_score", "corpus_doc", "corpus_id", "pt_start", "pt_end"])
            for m in matches:
                writer.writerow([
                    pt_filename,
                    f"{m['score']:.2f}",
                    f"{m.get('article_score', 0.0):.2f}",
                    m['corpus_doc']['title'],
                    m['corpus_doc']['doc_id'],
                    m['query_range'][0],
                    m['query_range'][1]
                ])

    def _write_html(self, pt_filename, text_pt: str, text_trans: str, all_volume_alignments: dict, secondary_matches: list, used_corpora: list):
        safe_name = Path(pt_filename).stem
        out_file = self.out_dir / f"{safe_name}_viz_report.html"
        
        pt_lines = text_pt.split('\n')
        primary_matches = all_volume_alignments.get(1, {}).get('matches', [])
        
        pt_line_offsets = []
        curr = 0
        for l in pt_lines:
            pt_line_offsets.append(curr)
            curr += len(l) + 1
            
        def highlight_block(block_text, block_start, matches):
            block_end = block_start + len(block_text)
            intersections = []
            for m in matches:
                m_s, m_e = m['query_range']
                if m_s < block_end and m_e > block_start:
                    ov_s = max(block_start, m_s) - block_start
                    ov_e = min(block_end, m_e) - block_start
                    if ov_e > ov_s:
                        intersections.append({'s': ov_s, 'e': ov_e, 'score': m['score']})
            
            if not intersections:
                return escape(block_text)
            
            points = set([0, len(block_text)])
            for i in intersections:
                points.add(i['s'])
                points.add(i['e'])
            sorted_points = sorted(list(points))
            
            res = ""
            for i in range(len(sorted_points) - 1):
                p_s = sorted_points[i]
                p_e = sorted_points[i+1]
                segment_text = block_text[p_s:p_e]
                covering_scores = [m['score'] for m in intersections if m['s'] <= p_s and m['e'] >= p_e]
                
                if covering_scores:
                    max_score = max(covering_scores)
                    alpha = max(0.05, min(0.95, (max_score - 40) / 60 * 0.9 + 0.05))
                    res += f'<mark style="background-color: rgba(255, 235, 59, {alpha:.2f});">{escape(segment_text)}</mark>'
                else:
                    res += escape(segment_text)
            return res

        total_coverage = 0.0
        if primary_matches:
            total_coverage = primary_matches[0].get('article_score', 0.0)

        cov_hue = min(120, max(0, (total_coverage - 20) * 1.5))
        cov_style = f"color: hsl({cov_hue}, 70%, 40%); font-weight: bold;"
        
        match_label = ""
        if not primary_matches and not secondary_matches:
            match_label = " <span style='color: #d32f2f; font-size: 0.9em; font-weight: bold; margin-left: 10px; padding: 2px 6px; border: 1px solid #d32f2f; border-radius: 4px;'>NO MATCH FOUND</span>"
        elif total_coverage < 50:
            match_label = " <span style='color: #d32f2f; font-size: 0.9em; font-style: italic; margin-left: 10px;'>(Low coverage)</span>"

        avg_score = 0.0
        if primary_matches:
            avg_score = sum(m['score'] for m in primary_matches) / len(primary_matches)

        if avg_score >= 80:
            avg_style = "color: #2e7d32; font-weight: bold;"
        elif avg_score >= 70:
            avg_style = "color: #7cb342; font-weight: bold;"
        elif avg_score >= 60:
            avg_style = "color: #f9a825; font-weight: bold;"
        else:
            red_hue = max(0, (avg_score / 60) * 30)
            avg_style = f"color: hsl({red_hue}, 85%, 40%); font-weight: bold;"

        html = [f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>Text Reuse Report</title>
            <style>
                body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; max-width: 1400px; margin: 0 auto; padding: 20px; }}
                h1 {{ border-bottom: 2px solid #333; padding-bottom: 10px; color: #2c3e50; }}
                table {{ border-collapse: collapse; width: 100%; table-layout: fixed; margin-top: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 15px; vertical-align: top; overflow-wrap: break-word; }}
                th {{ background-color: #f2f2f2; position: sticky; top: 0; text-align: left; }}
                mark {{ background-color: #ffeb3b; }}
                .match-entry {{ margin-bottom: 25px; padding: 12px; border-left: 5px solid #ccc; font-size: 0.95em; }}
                .match-meta {{ color: #555; font-size: 0.8em; margin-top: 10px; font-family: monospace; border-top: 1px solid #e0e0e0; padding-top: 5px; }}
                .row-num {{ color: #bbb; user-select: none; font-size: 0.8em; text-align: right; width: 45px; background: #fafafa; }}
                tr:nth-child(even) {{ background-color: #fafafa; }}
                .stats {{ margin-bottom: 25px; color: #666; font-size: 1em; background: #f0f4f8; padding: 15px; border-radius: 4px; border-left: 5px solid #2196f3; }}
                .na {{ color: #ccc; font-style: italic; }}
            </style>
        </head>
        <body>
            <h1>Text Reuse Report (Direct Cross-Lingual)</h1>
            <div class="stats">
                <strong>File:</strong> {escape(pt_filename)}<br>
                <strong>Corpora Used:</strong> {escape(", ".join(used_corpora))}<br>
                <strong>Overall Article Coverage:</strong> <span style="{cov_style}">{total_coverage:.1f}%</span>{match_label}<br>
                <strong>Average Match Similarity:</strong> <span style="{avg_style}">{avg_score:.1f}%</span><br>
                <strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
            <table>
                <thead>
                    <tr>
                        <th class="row-num">#</th>
                        <th style="width: 25%">Original</th>
                        <th style="width: 18%">Rank 1: {escape(all_volume_alignments.get(1, {}).get('vol_id', 'N/A'))}</th>
                        <th style="width: 18%">Rank 2: {escape(all_volume_alignments.get(2, {}).get('vol_id', 'N/A'))}</th>
                        <th style="width: 18%">Rank 3: {escape(all_volume_alignments.get(3, {}).get('vol_id', 'N/A'))}</th>
                        <th style="width: 18%">Rank 4: {escape(all_volume_alignments.get(4, {}).get('vol_id', 'N/A'))}</th>
                    </tr>
                </thead>
                <tbody>
        """]
        
        CHUNK_SIZE = 6
        row_count = 0
        for i in range(0, len(pt_lines), CHUNK_SIZE):
            row_count += 1
            idx_end = min(i + CHUNK_SIZE, len(pt_lines))
            chunk_pt = pt_lines[i : idx_end]
            b_pt = "\n".join(chunk_pt)
            b_start = pt_line_offsets[i]
            b_end = pt_line_offsets[idx_end] if idx_end < len(pt_line_offsets) else len(text_pt)
            
            row_cols = []
            for rank in range(1, 5):
                vol_data = all_volume_alignments.get(rank)
                if not vol_data:
                    row_cols.append('<span class="na">N/A</span>')
                    continue
                
                chunk_matches = sorted(
                    [m for m in vol_data['matches'] if b_start <= m['query_range'][0] < b_end],
                    key=lambda x: x['query_range'][0]
                )
                
                seen_txt = set()
                col_html = []
                for m in chunk_matches:
                    c_text = m['corpus_doc']['text']
                    if c_text in seen_txt: continue
                    seen_txt.add(c_text)
                    
                    score = m['score']
                    alpha = max(0.05, min(0.95, (score - 40) / 60 * 0.9 + 0.05))
                    hue = min(120, max(0, (score - 40) * 2))
                    col_html.append(f"""
                        <div class="match-entry" style="background: rgba(255, 235, 59, {alpha:.2f}); border-left-color: hsl({hue}, 100%, 40%);">
                            {escape(c_text)}
                            <div class="match-meta">
                                {escape(Reporter._format_doc_id(m['corpus_doc']['doc_id']))}<br>
                                Score: {m['score']}
                            </div>
                        </div>
                    """)
                row_cols.append("".join(col_html) if col_html else '<span class="na">N/A</span>')

            o_display = highlight_block(b_pt, b_start, primary_matches).replace('\n', '<br>')
            html.append(f"""
                <tr>
                    <td class="row-num">{row_count}</td>
                    <td>{o_display}</td>
                    <td>{row_cols[0]}</td>
                    <td>{row_cols[1]}</td>
                    <td>{row_cols[2]}</td>
                    <td>{row_cols[3]}</td>
                </tr>
            """)
            
        html.append("</tbody></table>")
        
        if secondary_matches:
            html.append("""
            <div style="margin-top: 30px; border-top: 1px dashed #ccc; padding-top: 20px;">
                <h3 style="color: #777;">Partial Sentence Overlaps (Candidates)</h3>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            """)
            for m in secondary_matches:
                hue = min(120, max(0, (m['score'] - 40) * 2))
                html.append(f"""
                    <div class="match-entry" style="background: hsl({hue}, 100%, 97%); border-left-color: hsl({hue}, 100%, 40%); margin-bottom: 0;">
                        {escape(m['text'])}
                        <div class="match-meta">
                            {escape(Reporter._format_doc_id(m['doc_id']))}<br>
                            Score: {m['score']}
                        </div>
                    </div>
                """)
            html.append("</div></div>")
            
        html.append("</body></html>")
        with open(out_file, 'w', encoding='utf-8') as f:
            f.write("".join(html))

    def _write_svg(self, pt_filename, text_pt: str, all_volume_alignments: dict, title_left: str = None, title_right: str = None):
        """Generates a high-fidelity SVG alignment visualization."""
        safe_name = Path(pt_filename).stem
        out_file = self.out_dir / f"{safe_name}_alignment.svg"
        
        from .utils import split_into_sentences
        pt_sentences = split_into_sentences(text_pt)
        total_lines = len(pt_sentences)
        
        if total_lines < 100:
            displayed_indices = list(range(total_lines))
        else:
            limit_idx = int(total_lines * 0.10)
            start_bottom_idx = int(total_lines * 0.90)
            if start_bottom_idx <= limit_idx:
                displayed_indices = list(range(total_lines))
            else:
                displayed_indices = list(range(0, limit_idx)) + [-1] + list(range(start_bottom_idx, total_lines))
            
        vol_data = all_volume_alignments.get(1, {})
        matches = vol_data.get('matches', [])
        vol_id = vol_data.get('vol_id', 'Matched Text')
        corpus_sentences = vol_data.get('corpus_sentences', [])
        
        visible_q_indices = set(i for i in displayed_indices if i != -1)
        visible_matches = [m for m in matches if m['q_idx'] in visible_q_indices]
        
        total_c_lines = len(corpus_sentences)
        if total_c_lines < 100:
            rhs_indices = list(range(total_c_lines))
        else:
            c_limit = int(total_c_lines * 0.10)
            c_start_bot = int(total_c_lines * 0.90)
            rhs_indices = list(range(0, c_limit)) + [-1] + list(range(c_start_bot, total_c_lines))

        FONT_SIZE, LINE_H, BLOCK_GAP, MARGIN_TOP, MARGIN_X, COL_W, GAP, CHARS_PER_LINE = 12, 16, 12, 50, 50, 400, 200, 60

        def wrap_text(text, limit):
            words = text.split()
            lines, curr_line, curr_len = [], [], 0
            for w in words:
                if curr_len + len(w) + 1 > limit:
                    lines.append(" ".join(curr_line))
                    curr_line, curr_len = [w], len(w)
                else:
                    curr_line.append(w)
                    curr_len += len(w) + 1
            if curr_line: lines.append(" ".join(curr_line))
            return lines

        pt_blocks, curr_y = [], MARGIN_TOP
        for item in displayed_indices:
            lines = ["(...)"] if item == -1 else wrap_text(pt_sentences[item], CHARS_PER_LINE)
            height = len(lines) * LINE_H
            pt_blocks.append((curr_y, curr_y + height/2, curr_y + height, lines))
            curr_y += height + BLOCK_GAP
        
        fr_blocks, curr_y = [], MARGIN_TOP
        for item in rhs_indices:
            lines = ["(...)"] if item == -1 else wrap_text(corpus_sentences[item][0], CHARS_PER_LINE)
            height = len(lines) * LINE_H
            fr_blocks.append((curr_y, curr_y + height/2, curr_y + height, lines))
            curr_y += height + BLOCK_GAP

        MAX_H = max(pt_blocks[-1][2] if pt_blocks else 0, fr_blocks[-1][2] if fr_blocks else 0) + MARGIN_TOP
        WIDTH = MARGIN_X * 2 + COL_W * 2 + GAP
        
        svg = [f'<svg viewBox="0 0 {WIDTH} {MAX_H}" xmlns="http://www.w3.org/2000/svg">']
        svg.append(f'<style>text {{ font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; font-size: {FONT_SIZE}px; fill: #333; }} .label {{ font-weight: bold; font-size: 14px; fill: #000; }} .gap {{ font-size: 16px; fill: #999; font-weight: bold; }}</style>')
        svg.append('<rect width="100%" height="100%" fill="#ffffff"/>')
        svg.append(f'<text x="{MARGIN_X}" y="30" class="label">{escape(title_left if title_left else str(pt_filename))}</text>')
        svg.append(f'<text x="{WIDTH - MARGIN_X - COL_W}" y="30" class="label">{escape(title_right if title_right else vol_id)}</text>')
        
        c_idx_to_rhs_idx = {idx: i for i, idx in enumerate(rhs_indices) if idx != -1}
        for m in visible_matches:
            q_idx_in_displayed = next((i for i, v in enumerate(displayed_indices) if v == m['q_idx']), None)
            c_idx_in_rhs = c_idx_to_rhs_idx.get(m['c_idx'])
            if q_idx_in_displayed is not None and c_idx_in_rhs is not None:
                y1, x1 = pt_blocks[q_idx_in_displayed][1], MARGIN_X + COL_W + 10 
                y2, x2 = fr_blocks[c_idx_in_rhs][1], WIDTH - MARGIN_X - COL_W - 10 
                
                score = m['score']
                hue = min(120, max(0, (score - 40) * 1.5))
                def hsl_to_rgb(h, s, l):
                    c = (1 - abs(2 * l - 1)) * s
                    x = c * (1 - abs((h / 60) % 2 - 1))
                    m = l - c / 2
                    if 0 <= h < 60: r, g, b = c, x, 0
                    elif 60 <= h < 120: r, g, b = x, c, 0
                    elif 120 <= h < 180: r, g, b = 0, c, x
                    else: r, g, b = 0, 0, 0
                    return int((r+m)*255), int((g+m)*255), int((b+m)*255)

                r, g, b = hsl_to_rgb(hue, 0.8, 0.6)
                opacity = max(0.4, min(0.9, (score - 30) / 70))
                svg.append(f'<path d="M {x1} {y1} C {x1+GAP*0.5} {y1}, {x2-GAP*0.5} {y2}, {x2} {y2}" stroke="rgb({r},{g},{b})" stroke-width="4" fill="none" opacity="{opacity}"/>')

        for y_start, _, _, lines in pt_blocks:
            for l_idx, line in enumerate(lines):
                y, is_gap = y_start + (l_idx * LINE_H) + (LINE_H * 0.8), line == "(...)"
                svg.append(f'<text x="{MARGIN_X + (20 if is_gap else 0)}" y="{y}"{" class=\"gap\"" if is_gap else ""}>{escape(line)}</text>')

        for y_start, _, _, lines in fr_blocks:
            for l_idx, line in enumerate(lines):
                y, is_gap = y_start + (l_idx * LINE_H) + (LINE_H * 0.8), line == "(...)"
                svg.append(f'<text x="{WIDTH - MARGIN_X - COL_W + (20 if is_gap else 0)}" y="{y}"{" class=\"gap\"" if is_gap else ""}>{escape(line)}</text>')
            
        svg.append('</svg>')
        with open(out_file, 'w', encoding='utf-8') as f: f.write("\n".join(svg))

    @staticmethod
    def _format_doc_id(doc_id):
        p_match = re.search(r'p(\d+)$', doc_id)
        if p_match:
            base = re.sub(r'[p/]\d+$', '', doc_id).replace('::', ' | ')
            return f"{base} | Page {p_match.group(1)}"
        return doc_id.replace('::', ' | ')
