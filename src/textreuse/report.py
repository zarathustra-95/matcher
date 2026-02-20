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
        
    def save_results(self, pt_filename: str, text_pt: str, text_trans: str, matches: list, secondary_matches: list = None, used_corpora: list = None):
        if secondary_matches is None: secondary_matches = []
        if used_corpora is None: used_corpora = ["All (Default)"]
        # 1. JSONL
        self._write_jsonl(pt_filename, matches)
        # 2. CSV
        self._write_csv(pt_filename, matches)
        # 3. HTML
        self._write_html(pt_filename, text_pt, text_trans, matches, secondary_matches, used_corpora)

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

    def _write_html(self, pt_filename, text_pt: str, text_trans: str, matches: list, secondary_matches: list, used_corpora: list):
        # In matcher2, text_pt and text_trans are identical
        safe_name = Path(pt_filename).stem
        out_file = self.out_dir / f"{safe_name}_report.html"
        
        pt_lines = text_pt.split('\n')
        
        # Calculate pt line offsets for highlighting
        pt_line_offsets = []
        curr = 0
        for l in pt_lines:
            pt_line_offsets.append(curr)
            curr += len(l) + 1
            
        def highlight_block(block_text, block_start, matches):
            block_end = block_start + len(block_text)
            # Find all intersections of matches with this block
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
            
            # To handle overlaps while varying intensity, 
            # we find all unique start/end points to create non-overlapping segments
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
                
                # Find the best score for any match covering this specific segment
                covering_scores = [m['score'] for m in intersections if m['s'] <= p_s and m['e'] >= p_e]
                
                if covering_scores:
                    max_score = max(covering_scores)
                    # Mapping score (usually 40-100) to alpha (0.1 to 0.9)
                    # alpha = 0.1 at score 40, alpha 0.9 at score 100
                    alpha = max(0.05, min(0.95, (max_score - 40) / 60 * 0.9 + 0.05))
                    res += f'<mark style="background-color: rgba(255, 235, 59, {alpha:.2f});">{escape(segment_text)}</mark>'
                else:
                    res += escape(segment_text)
            return res

        total_coverage = 0.0
        if matches:
            total_coverage = matches[0].get('article_score', 0.0)

        # Calculate overall coverage color (20%=Red, 100%=Green)
        cov_hue = min(120, max(0, (total_coverage - 20) * 1.5))
        cov_style = f"color: hsl({cov_hue}, 70%, 40%); font-weight: bold;"
        
        match_label = ""
        if not matches and not secondary_matches:
            match_label = " <span style='color: #d32f2f; font-size: 0.9em; font-weight: bold; margin-left: 10px; padding: 2px 6px; border: 1px solid #d32f2f; border-radius: 4px;'>NO MATCH FOUND</span>"
        elif total_coverage < 50:
            match_label = " <span style='color: #d32f2f; font-size: 0.9em; font-style: italic; margin-left: 10px;'>(Low coverage)</span>"

        avg_score = 0.0
        if matches:
            avg_score = sum(m['score'] for m in matches) / len(matches)

        # Dynamic Styling for Average Similarity
        if avg_score >= 80:
            avg_style = "color: #2e7d32; font-weight: bold;" # Green
        elif avg_score >= 70:
            avg_style = "color: #7cb342; font-weight: bold;" # Light Green
        elif avg_score >= 60:
            avg_style = "color: #f9a825; font-weight: bold;" # Yellow
        else:
            # Shades of Red/Orange for < 60
            # Hue 0 (Red) to 30 (Orange)
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
                        <th style="width: 45%">Original</th>
                        <th style="width: 50%">Matched Corpus Text</th>
                    </tr>
                </thead>
                <tbody>
        """]
        
        # BALANCED CHUNK SIZE for spatial alignment
        CHUNK_SIZE = 6
        row_count = 0
        displayed_match_ids = set()
        
        for i in range(0, len(pt_lines), CHUNK_SIZE):
            row_count += 1
            idx_end = min(i + CHUNK_SIZE, len(pt_lines))
            chunk_pt = pt_lines[i : idx_end]
            b_pt = "\n".join(chunk_pt)
            b_start = pt_line_offsets[i]
            
            if idx_end < len(pt_line_offsets):
                b_end = pt_line_offsets[idx_end]
            else:
                b_end = len(text_pt)
            
            # Find matches that START in this chunk to avoid duplication/leakage
            row_matches = [m for m in matches if b_start <= m['query_range'][0] < b_end]
            row_matches.sort(key=lambda x: x['query_range'][0])
            
            # Build Corpus column
            corpus_html = []
            if row_matches:
                seen_corpus_text = set()
                for m in row_matches:
                    c_text = m['corpus_doc']['text']
                    if c_text in seen_corpus_text: continue
                    seen_corpus_text.add(c_text)
                    
                    score = m['score']
                    hue = min(120, max(0, (score - 40) * 2))
                    bg_color = f"hsl({hue}, 100%, 97%)"
                    border_color = f"hsl({hue}, 100%, 40%)"
                    
                    corpus_html.append(f"""
                        <div class="match-entry" style="background: {bg_color}; border-left-color: {border_color};">
                            {escape(c_text)}
                            <div class="match-meta">
                                {escape(Reporter._format_doc_id(m['corpus_doc']['doc_id']))}<br>
                                Score: {m['score']} | {m.get('tier', 'N/A')}
                            </div>
                        </div>
                    """)
            
            matched_col = "".join(corpus_html) if corpus_html else '<span class="na">N/A</span>'
            # Highlight uses global matches to ensure highlights span correctly
            o_display = highlight_block(b_pt, b_start, matches).replace('\n', '<br>')
            
            html.append(f"""
                <tr>
                    <td class="row-num">{row_count}</td>
                    <td>{o_display}</td>
                    <td>{matched_col}</td>
                </tr>
            """)
            
        html.append("</tbody></table>")
        
        # Add Secondary Matches Section
        if secondary_matches:
            html.append("""
            <div style="margin-top: 50px; border-top: 3px solid #eee; padding-top: 20px;">
                <h2 style="color: #2c3e50;">Other High-Confidence Parallels (>70% Score)</h2>
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

    @staticmethod
    def _format_doc_id(doc_id):
        # Extract page number if present (e.g. "Vol/p123" or "Vol.pdf/p123")
        p_match = re.search(r'p(\d+)$', doc_id)
        if p_match:
            base = re.sub(r'[p/]\d+$', '', doc_id)
            base = base.replace('::', ' | ')
            return f"{base} | Page {p_match.group(1)}"
        return doc_id.replace('::', ' | ')
