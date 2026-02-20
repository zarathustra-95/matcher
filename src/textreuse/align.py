import difflib
import re
import numpy as np
from typing import List, Dict, Any
from rapidfuzz import fuzz
from .config import Config
from .utils import logger, clean_for_matching, split_into_sentences

class Aligner:
    def __init__(self, config: Config):
        self.config = config

    def get_score(self, query_chunk: str, candidate_chunk: str) -> int:
        """Fast lexical score check."""
        return int(fuzz.token_set_ratio(query_chunk, candidate_chunk))

    def get_details(self, query_chunk: str, candidate_chunk: str):
        """Detailed alignment for highlighting."""
        # Clean both for better alignment calculation
        matcher = difflib.SequenceMatcher(None, query_chunk, candidate_chunk)
        match = matcher.find_longest_match(0, len(query_chunk), 0, len(candidate_chunk))
        
        if match.size == 0:
            return None

        return {
            "query_start": match.a,
            "query_end": match.a + match.size,
            "cand_start": match.b,
            "cand_end": match.b + match.size,
            "match_len": match.size
        }

    def _find_best_window(self, hits: List[tuple], window_size: int = 40) -> tuple:
        """Find the page range with the highest density of hits to exclude outliers."""
        if not hits: return None
        pages = sorted([p for p, d in hits])
        
        best_count = 0
        best_start = pages[0]
        
        # Consider each unique hit page as a potential window start
        unique_pages = sorted(list(set(pages)))
        for p in unique_pages:
            count = sum(1 for hp in pages if p <= hp <= p + window_size)
            if count > best_count:
                best_count = count
                best_start = p
        
        # Return expanded window
        return (max(1, best_start - 5), best_start + window_size + 10)

    def process_document(self, translated_text: str, indexer, include_corpora: list = None) -> List[Dict[str, Any]]:
        """
        Process a full document using a localized and monotonic approach:
        1. Identify winning volume and localized page window.
        2. Deep sentence alignment using soft-monotonicity.
        """
        chunks = indexer._chunk_text(translated_text)
        # Phase 1: Identify winning volume and gather hits
        vol_coverage = {}
        page_hits = []
        
        logger.info(f"Identifying candidates for {len(chunks)} article chunks...")
        
        q_texts = [c[0] for c in chunks]
        # Increase top_k to 50 to ensure sources aren't buried by redundant compilations
        batch_candidates = indexer.query_batch(q_texts, top_k=50, include_corpora=include_corpora)
        
        for candidates in batch_candidates:
            # DEDUPLICATION: Each volume can only contribute its BEST chunk match for THIS query chunk.
            # This prevents compilations (like ESM) from winning purely by having 5 overlapping versions
            # of the same article text.
            chunk_vols = {} # vol_id -> best_distance
            
            for cand in candidates:
                doc_id = cand['doc_id']
                vol_id = re.sub(r'[p/]\d+$', '', doc_id)
                vol_id = re.sub(r'\.pdf$', '', vol_id)
                
                if cand['distance'] > chunk_vols.get(vol_id, 0):
                    chunk_vols[vol_id] = cand['distance']
            
            for vol_id, score in chunk_vols.items():
                vol_coverage[vol_id] = vol_coverage.get(vol_id, 0) + score
                
                # Keep tracking page hits for density window (Phase 2)
                # We still store these normally as density is about coverage spread
                # Note: We take best hit per vol per chunk here too
                # Search for original doc_id in candidates for this vol_id if needed
                pass

        # Re-populate page_hits for winning volume determination
        # We need the page numbers for the winner windowing logic
        for candidates in batch_candidates:
            for cand in candidates:
                doc_id = cand['doc_id']
                vol_id = re.sub(r'[p/]\d+$', '', doc_id)
                vol_id = re.sub(r'\.pdf$', '', vol_id)
                p_match = re.search(r'p(\d+)$', doc_id)
                p_num = int(p_match.group(1)) if p_match else 0
                page_hits.append((vol_id, p_num, cand['distance']))
        
        if not vol_coverage:
            logger.warning("No semantic candidates found.")
            return [], []
            
        winning_vol = max(vol_coverage.items(), key=lambda x: x[1])[0]
        logger.info(f"Volume Pairing: {winning_vol}")
        
        # Collect secondary high-confidence matches (>70 score) from other volumes
        # OVERHAUL: Only search using the first paragraph (first chunk)
        secondary_matches = []
        seen_secondary = set()
        
        if chunks:
            q_text, _, _ = chunks[0]
            candidates = indexer.query(q_text, top_k=10, include_corpora=include_corpora)
            for cand in candidates:
                doc_id = cand['doc_id']
                vol_id = re.sub(r'[p/]\d+$', '', doc_id)
                vol_id = re.sub(r'\.pdf$', '', vol_id)
                score = int(cand['distance'] * 100)
                
                if vol_id != winning_vol and score >= 70:
                    key = (vol_id, cand['text'][:50])
                    if key not in seen_secondary:
                        secondary_matches.append({
                            "doc_id": doc_id,
                            "score": score,
                            "text": cand['text']
                        })
                        seen_secondary.add(key)
                        
        # Phase 2: Identify likely page window within winning volume using density clustering
        # Filter for valid page numbers only
        winner_hits = [(p, d) for v, p, d in page_hits if v == winning_vol and p > 0]
        
        page_window = None
        if winner_hits:
            confident_hits = [h for h in winner_hits if h[1] >= 0.70]
            if not confident_hits:
                confident_hits = [h for h in winner_hits if h[1] > 0.5]
            
            # IMPROVEMENT: If hits are spread across many pages (common in journals/composites),
            # use a wide window encompassing all clusters. Otherwise, use densest window.
            pages = sorted([h[0] for h in confident_hits])
            if pages and (pages[-1] - pages[0]) > 50:
                page_window = (max(1, pages[0] - 10), pages[-1] + 10)
                logger.info(f"Scattered Hits Detected: Expanding window to span Pages {page_window[0]} to {page_window[1]}")
            else:
                page_window = self._find_best_window(confident_hits if confident_hits else winner_hits)
        
        if page_window:
            logger.info(f"Density Locality Window: Pages {page_window[0]} to {page_window[1]}")
            results = self._align_sentences(translated_text, winning_vol, indexer, page_window)
        else:
            results = self._align_sentences(translated_text, winning_vol, indexer)
            
        results = self._compute_global_scores(results, len(translated_text))
        return results, secondary_matches

    def _align_sentences(self, pt_trans: str, vol_id: str, indexer, page_window=None) -> List[Dict]:
        """Perform deep semantic sentence-level comparison against a likely window of pages."""
        doc_chunks = indexer.get_volume_chunks(vol_id)
        
        # Filter by page window if provided
        if page_window:
            filtered_chunks = []
            for c in doc_chunks:
                p_match = re.search(r'p(\d+)$', c['doc_id'])
                if p_match:
                    pno = int(p_match.group(1))
                    if page_window[0] <= pno <= page_window[1]:
                        filtered_chunks.append(c)
            doc_chunks = filtered_chunks

        # Sort by doc_id (page) then by start offset within page
        # This keeps the global corpus text in order across pages
        sorted_chunks = sorted(doc_chunks, key=lambda x: (x['doc_id'], x['start']))
        
        # We need to track which doc_id (page) each sentence belongs to
        # Reconstruct full corpus text and maintain a map
        # We MUST account for chunk overlap (usually 50 chars) to avoid repeating text
        full_corpus_text = ""
        sentence_to_doc = [] # List of (char_start, char_end, doc_id)
        
        # Group by doc_id to handle multiple documents (pages) correctly
        from collections import defaultdict
        docs = defaultdict(list)
        for c in sorted_chunks:
            docs[c['doc_id']].append(c)
            
        curr_global_offset = 0
        from .utils import normalize_text
        for d_id in sorted(docs.keys()):
            doc_chunks = sorted(docs[d_id], key=lambda x: x['start'])
            # Stitch chunks within the document
            doc_text = ""
            for i, c in enumerate(doc_chunks):
                txt = c['text']
                if i == 0:
                    doc_text = txt
                else:
                    # Chunks are expected to overlap by ~50 chars. 
                    # We use the 'start' offset to find exactly where to append.
                    relative_start = c['start'] - doc_chunks[0]['start']
                    if relative_start < len(doc_text):
                        # Overlap exists, take only the NEW part
                        overlap_len = len(doc_text) - relative_start
                        doc_text += txt[overlap_len:]
                    else:
                        # Gap exists (unlikely in contiguous indexing)
                        doc_text += " " + txt
            
            # Apply de-hyphenation and general normalization
            doc_text = normalize_text(doc_text)
            
            start_in_full = curr_global_offset
            full_corpus_text += doc_text + "\n"
            end_in_full = len(full_corpus_text)
            sentence_to_doc.append((start_in_full, end_in_full, d_id))
            curr_global_offset = end_in_full

        pt_sentences = split_into_sentences(pt_trans)
        corpus_sentences = split_into_sentences(full_corpus_text)
        
        # Map corpus sentences back to their original pages
        corpus_sentence_data = [] # List of (text, doc_id)
        last_found = 0
        for s in corpus_sentences:
            idx = full_corpus_text.find(s, last_found)
            if idx != -1:
                # Find which page this offset belongs to
                sent_doc_id = vol_id # Fallback
                for start, end, d_id in sentence_to_doc:
                    if idx < end:
                        sent_doc_id = d_id
                        break
                corpus_sentence_data.append((s, sent_doc_id))
                last_found = idx + len(s)
        
        if not pt_sentences or not corpus_sentences:
            return []

        # Map sentences to their character offsets
        pt_sentence_data = []
        last_found = 0
        for s in pt_sentences:
            idx = pt_trans.find(s, last_found)
            if idx != -1:
                pt_sentence_data.append((s, idx, idx + len(s)))
                last_found = idx + len(s)

        logger.info(f"Deep Alignment: Encoding {len(pt_sentences)} query sentences...")
        q_embeddings = indexer.model.encode([d[0] for d in pt_sentence_data], convert_to_numpy=True)
        
        logger.info(f"Deep Alignment: Encoding {len(corpus_sentences)} corpus sentences...")
        c_embeddings = indexer.model.encode([d[0] for d in corpus_sentence_data], convert_to_numpy=True)

        # Normalize for cosine similarity
        q_embeddings = q_embeddings / np.linalg.norm(q_embeddings, axis=1, keepdims=True)
        c_embeddings = c_embeddings / np.linalg.norm(c_embeddings, axis=1, keepdims=True)

        # Compute Similarity Matrix
        sim_matrix = np.matmul(q_embeddings, c_embeddings.T)
        
        # JUMP START: Find a confident anchor in the first few sentences to skip volume preambles.
        # This is critical for 500-page volumes indexed as single documents.
        
        # Pre-extract page numbers for all corpus sentences for fast windowing
        corpus_pages = []
        for _, d_id in corpus_sentence_data:
            p_match = re.search(r'p(\d+)$', d_id)
            corpus_pages.append(int(p_match.group(1)) if p_match else 0)

        anchor_idx = 0
        found_anchor = False
        # Look for a strong match in the first 8 sentences, ignoring very short ones (likely headers)
        for i in range(min(8, len(pt_sentences))):
            if len(pt_sentences[i]) < 30: # Ignore short titles/headers for anchoring
                continue
            
            best_hit_idx = np.argmax(sim_matrix[i])
            if sim_matrix[i][best_hit_idx] > 0.75: # Found a confident start point
                anchor_idx = best_hit_idx
                found_anchor = True
                break
        
        # If no strong anchor found, fallback to the very first sentence's best hit 
        # only if it's reasonably strong, otherwise start at the beginning of the volume.
        if not found_anchor:
            first_best = np.argmax(sim_matrix[0])
            if sim_matrix[0][first_best] > 0.70:
                anchor_idx = first_best
            else:
                anchor_idx = 0

        # Initialize tracking for last match position (and page)
        last_idx = max(0, int(anchor_idx) - 2)
        last_page = corpus_pages[last_idx]
        
        logger.info(f"Alignment Jump-Start: Article likely begins around page {last_page} (corpus sentence {last_idx})")

        matches = []
        for q_idx, (q_text, q_start, q_end) in enumerate(pt_sentence_data):
            # Hybrid Monotonicity: 
            # 1. SCOPE: Allow searching back and forward up to 5 pages from the last match.
            # 2. SOFT ORDER: Give a small semantic boost to sentences that appear AFTER the last match.
            
            min_p = last_page - 5
            max_p = last_page + 5
            
            # Identify indices within the +/- 5 page window
            valid_indices = [i for i, p in enumerate(corpus_pages) if min_p <= p <= max_p]
            
            if not valid_indices:
                # If window is empty (e.g. at edges or gap), broadening search as fallback
                valid_indices = list(range(len(corpus_sentence_data)))
            
            # Slice similarity matrix for this sentence and window
            search_sims = sim_matrix[q_idx][valid_indices]
            
            # Apply Soft Order Boost: favor forward progression (+0.10 score boost)
            boosted_sims = search_sims.copy()
            for i, idx in enumerate(valid_indices):
                if idx >= last_idx:
                    boosted_sims[i] += 0.10

            # Hybrid Search: Check top 25 candidates (ranked by boosted score)
            top_k_rel_idx = np.argsort(boosted_sims)[-25:][::-1]
            
            for rel_c_idx in top_k_rel_idx:
                c_idx = valid_indices[rel_c_idx]
                sem_score = search_sims[rel_c_idx] # Use ORIGINAL score for tier thresholds
                best_cand, best_doc_id = corpus_sentence_data[c_idx]
                
                # Hybrid Scoring Metrics
                lex_set_score = fuzz.token_set_ratio(q_text, best_cand) / 100.0
                l1, l2 = len(q_text), len(best_cand)
                len_ratio = min(l1, l2) / max(l1, l2)
                
                # Hierarchical Precision Filter
                is_match = False
                tier = None
                
                # Tier A: Text reuse (>=0.80) OR same-language high-fidelity reprint (>0.95 lexical)
                if (sem_score >= 0.80 and len_ratio > 0.5) or (lex_set_score > 0.95 and len_ratio > 0.8):
                    is_match = True
                    tier = "Tier A: Text reuse"
                # Tier B: Text adaptation (>=0.70) OR same-language adaptation (>0.80 lexical)
                elif (sem_score >= 0.70 and lex_set_score > 0.3 and len_ratio > 0.4) or (lex_set_score > 0.80 and len_ratio > 0.5):
                    is_match = True
                    tier = "Tier B: Text adaptation"
                # Tier C: Contextual overlap (< 0.70)
                elif (sem_score > 0.45 and lex_set_score > 0.2) or (sem_score > 0.40 and lex_set_score > 0.40):
                    is_match = True
                    tier = "Tier C: Contextual overlap"

                if is_match:
                    # Update anchor for the next sentence
                    last_idx = c_idx
                    last_page = corpus_pages[c_idx]
                    
                    matches.append({
                        "q_idx": q_idx,
                        "c_idx": c_idx,
                        "query_chunk": q_text,
                        "query_range": (q_start, q_end),
                        "corpus_doc": {
                            "doc_id": best_doc_id,
                            "title": best_doc_id.split('/')[-1],
                            "text": best_cand
                        },
                        "score": int(max(sem_score * 100, lex_set_score * 100)),
                        "tier": tier,
                        "details": self.get_details(q_text, best_cand)
                    })
                    break # Found the best qualifying candidate for this sentence
        
        return matches

    def _compute_global_scores(self, results: List[Dict], total_len: int):
        if not results: return []
        
        ranges = [m['query_range'] for m in results]
        sorted_ranges = sorted(ranges)
        merged = []
        if sorted_ranges:
            curr_start, curr_end = sorted_ranges[0]
            for next_start, next_end in sorted_ranges[1:]:
                if next_start <= curr_end: # Overlap or touch
                    curr_end = max(curr_end, next_end)
                else:
                    merged.append((curr_start, curr_end))
                    curr_start, curr_end = next_start, next_end
            merged.append((curr_start, curr_end))
        
        coverage_len = sum(end - start for start, end in merged)
        article_score = (coverage_len / total_len) * 100.0 if total_len > 0 else 0
        
        for m in results:
            m['article_score'] = article_score
            
        return results
