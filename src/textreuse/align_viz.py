import difflib
import re
import numpy as np
from typing import List, Dict, Any, Tuple
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
        
        unique_pages = sorted(list(set(pages)))
        for p in unique_pages:
            count = sum(1 for hp in pages if p <= hp <= p + window_size)
            if count > best_count:
                best_count = count
                best_start = p
        
        return (max(1, best_start - 5), best_start + window_size + 10)

    def process_document(self, translated_text: str, indexer, include_corpora: list = None) -> List[Dict[str, Any]]:
        """
        Process a full document using a localized and monotonic approach:
        1. Identify winning volume and localized page window.
        2. Deep sentence alignment using soft-monotonicity.
        """
        chunks = indexer._chunk_text(translated_text)
        vol_coverage = {}
        page_hits = []
        
        logger.info(f"Identifying candidates for {len(chunks)} article chunks...")
        
        q_texts = [c[0] for c in chunks]
        batch_candidates = indexer.query_batch(q_texts, top_k=50, include_corpora=include_corpora)
        
        for candidates in batch_candidates:
            chunk_vols = {} 
            for cand in candidates:
                doc_id = cand['doc_id']
                vol_id = re.sub(r'[p/]\d+$', '', doc_id)
                vol_id = re.sub(r'\.pdf$', '', vol_id)
                
                if cand['distance'] > chunk_vols.get(vol_id, 0):
                    chunk_vols[vol_id] = cand['distance']
            
            for vol_id, score in chunk_vols.items():
                vol_coverage[vol_id] = vol_coverage.get(vol_id, 0) + score

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
            
        top_vols = sorted(vol_coverage.items(), key=lambda x: x[1], reverse=True)[:4]
        logger.info(f"Top 4 Volumes: {[v[0] for v in top_vols]}")
        
        all_volume_alignments = {}
        secondary_matches = [] 
        
        for rank, (vol_id, _) in enumerate(top_vols, 1):
            logger.info(f"Aligning Rank {rank}: {vol_id}")
            v_hits = [(p, d) for v, p, d in page_hits if v == vol_id and p > 0]
            page_window = None
            if v_hits:
                confident_hits = [h for h in v_hits if h[1] >= 0.70]
                if not confident_hits: confident_hits = [h for h in v_hits if h[1] > 0.5]
                pages = sorted([h[0] for h in confident_hits])
                if pages and (pages[-1] - pages[0]) > 50:
                    page_window = (max(1, pages[0] - 10), pages[-1] + 10)
                else:
                    page_window = self._find_best_window(confident_hits if confident_hits else v_hits)
            
            if page_window:
                aligned, corpus_sentences = self._align_sentences(translated_text, vol_id, indexer, page_window)
            else:
                aligned, corpus_sentences = self._align_sentences(translated_text, vol_id, indexer)
            
            aligned = self._compute_global_scores(aligned, len(translated_text))
            
            all_volume_alignments[rank] = {
                "vol_id": vol_id,
                "matches": aligned,
                "corpus_sentences": corpus_sentences
            }

        return all_volume_alignments, secondary_matches

    def align_single_volume(self, translated_text: str, vol_id: str, indexer) -> dict:
        """Directly align against a specific volume without semantic search."""
        logger.info(f"Direct alignment against: {vol_id}")
        aligned, corpus_sentences = self._align_sentences(translated_text, vol_id, indexer)
        aligned = self._compute_global_scores(aligned, len(translated_text))
        
        return {
            1: {
                "vol_id": vol_id,
                "matches": aligned,
                "corpus_sentences": corpus_sentences
            }
        }

    def _align_sentences(self, pt_trans: str, vol_id: str, indexer, page_window=None) -> Tuple[List[Dict], List[Dict]]:
        """Perform deep semantic sentence-level comparison against a volume."""
        doc_chunks = indexer.get_volume_chunks(vol_id)
        
        if page_window:
            filtered_chunks = []
            for c in doc_chunks:
                p_match = re.search(r'p(\d+)$', c['doc_id'])
                if p_match:
                    pno = int(p_match.group(1))
                    if page_window[0] <= pno <= page_window[1]:
                        filtered_chunks.append(c)
            doc_chunks = filtered_chunks

        sorted_chunks = sorted(doc_chunks, key=lambda x: (x['doc_id'], x['start']))
        
        from collections import defaultdict
        docs = defaultdict(list)
        for c in sorted_chunks:
            docs[c['doc_id']].append(c)
            
        full_corpus_text = ""
        sentence_to_doc = [] 
        curr_global_offset = 0
        from .utils import normalize_text
        for d_id in sorted(docs.keys()):
            doc_chunks = sorted(docs[d_id], key=lambda x: x['start'])
            doc_text = ""
            for i, c in enumerate(doc_chunks):
                txt = c['text']
                if i == 0:
                    doc_text = txt
                else:
                    relative_start = c['start'] - doc_chunks[0]['start']
                    if relative_start < len(doc_text):
                        overlap_len = len(doc_text) - relative_start
                        doc_text += txt[overlap_len:]
                    else:
                        doc_text += " " + txt
            
            doc_text = normalize_text(doc_text)
            start_in_full = curr_global_offset
            full_corpus_text += doc_text + "\n"
            end_in_full = len(full_corpus_text)
            sentence_to_doc.append((start_in_full, end_in_full, d_id))
            curr_global_offset = end_in_full

        pt_sentences = split_into_sentences(pt_trans)
        corpus_sentences = split_into_sentences(full_corpus_text)
        
        corpus_sentence_data = [] 
        last_found = 0
        for s in corpus_sentences:
            idx = full_corpus_text.find(s, last_found)
            if idx != -1:
                sent_doc_id = vol_id 
                for start, end, d_id in sentence_to_doc:
                    if idx < end:
                        sent_doc_id = d_id
                        break
                corpus_sentence_data.append((s, sent_doc_id))
                last_found = idx + len(s)
        
        if not pt_sentences or not corpus_sentences:
            return [], []

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

        q_embeddings = q_embeddings / np.linalg.norm(q_embeddings, axis=1, keepdims=True)
        c_embeddings = c_embeddings / np.linalg.norm(c_embeddings, axis=1, keepdims=True)

        sim_matrix = np.matmul(q_embeddings, c_embeddings.T)
        
        corpus_pages = []
        for _, d_id in corpus_sentence_data:
            p_match = re.search(r'p(\d+)$', d_id)
            corpus_pages.append(int(p_match.group(1)) if p_match else 0)

        anchor_idx = 0
        found_anchor = False
        for i in range(min(8, len(pt_sentences))):
            if len(pt_sentences[i]) < 30:
                continue
            best_hit_idx = np.argmax(sim_matrix[i])
            if sim_matrix[i][best_hit_idx] > 0.75:
                anchor_idx = best_hit_idx
                found_anchor = True
                break
        
        if not found_anchor:
            first_best = np.argmax(sim_matrix[0])
            anchor_idx = first_best if sim_matrix[0][first_best] > 0.70 else 0

        last_idx = max(0, int(anchor_idx) - 2)
        last_page = corpus_pages[last_idx]
        
        logger.info(f"Alignment Jump-Start: Article likely begins around page {last_page}")

        matches = []
        for q_idx, (q_text, q_start, q_end) in enumerate(pt_sentence_data):
            min_p = last_page - 5
            max_p = last_page + 5
            
            valid_indices = [i for i, p in enumerate(corpus_pages) if min_p <= p <= max_p]
            if not valid_indices:
                valid_indices = list(range(len(corpus_sentence_data)))
            
            search_sims = sim_matrix[q_idx][valid_indices]
            
            boosted_sims = search_sims.copy()
            for i, idx in enumerate(valid_indices):
                if idx >= last_idx:
                    boosted_sims[i] += 0.10

            top_k_rel_idx = np.argsort(boosted_sims)[-25:][::-1]
            
            for rel_c_idx in top_k_rel_idx:
                c_idx = valid_indices[rel_c_idx]
                sem_score = search_sims[rel_c_idx] 
                best_cand, best_doc_id = corpus_sentence_data[c_idx]
                
                lex_set_score = fuzz.token_set_ratio(q_text, best_cand) / 100.0
                l1, l2 = len(q_text), len(best_cand)
                len_ratio = min(l1, l2) / max(l1, l2)
                
                is_match = False
                tier = None
                
                if (sem_score >= 0.80 and len_ratio > 0.5) or (lex_set_score > 0.95 and len_ratio > 0.8):
                    is_match = True
                    tier = "Tier A: Text reuse"
                elif (sem_score >= 0.70 and lex_set_score > 0.3 and len_ratio > 0.4) or (lex_set_score > 0.80 and len_ratio > 0.5):
                    is_match = True
                    tier = "Tier B: Text adaptation"
                elif (sem_score > 0.45 and lex_set_score > 0.2) or (sem_score > 0.40 and lex_set_score > 0.40):
                    is_match = True
                    tier = "Tier C: Contextual overlap"

                if is_match:
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
                    break 
        
        return matches, corpus_sentence_data

    def _compute_global_scores(self, results: List[Dict], total_len: int):
        if not results: return []
        
        ranges = sorted([m['query_range'] for m in results])
        merged = []
        if ranges:
            curr_start, curr_end = ranges[0]
            for next_start, next_end in ranges[1:]:
                if next_start <= curr_end:
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
