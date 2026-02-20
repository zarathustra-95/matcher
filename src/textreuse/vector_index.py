
import os
import sqlite3
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from .config import Config
from .utils import logger
from .ingest import Ingestor

class VectorIndexer:
    def __init__(self, config: Config, index_path: str = None):
        self.config = config
        self.index_dir = Path(index_path) if index_path else Path("index_data")
        self.db_path = self.index_dir / "metadata.db"
        self.faiss_path = self.index_dir / "index.faiss"
        
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        self.dimension = 768 
        self.index = None
        
    def _init_db(self):
        self.index_dir.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS metadata 
                     (key INTEGER PRIMARY KEY, doc_id TEXT, title TEXT, start INTEGER, end INTEGER, text TEXT)''')
        c.execute("CREATE INDEX IF NOT EXISTS idx_doc_id ON metadata (doc_id)")
        c.execute("PRAGMA synchronous = OFF")
        c.execute("PRAGMA journal_mode = WAL")
        conn.commit()
        conn.close()

    def clear(self):
        """Clear the existing index and metadata."""
        if self.faiss_path.exists():
            self.faiss_path.unlink()
        if self.db_path.exists():
            self.db_path.unlink()
        self.index = None
        logger.info("Index cleared.")

    def build_index(self, corpus_path: str, append: bool = False, corpus_prefix: str = None):
        corpus_path_p = Path(corpus_path)
        if corpus_prefix is None:
            corpus_prefix = corpus_path_p.stem 
        
        indexed_ids = set()
        if not append:
            self.clear()
            self._init_db()
            self.index = faiss.IndexFlatIP(self.dimension)
            start_key = 0
        else:
            if not self.load():
                logger.warning("Could not load existing index for append, starting fresh.")
                self._init_db()
                self.index = faiss.IndexFlatIP(self.dimension)
                start_key = 0
            else:
                conn = sqlite3.connect(self.db_path)
                c = conn.cursor()
                c.execute("SELECT MAX(key) FROM metadata")
                res = c.fetchone()
                start_key = (res[0] + 1) if res and res[0] is not None else 0
                
                logger.info("Fetching already indexed IDs...")
                c.execute("SELECT DISTINCT doc_id FROM metadata")
                indexed_ids = {r[0] for r in c.fetchall()}
                conn.close()
                logger.info(f"Resuming index. Found {len(indexed_ids)} documents already indexed.")
        
        logger.info(f"Building vector index from {corpus_path}...")
        
        current_chunks = []
        current_db_rows = []
        processed_count = 0
        
        def flush_batch():
            nonlocal processed_count
            if not current_chunks:
                return
            
            logger.info(f"Encoding batch of {len(current_chunks)} chunks...")
            embeddings = self.model.encode(current_chunks, batch_size=128, show_progress_bar=True, convert_to_numpy=True)
            faiss.normalize_L2(embeddings)
            self.index.add(embeddings)
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.executemany("INSERT OR REPLACE INTO metadata VALUES (?, ?, ?, ?, ?, ?)", current_db_rows)
            conn.commit()
            conn.close()
            
            processed_count += len(current_chunks)
            current_chunks.clear()
            current_db_rows.clear()

        for doc in Ingestor.iter_corpus(corpus_path):
            doc_id = doc['id']
            if not doc_id.startswith(f"{corpus_prefix}/") and not doc_id.startswith(f"{corpus_prefix}::"):
                doc_id = f"{corpus_prefix}/{doc_id}"
            
            if doc_id in indexed_ids:
                continue

            doc_chunks = self._chunk_text(doc['text'])
                
            for idx, (chunk_text, start, end) in enumerate(doc_chunks):
                current_chunks.append(chunk_text)
                key = start_key + processed_count + len(current_chunks) - 1
                current_db_rows.append((key, doc_id, doc.get('title', ''), start, end, chunk_text))
                
                if len(current_chunks) >= 100000:
                    flush_batch()
                    self.save() 
        
        flush_batch()
        
        if processed_count == 0 and not append:
            logger.warning("No chunks found to index.")
            return

        self.save()
        logger.info(f"Index updated. Total chunks in index: {self.index.ntotal}")

    def _chunk_text(self, text: str):
        res = []
        n = len(text)
        step = max(1, self.config.chunk_size - self.config.chunk_overlap)
        
        for i in range(0, n, step):
            end = min(i + self.config.chunk_size, n)
            chunk = text[i:end]
            if len(chunk) > 50:
                res.append((chunk, i, end))
        return res

    def save(self):
        faiss.write_index(self.index, str(self.faiss_path))

    def load(self):
        if self.faiss_path.exists() and self.db_path.exists():
            import os
            omp_threads = os.environ.get("OMP_NUM_THREADS", "16")
            os.environ["OMP_NUM_THREADS"] = omp_threads
            faiss.omp_set_num_threads(int(omp_threads))
            
            self.index = faiss.read_index(str(self.faiss_path))
            logger.info(f"Vector index loaded. OMP threads: {omp_threads}")
            return True
        return False

    def query(self, text: str, top_k: int = 5, include_corpora: list = None):
        results = self.query_batch([text], top_k=top_k, include_corpora=include_corpora)
        return results[0] if results else []

    def query_batch(self, texts: list[str], top_k: int = 5, include_corpora: list = None):
        if self.index is None:
            if not self.load():
                return [[] for _ in texts]
                
        query_vecs = self.model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(query_vecs)
        
        distances, indices = self.index.search(query_vecs, top_k * 10 if include_corpora else top_k)
        
        all_keys = [int(idx) for row in indices for idx in row if idx != -1]
        if not all_keys:
            return [[] for _ in texts]
            
        unique_keys = list(set(all_keys))
        key_to_meta = {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for i in range(0, len(unique_keys), 999):
            batch = unique_keys[i:i+999]
            placeholders = ",".join("?" * len(batch))
            cursor.execute(f"SELECT key, doc_id, title, start, end, text FROM metadata WHERE key IN ({placeholders})", batch)
            for r in cursor.fetchall():
                key_to_meta[r[0]] = r[1:]
        conn.close()
            
        batch_results = []
        for i in range(len(texts)):
            res_list = []
            for j, idx in enumerate(indices[i]):
                if idx in key_to_meta:
                    meta = key_to_meta[idx]
                    doc_id = meta[0]
                    if include_corpora and not any(doc_id.startswith(f"{c}/") or doc_id.startswith(f"{c}::") for c in include_corpora):
                        continue
                    res_list.append({
                        "doc_id": doc_id, "title": meta[1], "start": meta[2], "end": meta[3], "text": meta[4], "distance": float(distances[i][j])
                    })
                    if len(res_list) >= top_k: break
            batch_results.append(res_list)
        return batch_results

    def get_document_chunks(self, doc_id: str):
        results = []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT title, start, end, text, doc_id FROM metadata WHERE doc_id = ?", (doc_id,))
        for r in c.fetchall():
            results.append({"doc_id": r[4], "title": r[0], "start": r[1], "end": r[2], "text": r[3]})
        conn.close()
        return results

    def get_volume_chunks(self, volume_prefix: str):
        results = []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT title, start, end, text, doc_id FROM metadata WHERE doc_id LIKE ?", (f"{volume_prefix}%",))
        for r in c.fetchall():
            results.append({"doc_id": r[4], "title": r[0], "start": r[1], "end": r[2], "text": r[3]})
        conn.close()
        return results

class MultiLanguageIndexer:
    LANGUAGE_INDEX_MAP = {
        "French": "index_french", "English": "index_english", "Portuguese": "index_portuguese",
        "Italian": "index_italian", "Spanish": "index_spanish", "German": "index_german",
        "French_Newspapers": "index_french_newspapers"
    }
    
    def __init__(self, config: Config, base_dir: str = "."):
        self.config = config
        self.base_dir = Path(base_dir)
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer('sentence-transformers/LaBSE')
        self._indices = {}

    def _chunk_text(self, text: str):
        res = []
        n = len(text)
        step = self.config.chunk_size - self.config.chunk_overlap
        if step < 1: step = 1
        for i in range(0, n, step):
            end = min(i + self.config.chunk_size, n)
            chunk = text[i:end]
            if len(chunk) > 50:
                res.append((chunk, i, end))
        return res

    def _get_index(self, language: str):
        """Lazy-load a language-specific index with multithreading support."""
        if language not in self._indices:
            index_dir_name = self.LANGUAGE_INDEX_MAP.get(language)
            if not index_dir_name:
                raise ValueError(f"Unknown language: {language}")
            
            index_dir = self.base_dir / index_dir_name
            faiss_path, db_path = index_dir / "index.faiss", index_dir / "metadata.db"
            
            if not faiss_path.exists():
                raise FileNotFoundError(f"Index not found: {faiss_path}")
            
            logger.info(f"Loading {language} index from {index_dir}...")
            
            # Enable multi-threaded search
            omp_threads = os.environ.get("OMP_NUM_THREADS", "16")
            os.environ["OMP_NUM_THREADS"] = omp_threads
            faiss.omp_set_num_threads(int(omp_threads))
            
            index = faiss.read_index(str(faiss_path))
            logger.info(f"Loaded {language} index: {index.ntotal:,} vectors")
            
            self._indices[language] = (index, db_path)
        
        return self._indices[language]
    
    def query_batch(self, texts: list[str], top_k: int = 5, include_corpora: list = None):
        if not texts:
            return []
            
        if not include_corpora:
            include_corpora = list(self.LANGUAGE_INDEX_MAP.keys())
        
        query_vecs = self.model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True)
        faiss.normalize_L2(query_vecs)
        
        n_queries = len(texts)
        
        # Collect results from each language
        all_results = [[] for _ in range(n_queries)]
        
        for language in include_corpora:
            if language not in self.LANGUAGE_INDEX_MAP:
                logger.warning(f"Unknown language in include_corpora: {language}")
                continue
            
            try:
                index, db_path = self._get_index(language)
            except FileNotFoundError as e:
                logger.warning(f"Skipping {language}: {e}")
                continue
            
            # Search this language's index
            distances, indices = index.search(query_vecs, top_k)
            
            # Get metadata for results
            all_keys = [int(idx) for row in indices for idx in row if idx != -1]
            if not all_keys:
                continue
            
            unique_keys = list(set(all_keys))
            key_to_meta = {}
            
            conn = sqlite3.connect(db_path)
            try:
                cursor = conn.cursor()
                for i in range(0, len(unique_keys), 999):
                    batch = unique_keys[i:i+999]
                    placeholders = ",".join("?" * len(batch))
                    cursor.execute(
                        f"SELECT key, doc_id, title, start, end, text FROM metadata WHERE key IN ({placeholders})",
                        batch
                    )
                    for r in cursor.fetchall():
                        key_to_meta[r[0]] = r[1:]
            finally:
                conn.close()
            
            # Add to results
            for i in range(n_queries):
                for j, idx in enumerate(indices[i]):
                    if idx != -1 and idx in key_to_meta:
                        meta = key_to_meta[idx]
                        all_results[i].append({
                            "doc_id": meta[0],
                            "title": meta[1],
                            "start": meta[2],
                            "end": meta[3],
                            "text": meta[4],
                            "distance": float(distances[i][j])
                        })
        
        # Sort by distance (highest first for inner product) and take top_k
        final_results = []
        for results in all_results:
            sorted_results = sorted(results, key=lambda x: x["distance"], reverse=True)
            final_results.append(sorted_results[:top_k])
        
        return final_results

    def query(self, text: str, top_k: int = 5, include_corpora: list = None):
        return self.query_batch([text], top_k=top_k, include_corpora=include_corpora)[0]

    def get_volume_chunks(self, volume_prefix: str):
        # Determine language from prefix
        language = None
        for lang in self.LANGUAGE_INDEX_MAP.keys():
            if volume_prefix.startswith(f"{lang}/") or volume_prefix.startswith(f"{lang}::"):
                language = lang
                break
        if not language: return []
        try:
            _, db_path = self._get_index(language)
            results = []
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT title, start, end, text, doc_id FROM metadata WHERE doc_id LIKE ?", (f"{volume_prefix}%",))
            for r in c.fetchall():
                results.append({"doc_id": r[4], "title": r[0], "start": r[1], "end": r[2], "text": r[3]})
            conn.close()
            return results
        except Exception: return []
