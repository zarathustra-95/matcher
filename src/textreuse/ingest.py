import os
import json
from pypdf import PdfReader
from pathlib import Path
from typing import Iterator, Dict
from .utils import logger, normalize_text

class Ingestor:
    @staticmethod
    def read_file(path: str) -> str:
        """Read .txt, .md, or .pdf file."""
        p = Path(path)
        if not p.exists():
            logger.error(f"File not found: {path}")
            return ""
        
        try:
            if p.suffix.lower() == '.pdf':
                return Ingestor._read_pdf(p)
            else:
                with open(p, 'r', encoding='utf-8', errors='replace') as f:
                    return f.read()
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            return ""

    @staticmethod
    def _read_pdf(path: Path) -> str:
        text = []
        try:
            reader = PdfReader(str(path))
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text.append(extracted)
        except Exception as e:
            logger.error(f"Error parsing PDF {path}: {e}")
        return "\n".join(text)

    @staticmethod
    def iter_corpus(source: str) -> Iterator[Dict]:
        """Iterate over corpus files or a JSONL."""
        p = Path(source)
        if p.is_file() and p.suffix.lower() == '.jsonl':
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        # Be flexible with ID keys
                        doc_id = doc.get('id') or doc.get('filename') or doc.get('volume_id') or doc.get('relative_path')
                        
                        if 'text' in doc and doc_id:
                            doc['id'] = doc_id
                            doc['text'] = normalize_text(doc['text'])
                            yield doc
                    except json.JSONDecodeError:
                        continue
        elif p.is_dir():
            # Check for JSONL files first (standard structured corpus)
            jsonl_files = list(p.rglob('*.jsonl'))
            if jsonl_files:
                logger.info(f"Indexing {len(jsonl_files)} JSONL files from {source}...")
                for j_file in jsonl_files:
                    with open(j_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                doc = json.loads(line)
                                # Be flexible with ID keys
                                doc_id = doc.get('id') or doc.get('filename') or doc.get('volume_id') or doc.get('relative_path')
                                
                                if 'text' in doc and doc_id:
                                    doc['id'] = doc_id
                                    doc['text'] = normalize_text(doc['text'])
                                    yield doc
                            except json.JSONDecodeError:
                                continue
            else:
                # Text/PDF files fallback
                for fpath in p.rglob('*'):
                    if fpath.suffix.lower() in ['.txt', '.md', '.pdf']:
                        text = Ingestor.read_file(str(fpath))
                        if text:
                            yield {
                                "id": fpath.name,
                                "title": fpath.name,
                                "path": str(fpath),
                                "text": normalize_text(text)
                            }
        else:
            logger.error(f"Invalid corpus source: {source}")
