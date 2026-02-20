import click
import os
from pathlib import Path
from tqdm import tqdm
from .config import Config
from .utils import logger, clean_input_text

@click.group()
def main():
    """Text Reuse Detection Tool"""
    pass

@main.command()
@click.option('--corpus', required=True, help='Corpus directory or jsonl')
@click.option('--index_dir', default='index_data', help='Output directory for index')
@click.option('--append', is_flag=True, help='Append to existing index')
@click.option('--prefix', help='Custom prefix for doc_ids')
def index(corpus, index_dir, append, prefix):
    """Build Semantic FAISS index for corpus."""
    from .vector_index import VectorIndexer
    config = Config()
    idx = VectorIndexer(config, index_dir)
    idx.build_index(corpus, append=append, corpus_prefix=prefix)

@main.command()
@click.option('--pt_dir', required=True, help='Input Portuguese articles')
@click.option('--corpus', required=False, help='Corpus directory or jsonl (optional if index exists)')
@click.option('--out_dir', required=True, help='Output directory')
@click.option('--index_dir', default='index_data', help='Index directory')
@click.option('--min_score', default=50, help='Minimum match score')
@click.option('--include_corpora', help='Comma-separated list of corpora to include (e.g. French,Spanish)')
@click.option('--viz', is_flag=True, help='Generate interactive alignment visualizations')
def run(pt_dir, corpus, out_dir, index_dir, min_score, include_corpora, viz):
    """Direct Cross-Lingual Alignment: Match Portuguese to Multi-lingual Corpus."""
    from .ingest import Ingestor
    
    if viz:
        from .align_viz import Aligner
        from .report_viz import Reporter
    else:
        from .align import Aligner
        from .report import Reporter
        
    from .vector_index import MultiLanguageIndexer
    
    config = Config()
    config.min_score = min_score
    
    indexer = MultiLanguageIndexer(config, base_dir=index_dir)
        
    aligner = Aligner(config)
    reporter = Reporter(out_dir)
    
    include_list = None
    if include_corpora:
        include_list = [c.strip() for c in include_corpora.split(',')]
        logger.info(f"Filtering by corpora: {include_list}")
        
    pt_path = Path(pt_dir)
    pts = [pt_path] if pt_path.is_file() else list(pt_path.rglob("*"))
        
    for f in tqdm(pts, desc="Processing"):
        if f.suffix.lower() not in ['.txt', '.md', '.pdf']:
            continue
            
        text_pt = Ingestor.read_file(str(f))
        if not text_pt: continue
        
        text_pt = clean_input_text(text_pt)
        pt_file = f.name
        
        from .utils import split_into_sentences
        pt_sentences = split_into_sentences(text_pt)
        text_pt = "\n".join(pt_sentences)
        
        if viz:
            all_volume_alignments, secondary_matches = aligner.process_document(text_pt, indexer, include_corpora=include_list)
            reporter.save_results(pt_file, text_pt, text_pt, all_volume_alignments, secondary_matches, used_corpora=include_list)
        else:
            matches, secondary_matches = aligner.process_document(text_pt, indexer, include_corpora=include_list)
            reporter.save_results(pt_file, text_pt, text_pt, matches, secondary_matches, used_corpora=include_list)
            
    logger.info("Done.")

if __name__ == '__main__':
    main()
