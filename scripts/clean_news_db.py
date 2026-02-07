#!/usr/bin/env python
"""Clean up old news items without URLs from the news database."""

import json
from pathlib import Path
from datetime import datetime

NEWS_DB = Path(__file__).resolve().parents[1] / 'db' / 'news'

def clean_news_database(dry_run=True):
    """Remove news items without URLs."""
    
    # Get all news files
    news_files = sorted(NEWS_DB.glob('news_*.jsonl'))
    
    if not news_files:
        print("No news files found to clean.")
        return
    
    total_removed = 0
    total_items = 0
    
    for news_file in news_files:
        print(f"\nProcessing: {news_file.name}")
        
        # Read and filter
        with open(news_file) as f:
            lines = f.readlines()
        
        filtered_lines = []
        file_removed = 0
        file_total = len(lines)
        total_items += file_total
        
        for line in lines:
            rec = json.loads(line)
            src = rec.get('source', '')
            url = rec.get('url', '')
            
            # Filter out items without URLs (typically old finviz_general items)
            if url:  # Only keep items with URLs
                filtered_lines.append(json.dumps(rec, ensure_ascii=False) + '\n')
            else:
                file_removed += 1
        
        total_removed += file_removed
        
        print(f"  Total items: {file_total}")
        print(f"  Removed (no URL): {file_removed}")
        print(f"  Remaining: {len(filtered_lines)}")
        
        # Write back if not dry run
        if not dry_run:
            with open(news_file, 'w', encoding='utf-8') as f:
                f.writelines(filtered_lines)
            print(f"  ✅ File updated")
        else:
            print(f"  ⚠️  Dry run - file not modified")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total items processed: {total_items}")
    print(f"  Total items removed: {total_removed}")
    print(f"  Total items remaining: {total_items - total_removed}")
    print(f"{'='*60}")
    
    if dry_run:
        print("\n⚠️  DRY RUN - No files were modified.")
        print("   Run with --force to apply changes.")

if __name__ == '__main__':
    import sys
    
    dry_run = '--force' not in sys.argv
    
    if dry_run:
        print("=" * 60)
        print("DRY RUN MODE")
        print("=" * 60)
        print("\nThis will show what would be removed without making changes.")
        print("Run with --force to apply changes.\n")
    else:
        print("=" * 60)
        print("FORCE MODE - Will modify files!")
        print("=" * 60)
        print("\n⚠️  This will permanently remove items without URLs.\n")
    
    clean_news_database(dry_run=dry_run)
