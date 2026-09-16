"""
Run Demo — Adaptive Multi-Hop GraphRAG with Andrew Huberman Transcript
=======================================================================
Ingests 'Andrew_Huberman_Raj_Shamani_Transcript.txt' and queries the pipeline.
"""

import re
import os
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.adaptive_pipeline import AdaptiveGraphRAG, TextChunk


def parse_timestamped_transcript(file_path: str, chunk_size_words: int = 250) -> list:
    """Parse transcript with timestamp markers into TextChunk objects."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Transcript file not found: {file_path}")

    text_content = path.read_text(encoding="utf-8", errors="replace")
    lines = text_content.splitlines()

    # Extract lines with timestamps
    parsed_lines = []
    for line in lines:
        match = re.match(r"\[(\d{2}:\d{2}:\d{2})\]\s*(.*)", line.strip())
        if match:
            ts, text = match.group(1), match.group(2).strip()
            if text:
                parsed_lines.append((ts, text))
        elif line.strip() and not line.startswith("=") and not line.startswith("TRANSCRIPT:") and not line.startswith("Video URL:") and not line.startswith("Channel:"):
            # Plain text line without timestamp
            ts = parsed_lines[-1][0] if parsed_lines else "00:00:00"
            parsed_lines.append((ts, line.strip()))

    if not parsed_lines:
        # Fallback: treat whole file as plain text
        words = text_content.split()
        chunks = []
        for i in range(0, len(words), chunk_size_words):
            chunk_text = " ".join(words[i:i + chunk_size_words])
            chunks.append(TextChunk(
                chunk_id=len(chunks),
                text=chunk_text,
                start_time="00:00:00",
                end_time="00:00:00",
                char_start=0,
                char_end=len(chunk_text),
                source=path.stem,
                token_count=len(chunk_text.split()),
            ))
        return chunks

    # Group into ~250 word chunks
    chunks = []
    current_words = []
    start_ts = parsed_lines[0][0]
    end_ts = parsed_lines[0][0]
    chunk_id = 0

    for ts, line_text in parsed_lines:
        line_words = line_text.split()
        current_words.extend(line_words)
        end_ts = ts

        if len(current_words) >= chunk_size_words:
            chunk_text = " ".join(current_words)
            chunks.append(TextChunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start_time=start_ts,
                end_time=end_ts,
                char_start=0,
                char_end=len(chunk_text),
                source=path.stem,
                token_count=len(current_words),
            ))
            chunk_id += 1
            # Keep ~50 word overlap
            current_words = current_words[-50:]
            start_ts = ts

    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append(TextChunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start_time=start_ts,
            end_time=end_ts,
            char_start=0,
            char_end=len(chunk_text),
            source=path.stem,
            token_count=len(current_words),
        ))

    return chunks


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive Multi-Hop GraphRAG Demo")
    parser.add_argument("--transcript", type=str, default=None, help="Path to transcript file (.txt or .srt)")
    parser.add_argument("--query", type=str, default=None, help="Query to run against the pipeline")
    args = parser.parse_args()

    if not args.transcript:
        # Check if any .txt or .srt files exist in current directory
        local_files = [f for f in os.listdir(".") if f.endswith((".txt", ".srt")) and not f.startswith(".")]
        if local_files:
            transcript_path = local_files[0]
            print(f"[INFO] No --transcript argument supplied. Defaulting to local file: {transcript_path}")
        else:
            print("[USAGE] Please specify a transcript file:")
            print("  python run_demo.py --transcript path/to/transcript.txt")
            print("  python run_demo.py --transcript path/to/transcript.txt --query \"What are the main topics?\"")
            return
    else:
        transcript_path = args.transcript

    print(f"[INFO] Parsing transcript from: {transcript_path}")
    chunks = parse_timestamped_transcript(transcript_path)
    print(f"[INFO] Created {len(chunks)} text chunks.")

    # Initialize RAG system
    rag = AdaptiveGraphRAG(persist_dir="./graph_store")

    # Ingest chunks
    source_name = Path(transcript_path).stem
    stats = rag.ingest_chunks(chunks, source_name=source_name)
    print(f"[INFO] Ingest stats: {stats}")

    if args.query:
        sample_queries = [args.query]
    else:
        sample_queries = [
            "What are the main topics and key insights discussed in this transcript?",
            "What core ideas or recommendations does the speaker emphasize?",
        ]

    for q in sample_queries:
        print("\n" + "=" * 80)
        print(f"QUESTION: {q}")
        print("=" * 80)
        res = rag.query(q, mode="auto")
        print(f"Classification: {res['classification'].query_type} (confidence: {res['classification'].confidence:.2f})")
        print(f"Entities Involved: {res['entities'][:10]}")
        print("\nReasoning Chain:\n" + res["reasoning_path_text"])
        print("\nContext Preview:\n" + res["context"][:600] + "...\n")


if __name__ == "__main__":
    main()
