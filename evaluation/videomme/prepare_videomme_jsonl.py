import argparse
import json
import re
from pathlib import Path

import pandas as pd


def load_subtitles(sub_path: Path) -> str:
    """Parse an SRT-like subtitle file and return a single text string."""
    text_lines = []

    with sub_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # skip index lines ("17", "18", ...)
            if re.match(r"^\d+$", line):
                continue
            # skip timestamp lines ("00:00:32,790 --> 00:00:38,219")
            if "-->" in line:
                continue
            # remove <font ...> tags
            line = re.sub(r"</?font[^>]*>", "", line)
            if line:
                text_lines.append(line)

    text = " ".join(text_lines)
    # normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def row_options_to_list(opts):
    """Make sure options are a Python list of strings."""
    if hasattr(opts, "tolist"):  # numpy array / pandas Series
        opts = opts.tolist()
    if isinstance(opts, (list, tuple)):
        return [str(o).strip() for o in opts]
    # fallback: single string with separators
    return [s.strip() for s in str(opts).split("||") if s.strip()]


def main(parquet_path, subtitles_dir, output_path):
    parquet_path = Path(parquet_path)
    subtitles_dir = Path(subtitles_dir)
    output_path = Path(output_path)

    df = pd.read_parquet(parquet_path)

    with output_path.open("w", encoding="utf-8") as out_f:
        for _, row in df.iterrows():
            video_id = str(row["video_id"])
            videoID = str(row["videoID"])

            # Try several possible subtitle filenames; adapt if needed
            candidates = [
                subtitles_dir / f"{videoID}.srt",
                subtitles_dir / f"{videoID}.txt",
                subtitles_dir / f"{videoID}.vtt",
                subtitles_dir / f"{video_id}.srt",
                subtitles_dir / f"{video_id}.txt",
            ]

            subtitle_text = ""
            for cand in candidates:
                if cand.exists():
                    subtitle_text = load_subtitles(cand)
                    break  # found one, stop searching

            record = {
                "video_id": video_id,
                "duration": str(row["duration"]),
                "domain": str(row["domain"]),
                "sub_category": str(row["sub_category"]),
                "url": str(row["url"]),
                "videoID": videoID,
                "question_id": str(row["question_id"]),
                "task_type": str(row["task_type"]),
                "question": str(row["question"]),
                "options": row_options_to_list(row["options"]),
                "answer": str(row["answer"]),
                "subtitles": subtitle_text,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True,
                        help="Path to test-00000-of-00001.parquet")
    parser.add_argument("--subtitles_dir", required=True,
                        help="Directory containing subtitle files")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path")
    args = parser.parse_args()

    main(args.parquet, args.subtitles_dir, args.output)
