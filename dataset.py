from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from tokenizer import build_tokenizer, encode_midi_file_to_ids


def iter_midi(dir: Path):
    for ext in (".mid", ".midi", ".MID", ".MIDI"):
        yield from dir.glob(f"**/*{ext}")


def process(midi: Path, out: Path):
    try:
        tok = build_tokenizer()
        ids = encode_midi_file_to_ids(midi, tok)
        if not ids:
            return None
        out_path = out / (midi.stem + ".npy")
        np.save(out_path, np.asarray(ids, dtype=np.int32))
        return out_path
    
    except Exception as e:
        print(f"error {midi}: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", default="data_midi")
    parser.add_argument("--out", default="tokens")
    parser.add_argument("--single", default=None)
    args = parser.parse_args()

    midi_dir = Path(args.midi)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.single:
        p = process(Path(args.single), out_dir)
        print(f"saved: {p}" if p else "Failed")

    else:

        cnt = 0
        for midi in iter_midi(midi_dir):
            if process(midi, out_dir):
                cnt += 1
                if cnt % 50 == 0:
                    print(f"{cnt} files...")
        print(f"done {cnt} saved to {out_dir}")
