from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

try:
    import miditok
    from miditok import TokenizerConfig  
except Exception as exc: 
    raise RuntimeError(
        "MidiTok v3 must be installed. Try pip install miditok"
    ) from exc



DEFAULT_CONFIG_DIR = Path("tokenizer_config")
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_DIR / "config.json"


def _ensure_dirs() -> None:
    DEFAULT_CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def build_tokenizer(config_path: Optional[os.PathLike] = None):
    _ensure_dirs()
    cfg_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg_dict = json.load(f)
            config = _config_from_json(cfg_dict)
        except Exception:
            
            config = None  
    else:
        config = None  

    if config is None:
        
        config = TokenizerConfig(
            use_chords=False, 
            use_rests=True,
            use_tempos=True,
            use_time_signatures=False,
            use_programs=True,
            one_token_stream=True, 
     
            beat_res={(0, 2): 8, (2, 4): 8, (4, 8): 4, (8, 12): 4, (12, 16): 4},
            beat_res_rest={(0, 0.5): 1, (0.5, 1): 2, (1, 2): 4, (2, 4): 2},
            tempo_range=(60, 180), 
            default_note_duration=2.0,  
            use_pitch_intervals=False,
            use_pitchdrum_tokens=False,
        )

        _atomic_write_json(cfg_path, _config_to_json(config))

    if hasattr(miditok, "REMI"):
        tokenizer = miditok.REMI(config) 
    else:  
        raise RuntimeError("MidiTok REMI tokenizer not found in this version.")

    return tokenizer


def save_config(tokenizer, config_path: Optional[os.PathLike] = None) -> None:
    
    cfg = getattr(tokenizer, "config", None)
    if cfg is None:
        raise RuntimeError("Tokenizer has no 'config' attribute to save.")

    _ensure_dirs()
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    _atomic_write_json(path, _config_to_json(cfg))


def _to_jsonable(val: Any) -> Any:
    if isinstance(val, dict):
        return {str(k): _to_jsonable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(v) for v in val]
    if isinstance(val, set):
        return [_to_jsonable(v) for v in sorted(list(val), key=lambda x: str(x))]
    if isinstance(val, Path):
        return str(val)
    return val


def _config_to_json(cfg: TokenizerConfig) -> Dict[str, Any]:
    data = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    beat_res = data.get("beat_res")
    if isinstance(beat_res, dict):
        conv: Dict[str, Any] = {}
        for k, v in beat_res.items():
            if isinstance(k, tuple) and len(k) == 2:
                conv[f"{k[0]}-{k[1]}"] = v
            else:
                conv[str(k)] = v
        data["beat_res"] = conv
    tsr = data.get("time_signature_range")

    if isinstance(tsr, dict):
        data["time_signature_range"] = {str(k): _to_jsonable(v) for k, v in tsr.items()}
    brr = data.get("beat_res_rest")

    if isinstance(brr, dict):
        conv_brr: Dict[str, Any] = {}
        for k, v in brr.items():
            if isinstance(k, tuple) and len(k) == 2:
                conv_brr[f"({k[0]}, {k[1]})"] = v
            else:
                conv_brr[str(k)] = v
        data["beat_res_rest"] = conv_brr

    if isinstance(data.get("tempo_range"), tuple):
        a, b = data["tempo_range"]
        data["tempo_range"] = [a, b]
    return _to_jsonable(data)


def _config_from_json(data: Dict[str, Any]) -> TokenizerConfig:
    beat_res = data.get("beat_res")
    if isinstance(beat_res, dict):
        conv: Dict[Tuple[int, int], int] = {}
        for k, v in beat_res.items():

            if isinstance(k, str) and "-" in k:
                a, b = k.split("-", 1)
                conv[(int(a), int(b))] = int(v)
            else:

                pass

        data["beat_res"] = conv
    tempo_range = data.get("tempo_range")

    if isinstance(tempo_range, list) and len(tempo_range) == 2:
        data["tempo_range"] = (int(tempo_range[0]), int(tempo_range[1]))
    tsr = data.get("time_signature_range")

    if isinstance(tsr, dict):
        data["time_signature_range"] = {int(k): v for k, v in tsr.items()}
    brr = data.get("beat_res_rest")
    
    if isinstance(brr, dict):
        conv_brr: Dict[Tuple[int, int], int] = {}
        for k, v in brr.items():

            if isinstance(k, str) and k.startswith("(") and "," in k and k.endswith(")"):

                try:
                    a_str, b_str = k[1:-1].split(",")
                    conv_brr[(int(a_str.strip()), int(b_str.strip()))] = int(v)
                except Exception:
                    continue

        if conv_brr:
            data["beat_res_rest"] = conv_brr
    return TokenizerConfig(**data)


def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def encode_midi_file_to_ids(midi_path: os.PathLike, tokenizer) -> List[int]:
    token_seqs = tokenizer.encode(str(midi_path))
    streams: List[List[int]] = []
    if isinstance(token_seqs, list):
        for seq in token_seqs:

            if hasattr(seq, "ids"):
                streams.append(list(map(int, seq.ids)))
            else:
                streams.append([int(x) for x in seq])

    else:
        seq = token_seqs
        if hasattr(seq, "ids"):
            streams.append(list(map(int, seq.ids)))
        else:
            streams.append([int(x) for x in seq])

    flat: List[int] = [tok for stream in streams for tok in stream]
    return flat


def decode_ids_to_midi(token_ids: List[int], tokenizer, out_path: os.PathLike) -> None:
 
    if hasattr(miditok, "TokSequence"):
        seq = miditok.TokSequence(ids=[int(t) for t in token_ids]) 
        score = tokenizer.decode(seq)

    else: 

        score = tokenizer.decode([int(t) for t in token_ids])

    if hasattr(score, "dump_midi"):
        score.dump_midi(str(out_path))
        return
    
    try:

        midi = tokenizer.tokens_to_midi([seq]) 
        midi.dump(str(out_path))

    except Exception as exc: 

        raise RuntimeError("Failed to save decoded MIDI") from exc


def load_vocab_size(tokenizer) -> int:
   
    if hasattr(tokenizer, "vocab_size"):
        return int(tokenizer.vocab_size) 
    if hasattr(tokenizer, "vocab") and hasattr(tokenizer.vocab, "__len__"):
        return int(len(tokenizer.vocab))
    if hasattr(tokenizer, "_vocab_base") and hasattr(tokenizer._vocab_base, "__len__"):
        return int(len(tokenizer._vocab_base))
    raise RuntimeError("Unable to determine tokenizer vocabulary size.")


__all__ = [
    "build_tokenizer",
    "save_config",
    "encode_midi_file_to_ids",
    "decode_ids_to_midi",
    "load_vocab_size",
]


