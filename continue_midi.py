
from __future__ import annotations
import torch
import random
from pathlib import Path
from miditok import REMI
from miditoolkit import MidiFile

from model import GPTModel
from tokenizer import build_tokenizer, load_vocab_size, decode_ids_to_midi


def continue_midi(input_midi: str, output_midi: str, continue_bars: int = 8, 
                  temperature: float = 0.7, scale: str = 'C_major', preview_callback=None):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = build_tokenizer()
    
    ckpt_data = torch.load("checkpoints/best_model.pt", map_location=device)
    model = GPTModel(vocab_size=407, d_model=512, n_heads=8, n_layers=12, 
                   max_seq_len=1024, d_ff_mult=4)
    model.load_state_dict(ckpt_data["model"], strict=False)
    model = model.to(device)
    model.eval()
    
    midi = MidiFile(input_midi)
    
    input_tokens = tokenizer(midi)
    
    if hasattr(input_tokens, 'ids'):
        original_seq = input_tokens.ids
    elif isinstance(input_tokens, list):
        original_seq = input_tokens
    else:
        original_seq = input_tokens[0].ids if len(input_tokens) > 0 else []
    
    bar_token_id = None
    for token_name, token_id in tokenizer.vocab.items():
        if token_name == 'Bar_None':
            bar_token_id = token_id
            break
    
    context_len = min(400, len(original_seq))
    start_idx = len(original_seq) - context_len
    
    if bar_token_id is not None:
        for i in range(start_idx, min(start_idx + 50, len(original_seq))):
            if original_seq[i] == bar_token_id:
                start_idx = i
                break
    
    token_sequence = original_seq[start_idx:].copy() 
    actual_context_len = len(token_sequence)
    
    
    pitch_classes_used = []
    recent_pitches = []
    for tid in original_seq[-100:]: 
        tname = [k for k, v in tokenizer.vocab.items() if v == tid]
        if tname:
            tname = tname[0]
            if tname.startswith('Pitch_'):
                pitch = int(tname.split('_')[1])
                recent_pitches.append(pitch)
                pitch_classes_used.append(pitch % 12)
    
    from music_theory import SCALES
    if pitch_classes_used:
   
        best_scale = scale
        best_match = 0
        for scale_name, scale_notes_temp in SCALES.items():
            match_count = sum(1 for pc in pitch_classes_used if pc in scale_notes_temp)
            if match_count > best_match:
                best_match = match_count
                best_scale = scale_name
        scale = best_scale
    
    scale_notes = SCALES.get(scale, SCALES['C_major'])
    
    min_pitch = min(recent_pitches) if recent_pitches else 48
    max_pitch = max(recent_pitches) if recent_pitches else 84
    
    durations_used = []
    positions_used = []
    velocities_used = []
    for tid in original_seq[-200:]:
        tname = [k for k, v in tokenizer.vocab.items() if v == tid]
        if tname:
            tname = tname[0]
            if tname.startswith('Duration_'):
                durations_used.append(tname)
            elif tname.startswith('Position_'):
                positions_used.append(int(tname.split('_')[1]))
            elif tname.startswith('Velocity_'):
                velocities_used.append(int(float(tname.split('_')[1])))
    
    avg_velocity = sum(velocities_used) // len(velocities_used) if velocities_used else 70
    
    max_new_tokens = continue_bars * 30  
    starting_length = len(token_sequence) 
    
    
    for step in range(max_new_tokens):
        if len(token_sequence) >= starting_length + max_new_tokens: 
            break
            
        inp = torch.tensor([token_sequence[-512:]], dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, _ = model(inp, None)
        

        nxt_logits = logits[0, -1] / (temperature * 0.8) 
        
   
        for tn, tid in tokenizer.vocab.items():
            if tn.startswith('Pitch_'):
                pitch = int(tn.split('_')[1])
                pitch_class = pitch % 12
                
          
                if pitch_class in scale_notes:
                    nxt_logits[tid] *= 2.0  
                
      
                if min_pitch <= pitch <= max_pitch:
                    nxt_logits[tid] *= 1.5
                
           
                if pitch < min_pitch - 12 or pitch > max_pitch + 12:
                    nxt_logits[tid] *= 0.1
                
           
                if tid in token_sequence[-3:]:
                    nxt_logits[tid] *= 0.2
        
      
        probs = torch.softmax(nxt_logits, dim=-1)
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cum = sorted_probs.cumsum(dim=-1)
        mask = cum > 0.9
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        nxt_logits[sorted_idx[mask]] = float('-inf')
        
        probs = torch.softmax(nxt_logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        
      
        tn = [k for k, v in tokenizer.vocab.items() if v == nxt][0]
        tt = tn.split('_')[0]
        
        token_sequence.append(nxt)
        
  
        if tt == 'Pitch':
            import random
            base_pitch = int(tn.split('_')[1])
            
            cur_pos = 0
            for t in token_sequence[-32:]:
                tname = [k for k, v in tokenizer.vocab.items() if v == t][0]
                if tname.startswith('Position_'):
                    cur_pos = int(tname.split('_')[1])
            
            if durations_used:
                dur = random.choice(durations_used)
                dur_tok = tokenizer.vocab[dur]
                dur_str = dur
            else:
                dur_opts = ['Duration_0.4.8', 'Duration_0.5.8', 'Duration_0.6.8']
                avail = [d for d in dur_opts if d in tokenizer.vocab]
                if avail:
                    dur = random.choice(avail)
                    dur_tok = tokenizer.vocab[dur]
                    dur_str = dur
                else:
                    dur_tok = tokenizer.vocab['Duration_0.5.8']
                    dur_str = 'Duration_0.5.8'
            
            vel = random.randint(max(50, avg_velocity-10), min(100, avg_velocity+10))
            vel_toks = [k for k in tokenizer.vocab.keys() if k.startswith('Velocity_')]
            closest = min(vel_toks, key=lambda x: abs(float(x.split('_')[1]) - vel))
            token_sequence.append(tokenizer.vocab[closest])
            
            token_sequence.append(dur_tok)
            
            if positions_used and len(positions_used) >= 2:
                prev_positions = [positions_used[i+1] - positions_used[i] for i in range(len(positions_used)-1) if positions_used[i+1] >= positions_used[i]]
                if prev_positions:
                    step = random.choice(prev_positions[:10])
                    nxt_pos = (cur_pos + step) % 32
                else:
                    nxt_pos = (cur_pos + 4) % 32
            else:
                nxt_pos = (cur_pos + 4) % 32
            
            if preview_callback:
                duration_beats = float(dur_str.split('_')[1].split('.')[0] + '.' + dur_str.split('_')[1].split('.')[1])
                time_position = step * 0.15
                preview_callback(base_pitch, time_position, duration_beats, 'melody')
            
            token_sequence.append(tokenizer.vocab[f'Position_{nxt_pos}'])
            
            if nxt_pos < cur_pos:
                token_sequence.append(tokenizer.vocab['Bar_None'])
    
    new_tokens = token_sequence[starting_length:] 
    full_sequence = original_seq + new_tokens
    
    token_names_orig = [list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(tid)] if tid in tokenizer.vocab.values() else f"UNK_{tid}" for tid in original_seq[-10:]]
    token_names_new = [list(tokenizer.vocab.keys())[list(tokenizer.vocab.values()).index(tid)] if tid in tokenizer.vocab.values() else f"UNK_{tid}" for tid in new_tokens[:20]]

    print(f"\nSaving to {output_midi}")
    decode_ids_to_midi(full_sequence, tokenizer, output_midi)
    
    return output_midi


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "decoded/continued.mid"
    bars = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    
    continue_midi(input_file, output_file, bars)

