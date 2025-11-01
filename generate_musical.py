#generate music with light music theory guidance (not forced)
from __future__ import annotations
import random
from pathlib import Path
import torch
from model import GPTModel
from tokenizer import build_tokenizer, load_vocab_size, decode_ids_to_midi


SCALES = {
    'C_major': [0, 2, 4, 5, 7, 9, 11],  
    'G_major': [7, 9, 11, 0, 2, 4, 6],
    'A_minor': [9, 11, 0, 2, 4, 5, 7],
    'E_minor': [4, 6, 7, 9, 11, 0, 2],
}


def generate_musical(ckpt="checkpoints/best_model.pt", n=1, max_tok=450, scale='C_major', preview_callback=None):
   
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {ckpt}")
    
    tok = build_tokenizer()
    
    ckpt_data = torch.load(ckpt, map_location=dev)
    mdl = GPTModel(vocab_size=407, d_model=512, n_heads=8, n_layers=12, max_seq_len=1024, d_ff_mult=4)
    mdl.load_state_dict(ckpt_data["model"], strict=False)
    mdl = mdl.to(dev)
    mdl.eval()
    
    out_dir = Path("decoded")
    out_dir.mkdir(exist_ok=True)
    
    scale_notes = SCALES.get(scale, SCALES['C_major'])
    
    for i in range(n):
        print(f"\nGenerating {i+1}/{n} in {scale}")
        
       
        seq = [tok.vocab['BOS_None'], tok.vocab['Bar_None'], tok.vocab['Program_0']]
        
        tempo = sorted([k for k in tok.vocab if k.startswith('Tempo_')],
                       key=lambda x: abs(float(x.split('_')[1]) - 115))
        if tempo:
            seq.append(tok.vocab[tempo[0]])
        
        seq.append(tok.vocab['Position_0'])
        
        current_time = 0.0
        last_pitch = None
        last_duration = 0.5
        
     
        for step in range(max_tok):
            inp = torch.tensor([seq[-512:]], dtype=torch.long, device=dev)
            
            with torch.no_grad():
                logits, _ = mdl(inp, None)
            
            nxt_logits = logits[0, -1] / 0.7  
            
         
            for tn, tid in tok.vocab.items():
                if tn.startswith('Pitch_'):
                    pitch = int(tn.split('_')[1])
                    pitch_class = pitch % 12
                    
                
                    if pitch_class in scale_notes:
                        nxt_logits[tid] *= 1.5  
                    
                   
                    recent = seq[-10:]
                    if tid in recent[-3:]:
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
            
            tn = [k for k, v in tok.vocab.items() if v == nxt][0]
            tt = tn.split('_')[0]
            
            seq.append(nxt)
            
          
            if tt == 'Pitch':
                base_pitch = int(tn.split('_')[1])
                
                
                cur_pos = 0
                for t in seq[-32:]:
                    tname = [k for k, v in tok.vocab.items() if v == t][0]
                    if tname.startswith('Position_'):
                        cur_pos = int(tname.split('_')[1])
                
              
                chord_notes = [base_pitch]
                if random.random() < 0.4:
                  
                    interval = 4 if random.random() < 0.7 else 3
                    third = base_pitch + interval
                    if f'Pitch_{third}' in tok.vocab:
                        chord_notes.append(third)
                    
                 
                    if random.random() < 0.6:
                        fifth = base_pitch + 7
                        if f'Pitch_{fifth}' in tok.vocab:
                            chord_notes.append(fifth)
                
               
                if len(chord_notes) > 1:
                    dur_opts = ['Duration_0.8.8', 'Duration_1.0', 'Duration_1.2']
                    weights = [0.4, 0.4, 0.2]
                else:
                    dur_opts = ['Duration_0.4.8', 'Duration_0.5.8', 'Duration_0.6.8', 'Duration_0.8.8', 'Duration_1.0']
                    weights = [0.25, 0.3, 0.2, 0.15, 0.1]
                
                avail = [d for d in dur_opts if d in tok.vocab]
                if avail:
                    dur = random.choices(avail[:len(weights)], weights=weights[:len(avail)])[0]
                    dur_tok = tok.vocab[dur]
                else:
                    dur_tok = tok.vocab['Duration_0.5.8']
                

                dur_str = [k for k, v in tok.vocab.items() if v == dur_tok][0]
                duration_beats = float(dur_str.split('_')[1].split('.')[0] + '.' + dur_str.split('_')[1].split('.')[1]) if '.' in dur_str.split('_')[1] else 0.5
            

                if preview_callback and last_pitch:
                    preview_callback(last_pitch, current_time, last_duration, 'melody')
                
                last_pitch = base_pitch
                last_duration = duration_beats
                
            
                for idx, pitch in enumerate(chord_notes):
               
                    if idx > 0:
                        seq.append(tok.vocab[f'Pitch_{pitch}'])
                    
                  
                    vel = random.randint(60, 80)
                    vel_toks = [k for k in tok.vocab.keys() if k.startswith('Velocity_')]
                    closest = min(vel_toks, key=lambda x: abs(float(x.split('_')[1]) - vel))
                    seq.append(tok.vocab[closest])
                    
                 
                    seq.append(dur_tok)
                    
                    
                    seq.append(tok.vocab[f'Position_{cur_pos}'])
                
                
                nxt_pos = (cur_pos + random.randint(3, 6)) % 32
                time_delta = (nxt_pos - cur_pos) * (4.0 / 32) if nxt_pos >= cur_pos else (32 - cur_pos + nxt_pos) * (4.0 / 32)
                current_time += time_delta
                
                seq.append(tok.vocab[f'Position_{nxt_pos}'])
                
             
                if nxt_pos < cur_pos:
                    seq.append(tok.vocab['Bar_None'])
        
        
        midi_file = out_dir / f"musical_{i+1:03d}.mid"
        decode_ids_to_midi(seq, tok, str(midi_file))
        print(f"Saved to {midi_file}")


if __name__ == "__main__":
    scales = ['C_major', 'G_major', 'A_minor']
    generate_musical(n=5, max_tok=400, scale=random.choice(scales))
