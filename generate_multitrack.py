# creates multitrack melodies , sounds much more complex!
from __future__ import annotations
import torch
import random
from pathlib import Path
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

from model import GPTModel
from tokenizer import build_tokenizer, decode_ids_to_midi
from music_theory import SCALES, PROGRESSIONS, get_chord_notes


def generate_multitrack(num_samples: int = 1, max_tokens: int = 500, scale: str = 'C_major',
                       temperature: float = 0.7, chord_density: int = 40, preview_callback=None):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
   
    tokenizer = build_tokenizer()
    
    checkpoint_data = torch.load("checkpoints/best_model.pt", map_location=device)
    model = GPTModel(vocab_size=407, d_model=512, n_heads=8, n_layers=12, 
                     max_seq_len=1024, d_ff_mult=4)
    model.load_state_dict(checkpoint_data["model"], strict=False)
    model = model.to(device)
    model.eval()
    
    scale_notes = SCALES.get(scale, SCALES['C_major'])
    
    output_dir = Path("decoded")
    output_dir.mkdir(exist_ok=True)
    
    for sample_idx in range(num_samples):
        print(f"\nStarting generation for sample {sample_idx + 1}")
      
        token_sequence = [tokenizer.vocab['Bar_None']]
        
        tempo_tokens = sorted([k for k in tokenizer.vocab if k.startswith('Tempo_')],
                             key=lambda x: abs(float(x.split('_')[1]) - 115))
        if tempo_tokens:
            token_sequence.append(tokenizer.vocab[tempo_tokens[0]])
        
        progression = random.choice(PROGRESSIONS)
        
        
        scale_root_note = {'C_major': 60, 'G_major': 67, 'D_major': 62, 
                          'A_major': 69, 'A_minor': 69, 'E_minor': 64, 
                          'D_minor': 62}.get(scale, 60)
        
        melody_track = generate_melody_track(model, tokenizer, scale_notes, scale_root_note, 
                                            max_tokens, temperature, device, preview_callback)
        
        bass_track = generate_bass_track(model, tokenizer, scale_notes, scale_root_note,
                                        progression, max_tokens, temperature, 
                                        chord_density, device, preview_callback)
        
        output_file = output_dir / f"multitrack_{sample_idx+1:03d}.mid"
        save_multitrack_midi(melody_track, bass_track, str(output_file), tokenizer)
        
        print(f"  Saved to {output_file}")

def generate_melody_track(model, tokenizer, scale_notes, root_note, max_tokens, 
                         temperature, device, preview_callback=None):
    
    print(f"  generating melody track ")
    
    token_sequence = []
    token_sequence.append(tokenizer.vocab['Position_0'])
    
    melody_pitch_range = (root_note + 12, root_note + 36)
    current_time = 0.0 
    
    num_notes_to_generate = max_tokens // 8
    for note_idx in range(num_notes_to_generate):
        if len(token_sequence) >= max_tokens:
            break
        
        inp = torch.tensor([[tokenizer.vocab['Bar_None']] + token_sequence[-min(len(token_sequence), 400):]], 
                          dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, _ = model(inp, None)
        
        nxt_logits = logits[0, -1] / temperature
        
        recent_pitches = []
        for t in token_sequence[-10:]:
            tn_check = [k for k, v in tokenizer.vocab.items() if v == t]
            if tn_check and tn_check[0].startswith('Pitch_'):
                recent_pitches.append(int(tn_check[0].split('_')[1]))
        
        for tn, tid in tokenizer.vocab.items():
            if tn.startswith('Pitch_'):
                pitch = int(tn.split('_')[1])
                pitch_class = pitch % 12
                
                if pitch_class in scale_notes and melody_pitch_range[0] <= pitch <= melody_pitch_range[1]:
                    nxt_logits[tid] *= 3.0  
                elif melody_pitch_range[0] <= pitch <= melody_pitch_range[1]:
                    nxt_logits[tid] *= 1.2
                else:
                    nxt_logits[tid] *= 0.1  
                
          
                if tid in token_sequence[-3:]:
                    nxt_logits[tid] *= 0.2  
                elif tid in token_sequence[-8:-3]:
                    nxt_logits[tid] *= 0.8  
                
                
                if recent_pitches:
                    last_pitch = recent_pitches[-1]
                    interval = abs(pitch - last_pitch)
                    
                    if interval <= 2: 
                        nxt_logits[tid] *= 1.4 
                    elif interval <= 4: 
                        nxt_logits[tid] *= 1.1 
                    elif interval >= 12: 
                        nxt_logits[tid] *= 0.6  
                    
            
                    if len(recent_pitches) >= 3:
                       
                        directions = []
                        for i in range(len(recent_pitches) - 1):
                            directions.append(recent_pitches[i+1] - recent_pitches[i])
                        
                  
                        if len(directions) >= 2:
                            if directions[-1] > 0 and directions[-2] > 0: 
                                if pitch < last_pitch:  
                                    nxt_logits[tid] *= 1.2
                            elif directions[-1] < 0 and directions[-2] < 0: 
                                if pitch > last_pitch: 
                                    nxt_logits[tid] *= 1.2
        

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
            pitch = int(tn.split('_')[1])
            
          
            cur_pos = 0
            for t in token_sequence[-32:]:
                tname = [k for k, v in tokenizer.vocab.items() if v == t][0]
                if tname.startswith('Position_'):
                    cur_pos = int(tname.split('_')[1])
            
       

            vel_choices = [
                (75, 85, 0.15),
                (86, 95, 0.40),
                (96, 105, 0.30),
                (106, 115, 0.15),
            ]
            
            vel_range = random.choices(vel_choices, weights=[w for _, _, w in vel_choices])[0]
            vel = random.randint(vel_range[0], vel_range[1])
            
            vel_toks = [k for k in tokenizer.vocab.keys() if k.startswith('Velocity_')]
            closest = min(vel_toks, key=lambda x: abs(float(x.split('_')[1]) - vel))
            token_sequence.append(tokenizer.vocab[closest])
            
        
            dur_opts = ['Duration_0.4.8', 'Duration_0.5.8', 'Duration_0.6.8', 'Duration_0.8.8', 'Duration_1.0.8']
            weights = [0.25, 0.3, 0.2, 0.15, 0.1]
            
            avail = [d for d in dur_opts if d in tokenizer.vocab]
            if avail:
                dur = random.choices(avail[:len(weights)], weights=weights[:len(avail)])[0]
            else:
                dur = 'Duration_0.5.8'

            dur_str = dur
            token_sequence.append(tokenizer.vocab[dur])
            
         
            duration_beats = float(dur_str.split('_')[1].split('.')[0] + '.' + dur_str.split('_')[1].split('.')[1])
            
            step = random.randint(4, 10)
            
            if random.random() < 0.10:
                step += random.choice([-1, 1]) if step > 1 else 0
            
            nxt_pos = (cur_pos + step) % 32
            
        
            if nxt_pos >= cur_pos:
                time_delta = (nxt_pos - cur_pos) * (4.0 / 32)
            else: 
                time_delta = (32 - cur_pos + nxt_pos) * (4.0 / 32)
            
      
            if preview_callback:
                preview_callback(pitch, current_time, duration_beats, 'melody')
            
            current_time += time_delta
            
            token_sequence.append(tokenizer.vocab[f'Position_{nxt_pos}'])
            
          
            if nxt_pos < cur_pos:
                token_sequence.append(tokenizer.vocab['Bar_None'])
    
    return token_sequence


def generate_bass_track(model, tokenizer, scale_notes, root, progression, max_tokens, 
                       temperature, chord_density, device, preview_callback=None):
    
    token_sequence = []
    token_sequence.append(tokenizer.vocab['Position_0'])
    
    bass_range = (root - 24, root)  
    current_time = 0.0 
    
    chord_prob = chord_density / 100.0
    bars_per_chord = 4 if progression else 2
    current_chord_idx = 0
    bar_count = 0
    
    for step in range(max_tokens // 6):
        if len(token_sequence) >= max_tokens:
            break
        
        inp = torch.tensor([[tokenizer.vocab['Bar_None']] + token_sequence[-min(len(token_sequence), 400):]], 
                          dtype=torch.long, device=device)
        
        with torch.no_grad():
            logits, _ = model(inp, None)
        
        nxt_logits = logits[0, -1] / (temperature * 0.9) 
        
    
        current_chord = None
        if progression and bar_count // bars_per_chord < len(progression['chords']):
            current_chord = progression['chords'][bar_count // bars_per_chord]
        
     
        for tn, tid in tokenizer.vocab.items():
            if tn.startswith('Pitch_'):
                pitch = int(tn.split('_')[1])
                pitch_class = pitch % 12
                
          
                if bass_range[0] <= pitch <= bass_range[1]:
                    nxt_logits[tid] *= 2.0
                    
                
                    if pitch_class in scale_notes:
                        nxt_logits[tid] *= 2.0
                    
                  
                    if current_chord and pitch_class in get_chord_notes(current_chord, root):
                        nxt_logits[tid] *= 3.0 
                else:
                    nxt_logits[tid] *= 0.05  
                
             
                if tid in token_sequence[-4:]:
                    nxt_logits[tid] *= 0.1
        
       
        probs = torch.softmax(nxt_logits, dim=-1)
        sorted_probs, sorted_idx = probs.sort(descending=True)
        cum = sorted_probs.cumsum(dim=-1)
        mask = cum > 0.92  
        mask[1:] = mask[:-1].clone()
        mask[0] = False
        nxt_logits[sorted_idx[mask]] = float('-inf')
        
        probs = torch.softmax(nxt_logits, dim=-1)
        nxt = torch.multinomial(probs, 1).item()
        
        tn = [k for k, v in tokenizer.vocab.items() if v == nxt][0]
        tt = tn.split('_')[0]
        
        token_sequence.append(nxt)
        
  
        if tt == 'Pitch':
            base_pitch = int(tn.split('_')[1])
            
     
            cur_pos = 0
            for t in token_sequence[-32:]:
                tname = [k for k, v in tokenizer.vocab.items() if v == t][0]
                if tname.startswith('Position_'):
                    cur_pos = int(tname.split('_')[1])
            
      
            chord_notes = [base_pitch]
            if random.random() < chord_prob:
             
                if current_chord:
                    chord_tone_classes = get_chord_notes(current_chord, root)
                    for offset in [3, 4, 7, 12]:
                        candidate = base_pitch + offset
                        if (candidate % 12) in chord_tone_classes and candidate <= bass_range[1]:
                            if f'Pitch_{candidate}' in tokenizer.vocab:
                                chord_notes.append(candidate)
                                if len(chord_notes) >= 3:
                                    break
            
        
            if len(chord_notes) > 1:
                dur_opts = ['Duration_1.0.8', 'Duration_1.2.8', 'Duration_2.0.8']
            else:
                dur_opts = ['Duration_0.6.8', 'Duration_1.0.8', 'Duration_1.2.8']
            avail_durs = [d for d in dur_opts if d in tokenizer.vocab]
            if not avail_durs:
               
                avail_durs = [k for k in tokenizer.vocab.keys() if k.startswith('Duration_')][:5]
            dur = random.choice(avail_durs)
            dur_str = dur
            dur_tok = tokenizer.vocab[dur]
            
          
            duration_beats = float(dur_str.split('_')[1].split('.')[0] + '.' + dur_str.split('_')[1].split('.')[1])
            
           
            for idx, pitch in enumerate(chord_notes):
                if idx > 0:
                    token_sequence.append(tokenizer.vocab[f'Pitch_{pitch}'])
                
                

                vel = random.randint(45, 60)
                vel_toks = [k for k in tokenizer.vocab.keys() if k.startswith('Velocity_')]
                closest = min(vel_toks, key=lambda x: abs(float(x.split('_')[1]) - vel))
                token_sequence.append(tokenizer.vocab[closest])
                
                token_sequence.append(dur_tok)
                token_sequence.append(tokenizer.vocab[f'Position_{cur_pos}'])
                
            
                if preview_callback:
                    preview_callback(pitch, current_time, duration_beats, 'bass')
            
          
            nxt_pos = (cur_pos + random.randint(3, 6)) % 32
            
  
            if nxt_pos >= cur_pos:
                time_delta = (nxt_pos - cur_pos) * (4.0 / 32)
            else:  
                time_delta = (32 - cur_pos + nxt_pos) * (4.0 / 32)
            
            current_time += time_delta
            
            token_sequence.append(tokenizer.vocab[f'Position_{nxt_pos}'])
            
        
            if nxt_pos < cur_pos:
                token_sequence.append(tokenizer.vocab['Bar_None'])
                bar_count += 1
    
    return token_sequence


def save_multitrack_midi(melody_tokens, bass_tokens, output_path, tokenizer):
    print("  Creating multi-track MIDI file")
    

    mid = MidiFile(type=1, ticks_per_beat=480)
    
  
    meta_track = MidiTrack()
    mid.tracks.append(meta_track)
    meta_track.append(MetaMessage('set_tempo', tempo=1500000, time=0))
    meta_track.append(MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    
   
    bass_track = MidiTrack()
    mid.tracks.append(bass_track)
    bass_track.append(Message('program_change', program=0, time=0, channel=0))
    tokens_to_midi_track(bass_tokens, bass_track, tokenizer, channel=0)
    
 
    melody_track = MidiTrack()
    mid.tracks.append(melody_track)
    melody_track.append(Message('program_change', program=0, time=0, channel=1))
    tokens_to_midi_track(melody_tokens, melody_track, tokenizer, channel=1)
    
    mid.save(output_path)
    print(f"  Saved multi-track MIDI: {output_path}")


def tokens_to_midi_track(tokens, track, tokenizer, channel=0):

    current_position = 0
    last_absolute_time = 0
    all_events = []  
    
    i = 0
    while i < len(tokens):
        tid = tokens[i]
        tname = [k for k, v in tokenizer.vocab.items() if v == tid]
        if not tname:
            i += 1
            continue
        tname = tname[0]
        ttype = tname.split('_')[0]
        
        if ttype == 'Position':
          
            pos = int(tname.split('_')[1])
       
            current_position = pos * 60
            
        elif ttype == 'Bar':
         
            current_position += 4 * 480  
            
        elif ttype == 'Pitch':
           
            pitch = int(tname.split('_')[1])
            
           
            velocity = 64  
            duration_ticks = 240  
            
            if i + 1 < len(tokens):
                vel_tid = tokens[i + 1]
                vel_tname = [k for k, v in tokenizer.vocab.items() if v == vel_tid]
                if vel_tname and vel_tname[0].startswith('Velocity_'):
                    velocity = int(float(vel_tname[0].split('_')[1]))
                    i += 1
            
            if i + 1 < len(tokens):
                dur_tid = tokens[i + 1]
                dur_tname = [k for k, v in tokenizer.vocab.items() if v == dur_tid]
                if dur_tname and dur_tname[0].startswith('Duration_'):
                   
                    parts = dur_tname[0].replace('Duration_', '').split('.')
                    try:
                        dur_beats = float(parts[0]) + float(parts[1]) / 10.0
                        duration_ticks = int(dur_beats * 480)
                    except:
                        duration_ticks = 240
                    i += 1
            
          
            all_events.append((current_position, 'note_on', (pitch, velocity)))
            all_events.append((current_position + duration_ticks, 'note_off', (pitch, 0)))
        
        i += 1
    
    all_events.sort(key=lambda x: x[0])
    
  
    last_time = 0
    for abs_time, event_type, data in all_events:
        delta = abs_time - last_time
        if delta < 0:
            delta = 0 
        
        if event_type == 'note_on':
            track.append(Message('note_on', note=data[0], velocity=data[1], 
                               time=delta, channel=channel))
        elif event_type == 'note_off':
            track.append(Message('note_off', note=data[0], velocity=data[1], 
                               time=delta, channel=channel))
        
        last_time = abs_time
    
    
    track.append(MetaMessage('end_of_track', time=480))


if __name__ == "__main__":
    scales = ['C_major', 'G_major', 'A_minor', 'D_major']
    generate_multitrack(num_samples=3, max_tokens=500, scale=random.choice(scales), 
                       temperature=0.7, chord_density=50)

