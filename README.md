Welcome to this project! Thanks for checking it out
I created MIDI Transformer, a deep learning model that generates realistic piano music from scratch or by continuing existing compositions. The project explores how transformer architectures can learn musical patterns from classical MIDI files and produce high-quality multi-track music.

## Overview

This project is a solo effort. The core idea: apply transformers (GPT-style) to symbolic music. I trained the model mostly on the MAESTRO classical piano dataset. Unlike text models, this one learns pitch, timing, structure, tempo, and melodic motion all mapped using REMI tokenization. Outputs are proper MIDI files, ready for playback or editing.

## Features

- Multi-track generation (melody and bass split into separate synchronized MIDI tracks)
- Real-time piano roll preview in the GUI
- Continuation mode lets you upload a MIDI and instantly extend it in the same style
- Batch generation for creating many samples at once
- GUI with dark theme and sliders for scale, length, creativity, chord density, and mode switching
- Customizable musical scale, tempo, and chord progressions
- Style matching for seamless MIDI continuation

## WARNING: IF THE SOUND IS TOO LOW , please use "midiano" online to boost volume!"

## Installation

## Important: Model File Download

**If you downloaded this as a ZIP file**, the model won't work because GitHub doesn't include Git LFS files in ZIP downloads.

**Option A: Clone with Git (Recommended)**
```bash
git clone https://github.com/moka492/midi-transformer.git
cd midi-transformer
git lfs pull  # Downloads the actual 445MB model file
pip install -r requirements.txt
```

**Option B: Manual Download (If you used ZIP)**
1. Download the repo as ZIP and extract
2. The `checkpoints/best_model.pt` file will be only 134 bytes (LFS pointer)
3. **You need the real model file** - contact the repo owner for a direct download link
4. Replace the 134-byte file with the actual 445MB model file
5. Install dependencies: `pip install -r requirements.txt`

**Verify Installation:**
Check that `checkpoints/best_model.pt` is approximately 445MB, not 134 bytes.

Model checkpoints live in `checkpoints/best_model.pt`. You can retrain if you want experimental results.

## Usage

### GUI

Launch the graphical interface:

```bash
python gui_app.py
```

Generate new music, continue MIDI files, batch create, and set all key parameters interactively. All outputs save to `decoded/` for easy access.

### Command-Line

Generate single-track music:
```bash
python generate_musical.py
```

Multi-track (melody + bass):
```bash
python generate_multitrack.py
```

Continue an existing MIDI:
```bash
python continue_midi.py input.mid output.mid --bars 8
```

Batch generate:
```bash
python batch_generate.py --count 10 --multitrack
```

## Technical Details

- GPT-style transformer: 12 layers, 8 attention heads, 512-dim, ~11M params
- REMI tokenization: 407-token vocab for Bar, Position, Pitch, Velocity, Duration, Tempo
- Context window: up to 1024 tokens for continuity
- Training: MAESTRO dataset, cross-entropy on next-token prediction, checkpoints every epoch
- Generation: nucleus sampling (p=0.9), scale guidance boosts in-key notes, anti-repetition penalizes overused notes, automatic completion fills in pitch/velocity/duration/position for each token
- Multi-track mode: melody covers higher register, expressive/short notes; bass uses lower register and chord progressions; tracked and merged in MIDI Type 1 format

## Advanced Generation

- Anti-repetition: output avoids recent notes, keeping patterns musical
- Chord progressions: bass follows patterns like I-IV-V-I or I-V-vi-IV
- Melodic motion: stepwise moves preferred, octaves penalized
- Scale detection: continuation mode reads scale and pitch from input file
- Outputs can be randomized for creative/varied generations

## Training

Prepare dataset tokens:

```bash
python dataset.py --midi_dir data_midi/ --output_dir tokens/
```

Train the transformer:

```bash
python train_model.py --tokens tokens/ --epochs 20
```

Loss and status reported every 100 steps. Model is saved to `checkpoints/`.

## Folder Structure

- `gui_app.py` — Main graphical UI
- `model.py` — Transformer architecture
- `tokenizer.py` — MIDI-to-token logic
- `dataset.py` — Preprocessing and tokenization
- `train_model.py` — Training script
- `generate_musical.py` — Single-track generation
- `generate_multitrack.py` — Multi-track generation
- `continue_midi.py` — MIDI continuation
- `batch_generate.py` — Batch generation
- `realtime_preview.py` — Piano roll visualization
- `music_theory.py` — Scales, chords, progressions
- `demos/` — Example outputs
- `data_midi/` — Training files
- `checkpoints/` — Saved models
- `decoded/` — Generated music
- `requirements.txt` — Dependencies

## Limitations

Right now, the best results are with classical piano styles (the dataset), no explicit controls for emotion, and continuation is best if the input matches the training style. Real-time preview adds a bit of generation overhead.

## Notebook Challenges

Technical hurdles:

1. Token order had a big effect on timing—getting pitch, velocity, duration, and position sequenced right was tricky
2. Multi-track required separate passes, careful MIDI formatting, and synchronizing events
3. Real-time preview to the UI needed thread-safe communication
4. Style matching (scale/key) for continuation depended on input MIDI parsing
5. Humanizing velocity and timing is an ongoing challenge

## Recent Improvements

- New GUI with full dark theme
- True multi-track generation
- Real-time piano roll preview
- Automatic scale/key detection for continuation
- Music theory helpers (chord progressions, randomization)
- Better transformer architecture (12 layers, bigger context window)
- Anti-repetition and melodic movement tweaks

## Motivation

I built this to understand generative models for music, especially how transformers work outside text. It taught me a ton about attention, tokenization, and balancing creativity with musical coherence. Multi-track music was the toughest—a breakthrough came with MIDI Type 1 and pitch range separation.

## Credits

This is a personal solo project. Everything is open-source, using PyTorch and MidiTok. Trained on the MAESTRO dataset. If you use MIDI Transformer for something interesting, I’d love to hear about it!

