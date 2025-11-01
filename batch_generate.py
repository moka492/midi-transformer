#Batch generation, create multiple samples at once to find the best ones,Useful for demos and exploring different parameter combinations 
import argparse
from pathlib import Path
import random

from generate_musical import generate_musical
from generate_multitrack import generate_multitrack


def batch_generate(count=10, multitrack=False, scale='C_major', length=16, 
                   temperature=0.7, chord_density=40, progress_callback=None):
    
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"Settings:")
    print(f"  Mode: {'Multi-track' if multitrack else 'Single-track'}")
    print(f"  Scale: {scale}")
    print(f"  Length: {length} bars")
    print(f"  Temperature: {temperature}")
    print(f"  Chord Density: {chord_density}%")
    print(f"{'='*60}\n")
    
    output_dir = Path("decoded/batch")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for old_file in output_dir.glob("*.mid"):
        old_file.unlink()
    
    generated_files = []
    
    for i in range(count):
        print(f"\n[{i+1}/{count}] Generating sample {i+1}...")
        
        if progress_callback:
            progress_callback(i, count)
        
        try:
            if multitrack:
            
                generate_multitrack(num_samples=1, max_tokens=length * 30, scale=scale,
                                  temperature=temperature, chord_density=chord_density)
                
         
                files = sorted(Path("decoded").glob("multitrack_*.mid"), 
                             key=lambda x: x.stat().st_mtime)
                if files:
                    latest = files[-1]
         
                    new_path = output_dir / f"batch_{i+1:03d}.mid"
                    latest.rename(new_path)
                    generated_files.append(new_path)
                    print(f"  Saved as {new_path.name}")
            else:
         
                generate_multitrack(num_samples=1, max_tokens=length * 30, scale=scale,
                                  temperature=temperature, chord_density=chord_density)
                
        
                files = sorted(Path("decoded").glob("multitrack_*.mid"), 
                             key=lambda x: x.stat().st_mtime)
                if files:
                    latest = files[-1]
        
                    new_path = output_dir / f"batch_{i+1:03d}.mid"
                    latest.rename(new_path)
                    generated_files.append(new_path)
                    print(f"  Saved as {new_path.name}")
                    
        except Exception as e:
            print(f"  Error generating sample {i+1}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f" BATCH COMPLETE!")
    print(f"{'='*60}")
    print(f"Generated {len(generated_files)}/{count} samples successfully")
    print(f"Location: {output_dir.absolute()}")
    print(f"{'='*60}\n")
    
    return generated_files


def batch_generate_varied(count=10, multitrack=False, base_length=16, progress_callback=None):
    scales = ['C_major', 'G_major', 'D_major', 'A_minor', 'E_minor']
    temperatures = [0.6, 0.7, 0.8]
    chord_densities = [30, 40, 50, 60]
    lengths = [12, 16, 20, 24]
    
    print(f"\n{'='*60}")
    print(f"{'='*60}")
    print(f"generate {count} samples with random parameter variations")
    print(f"{'='*60}\n")
    
    output_dir = Path("decoded/batch_varied")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for old_file in output_dir.glob("*.mid"):
        old_file.unlink()
    
    generated_files = []
    
    for i in range(count):
      
        scale = random.choice(scales)
        temp = random.choice(temperatures)
        chord = random.choice(chord_densities)
        length = random.choice(lengths)
        
        print(f"\n[{i+1}/{count}] Sample {i+1}")
        print(f"  Scale: {scale}, Temp: {temp}, Chords: {chord}%, Length: {length} bars")
        
        if progress_callback:
            progress_callback(i, count)
        
        try:
            generate_multitrack(num_samples=1, max_tokens=length * 30, scale=scale,
                              temperature=temp, chord_density=chord)
            files = sorted(Path("decoded").glob("multitrack_*.mid"), 
                         key=lambda x: x.stat().st_mtime)
            
            if files:
                latest = files[-1]
            
                filename = f"batch_{i+1:03d}_{scale}_t{int(temp*10)}_c{chord}_l{length}.mid"
                new_path = output_dir / filename
                latest.rename(new_path)
                generated_files.append(new_path)
                print(f" Saved as {new_path.name}")
                
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"VARIED BATCH COMPLETE!")
    print(f"{'='*60}")
    print(f"Generated {len(generated_files)}/{count} samples")
    print(f"Location: {output_dir.absolute()}")
    print(f"{'='*60}\n")
    
    return generated_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Batch generate MIDI samples')
    parser.add_argument('--count', type=int, default=10, help='Number of samples')
    parser.add_argument('--multitrack', action='store_true', help='Use multi-track')
    parser.add_argument('--varied', action='store_true', help='Use varied parameters')
    parser.add_argument('--scale', type=str, default='C_major', help='Musical scale')
    parser.add_argument('--length', type=int, default=16, help='Length in bars')
    parser.add_argument('--temp', type=float, default=0.7, help='Temperature')
    parser.add_argument('--chord', type=int, default=40, help='Chord density (0-100)')
    
    args = parser.parse_args()
    
    if args.varied:
        batch_generate_varied(count=args.count, multitrack=args.multitrack)
    else:
        batch_generate(count=args.count, multitrack=args.multitrack, 
                      scale=args.scale, length=args.length,
                      temperature=args.temp, chord_density=args.chord)
