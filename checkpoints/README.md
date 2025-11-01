# Model Checkpoint

**best_model.pt** - trained for 6 epochs on classical piano midi files

this checkpoint contains:
- model weights
- optimizer state
- training epoch info
- final loss: check the file if you need exact numbers

## using the checkpoint

the `generate.py` script automatically loads this checkpoint. just run:

```bash
python generate.py
```

## training stats

- epochs: 6
- model size: ~445 MB (2.6M parameters)
- trained on classical piano pieces
- uses mixed precision training for faster convergence

if you want to continue training from this checkpoint, use the `--resume` flag:

```bash
python train.py --resume checkpoints/best_model.pt
```
