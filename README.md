# Scrumpler

Audio sample processor for experimental music production. Chops, splits, and extracts samples from long-form audio.

## Installation

```bash
pip install soundfile scipy numpy
```

## Usage

```bash
# Process a single file
python sample_processor.py file.wav --grid
python sample_processor.py file.wav --transient
python sample_processor.py file.wav --texture
python sample_processor.py file.wav --all

# Process all files in _incoming_rips/
python sample_processor.py --batch --all

# Use presets
python batch_processor.py help          # List presets
python batch_processor.py drums file.wav
python batch_processor.py loops_128 --batch
```

## Processing Modes

### Grid (`--grid`)
Slices audio into equal-length chunks.

```bash
python sample_processor.py file.wav --grid --chunk-length 2.0
python sample_processor.py file.wav --grid --bpm 128 --bars 4  # Musical timing
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chunk-length` | 2.0 | Chunk duration in seconds |
| `--bpm` | - | BPM for musical timing (overrides chunk-length) |
| `--bars` | 4 | Bars per chunk when using BPM |

### Transient (`--transient`)
Extracts percussive hits and sharp transients.

```bash
python sample_processor.py file.wav --transient --delta 0.05
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--delta` | 0.07 | Sensitivity (lower = more sensitive) |
| `--min-length` | 0.05 | Minimum segment length (seconds) |
| `--max-length` | 10.0 | Maximum segment length (seconds) |

### Texture (`--texture`)
Isolates sustained ambient sections.

```bash
python sample_processor.py file.wav --texture --min-duration 3.0
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--min-duration` | 1.0 | Minimum texture length (seconds) |
| `--max-duration` | 30.0 | Maximum texture length (seconds) |
| `--rms-threshold` | 0.1 | Energy threshold (0.0-1.0) |
| `--stability-threshold` | 0.15 | Timbre stability (lower = more stable) |

## General Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--channel` | mono | Channel selection: `mono`, `left`, `right` |
| `--sr` | 44100 | Sample rate |
| `-i` | _incoming_rips | Input directory |
| `-o` | _chopped_samples | Output directory |

## Output Structure

```
_chopped_samples/
└── filename/
    ├── grid_bpm128_bars4/
    ├── transient_delta0p07_min0p05s_max10p0s/
    └── texture_min1p0s_max30p0s_rms0p10_stab0p15/
```

## Presets

| Preset | Description |
|--------|-------------|
| `quick` | All modes, 1.5s grid chunks |
| `loops_90` | 4-bar loops at 90 BPM |
| `loops_120` | 4-bar loops at 120 BPM |
| `loops_128` | 4-bar loops at 128 BPM |
| `loops_140` | 4-bar loops at 140 BPM |
| `loops_174` | 2-bar loops at 174 BPM |
| `drums` | Sensitive transient detection |
| `drones` | Long ambient textures (3-60s) |
| `pads` | Medium ambient textures (1-15s) |
| `granular` | 250ms chunks |

Run `python batch_processor.py help` for full list.
