# src/evaluation/metrics.py
"""
Unified evaluation metrics for all tasks.
Uses MIDI files directly as per faculty guide.
"""

import os
import numpy as np
import pretty_midi
from typing import List

from evaluation.pitch_histogram import (
    get_pitch_class_histogram,
    pitch_histogram_similarity,
    pitch_histogram_from_roll,
)
from evaluation.rhythm_score import (
    rhythm_diversity_from_midi,
    rhythm_diversity_from_roll,
    repetition_ratio_from_midi,
    quantize_duration,
)


def evaluate_midi_files(
    generated_paths : List[str],
    reference_paths : List[str] = None,
    task_label      : str       = "Task",
) -> dict:
    """
    Evaluate a list of generated MIDI files.
    Computes all metrics from faculty guide.

    Args:
        generated_paths: list of paths to generated MIDI files
        reference_paths: list of reference MIDI paths for pitch similarity
                        if None, compares generated files against each other
        task_label: label for printing

    Returns:
        dict of mean metric values
    """
    rd_scores  = []
    rep_scores = []
    ph_scores  = []

    for path in generated_paths:
        if not os.path.exists(path):
            print(f"  File not found: {path}")
            continue

        rd  = rhythm_diversity_from_midi(path)
        rep = repetition_ratio_from_midi(path)
        rd_scores.append(rd)
        rep_scores.append(rep)

    # Pitch histogram similarity
    if reference_paths:
        for gen_path, ref_path in zip(generated_paths, reference_paths):
            if os.path.exists(gen_path) and os.path.exists(ref_path):
                ph = pitch_histogram_similarity(gen_path, ref_path)
                ph_scores.append(ph)
    else:
        # Compare consecutive generated files
        for i in range(len(generated_paths) - 1):
            if os.path.exists(generated_paths[i]) and \
               os.path.exists(generated_paths[i+1]):
                ph = pitch_histogram_similarity(
                    generated_paths[i],
                    generated_paths[i+1]
                )
                ph_scores.append(ph)

    results = {
        "rhythm_diversity"         : np.mean(rd_scores)  if rd_scores  else 0.0,
        "rhythm_diversity_std"     : np.std(rd_scores)   if rd_scores  else 0.0,
        "repetition_ratio"         : np.mean(rep_scores) if rep_scores else 0.0,
        "repetition_ratio_std"     : np.std(rep_scores)  if rep_scores else 0.0,
        "pitch_histogram_similarity": np.mean(ph_scores) if ph_scores  else 0.0,
    }

    print("=" * 60)
    print(f"  {task_label} — Evaluation ({len(generated_paths)} samples)")
    print("=" * 60)
    print(f"  Rhythm Diversity    : {results['rhythm_diversity']:.4f}"
          f" ± {results['rhythm_diversity_std']:.4f}")
    print(f"  Repetition Ratio    : {results['repetition_ratio']:.4f}"
          f" ± {results['repetition_ratio_std']:.4f}")
    print(f"  Pitch Histogram Sim : {results['pitch_histogram_similarity']:.4f}")
    print("=" * 60)

    return results


def note_stats_from_midi(midi_path: str) -> dict:
    """Extract basic stats from a MIDI file for verification."""
    try:
        midi  = pretty_midi.PrettyMIDI(midi_path)
        notes = sum(len(i.notes) for i in midi.instruments)
        dur   = midi.get_end_time()
        pitches = [n.pitch for i in midi.instruments for n in i.notes]
        return {
            "notes"    : notes,
            "duration" : dur,
            "pitch_min": min(pitches) if pitches else 0,
            "pitch_max": max(pitches) if pitches else 0,
            "pitch_mean": np.mean(pitches) if pitches else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def generate_random_baseline(n_samples: int = 5,
                              out_dir: str = "outputs/generated_midis/baseline_random",
                              duration: float = 8.0,
                              fs: int = 16) -> List[str]:
    """
    Random Note Generator baseline.
    Faculty: sample pitches uniformly from 88-key range,
    durations from fixed set, random onset times.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for i in range(n_samples):
        midi       = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        n_notes    = np.random.randint(50, 150)
        onsets     = sorted(np.random.uniform(0, duration * 0.9, n_notes))
        dur_choices = [0.125, 0.25, 0.5, 1.0]

        for onset in onsets:
            pitch    = np.random.randint(21, 109)
            dur      = np.random.choice(dur_choices)
            velocity = np.random.randint(40, 100)
            note     = pretty_midi.Note(
                velocity = velocity,
                pitch    = pitch,
                start    = onset,
                end      = min(onset + dur, duration),
            )
            instrument.notes.append(note)

        midi.instruments.append(instrument)
        path = os.path.join(out_dir, f"random_sample_{i+1:02d}.mid")
        midi.write(path)
        paths.append(path)

    print(f"{n_samples} random baseline samples saved to {out_dir}")
    return paths


def generate_markov_baseline(train_midi_paths: List[str],
                              n_samples: int = 5,
                              out_dir: str = "outputs/generated_midis/baseline_markov",
                              duration: float = 8.0) -> List[str]:
    """
    Markov Chain baseline.
    Faculty: first-order transition matrix from training data.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Build transition matrix from training MIDIs
    transitions = np.zeros((128, 128))
    durations_by_pitch = {p: [] for p in range(128)}

    print("Building Markov transition matrix...")
    for path in train_midi_paths[:100]:  # use first 100 files
        try:
            midi = pretty_midi.PrettyMIDI(path)
            for inst in midi.instruments:
                notes = sorted(inst.notes, key=lambda n: n.start)
                for j in range(len(notes) - 1):
                    curr = notes[j].pitch
                    next_ = notes[j+1].pitch
                    transitions[curr, next_] += 1
                    dur = notes[j].end - notes[j].start
                    durations_by_pitch[curr].append(dur)
        except:
            continue

    # Normalize rows
    row_sums = transitions.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    transitions = transitions / row_sums

    paths = []
    for i in range(n_samples):
        midi       = pretty_midi.PrettyMIDI()
        instrument = pretty_midi.Instrument(program=0)

        current_pitch = np.random.randint(48, 72)
        current_time  = 0.0

        while current_time < duration:
            row = transitions[current_pitch]
            if row.sum() == 0:
                current_pitch = np.random.randint(48, 72)
                continue

            next_pitch = np.random.choice(128, p=row)
            durs = durations_by_pitch.get(current_pitch, [0.25])
            dur  = float(np.random.choice(durs)) if durs else 0.25
            dur  = max(0.05, min(dur, 2.0))

            note = pretty_midi.Note(
                velocity = np.random.randint(40, 100),
                pitch    = int(next_pitch),
                start    = current_time,
                end      = min(current_time + dur, duration),
            )
            instrument.notes.append(note)
            current_time  += dur * 0.8
            current_pitch  = next_pitch

        midi.instruments.append(instrument)
        path = os.path.join(out_dir, f"markov_sample_{i+1:02d}.mid")
        midi.write(path)
        paths.append(path)

    print(f"{n_samples} Markov baseline samples saved to {out_dir}")
    return paths