# src/evaluation/metrics.py
"""
Evaluation metrics as defined in the project specification.

All functions operate on numpy piano-roll arrays of shape (T, 88)
with values normalised to [0, 1].
"""

import numpy as np
from typing import List


# ─────────────────────────────────────────────────────────────
# 1. Pitch Histogram Similarity
#    H(p, q) = Σ_{i=1}^{12} |p_i − q_i|
# ─────────────────────────────────────────────────────────────
def pitch_class_histogram(roll: np.ndarray, threshold: float = 0.15) -> np.ndarray:
    """
    Returns a 12-dim normalised pitch-class histogram (mod-12 folding).
    Counts the total active frames per pitch class.
    """
    active = (roll > threshold).astype(float)   # (T, 88)
    hist   = np.zeros(12)
    for key in range(88):
        hist[key % 12] += active[:, key].sum()
    total = hist.sum()
    return hist / total if total > 0 else hist


def pitch_histogram_similarity(roll1: np.ndarray, roll2: np.ndarray) -> float:
    """
    H(p, q) = Σ |p_i − q_i|    lower = more similar.
    """
    p = pitch_class_histogram(roll1)
    q = pitch_class_histogram(roll2)
    return float(np.sum(np.abs(p - q)))


# ─────────────────────────────────────────────────────────────
# 2. Rhythm Diversity Score
#    D_rhythm = #unique durations / #total notes
# ─────────────────────────────────────────────────────────────
def rhythm_diversity(roll: np.ndarray, threshold: float = 0.15) -> float:
    """
    Extracts note durations (in frames) and returns the ratio of
    unique durations to total note events.
    Higher = more rhythmically diverse.
    """
    durations = []
    for key in range(roll.shape[1]):
        active  = roll[:, key] > threshold
        note_on = None
        for t in range(len(active)):
            if active[t] and note_on is None:
                note_on = t
            elif not active[t] and note_on is not None:
                durations.append(t - note_on)
                note_on = None
        if note_on is not None:
            durations.append(len(active) - note_on)

    if not durations:
        return 0.0
    return len(set(durations)) / len(durations)


# ─────────────────────────────────────────────────────────────
# 3. Repetition Ratio
#    R = #repeated patterns / #total patterns
# ─────────────────────────────────────────────────────────────
def repetition_ratio(
    roll: np.ndarray,
    threshold: float = 0.15,
    window: int = 16,
) -> float:
    """
    Slides a window of `window` frames over the binary piano-roll,
    counts how many windows are exact duplicates of another.
    Higher = more repetitive.
    """
    binary   = (roll > threshold).astype(np.uint8)
    T        = binary.shape[0]
    patterns = [tuple(binary[i : i + window].flatten()) for i in range(T - window)]

    if not patterns:
        return 0.0

    from collections import Counter
    counts         = Counter(patterns)
    repeated_count = sum(v for v in counts.values() if v > 1)
    return repeated_count / len(patterns)


# ─────────────────────────────────────────────────────────────
# 4. Reconstruction Loss (task-level comparison helper)
# ─────────────────────────────────────────────────────────────
def mean_squared_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    return float(np.mean((original - reconstructed) ** 2))


# ─────────────────────────────────────────────────────────────
# 5. Evaluate a list of generated rolls and print a summary
# ─────────────────────────────────────────────────────────────
def evaluate_rolls(rolls: List[np.ndarray], task_label: str = "Task"):
    """
    Given a list of generated piano rolls, compute and print all metrics.
    Returns a dict of mean values for logging.
    """
    rd_scores  = [rhythm_diversity(r)  for r in rolls]
    rep_scores = [repetition_ratio(r)  for r in rolls]
    ph_scores  = [
        pitch_histogram_similarity(rolls[i], rolls[i + 1])
        for i in range(len(rolls) - 1)
    ]

    results = {
        "rhythm_diversity":          np.mean(rd_scores),
        "rhythm_diversity_std":      np.std(rd_scores),
        "repetition_ratio":          np.mean(rep_scores),
        "repetition_ratio_std":      np.std(rep_scores),
        "pitch_histogram_similarity": np.mean(ph_scores) if ph_scores else 0.0,
    }

    print("=" * 55)
    print(f"  {task_label} – Evaluation Metrics ({len(rolls)} samples)")
    print("=" * 55)
    print(f"  Rhythm Diversity        : {results['rhythm_diversity']:.4f}"
          f" ± {results['rhythm_diversity_std']:.4f}")
    print(f"  Repetition Ratio        : {results['repetition_ratio']:.4f}"
          f" ± {results['repetition_ratio_std']:.4f}")
    print(f"  Pitch Histogram Sim (H) : {results['pitch_histogram_similarity']:.4f}")
    print("=" * 55)

    return results