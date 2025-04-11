import numpy as np
import librosa
from scipy.linalg import det
import matplotlib.pyplot as plt

# Compute MFCC features from the audio signal
def compute_mfcc(audio, sr):
    return librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13).T

# Compute the Bayesian Information Criterion (BIC) for a segment to determine the optimal split point
def compute_bic(X, start, end, penalty_coef=1.0):
    best_bic = float('inf')
    best_k = None
    for k in range(start + 10, end - 10):
        X1 = X[start:k]
        X2 = X[k:end]
        X_full = X[start:end]

        # Compute log-determinant of covariance matrix
        def cov_log_det(X):
            cov = np.cov(X.T)
            return np.log(det(cov) + 1e-6), cov.shape[0] * np.log(X.shape[0])

        l1, p1 = cov_log_det(X1)
        l2, p2 = cov_log_det(X2)
        l_full, p_full = cov_log_det(X_full)

        # BIC calculation
        bic = 0.5 * (X_full.shape[0]*l_full - X1.shape[0]*l1 - X2.shape[0]*l2) - penalty_coef * 0.5 * (p_full - p1 - p2)

        if bic < best_bic:
            best_bic = bic
            best_k = k
    return best_k, best_bic

# Detect change points using BIC within a sliding window
def detect_change_points(mfcc, win_size=100, hop_size=50):
    change_points = []
    for start in range(0, len(mfcc) - win_size, hop_size):
        end = start + win_size
        k, bic = compute_bic(mfcc, start, end)
        if k:
            change_points.append(k)
    return sorted(set(change_points))

# Diarize speakers using BIC change point detection
def diarize_bic(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    mfcc = compute_mfcc(y, sr)
    cps = detect_change_points(mfcc)

    # Convert frame indices to time in seconds
    times = [c * 0.01 for c in cps]

    # Print estimated speaker segments
    print("Estimated Speaker Segments:")
    start_time = 0.0
    speaker_id = 1
    for t in times:
        print(f"Speaker {speaker_id}: {start_time:.2f}s - {t:.2f}s")
        start_time = t
        speaker_id += 1
    print(f"Speaker {speaker_id}: {start_time:.2f}s - {len(y)/sr:.2f}s")

    return times

# Example usage
ans = diarize_bic("k3g_testaudio.mp3")
print(ans)
