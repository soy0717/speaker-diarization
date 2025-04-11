def remove_silence(y, sr, top_db=25):
    """
    Removes silence using librosa.effects.split
    Returns non-silent audio concatenated together.
    """
    intervals = librosa.effects.split(y, top_db=top_db)
    nonsilent = np.concatenate([y[start:end] for start, end in intervals])
    return nonsilent

import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# CONFIG
FRAME_DURATION = 2.0  # seconds
SR = 16000            # sampling rate

# Feature Extraction
def extract_features_auto(audio_path, sr=SR, frame_duration=1.0):
    y, sr = librosa.load(audio_path, sr=sr)
    y = remove_silence(y, sr, top_db=25)

    hop_length = int(sr * frame_duration)
    features, timestamps = [], []

    for i in range(0, len(y), hop_length):
        chunk = y[i:i + hop_length]
        if len(chunk) < hop_length:
            break
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
        features.append(np.mean(mfcc, axis=1))
        timestamps.append((i / sr, (i + hop_length) / sr))

    return np.array(features), timestamps, y, sr

# Estimate number of speakers
def estimate_num_clusters(features, max_k=10):
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        scores.append(score)
    best_k = np.argmax(scores) + 2
    return best_k

# Final Clustering
def cluster_speakers(features, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(features)

# Plot Results
def plot_diarization(y, sr, timestamps, labels):
    time = np.linspace(0, len(y) / sr, num=len(y))
    plt.figure(figsize=(14, 6))
    plt.plot(time, y, color='lightgray', label='Audio')

    colors = ['skyblue', 'salmon', 'lightgreen', 'plum', 'khaki', 'orange']
    for (start, end), label in zip(timestamps, labels):
        plt.axvspan(start, end, alpha=0.4, color=colors[label % len(colors)], label=f"Speaker {label}")

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Unsupervised Speaker Diarization (KMeans + Silhouette)")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()

# Main
def main():
    audio_path = "k3g_testaudio.mp3"  # replace with your actual path
    features_raw, timestamps, y, sr = extract_features_auto(audio_path)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_raw)

    pca = PCA(n_components=3)
    features_reduced = pca.fit_transform(features_scaled)

    best_k = estimate_num_clusters(features_reduced, max_k=10)
    print(f"Estimated Speakers: {best_k}")

    final_labels = cluster_speakers(features_reduced, best_k)

    plot_diarization(y, sr, timestamps, final_labels)

    print("\nDiarization Output (start_time, end_time, speaker_id):")
    for (start, end), label in zip(timestamps, final_labels):
        print(f"{start:.2f}s - {end:.2f}s: Speaker {label}")

if __name__ == "__main__":
    main()