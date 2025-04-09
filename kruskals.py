import numpy as np
import librosa
import librosa.display
import networkx as nx
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform
from IPython.display import Audio

# ✅ Load One Full Audio Sample
print("Loading one full audio sample...")
dataset = load_dataset("talkbank/callhome", "eng", split="data", streaming=True)
audio_sample = next(iter(dataset))["audio"]
y, sr = audio_sample["array"], audio_sample["sampling_rate"]

# Display audio sample
print(f"✅ Audio Sample Loaded (Duration: {len(y) / sr:.2f} sec)")
display(Audio(data=y, rate=sr))

# ✅ Plot Audio Waveform
plt.figure(figsize=(10, 4))
librosa.display.waveshow(y, sr=sr)
plt.title("Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# ✅ Frame Segmentation with Increased Hop Length
frame_length = int(sr * 0.025)  # 25ms
hop_length = int(sr * 0.050)  # Increased to 50ms

def frame_audio(signal, frame_length, hop_length):
    return librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length).T

frames = frame_audio(y, frame_length, hop_length)

# ✅ Compute Energy and Zero-Crossing Rate
energy = np.sum(frames ** 2, axis=1)
zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)), axis=1)

# ✅ Speech/Silence Classification
energy_threshold = np.percentile(energy, 60)
zcr_threshold = np.percentile(zcr, 60)
speech_indices = np.where((energy > energy_threshold) & (zcr > zcr_threshold))[0]

# ✅ Extract and Normalize MFCC Features
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length).T
mfccs = mfccs[speech_indices]

# ✅ Downsample MFCCs (Avoid Too Many Small Segments)
def downsample_features(features, factor=5):
    return np.array([np.mean(features[i : i + factor], axis=0) for i in range(0, len(features) - factor, factor)])

mfccs = downsample_features(mfccs, factor=5)
scaler = StandardScaler()
mfccs = scaler.fit_transform(mfccs)

# ✅ Construct Similarity Graph Using KNN
G = nx.Graph()
n_segments = len(mfccs)
k = 7  # Moderate K to avoid over-fragmentation
knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
knn.fit(mfccs)
distances, indices = knn.kneighbors(mfccs)

# ✅ Add Edges Based on KNN
for i in range(n_segments):
    for j in indices[i]:
        if i != j:
            dist = distances[i][np.where(indices[i] == j)[0][0]]
            G.add_edge(i, j, weight=dist)

# ✅ Compute Minimum Spanning Tree (MST)
mst = nx.minimum_spanning_tree(G)

# ✅ Agglomerative Clustering (Fix Distance Threshold)
dist_matrix = squareform(pdist(mfccs, metric='euclidean'))
clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=5.0, linkage='ward')  # Increased threshold
speaker_labels = clustering.fit_predict(dist_matrix)

# ✅ Fix Over-Segmentation in Speaker Timeline
speaker_timeline = {}
for i, speaker in enumerate(speaker_labels):
    start_time = speech_indices[i] * hop_length / sr
    end_time = (speech_indices[i] + 1) * hop_length / sr
    if speaker not in speaker_timeline:
        speaker_timeline[speaker] = []
    speaker_timeline[speaker].append((start_time, end_time))

# ✅ Merge Consecutive Speaker Segments
merged_timeline = {}
for speaker, times in speaker_timeline.items():
    times.sort()
    merged = [times[0]]
    for curr_start, curr_end in times[1:]:
        prev_start, prev_end = merged[-1]
        if curr_start - prev_end <= 0.5:  # Merge segments within 500ms
            merged[-1] = (prev_start, curr_end)
        else:
            merged.append((curr_start, curr_end))
    merged_timeline[speaker] = merged

# ✅ Display Speaker Diarization Results
# print("\n✅ Speaker Diarization Results:")
# for speaker, times in merged_timeline.items():
#     print(f"🗣️ Speaker {speaker}: {times}")

print(f"\n🔍 Total Speakers Detected: {len(merged_timeline)}")  # Should be realistic (~2-5 speakers)

# ✅ Plot Speaker Timeline
plt.figure(figsize=(10, 4))
for speaker, times in merged_timeline.items():
    for (start, end) in times:
        plt.plot([start, end], [speaker, speaker], marker='o', markersize=4)

plt.xlabel("Time (seconds)")
plt.ylabel("Speakers")
plt.title("Speaker Diarization Timeline")
plt.legend()
plt.show()
