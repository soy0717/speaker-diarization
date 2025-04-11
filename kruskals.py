import numpy as np
import librosa
# import librosa.display
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree

# Load audio file
file_path = "k3g_testaudio.mp3"
y, sr = librosa.load(file_path, sr=None)

# Define frame parameters
frame_length = int(sr * 0.025)
hop_length = int(sr * 0.010)

# Function to segment audio into frames
def frame_audio(signal, frame_length, hop_length):
    return librosa.util.frame(signal, frame_length=frame_length, hop_length=hop_length).T

# Segment the audio into frames
frames = frame_audio(y, frame_length, hop_length)

# Compute Energy
energy = np.sum(frames ** 2, axis=1)

# Compute Zero-Crossing Rate (ZCR)
zcr = np.sum(np.abs(np.diff(np.sign(frames), axis=1)), axis=1)

# Set classification thresholds
energy_threshold = np.percentile(energy, 70)
zcr_threshold = np.percentile(zcr, 70)

# Select speech frames
speech_frames = frames[(energy > energy_threshold) & (zcr > zcr_threshold)]

# Limit the number of speech frames to avoid high memory usage
max_speech_frames = 500
speech_frames = speech_frames[:max_speech_frames]

# Reconstruct the speech signal from selected frames
speech_signal = speech_frames.flatten()

# Extract MFCC features (Reduced number from 13 → 10 for efficiency)
mfccs = librosa.feature.mfcc(y=speech_signal, sr=sr, n_mfcc=10, hop_length=hop_length).T

# Normalize features
scaler = StandardScaler()
mfccs = scaler.fit_transform(mfccs)

# Create graph and build MST
G = nx.Graph()
n_segments = len(mfccs)
G.add_nodes_from(range(n_segments))

# Use KDTree for fast neighbor search
k = 5
tree = KDTree(mfccs)
_, indices = tree.query(mfccs, k=k + 1)  # +1 for self

for i in range(n_segments):
    for j in indices[i][1:]:  # Skip self
        dist = euclidean(mfccs[i], mfccs[j])
        G.add_edge(i, j, weight=dist)

# Compute MST
mst = nx.minimum_spanning_tree(G)

# Extract connected components as clusters
clusters = nx.connected_components(mst)
speaker_segments = {i: list(cluster) for i, cluster in enumerate(clusters)}

# Generate speaker diarization timeline
speaker_timeline = {}
for speaker, segments in speaker_segments.items():
    start_times = [s * hop_length / sr for s in segments]
    end_times = [(s + 1) * hop_length / sr for s in segments]
    speaker_timeline[speaker] = list(zip(start_times, end_times))

# Print results
print("\nSpeaker Diarization Results:")
for speaker, times in speaker_timeline.items():
    print(f"🗣️ Speaker {speaker}: {times}")

# Plot the timeline
plt.figure(figsize=(10, 4))
for speaker, times in speaker_timeline.items():
    for (start, end) in times:
        plt.plot([start, end], [speaker, speaker], marker='o', markersize=4,
                 label=f'Speaker {speaker}' if start == times[0][0] else "")

plt.xlabel("Time (seconds)")
plt.ylabel("Speakers")
plt.title("Speaker Diarization Timeline")
plt.legend()
plt.show()

print(f"\nTotal Speakers Detected: {len(speaker_timeline)}")
