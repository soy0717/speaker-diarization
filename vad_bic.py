import numpy as np
import scipy.io.wavfile as wavfile
from scipy.signal import stft, butter, filtfilt
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
import librosa
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random

def preprocess_audio(audio_path, target_sr=16000):
    """
    Preprocess the audio file:
    1. Load and convert to mono if stereo
    2. Resample to target sample rate
    3. Apply pre-emphasis filter to enhance high frequencies
    4. Normalize audio amplitude
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=target_sr, mono=True)

    # Calculate duration
    duration = librosa.get_duration(y=y, sr=sr)

    # Apply pre-emphasis filter (emphasize higher frequencies)
    pre_emphasis = 0.97
    emphasized_signal = np.append(y[0], y[1:] - pre_emphasis * y[:-1])

    # Normalize audio
    emphasized_signal = emphasized_signal / np.max(np.abs(emphasized_signal))

    return emphasized_signal, target_sr, duration

def extract_mfcc_features(signal, sample_rate, frame_length=0.025, frame_step=0.01, num_mfcc=13):
    """
    Extract MFCC features from the audio signal
    """
    # Convert frame sizes from seconds to samples
    frame_length_samples = int(frame_length * sample_rate)
    frame_step_samples = int(frame_step * sample_rate)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=num_mfcc,
        hop_length=frame_step_samples,
        n_fft=frame_length_samples
    )

    # Add delta and delta-delta (1st and 2nd derivatives) to capture dynamics
    delta_mfccs = librosa.feature.delta(mfccs)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)

    # Stack all features
    features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])

    return features.T  # Return with frames as rows, features as columns

def voice_activity_detection(signal, sample_rate, frame_length=0.025, frame_step=0.01,
                             energy_threshold=0.1, zero_crossing_threshold=0.2):
    """
    Simple energy and zero-crossing based Voice Activity Detection (VAD)
    Returns VAD decisions with the same length as the feature vectors
    """
    # Convert frame sizes from seconds to samples
    frame_length_samples = int(frame_length * sample_rate)
    frame_step_samples = int(frame_step * sample_rate)

    # Calculate number of frames to match MFCC feature extraction
    num_frames = 1 + int((len(signal) - frame_length_samples) / frame_step_samples)

    # Calculate energy for each frame
    energy = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * frame_step_samples
        end = start + frame_length_samples
        if end <= len(signal):
            energy[i] = np.sum(signal[start:end]**2) / frame_length_samples

    energy = energy / np.max(energy) if np.max(energy) > 0 else energy  # Normalize

    # Calculate zero crossing rate for each frame
    zcr = np.zeros(num_frames)
    for i in range(num_frames):
        start = i * frame_step_samples
        end = start + frame_length_samples
        if end <= len(signal):
            zcr[i] = np.sum(np.abs(np.diff(np.signbit(signal[start:end])))) / (2 * frame_length_samples)

    zcr = zcr / np.max(zcr) if np.max(zcr) > 0 else zcr  # Normalize

    # Combine energy and ZCR for VAD decision
    vad_decision = (energy > energy_threshold) & (zcr < zero_crossing_threshold)

    return vad_decision

def speaker_segmentation(features, vad_mask, window_size=100, step_size=50, threshold=1.5):
    """
    Segment audio into homogeneous speaker segments using Bayesian Information Criterion (BIC)
    """
    # Make sure vad_mask has the same length as features
    if len(vad_mask) != len(features):
        # Trim or pad the VAD mask to match features length
        if len(vad_mask) > len(features):
            vad_mask = vad_mask[:len(features)]
        else:
            # Pad with False
            vad_mask = np.pad(vad_mask, (0, len(features) - len(vad_mask)), 'constant', constant_values=False)

    if not np.any(vad_mask):
        return [0, len(features)]  # No voice activity detected, return whole file as one segment

    # Apply VAD mask to features
    active_features = features[vad_mask]

    # If not enough active frames, return simple segmentation
    if len(active_features) < window_size * 2:
        return [0, len(features)]

    # Initialize segments
    change_points = [0]

    # Sliding window approach for change point detection
    for i in range(window_size, len(active_features) - window_size, step_size):
        left_window = active_features[i-window_size:i]
        right_window = active_features[i:i+window_size]

        # Calculate BIC
        bic_value = calculate_bic(left_window, right_window)

        # If BIC exceeds threshold, mark as change point
        if bic_value > threshold:
            change_points.append(i)

    # Add the final point
    change_points.append(len(active_features))

    # Convert change points from VAD indices back to original feature indices
    original_change_points = []
    vad_indices = np.where(vad_mask)[0]

    for cp in change_points:
        if cp < len(vad_indices):
            idx = vad_indices[min(cp, len(vad_indices)-1)]
            original_change_points.append(idx)
        else:
            original_change_points.append(len(features) - 1)

    # Make sure the change points are unique and sorted
    original_change_points = sorted(list(set(original_change_points)))

    # Make sure the first change point is 0
    if original_change_points[0] != 0:
        original_change_points.insert(0, 0)

    return original_change_points

def calculate_bic(left_segment, right_segment):
    """
    Calculate Bayesian Information Criterion (BIC) for two segments
    """
    if len(left_segment) < 2 or len(right_segment) < 2:
        return 0

    # Get dimensions
    n1, d = left_segment.shape
    n2 = right_segment.shape[0]
    n = n1 + n2

    # Calculate covariance matrices
    try:
        cov1 = np.cov(left_segment, rowvar=False)
        cov2 = np.cov(right_segment, rowvar=False)

        # Handle singleton dimension
        if np.isscalar(cov1):
            cov1 = np.array([[cov1]])
        if np.isscalar(cov2):
            cov2 = np.array([[cov2]])

        # Handle potential numerical issues
        cov1 = cov1 + np.eye(d) * 1e-6
        cov2 = cov2 + np.eye(d) * 1e-6

        # Calculate full segment
        full_segment = np.vstack((left_segment, right_segment))
        cov_full = np.cov(full_segment, rowvar=False) + np.eye(d) * 1e-6

        # Calculate BIC
        penalty = 0.5 * (d + 0.5*d*(d+1)) * np.log(n)

        # Use safe determinant calculation
        det_full = np.linalg.slogdet(cov_full)[1]
        det_cov1 = np.linalg.slogdet(cov1)[1]
        det_cov2 = np.linalg.slogdet(cov2)[1]

        bic = n * det_full - n1 * det_cov1 - n2 * det_cov2 - penalty

    except (np.linalg.LinAlgError, ValueError):
        # In case of numerical issues, return a safe value
        bic = 0

    return bic

def cluster_segments(features, change_points, n_clusters=None):
    """
    Cluster segments into speaker identities using agglomerative clustering
    """
    # Extract mean features for each segment
    segment_features = []
    for i in range(len(change_points) - 1):
        start, end = change_points[i], change_points[i+1]
        if start == end:
            continue  # Skip empty segments
        segment_mean = np.mean(features[start:end], axis=0)
        segment_features.append(segment_mean)

    # If not enough segments to cluster, return a single speaker
    if len(segment_features) <= 1:
        return [1]

    # Standardize features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(segment_features)

    # Perform hierarchical clustering
    Z = linkage(standardized_features, method='ward')

    # Determine optimal number of clusters if not specified
    if n_clusters is None:
        # Simple elbow method based on distances in linkage matrix
        distances = Z[:, 2]
        if len(distances) > 2:
            acceleration = np.diff(distances, 2)
            n_clusters = np.argmax(acceleration) + 2  # Add 2 for correct indexing
            n_clusters = min(max(n_clusters, 2), 8)  # Constrain between 2 and 8 speakers
        else:
            n_clusters = min(len(segment_features), 2)  # Default to 2 if not enough data

    # Assign cluster labels
    labels = fcluster(Z, n_clusters, criterion='maxclust')

    return labels

def visualize_diarization(diarization_result, total_duration, num_speakers):
    """
    Create a visualization of the speaker diarization result
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))

    # Generate colors for each speaker
    colors = plt.cm.Paired(np.linspace(0, 1, num_speakers))

    # Plot segments
    y_height = 0.8
    y_pos = 0.1

    # Track speaking time per speaker
    speaker_times = {}

    for segment in diarization_result:
        start = segment['start_time']
        end = segment['end_time']
        speaker = segment['speaker_id']
        speaker_idx = int(speaker.split('_')[1]) - 1  # Extract speaker number

        # Track speaking time
        if speaker not in speaker_times:
            speaker_times[speaker] = 0
        speaker_times[speaker] += (end - start)

        # Draw rectangle for segment
        rect = patches.Rectangle(
            (start, y_pos), end - start, y_height,
            linewidth=1, edgecolor='black', facecolor=colors[speaker_idx % len(colors)]
        )
        ax.add_patch(rect)

        # Add speaker label to long enough segments
        if (end - start) > total_duration / 50:  # Only label segments that are long enough
            ax.text(
                (start + end) / 2, y_pos + y_height / 2,
                speaker, ha='center', va='center', fontsize=9
            )

    # Set axis limits
    ax.set_xlim(0, total_duration)
    ax.set_ylim(0, 1)

    # Set labels and title
    ax.set_xlabel('Time (seconds)')
    ax.set_yticks([])  # No y-ticks needed
    ax.set_title(f'Speaker Diarization Timeline (Total Duration: {total_duration:.2f}s, Speakers: {num_speakers})')

    # Add legend for speakers
    for i in range(num_speakers):
        speaker = f"Speaker_{i+1}"
        duration = speaker_times.get(speaker, 0)
        percentage = (duration / total_duration) * 100
        ax.bar(0, 0, color=colors[i % len(colors)],
               label=f"{speaker}: {duration:.1f}s ({percentage:.1f}%)")

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(5, num_speakers))

    plt.tight_layout()
    plt.savefig('speaker_diarization_timeline.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig

def analyze_diarization_results(diarization_result, total_duration):
    """
    Analyze the diarization results to provide statistics
    """
    # Get unique speakers
    speakers = set()
    for segment in diarization_result:
        speakers.add(segment['speaker_id'])

    num_speakers = len(speakers)

    # Calculate speaking time per speaker
    speaker_times = {}
    for segment in diarization_result:
        speaker = segment['speaker_id']
        duration = segment['end_time'] - segment['start_time']
        if speaker not in speaker_times:
            speaker_times[speaker] = 0
        speaker_times[speaker] += duration

    # Calculate overlap (not precise without timestamp analysis)
    total_speaking_time = sum(speaker_times.values())
    overlap_estimate = max(0, total_speaking_time - total_duration)

    # Count speaker turns (transition from one speaker to another)
    speaker_turns = 0
    for i in range(1, len(diarization_result)):
        if diarization_result[i]['speaker_id'] != diarization_result[i-1]['speaker_id']:
            speaker_turns += 1

    # Average segment duration
    avg_segment_duration = sum(segment['end_time'] - segment['start_time'] for segment in diarization_result) / len(diarization_result)

    # Results
    analysis = {
        'num_speakers': num_speakers,
        'total_duration': total_duration,
        'speaker_times': speaker_times,
        'speaker_percentages': {s: (t/total_duration)*100 for s, t in speaker_times.items()},
        'overlap_estimate': overlap_estimate,
        'speaker_turns': speaker_turns,
        'avg_segment_duration': avg_segment_duration,
        'num_segments': len(diarization_result)
    }

    return analysis

def perform_diarization(audio_path, num_speakers=None):
    """
    Main function to perform speaker diarization and analysis
    """
    print("Step 1: Preprocessing audio...")
    # Preprocess audio
    signal, sample_rate, duration = preprocess_audio(audio_path)
    print(f"Audio duration: {duration:.2f} seconds")

    print("Step 2: Extracting MFCC features...")
    # Extract MFCC features
    features = extract_mfcc_features(signal, sample_rate)

    print("Step 3: Performing Voice Activity Detection...")
    # Perform VAD
    vad_decisions = voice_activity_detection(signal, sample_rate)
    print(f"Features shape: {features.shape}, VAD decisions shape: {vad_decisions.shape}")

    print("Step 4: Segmenting audio based on speaker changes...")
    # Segment audio based on speaker changes
    change_points = speaker_segmentation(features, vad_decisions)
    print(f"Found {len(change_points)-1} segments")

    print("Step 5: Clustering segments into speaker identities...")
    # Cluster segments into speaker identities
    speaker_labels = cluster_segments(features, change_points, n_clusters=num_speakers)
    detected_speakers = len(np.unique(speaker_labels))
    print(f"Identified {detected_speakers} speakers")

    # Convert to timeline format (start_time, end_time, speaker_id)
    frame_step = 0.01  # 10ms step size
    diarization_result = []

    for i in range(len(speaker_labels)):
        if i >= len(change_points) - 1:
            break  # Safety check

        start_idx = change_points[i]
        end_idx = change_points[i+1]
        start_time = start_idx * frame_step
        end_time = end_idx * frame_step
        speaker_id = speaker_labels[i]

        diarization_result.append({
            'start_time': start_time,
            'end_time': end_time,
            'speaker_id': f"Speaker_{speaker_id}"
        })

    # Analyze results
    analysis = analyze_diarization_results(diarization_result, duration)

    return diarization_result, analysis

if __name__ == "__main__":
    audio_file = "k3g_testaudio.mp3"

    try:
        results, analysis = perform_diarization(audio_file, num_speakers=None)

        print("\nDiarization Analysis:")
        print(f"Total audio duration: {analysis['total_duration']:.2f} seconds")
        print(f"Number of speakers detected: {analysis['num_speakers']}")
        print(f"Number of speaker segments: {analysis['num_segments']}")
        print(f"Average segment duration: {analysis['avg_segment_duration']:.2f} seconds")
        print(f"Number of speaker turns: {analysis['speaker_turns']}")

        print("\nSpeaking time per speaker:")
        for speaker, time in analysis['speaker_times'].items():
            percentage = analysis['speaker_percentages'][speaker]
            print(f"{speaker}: {time:.2f}s ({percentage:.1f}%)")

        visualize_diarization(results, analysis['total_duration'], analysis['num_speakers'])

    except Exception as e:
        print(f"Error during diarization: {str(e)}")
        import traceback
        traceback.print_exc()
