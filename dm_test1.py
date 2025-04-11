import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

class SpeakerDiarizer:
    def __init__(self, filepath, min_speakers=2, max_speakers=5):
        self.filepath = filepath
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.frame_size = 0.025  # 25ms
        self.frame_shift = 0.01  # 10ms
        self.segments = []
        self.labels = []
        self.num_speakers = 0
        
    def preprocess_audio(self):
        """Load and preprocess the audio file"""
        print("Loading and preprocessing audio...")
        # Load audio file
        self.audio, self.sr = librosa.load(self.filepath, sr=16000, mono=True)
        
        # Normalize audio
        self.audio = librosa.util.normalize(self.audio)
        
        # Apply pre-emphasis filter
        self.audio = librosa.effects.preemphasis(self.audio, coef=0.97)
        
        # Voice Activity Detection (simple energy-based)
        energy = librosa.feature.rms(y=self.audio, frame_length=int(self.frame_size * self.sr), 
                                  hop_length=int(self.frame_shift * self.sr))[0]
        energy_threshold = 0.1 * np.max(energy)
        self.speech_frames = energy > energy_threshold
        
        # Create time axis for frames
        self.frame_time = librosa.frames_to_time(np.arange(len(energy)), 
                                               sr=self.sr, 
                                               hop_length=int(self.frame_shift * self.sr))
        
        print(f"Audio duration: {len(self.audio)/self.sr:.2f} seconds")
        print(f"Number of frames: {len(self.speech_frames)}")
        
    def extract_features(self):
        """Extract MFCC features from the audio"""
        print("Extracting MFCC features...")
        
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=19,
                                   hop_length=int(self.frame_shift * self.sr),
                                   n_fft=int(self.frame_size * self.sr))
        
        # Calculate delta and delta-delta features
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Combine features
        self.features = np.vstack([mfccs, delta_mfccs, delta2_mfccs])
        self.features = self.features.T  # Shape: (num_frames, num_features)
        
        # Keep only speech frames
        self.features = self.features[self.speech_frames]
        self.speech_time = self.frame_time[self.speech_frames]
        
        print(f"Feature shape: {self.features.shape}")
        
    def segmentation(self, segment_length=1.0):
        """Segment the audio into fixed-length segments"""
        print("Segmenting audio...")
        
        # Calculate number of frames per segment
        frames_per_segment = int(segment_length / self.frame_shift)
        
        # Create segments
        self.segments = []
        self.segment_times = []
        
        i = 0
        while i + frames_per_segment <= len(self.features):
            segment_features = self.features[i:i+frames_per_segment]
            self.segments.append(np.mean(segment_features, axis=0))
            self.segment_times.append((self.speech_time[i], self.speech_time[min(i+frames_per_segment-1, len(self.speech_time)-1)]))
            i += frames_per_segment // 2  # 50% overlap
            
        self.segments = np.array(self.segments)
        
        # Normalize segment features
        scaler = StandardScaler()
        self.segments = scaler.fit_transform(self.segments)
        
        print(f"Number of segments: {len(self.segments)}")
        
    def find_optimal_num_speakers(self):
        """Find optimal number of speakers using BIC or Silhouette score"""
        print("Finding optimal number of speakers...")

        # Calculate distance matrix
        distance_matrix = squareform(pdist(self.segments, metric='euclidean'))
        
        # Compute linkage
        Z = linkage(distance_matrix, method='ward')
        
        # Calculate BIC and Silhouette scores for different numbers of clusters
        bic_scores = []
        silhouette_scores = []
        
        for k in range(self.min_speakers, self.max_speakers + 1):
            clusters = fcluster(Z, k, criterion='maxclust')
            # Use silhouette score to measure cluster validity
            sil_score = silhouette_score(self.segments, clusters)
            silhouette_scores.append(sil_score)
            print(f"Number of speakers: {k}, Silhouette Score: {sil_score:.2f}")
        
        # Select number of speakers based on the highest silhouette score
        self.num_speakers = np.argmax(silhouette_scores) + self.min_speakers
        print(f"Optimal number of speakers: {self.num_speakers}")
        
        return self.num_speakers
    
    def cluster_segments(self):
        """Cluster segments using agglomerative clustering"""
        print("Clustering segments...")
        
        # Apply agglomerative clustering
        clustering = AgglomerativeClustering(
            n_clusters=self.num_speakers,
            metric='euclidean',  # Changed from 'affinity'
            linkage='ward'
        )
        
        self.labels = clustering.fit_predict(self.segments)
        
        # Detect potential overlapping speech
        self.overlaps = []
        for i in range(1, len(self.labels)):
            if self.labels[i] != self.labels[i-1]:
                # Mark as potential overlap
                overlap_start = self.segment_times[i-1][1] - 0.25  # 250ms before end of previous segment
                overlap_end = self.segment_times[i][0] + 0.25      # 250ms after start of current segment
                self.overlaps.append((overlap_start, overlap_end, [self.labels[i-1], self.labels[i]]))
    
        print(f"Found {len(self.overlaps)} potential overlapping regions")
        
    def refine_diarization(self):
        """Refine diarization using Viterbi smoothing (simplified)"""
        print("Refining diarization...")
        
        # Simple smoothing using sliding window majority vote
        window_size = 3
        smoothed_labels = np.copy(self.labels)
        
        for i in range(window_size, len(self.labels) - window_size):
            window = self.labels[i-window_size:i+window_size+1]
            # Get the most common label in the window
            unique_labels, counts = np.unique(window, return_counts=True)
            smoothed_labels[i] = unique_labels[np.argmax(counts)]
            
        self.labels = smoothed_labels
    
    def visualize_diarization(self):
        """Visualize diarization results"""
        print("Generating visualization...")
        
        plt.figure(figsize=(10, 6))
        
        # Plot waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(self.audio, sr=self.sr, alpha=0.5)
        plt.title("Audio Waveform")
        
        # Plot speaker timeline
        plt.subplot(2, 1, 2)
        
        # Create colormap
        colors = plt.cm.tab10(np.arange(self.num_speakers))
        
        # Plot segments
        for i, (label, ((start, end))) in enumerate(zip(self.labels, self.segment_times)):
            plt.barh(y=label+1, width=end-start, left=start, height=0.8, color=colors[label], alpha=0.6)
        
        # Plot potential overlaps
        for start, end, speakers in self.overlaps:
            plt.axvspan(start, end, color='red', alpha=0.2)
            plt.text(start, 0.5, "Overlap", fontsize=8, color='red')
        
        plt.yticks(np.arange(1, self.num_speakers + 1), [f"Speaker {i+1}" for i in range(self.num_speakers)])
        plt.xlabel("Time (s)")
        plt.title("Speaker Diarization")
        plt.tight_layout()
        
        plt.savefig("diarization_result.png")
        plt.show()
        
    def print_timeline(self):
        """Print the speaker timeline"""
        print("\nSpeaker Timeline:")
        print("================")
        
        current_speaker = self.labels[0]
        start_time = self.segment_times[0][0]
        
        for i in range(1, len(self.labels)):
            if self.labels[i] != current_speaker:
                end_time = self.segment_times[i-1][1]
                print(f"{start_time:.2f}s - {end_time:.2f}s: Speaker {current_speaker + 1}")
                current_speaker = self.labels[i]
                start_time = self.segment_times[i][0]
                
        # Last segment
        end_time = self.segment_times[-1][1]
        print(f"{start_time:.2f}s - {end_time:.2f}s: Speaker {current_speaker + 1}")
        
        # Print overlaps
        if self.overlaps:
            print("\nPotential Overlaps:")
            for start, end, speakers in self.overlaps:
                speakers_str = " and ".join([f"Speaker {s+1}" for s in speakers])
                print(f"{start:.2f}s - {end:.2f}s: {speakers_str}")
                
    def run_diarization(self):
        """Run the complete diarization pipeline"""
        self.preprocess_audio()
        self.extract_features()
        self.segmentation()
        self.find_optimal_num_speakers()
        self.cluster_segments()
        self.refine_diarization()
        self.visualize_diarization()
        self.print_timeline()
        
        return {
            'num_speakers': self.num_speakers,
            'timeline': list(zip(self.segment_times, self.labels)),
            'overlaps': self.overlaps
        }

# Usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        audio_file = "k3g_testaudio.mp3"  # Default audio file
        
    print(f"Processing file: {audio_file}")
    
    diarizer = SpeakerDiarizer(audio_file)
    results = diarizer.run_diarization()
    
    print(f"\nDiarization complete. Found {results['num_speakers']} speakers.")
