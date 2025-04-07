import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import scipy
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score
import warnings
import os
warnings.filterwarnings('ignore')

class ImprovedSpeakerDiarizer:
    def __init__(self, filepath, min_speakers=2, max_speakers=8):
        self.filepath = filepath
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.frame_size = 0.025  # 25ms
        self.frame_shift = 0.01  # 10ms
        self.segments = []
        self.labels = []
        # Advanced parameters
        self.refinement_iterations = 3
        self.segment_length = 1.5  # seconds
        self.segment_overlap = 0.75  # 75% overlap for better resolution
        
    def preprocess_audio(self):
        """Load and preprocess the audio file with improved techniques"""
        print("Loading and preprocessing audio...")
        # Load audio file
        self.audio, self.sr = librosa.load(self.filepath, sr=16000, mono=True)
        
        # Normalize audio
        self.audio = librosa.util.normalize(self.audio)
        
        # Apply pre-emphasis filter
        self.audio = librosa.effects.preemphasis(self.audio, coef=0.97)
        
        # Advanced Voice Activity Detection using both energy and spectral features
        # This improves accuracy in detecting speech vs. non-speech
        frame_length = int(self.frame_size * self.sr)
        hop_length = int(self.frame_shift * self.sr)
        
        # Energy-based features
        energy = librosa.feature.rms(y=self.audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # Spectral features for better VAD
        zcr = librosa.feature.zero_crossing_rate(y=self.audio, frame_length=frame_length, hop_length=hop_length)[0]
        spectral_centroid = librosa.feature.spectral_centroid(y=self.audio, sr=self.sr, 
                                                           n_fft=frame_length, hop_length=hop_length)[0]
        
        # Combine features for VAD decision
        energy_threshold = 0.1 * np.mean(energy[energy > 0.05 * np.max(energy)])
        zcr_threshold = 0.8 * np.mean(zcr)
        
        # Combined VAD decision
        self.speech_frames = (energy > energy_threshold) & (zcr < zcr_threshold) 
        
        # Apply smoothing to VAD decisions using a median filter
        # This reduces sporadic false positives/negatives
        self.speech_frames = scipy.signal.medfilt(self.speech_frames.astype(int), kernel_size=5).astype(bool)
        
        # Create time axis for frames
        self.frame_time = librosa.frames_to_time(np.arange(len(energy)), 
                                               sr=self.sr, 
                                               hop_length=hop_length)
        
        print(f"Audio duration: {len(self.audio)/self.sr:.2f} seconds")
        print(f"Number of total frames: {len(self.speech_frames)}")
        print(f"Number of speech frames: {np.sum(self.speech_frames)}")
        
    def extract_features(self):
        """Extract enhanced acoustic features for speaker differentiation"""
        print("Extracting advanced acoustic features...")
        
        frame_length = int(self.frame_size * self.sr)
        hop_length = int(self.frame_shift * self.sr)
        
        # Extract MFCCs with more coefficients (19) for better speaker characteristics
        mfccs = librosa.feature.mfcc(y=self.audio, sr=self.sr, n_mfcc=19,
                                   hop_length=hop_length,
                                   n_fft=frame_length)
        
        # Calculate delta and delta-delta features to capture temporal dynamics
        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)
        
        # Add spectral features for better discrimination
        spectral_contrast = librosa.feature.spectral_contrast(y=self.audio, sr=self.sr,
                                                           n_fft=frame_length, hop_length=hop_length)
        
        # Pitch features (fundamental frequency) for better speaker discrimination
        pitch = librosa.yin(y=self.audio, fmin=50, fmax=500, sr=self.sr, 
                          frame_length=frame_length, hop_length=hop_length)
        
        # Convert pitch to log scale and handle zeros
        pitch[pitch == 0] = np.nan  # Replace zeros with NaN
        log_pitch = np.zeros_like(pitch)
        non_zero_idx = ~np.isnan(pitch)
        log_pitch[non_zero_idx] = np.log(pitch[non_zero_idx])
        log_pitch[np.isnan(log_pitch)] = np.nanmean(log_pitch)  # Replace NaNs with mean
        
        # Normalize and reshape to match other features
        log_pitch = (log_pitch - np.mean(log_pitch)) / np.std(log_pitch)
        log_pitch = log_pitch.reshape(1, -1)
        
        # Combine all features
        self.features = np.vstack([
            mfccs,        # 19 features
            delta_mfccs,  # 19 features  
            delta2_mfccs, # 19 features
            spectral_contrast[:5, :],  # First 5 bands of spectral contrast
            log_pitch     # 1 feature (pitch)
        ])
        
        self.features = self.features.T  # Shape: (num_frames, num_features)
        
        # Keep only speech frames
        self.features = self.features[self.speech_frames]
        self.speech_time = self.frame_time[self.speech_frames]
        
        # Handle NaN values that might have occurred
        self.features = np.nan_to_num(self.features)
        
        print(f"Feature shape: {self.features.shape}")
        
    def bic_segmentation(self):
        """Perform BIC-based segmentation to find speaker change points"""
        print("Performing BIC-based segmentation...")
        
        # Initialize segment boundaries with fixed initial segmentation
        window_size = int(self.segment_length / self.frame_shift)  # Convert seconds to frames
        step_size = int(window_size * (1 - self.segment_overlap))  # Using overlap for better coverage
        
        # Initial segmentation
        self.segment_boundaries = []
        frame_idx = 0
        while frame_idx + window_size < len(self.features):
            self.segment_boundaries.append(frame_idx)
            frame_idx += step_size
        self.segment_boundaries.append(len(self.features) - 1)  # Add last frame
        
        # Convert to time
        self.segment_times = [(self.speech_time[start], 
                             self.speech_time[min(end, len(self.speech_time)-1)]) 
                            for start, end in zip(self.segment_boundaries[:-1], self.segment_boundaries[1:])]
        
        print(f"Initial number of segments: {len(self.segment_times)}")
        
        # BIC refinement for change point detection
        if len(self.segment_boundaries) > 30:  # Only do BIC refinement if we have many segments
            # BIC parameters
            lambda_factor = 1.0  
            min_segment_duration = 1.0  # seconds
            min_segment_frames = int(min_segment_duration / self.frame_shift)
            
            refined_boundaries = [0]  # Always keep the first boundary
            
            for i in range(1, len(self.segment_boundaries) - 1):
                start = self.segment_boundaries[i-1]
                mid = self.segment_boundaries[i]
                end = self.segment_boundaries[i+1]
                
                # Only consider segments that are long enough
                if end - start < min_segment_frames:
                    continue
                
                # Get data for the combined segment and potential subsegments
                X_all = self.features[start:end]
                X_left = self.features[start:mid]
                X_right = self.features[mid:end]
                
                # Ensure we have enough data in each subsegment
                if len(X_left) < 5 or len(X_right) < 5:
                    continue
                
                # Calculate BIC
                d = X_all.shape[1]  # Feature dimension
                n = len(X_all)
                n1 = len(X_left)
                n2 = len(X_right)
                
                # Calculate full covariance
                cov_all = np.cov(X_all, rowvar=False) + np.eye(d) * 1e-10
                cov_left = np.cov(X_left, rowvar=False) + np.eye(d) * 1e-10
                cov_right = np.cov(X_right, rowvar=False) + np.eye(d) * 1e-10
                
                # Calculate log determinants
                try:
                    logdet_all = np.log(np.linalg.det(cov_all))
                    logdet_left = np.log(np.linalg.det(cov_left))
                    logdet_right = np.log(np.linalg.det(cov_right))
                    
                    # BIC formula
                    penalty = 0.5 * (d + 0.5 * d * (d + 1)) * np.log(n) * lambda_factor
                    bic = 0.5 * (n * logdet_all - n1 * logdet_left - n2 * logdet_right) - penalty
                    
                    # If BIC is positive, there's a speaker change point
                    if bic > 0:
                        refined_boundaries.append(mid)
                except:
                    # Skip in case of numerical issues
                    pass
            
            refined_boundaries.append(self.segment_boundaries[-1])  # Always keep the last boundary
            self.segment_boundaries = sorted(refined_boundaries)
            
            # Recalculate segment times
            self.segment_times = [(self.speech_time[start], 
                                 self.speech_time[min(end, len(self.speech_time)-1)]) 
                                for start, end in zip(self.segment_boundaries[:-1], self.segment_boundaries[1:])]
            
            print(f"Number of segments after BIC refinement: {len(self.segment_times)}")
        
        # Create feature vectors for each segment by taking the mean
        self.segments = []
        for start, end in zip(self.segment_boundaries[:-1], self.segment_boundaries[1:]):
            segment_features = np.mean(self.features[start:end], axis=0)
            self.segments.append(segment_features)
        
        self.segments = np.array(self.segments)
        
        # Normalize segment features
        scaler = StandardScaler()
        self.segments = scaler.fit_transform(self.segments)
        
    def find_optimal_num_speakers(self):
        """Find optimal number of speakers using multiple criteria"""
        print("Finding optimal number of speakers using enhanced methods...")
        
        # Calculate distance matrix
        distance_matrix = squareform(pdist(self.segments, metric='euclidean'))
        
        # Compute linkage for hierarchical clustering
        Z = linkage(distance_matrix, method='ward')
        
        # Calculate multiple metrics for different numbers of clusters
        bic_scores = []
        silhouette_scores = []
        
        for k in range(self.min_speakers, self.max_speakers + 1):
            # Get cluster assignments
            clusters = fcluster(Z, k, criterion='maxclust')
            
            # Calculate BIC
            bic = self._calculate_bic(clusters)
            bic_scores.append(bic)
            
            # Calculate silhouette score if we have enough clusters
            if k > 1 and len(np.unique(clusters)) > 1:
                try:
                    silhouette = silhouette_score(self.segments, clusters)
                    silhouette_scores.append(silhouette)
                except:
                    silhouette_scores.append(-1)  # Error case
            else:
                silhouette_scores.append(-1)
            
            print(f"Number of speakers: {k}, BIC: {bic:.2f}, Silhouette: {silhouette_scores[-1]:.4f}")
        
        # Normalize scores for fair comparison
        bic_scores = np.array(bic_scores)
        silhouette_scores = np.array(silhouette_scores)
        
        # Convert BIC to a score to maximize (like silhouette)
        bic_scores = (max(bic_scores) - bic_scores) / (max(bic_scores) - min(bic_scores) + 1e-10)
        
        # Combine scores (weighted average)
        combined_scores = 0.7 * bic_scores + 0.3 * silhouette_scores
        
        # Get optimal k (adding min_speakers because index starts at 0)
        self.num_speakers = np.argmax(combined_scores) + self.min_speakers
        print(f"Optimal number of speakers: {self.num_speakers}")
        
        return self.num_speakers
    
    def _calculate_bic(self, clusters):
        """Calculate BIC for a given clustering"""
        n_clusters = len(np.unique(clusters))
        n_features = self.segments.shape[1]
        n_samples = self.segments.shape[0]
        
        # Calculate within-cluster dispersion with regularization
        wcd = 0
        for i in range(1, n_clusters + 1):
            cluster_samples = self.segments[clusters == i]
            if len(cluster_samples) > 1:
                cluster_mean = np.mean(cluster_samples, axis=0)
                wcd += np.sum((cluster_samples - cluster_mean) ** 2)
        
        # Calculate BIC with a penalty term proportional to model complexity
        bic = wcd + np.log(n_samples) * (n_clusters * n_features) * 0.5
        
        return bic
        
    def cluster_segments(self):
        """Cluster segments using advanced techniques"""
        print("Clustering segments using refined methods...")
        
        # Calculate distance matrix
        distance_matrix = squareform(pdist(self.segments, metric='euclidean'))
        
        # Compute linkage for hierarchical clustering
        Z = linkage(distance_matrix, method='ward')
        
        # Get cluster labels
        self.labels = fcluster(Z, self.num_speakers, criterion='maxclust') - 1  # Convert to 0-based indexing
        
        # Detect potential overlapping speech using acoustic features
        self.overlaps = []
        
        # Look at pairs of adjacent segments
        for i in range(1, len(self.labels)):
            if self.labels[i] != self.labels[i-1]:
                # Get the boundary region
                prev_end = self.segment_times[i-1][1]
                curr_start = self.segment_times[i][0]
                
                # Find frames within this boundary
                boundary_start_idx = np.argmin(np.abs(self.speech_time - prev_end))
                boundary_end_idx = np.argmin(np.abs(self.speech_time - curr_start))
                
                # Get features in this region
                if boundary_end_idx > boundary_start_idx:
                    boundary_features = self.features[boundary_start_idx:boundary_end_idx]
                    
                    # Use variance of features as an overlap indicator
                    # High variance often indicates multiple speakers
                    if len(boundary_features) > 0:
                        feature_variance = np.var(boundary_features, axis=0).mean()
                        
                        # Compare to average variance of single-speaker segments
                        single_speaker_var = 0
                        count = 0
                        for j in range(len(self.segment_boundaries) - 1):
                            seg_features = self.features[self.segment_boundaries[j]:self.segment_boundaries[j+1]]
                            if len(seg_features) > 5:  # Minimum size for variance calculation
                                single_speaker_var += np.var(seg_features, axis=0).mean()
                                count += 1
                        
                        if count > 0:
                            single_speaker_var /= count
                            
                            # If variance is significantly higher, likely an overlap
                            if feature_variance > 1.4 * single_speaker_var:
                                self.overlaps.append((
                                    prev_end - 0.1,  # Add small margin
                                    curr_start + 0.1,
                                    [self.labels[i-1], self.labels[i]]
                                ))
        
        print(f"Found {len(self.overlaps)} potential overlapping regions")
        
    def refine_diarization(self):
        """Refine diarization using advanced techniques"""
        print("Refining diarization results...")
        
        # 1. GMM refinement
        # Train a GMM for each speaker
        gmms = []
        for speaker in range(self.num_speakers):
            # Collect all features for this speaker
            speaker_features = []
            for i, label in enumerate(self.labels):
                if label == speaker:
                    start, end = self.segment_boundaries[i], self.segment_boundaries[i+1]
                    speaker_features.extend(self.features[start:end])
            
            speaker_features = np.array(speaker_features)
            
            # Only train if we have enough data
            if len(speaker_features) > self.num_speakers * 2:
                # Train a GMM with 4 components (typical for speaker modeling)
                gmm = GaussianMixture(n_components=min(4, len(speaker_features)//10), 
                                     covariance_type='diag', 
                                     random_state=0)
                gmm.fit(speaker_features)
                gmms.append(gmm)
            else:
                gmms.append(None)
        
        # If we have trained GMMs for each speaker, perform refinement
        if all(gmm is not None for gmm in gmms):
            print("Performing GMM-based refinement...")
            
            # Refine segment labels
            for i in range(len(self.segments)):
                # Get segment boundaries
                start, end = self.segment_boundaries[i], self.segment_boundaries[i+1]
                segment_features = self.features[start:end]
                
                # Skip if not enough features
                if len(segment_features) < 5:
                    continue
                
                # Calculate log likelihood for each speaker
                loglikes = []
                for gmm in gmms:
                    # Compute average log likelihood per frame
                    loglike = np.mean(gmm.score_samples(segment_features))
                    loglikes.append(loglike)
                
                # Assign to the speaker with highest likelihood
                self.labels[i] = np.argmax(loglikes)
        
        # 2. Temporal smoothing
        # Apply median filtering to reduce label noise
        print("Applying temporal smoothing...")
        window_size = min(5, len(self.labels) // 10)  # Adaptive window size
        if window_size >= 3:  # Only apply if we have enough segments
            self.labels = scipy.signal.medfilt(self.labels, kernel_size=window_size)
        
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
        for i, (label, (start, end)) in enumerate(zip(self.labels, self.segment_times)):
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
        
    def visualize_diarization_with_segments(self):
        """Visualize diarization results with audio segments for each speaker"""
        print("Generating enhanced visualization with audio segments...")
        
        # Create a larger figure to accommodate the audio segments
        plt.figure(figsize=(12, 8))
        
        # Plot main waveform
        plt.subplot(self.num_speakers + 2, 1, 1)
        librosa.display.waveshow(self.audio, sr=self.sr, alpha=0.5)
        plt.title("Full Audio Waveform")
        plt.xlabel("")  # Remove x-label for this subplot
        
        # Plot speaker timeline
        plt.subplot(self.num_speakers + 2, 1, 2)
        
        # Create colormap
        colors = plt.cm.tab10(np.arange(self.num_speakers))
        
        # Plot segments
        for i, (label, (start, end)) in enumerate(zip(self.labels, self.segment_times)):
            plt.barh(y=label+1, width=end-start, left=start, height=0.8, color=colors[label], alpha=0.6)
        
        # Plot potential overlaps
        for start, end, speakers in self.overlaps:
            plt.axvspan(start, end, color='red', alpha=0.2)
            plt.text(start, 0.5, "Overlap", fontsize=8, color='red')
        
        plt.yticks(np.arange(1, self.num_speakers + 1), [f"Speaker {i+1}" for i in range(self.num_speakers)])
        plt.title("Speaker Timeline")
        plt.xlabel("")  # Remove x-label for this subplot
        
        # Create separate waveform for each speaker
        for speaker_idx in range(self.num_speakers):
            plt.subplot(self.num_speakers + 2, 1, speaker_idx + 3)
            
            # Create a masked version of the audio for this speaker
            masked_audio = np.zeros_like(self.audio)
            
            # For each segment assigned to this speaker
            for i, (label, (start, end)) in enumerate(zip(self.labels, self.segment_times)):
                if label == speaker_idx:
                    # Convert time to samples
                    start_sample = int(start * self.sr)
                    end_sample = int(end * self.sr)
                    # Ensure we don't go out of bounds
                    if end_sample > len(self.audio):
                        end_sample = len(self.audio)
                    # Apply the audio for this segment
                    masked_audio[start_sample:end_sample] = self.audio[start_sample:end_sample]
            
            # Display the masked audio
            librosa.display.waveshow(masked_audio, sr=self.sr, color=colors[speaker_idx], alpha=0.7)
            plt.title(f"Speaker {speaker_idx + 1}")
            if speaker_idx < self.num_speakers - 1:
                plt.xlabel("")  # Remove x-label except for the last subplot
        
        plt.xlabel("Time (s)")
        plt.tight_layout()
        
        plt.savefig("diarization_result_with_segments.png", dpi=300)
        plt.show()
    
    def save_speaker_segments(self, output_dir="./speaker_segments"):
        """Save audio segments for each speaker"""
        print(f"Saving individual speaker segments to {output_dir}...")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # For each speaker, create a separate audio file
        for speaker_idx in range(self.num_speakers):
            # Create an empty audio array
            speaker_audio = np.zeros_like(self.audio)
            
            # Collect segments for this speaker
            for i, (label, (start, end)) in enumerate(zip(self.labels, self.segment_times)):
                if label == speaker_idx:
                    # Convert time to samples
                    start_sample = int(start * self.sr)
                    end_sample = int(end * self.sr)
                    # Ensure we don't go out of bounds
                    if end_sample > len(self.audio):
                        end_sample = len(self.audio)
                    # Apply the audio for this segment
                    speaker_audio[start_sample:end_sample] = self.audio[start_sample:end_sample]
            
            # Save the audio file using soundfile
            output_file = os.path.join(output_dir, f"speaker_{speaker_idx + 1}.wav")
            try:
                import soundfile as sf
                sf.write(output_file, speaker_audio, self.sr)
            except ImportError:
                # Fallback to scipy.io.wavfile
                from scipy.io import wavfile
                wavfile.write(output_file, self.sr, (speaker_audio * 32767).astype(np.int16))
            
            print(f"Saved {output_file}")
        
        # Also save overlap segments if any
        if self.overlaps:
            overlap_audio = np.zeros_like(self.audio)
            
            for start, end, _ in self.overlaps:
                # Convert time to samples
                start_sample = max(0, int((start - 0.1) * self.sr))  # Add a small buffer
                end_sample = min(len(self.audio), int((end + 0.1) * self.sr))
                # Apply the audio for this segment
                overlap_audio[start_sample:end_sample] = self.audio[start_sample:end_sample]
            
            # Save the overlaps audio file
            output_file = os.path.join(output_dir, "overlaps.wav")
            try:
                import soundfile as sf
                sf.write(output_file, overlap_audio, self.sr)
            except ImportError:
                # Fallback to scipy.io.wavfile
                from scipy.io import wavfile
                wavfile.write(output_file, self.sr, (overlap_audio * 32767).astype(np.int16))
            
            print(f"Saved {output_file}")
    
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
        
        # Calculate and print speaker statistics
        total_duration = sum(end - start for start, end in self.segment_times)
        speaker_durations = {}
        
        for i, (label, (start, end)) in enumerate(zip(self.labels, self.segment_times)):
            speaker = label + 1  # Convert to 1-based indexing for display
            duration = end - start
            
            if speaker not in speaker_durations:
                speaker_durations[speaker] = 0
            speaker_durations[speaker] += duration
        
        print("\nSpeaker Statistics:")
        print("=================")
        for speaker, duration in sorted(speaker_durations.items()):
            percentage = (duration / total_duration) * 100
            print(f"Speaker {speaker}: {duration:.2f}s ({percentage:.1f}% of speech)")
    
    def run_diarization(self):
        """Run the complete diarization pipeline"""
        self.preprocess_audio()
        self.extract_features()
        self.bic_segmentation()  # Using improved BIC segmentation
        self.find_optimal_num_speakers()
        self.cluster_segments()
        
        # Multiple refinement iterations for better accuracy
        for i in range(self.refinement_iterations):
            print(f"Refinement iteration {i+1}/{self.refinement_iterations}")
            self.refine_diarization()
        
        # Visualizations and output
        self.visualize_diarization()
        self.visualize_diarization_with_segments()
        self.print_timeline()
        self.save_speaker_segments()
        
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
        audio_file = 'D:\\Soy\\New folder\\kkhh.mp3'  # Default audio file
        
    print(f"Processing file: {audio_file}")
    
    diarizer = ImprovedSpeakerDiarizer(audio_file)
    results = diarizer.run_diarization()
    
    print(f"\nDiarization complete. Found {results['num_speakers']} speakers.")