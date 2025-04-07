import numpy as np
import sounddevice as sd
import soundfile as sf
import ddsp
import librosa
import time
import os

# Config
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SIZE = 1024
HOP_SIZE = 64
CREPE_FRAME_RATE = 60
CONFIDENCE_THRESHOLD = 0.6
RECORDING_PATH = "kriti_audio.wav"
OUTPUT_PATH = "synthesized_audio.wav"


# Recording buffer
recorded_audio = []
"""
def audio_callback(indata, frames, time_info, status):
    recorded_audio.append(indata.copy())

# --- Start recording ---
print("Recording... Press Ctrl+C to stop.")

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS,
                        blocksize=FRAME_SIZE, callback=audio_callback):
        while True:
            time.sleep(0.1)
except KeyboardInterrupt:
    print("\nRecording stopped.")
except Exception as e:
    print(f"Error during recording: {e}")

# --- Save recorded audio ---
recorded_audio = np.concatenate(recorded_audio, axis=0)
sf.write(RECORDING_PATH, recorded_audio, SAMPLE_RATE)
print(f"Saved recording to {RECORDING_PATH}")
"""

# --- Load and preprocess ---
audio, sr = librosa.load(RECORDING_PATH, sr=None, mono=True)
audio = audio[np.newaxis, :]

# ✅ Limit to 30 seconds to avoid OOM
max_samples = sr * 30  # 30 seconds worth of samples
audio = audio[:, :min(audio.shape[1], max_samples)]

n_samples = audio.shape[1]
time_steps = n_samples // HOP_SIZE
n_samples = time_steps * HOP_SIZE
audio = audio[:, :n_samples]

# --- Extract f0 ---
ddsp.spectral_ops.reset_crepe()
print("Extracting pitch with CREPE...")
start_time = time.time()

f0_crepe, f0_confidence = ddsp.spectral_ops.compute_f0(
    audio[0], frame_rate=CREPE_FRAME_RATE, viterbi=True
)

print(f"Pitch extraction took {time.time() - start_time:.1f} seconds")

# --- Filter by confidence ---
f0_filtered = np.where(f0_confidence >= CONFIDENCE_THRESHOLD, f0_crepe, 0.0)

# --- Synthesize ---
synth = ddsp.synths.Wavetable(n_samples=n_samples, scale_fn=None)
wavetable = np.sin(np.linspace(0, 2.0 * np.pi, 2048))[np.newaxis, np.newaxis, :]
amps = np.ones([1, time_steps, 1]) * 0.1

audio_synth = synth(
    amps, wavetable, f0_filtered[np.newaxis, :time_steps, np.newaxis]
)
synth_audio = np.squeeze(audio_synth)

# --- Align synthesized audio with 87ms delay ---
delay_sec = 0.087
delay_samples = int(delay_sec * sr)

# Pad the synthesized audio with silence
synth_padded = np.pad(synth_audio, (delay_samples, 0), mode='constant')

# Trim to match lengths (if synth is longer now)
min_len = min(len(audio[0]), len(synth_padded))
original = audio[0][:min_len]
synth_padded = synth_padded[:min_len]

# --- Mix both signals together ---
mixed = original + synth_padded

# Normalize to avoid clipping
max_val = np.max(np.abs(mixed))
if max_val > 1.0:
    mixed = mixed / max_val

# --- Save final mix ---
FINAL_MIX_PATH = "kriti_mixed_audio.wav"
sf.write(FINAL_MIX_PATH, mixed, sr)
print(f"Final mix with delay saved to {FINAL_MIX_PATH}")

