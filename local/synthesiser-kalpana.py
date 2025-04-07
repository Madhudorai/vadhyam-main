import numpy as np
import librosa
import soundfile as sf
import ddsp
import time
import sounddevice as sd

# Config
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SIZE = 1024
HOP_SIZE = 64
CREPE_FRAME_RATE = 60
CONFIDENCE_THRESHOLD = 0.6
RECORDING_PATH = "recorded_audio.wav"
OUTPUT_PATH = "synthesized_audio.wav"


# Recording buffer
recorded_audio = []

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


INPUT_PATH = "recorded_audio.wav"
OUTPUT_PATH = "final_mixed_audio.wav"
SAMPLE_RATE = 16000
HOP_SIZE = 512
ENERGY_THRESHOLD = 0.001
MIN_SILENCE_DURATION_SEC = 0.2
INVERSE_DELAY_SEC = 0.08

# --- Load audio ---
audio, sr = librosa.load(INPUT_PATH, sr=SAMPLE_RATE, mono=True)
frame_len = HOP_SIZE
hop_len = HOP_SIZE
energy = librosa.feature.rms(y=audio, frame_length=frame_len, hop_length=hop_len)[0]

# --- Find silent frames ---
is_silent = energy < ENERGY_THRESHOLD
frame_times = librosa.frames_to_time(np.arange(len(is_silent)), sr=sr, hop_length=hop_len)

# --- Detect silent segments ---
silent_segments = []
start_idx = None
for i, silent in enumerate(is_silent):
    if silent and start_idx is None:
        start_idx = i
    elif not silent and start_idx is not None:
        end_idx = i
        duration = (end_idx - start_idx) * hop_len / sr
        if duration >= MIN_SILENCE_DURATION_SEC:
            silent_segments.append((start_idx, end_idx))
        start_idx = None

# --- Synthesize and insert into silence ---
output_audio = audio.copy()
for start_f, end_f in silent_segments:
    start_sample = start_f * hop_len
    end_sample = end_f * hop_len
    duration = end_sample - start_sample

    buffer_start = max(0, start_sample - duration)
    buffer_audio = audio[buffer_start:start_sample]

    if len(buffer_audio) < duration:
        continue

    # Synthesize buffer
    buffer = buffer_audio[np.newaxis, :]
    n_samples = buffer.shape[1]
    time_steps = n_samples // 64
    n_samples = time_steps * 64
    buffer = buffer[:, :n_samples]

    ddsp.spectral_ops.reset_crepe()
    f0_crepe, f0_confidence = ddsp.spectral_ops.compute_f0(buffer[0], frame_rate=60, viterbi=True)
    f0_filtered = np.where(f0_confidence >= 0.6, f0_crepe, 0.0)

    synth = ddsp.synths.Wavetable(n_samples=n_samples, scale_fn=None)
    wavetable = np.sin(np.linspace(0, 2 * np.pi, 2048))[np.newaxis, np.newaxis, :]
    amps = np.ones([1, time_steps, 1]) * 0.1
    synth_audio = synth(amps, wavetable, f0_filtered[np.newaxis, :time_steps, np.newaxis])
    synth_audio = np.squeeze(synth_audio).copy()

    # Trim and fade synth
    synth_audio = synth_audio[:duration]
    fade_len = int(0.1 * SAMPLE_RATE)
    fade_len = min(fade_len, len(synth_audio) // 2)
    fade_in = np.linspace(0, 1, fade_len)
    fade_out = np.linspace(1, 0, fade_len)
    envelope = np.ones_like(synth_audio)
    envelope[:fade_len] = fade_in
    envelope[-fade_len:] = fade_out
    synth_audio *= envelope

    # Inverse delay (left shift synth)
    inverse_delay_samples = int(INVERSE_DELAY_SEC * SAMPLE_RATE)
    synth_audio = np.roll(synth_audio, -inverse_delay_samples)
    synth_audio[-inverse_delay_samples:] = 0  # Zero out tail

    # Insert synth only into silent segment (keep rest of audio untouched)
    insert_len = min(len(synth_audio), end_sample - start_sample)
    output_audio[start_sample:start_sample + insert_len] += synth_audio[:insert_len]

# --- Save ---
sf.write(OUTPUT_PATH, output_audio, sr)
print(f"Saved gap-filled audio to {OUTPUT_PATH}")
