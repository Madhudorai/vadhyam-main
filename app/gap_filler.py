import numpy as np
import librosa
import soundfile as sf
import ddsp
import os

def fill_gaps(input_path: str, output_path: str) -> str:
    SAMPLE_RATE = 16000
    HOP_SIZE = 512
    ENERGY_THRESHOLD = 0.001
    MIN_SILENCE_DURATION_SEC = 0.2
    INVERSE_DELAY_SEC = 0.08

    audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    energy = librosa.feature.rms(y=audio, frame_length=HOP_SIZE, hop_length=HOP_SIZE)[0]
    is_silent = energy < ENERGY_THRESHOLD
    silent_segments = []
    start_idx = None
    for i, silent in enumerate(is_silent):
        if silent and start_idx is None:
            start_idx = i
        elif not silent and start_idx is not None:
            end_idx = i
            duration = (end_idx - start_idx) * HOP_SIZE / sr
            if duration >= MIN_SILENCE_DURATION_SEC:
                silent_segments.append((start_idx, end_idx))
            start_idx = None

    output_audio = audio.copy()
    for start_f, end_f in silent_segments:
        start_sample = start_f * HOP_SIZE
        end_sample = end_f * HOP_SIZE
        duration = end_sample - start_sample
        buffer_start = max(0, start_sample - duration)
        buffer_audio = audio[buffer_start:start_sample]
        if len(buffer_audio) < duration:
            continue
        buffer = buffer_audio[np.newaxis, :]
        n_samples = buffer.shape[1]
        time_steps = n_samples // 64
        n_samples = time_steps * 64
        buffer = buffer[:, :n_samples]

        ddsp.spectral_ops.reset_crepe()
        f0_crepe, f0_conf = ddsp.spectral_ops.compute_f0(buffer[0], frame_rate=60, viterbi=True)
        f0_filtered = np.where(f0_conf >= 0.6, f0_crepe, 0.0)

        synth = ddsp.synths.Wavetable(n_samples=n_samples, scale_fn=None)
        wavetable = np.sin(np.linspace(0, 2 * np.pi, 2048))[np.newaxis, np.newaxis, :]
        amps = np.ones([1, time_steps, 1]) * 0.1
        synth_audio = synth(amps, wavetable, f0_filtered[np.newaxis, :time_steps, np.newaxis])
        synth_audio = np.squeeze(synth_audio)[:duration].copy()

        fade_len = int(0.1 * SAMPLE_RATE)
        fade_len = min(fade_len, len(synth_audio) // 2)
        envelope = np.ones_like(synth_audio)
        envelope[:fade_len] = np.linspace(0, 1, fade_len)
        envelope[-fade_len:] = np.linspace(1, 0, fade_len)
        synth_audio *= envelope

        synth_audio = np.roll(synth_audio, -int(INVERSE_DELAY_SEC * SAMPLE_RATE))
        synth_audio[-int(INVERSE_DELAY_SEC * SAMPLE_RATE):] = 0
        output_audio[start_sample:start_sample + len(synth_audio)] += synth_audio

    sf.write(output_path, output_audio, sr)
    return output_path
