import numpy as np
import soundfile as sf
import ddsp
import librosa

def synth_with_delay(input_path: str, output_path: str) -> str:
    SAMPLE_RATE = 16000
    HOP_SIZE = 64
    DELAY_SEC = 0.087

    audio, sr = librosa.load(input_path, sr=SAMPLE_RATE, mono=True)
    audio = audio[np.newaxis, :]
    n_samples = audio.shape[1]
    time_steps = n_samples // HOP_SIZE
    n_samples = time_steps * HOP_SIZE
    audio = audio[:, :n_samples]

    ddsp.spectral_ops.reset_crepe()
    f0_crepe, f0_conf = ddsp.spectral_ops.compute_f0(audio[0], frame_rate=60, viterbi=True)
    f0_filtered = np.where(f0_conf >= 0.6, f0_crepe, 0.0)

    synth = ddsp.synths.Wavetable(n_samples=n_samples, scale_fn=None)
    wavetable = np.sin(np.linspace(0, 2 * np.pi, 2048))[np.newaxis, np.newaxis, :]
    amps = np.ones([1, time_steps, 1]) * 0.1
    synth_audio = synth(amps, wavetable, f0_filtered[np.newaxis, :time_steps, np.newaxis])
    synth_audio = np.squeeze(synth_audio)

    delay_samples = int(DELAY_SEC * sr)
    synth_padded = np.pad(synth_audio, (delay_samples, 0), mode='constant')
    synth_padded = synth_padded[:len(audio[0])]
    mixed = audio[0] + synth_padded
    mixed /= np.max(np.abs(mixed)) + 1e-8

    sf.write(output_path, mixed, sr)
    return output_path
