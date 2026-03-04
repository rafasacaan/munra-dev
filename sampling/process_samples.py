"""
Procesa one-shots de batería:
1. Normaliza a -1 dBFS
2. Agrega 5–10 ms antes del primer zero-crossing y corta en el último zero-crossing + 50 ms
"""

import os
import glob
import numpy as np
import soundfile as sf

# --- Configuración ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDERS = ["kick", "snare", "hh"]
PRE_MS = 7        # ms de silencio antes del primer zero-crossing (entre 5 y 10)
POST_MS = 50      # ms después del último zero-crossing
TARGET_DBFS = -1.0


def dbfs_to_linear(dbfs: float) -> float:
    return 10 ** (dbfs / 20)


def find_zero_crossings(audio: np.ndarray) -> np.ndarray:
    """Retorna índices donde la señal cruza cero (cambio de signo)."""
    signs = np.sign(audio)
    signs[signs == 0] = 1  # tratar exactos-cero como positivos
    crossings = np.where(np.diff(signs))[0]
    return crossings


def process_file(filepath: str) -> None:
    audio, sr = sf.read(filepath, always_2d=True)  # shape: (samples, channels)

    # --- Mono mix para detectar zero-crossings ---
    mono = audio.mean(axis=1)

    # --- Normalizar a -1 dBFS ---
    peak = np.max(np.abs(audio))
    if peak == 0:
        print(f"  SKIP (silencio): {filepath}")
        return
    target_peak = dbfs_to_linear(TARGET_DBFS)
    audio = audio * (target_peak / peak)
    mono = mono * (target_peak / peak)

    # --- Encontrar zero-crossings ---
    crossings = find_zero_crossings(mono)

    n_samples = len(mono)
    pre_samples = int(PRE_MS / 1000 * sr)
    post_samples = int(POST_MS / 1000 * sr)

    # Primer zero-crossing con señal antes del pico principal
    # Buscamos la muestra con mayor amplitud y luego el ZC anterior más cercano
    peak_idx = np.argmax(np.abs(mono))
    zc_before = crossings[crossings < peak_idx]
    if len(zc_before) == 0:
        first_zc = 0
    else:
        first_zc = zc_before[-1]  # el más cercano al pico por la izquierda

    # Último zero-crossing
    zc_after = crossings[crossings > peak_idx]
    if len(zc_after) == 0:
        last_zc = n_samples - 1
    else:
        last_zc = zc_after[-1]

    # --- Calcular rango de corte ---
    start = max(0, first_zc - pre_samples)
    end = min(n_samples, last_zc + post_samples)

    audio_trimmed = audio[start:end]

    # --- Sobrescribir archivo ---
    sf.write(filepath, audio_trimmed, sr)
    duration_ms = len(audio_trimmed) / sr * 1000
    print(f"  OK  {os.path.basename(filepath)}  "
          f"({duration_ms:.1f} ms, peak {20*np.log10(np.max(np.abs(audio_trimmed))):.2f} dBFS)")


def main():
    for folder in FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        if not os.path.isdir(folder_path):
            print(f"Carpeta no encontrada: {folder_path}")
            continue

        files = sorted(
            glob.glob(os.path.join(folder_path, "*.wav")) +
            glob.glob(os.path.join(folder_path, "*.WAV")) +
            glob.glob(os.path.join(folder_path, "*.aiff")) +
            glob.glob(os.path.join(folder_path, "*.AIFF")) +
            glob.glob(os.path.join(folder_path, "*.flac"))
        )

        print(f"\n[{folder.upper()}] {len(files)} archivos")
        for f in files:
            process_file(f)

    print("\nListo.")


if __name__ == "__main__":
    main()

