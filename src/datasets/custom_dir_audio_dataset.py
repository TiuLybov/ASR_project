from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        data = []
        for path in sorted(Path(audio_dir).iterdir()):
            if path.suffix not in [".mp3", ".wav", ".flac", ".m4a"]:
                continue
            entry = {"path": str(path)}
            waveform, sr = torchaudio.load(str(path))
            entry["audio_len"] = waveform.shape[1] / sr

            if transcription_dir and Path(transcription_dir).exists():
                transc_path = Path(transcription_dir) / (path.stem + ".txt")
                if transc_path.exists():
                    with transc_path.open() as f:
                        entry["text"] = f.read().strip()

            if "text" not in entry:
                entry["text"] = ""

            data.append(entry)
        super().__init__(data, *args, **kwargs)
