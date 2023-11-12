import logging
from pathlib import Path

import torchaudio

from src.base.base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class CustomAudioDataset(BaseDataset):
    def __init__(self, data, *args, **kwargs):
        index = data
        speakers_class = {}
        num_speakers = -1
        for entry in data:
            assert "path_mix" in entry
            assert "path_ref" in entry
            assert "path_target" in entry
            assert Path(entry["path_mix"]).exists(), f"Path {entry['path_mix']} doesn't exist"
            assert Path(entry["path_ref"]).exists(), f"Path {entry['path_ref']} doesn't exist"
            assert Path(entry["path_target"]).exists(), f"Path {entry['path_target']} doesn't exist"
            entry["path_mix"] = str(Path(entry["path_mix"]).absolute().resolve())
            t_info = torchaudio.info(entry["path_mix"])
            entry["audio_len_mix"] = t_info.num_frames / t_info.sample_rate

            entry["path_ref"] = str(Path(entry["path_ref"]).absolute().resolve())
            t_info = torchaudio.info(entry["path_ref"])
            entry["audio_len_ref"] = t_info.num_frames / t_info.sample_rate

            entry["path_target"] = str(Path(entry["path_target"]).absolute().resolve())
            t_info = torchaudio.info(entry["path_target"])
            entry["audio_len_target"] = t_info.num_frames / t_info.sample_rate

            assert "speaker_id" in entry
            speaker_id = entry["speaker_id"]
            if speaker_id in speakers_class:
                entry["speaker"] = speakers_class[speaker_id]
            else:
                num_speakers += 1
                speakers_class[speaker_id] = num_speakers
                entry["speaker"] = speakers_class[speaker_id]

        super().__init__(index, *args, **kwargs)
