import logging
from pathlib import Path

from src.datasets.custom_audio_dataset import CustomAudioDataset

logger = logging.getLogger(__name__)


class CustomDirAudioDataset(CustomAudioDataset):
    def __init__(self, mix_dir, ref_dir=None, target_dir=None, *args, **kwargs):
        data = []
        is_test = kwargs["is_test"]
        del kwargs["is_test"]
        if is_test is None:
            is_test = 0
        for path in Path(mix_dir).iterdir():
            entry = {}
            if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                entry["path_mix"] = str(path)
                if ref_dir and Path(ref_dir).exists():
                    ref_path = Path(ref_dir) / (path.stem.split('-')[0] + '-ref' + path.suffix)
                    if ref_path.exists():
                        entry["path_ref"] = str(ref_path)
                        if is_test == 1:
                            entry['speaker_id'] = -1
                        else:
                            entry['speaker_id'] = int(ref_path.stem.split('_')[0])
                if target_dir and Path(target_dir).exists():
                    target_path = Path(target_dir) / (path.stem.split('-')[0] + '-target' + path.suffix)
                    if target_path.exists():
                        entry["path_target"] = str(target_path)
            if len(entry) > 0:
                data.append(entry)
        super().__init__(data, *args, **kwargs)
