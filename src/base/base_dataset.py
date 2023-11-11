import logging
import random
from typing import List

import numpy as np
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

from src.utils.parse_config import ConfigParser

logger = logging.getLogger(__name__)


class BaseDataset(Dataset):
    def __init__(
            self,
            index,
            config_parser: ConfigParser,
            wave_augs=None,
            limit=None,
            max_audio_length_mix=None,
            max_audio_length_ref=None,
            max_audio_length_target=None
    ):
        self.config_parser = config_parser
        self.wave_augs = wave_augs

        self._assert_index_is_valid(index)
        index = self._filter_records_from_dataset(index, max_audio_length_mix, max_audio_length_ref, max_audio_length_target, limit)
        # it's a good idea to sort index by audio length
        # It would be easier to write length-based batch samplers later
        index = self._sort_index(index)
        self._index: List[dict] = index

    def __getitem__(self, ind):
        data_dict = self._index[ind]
        audio_path_mix = data_dict["path_mix"]
        audio_path_ref = data_dict["path_ref"]
        audio_path_target = data_dict["path_target"]
        audio_wave_mix = self.load_audio(audio_path_mix)
        audio_wave_ref = self.load_audio(audio_path_ref)
        audio_wave_target = self.load_audio(audio_path_target)
        audio_wave_mix = self.process_wave(audio_wave_mix)
        audio_wave_ref = self.process_wave(audio_wave_ref)
        # no augs for target
        return {
            "audio_mix": audio_wave_mix,
            "audio_ref": audio_wave_ref,
            "audio_target": audio_wave_target,
            "duration_mix": audio_wave_mix.size(1) / self.config_parser["preprocessing"]["sr"],
            "duration_ref": audio_wave_ref.size(1) / self.config_parser["preprocessing"]["sr"],
            "duration_target": audio_wave_target.size(1) / self.config_parser["preprocessing"]["sr"],
            "audio_path_mix": audio_path_mix,
            "audio_path_ref": audio_path_ref,
            "audio_path_target": audio_path_target,
            'speaker_id': data_dict['speaker_id']
        }

    @staticmethod
    def _sort_index(index):
        return sorted(index, key=lambda x: x["audio_len_mix"])

    def __len__(self):
        return len(self._index)

    def load_audio(self, path):
        audio_tensor, sr = torchaudio.load(path)
        audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
        target_sr = self.config_parser["preprocessing"]["sr"]
        if sr != target_sr:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
        return audio_tensor

    def process_wave(self, audio_tensor_wave: Tensor):
        with torch.no_grad():
            if self.wave_augs is not None:
                audio_tensor_wave = self.wave_augs(audio_tensor_wave)
            return audio_tensor_wave

    @staticmethod
    def _filter_records_from_dataset(
            index: list, max_audio_length_mix, max_audio_length_ref, max_audio_length_target, limit
    ) -> list:
        initial_size = len(index)
        if max_audio_length_mix is not None:
            exceeds_audio_length_mix = np.array([el["audio_len_mix"] for el in index]) >= max_audio_length_mix
            _total = exceeds_audio_length_mix.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) mix records are longer then "
                f"{max_audio_length_mix} seconds. Excluding them."
            )
        else:
            exceeds_audio_length_mix = False

        initial_size = len(index)
        if max_audio_length_ref is not None:
            exceeds_audio_length_ref = np.array([el["audio_len_ref"] for el in index]) >= max_audio_length_ref
            _total = exceeds_audio_length_ref.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) ref records are longer then "
                f"{max_audio_length_ref} seconds. Excluding them."
            )
        else:
            exceeds_audio_length_ref = False

        initial_size = len(index)
        if max_audio_length_target is not None:
            exceeds_audio_length_target = np.array([el["audio_len_target"] for el in index]) >= max_audio_length_target
            _total = exceeds_audio_length_target.sum()
            logger.info(
                f"{_total} ({_total / initial_size:.1%}) target records are longer then "
                f"{max_audio_length_target} seconds. Excluding them."
            )
        else:
            exceeds_audio_length_target = False

        records_to_filter = exceeds_audio_length_mix | exceeds_audio_length_ref | exceeds_audio_length_target

        if records_to_filter is not False and records_to_filter.any():
            _total = records_to_filter.sum()
            index = [el for el, exclude in zip(index, records_to_filter) if not exclude]
            logger.info(
                f"Filtered {_total}({_total / initial_size:.1%}) records  from dataset"
            )

        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_len_mix" in entry, (
                "Each dataset item should include field 'audio_len_mix'"
                " - duration of audio (in seconds)."
            )
            assert "audio_len_ref" in entry, (
                "Each dataset item should include field 'audio_len_ref'"
                " - duration of audio (in seconds)."
            )
            assert "audio_len_target" in entry, (
                "Each dataset item should include field 'audio_len_target'"
                " - duration of audio (in seconds)."
            )
            assert "path_mix" in entry, (
                "Each dataset item should include field 'path_mix'" " - path to mixed audio file."
            )

            assert "path_ref" in entry, (
                "Each dataset item should include field 'path_ref'" " - path to ref audio file."
            )

            assert "path_target" in entry, (
                "Each dataset item should include field 'path_target'" " - path to target audio file."
            )
