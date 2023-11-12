import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    mix_lengths = []
    ref_lengths = []
    target_lengths = []
    speakers = []

    for ds in dataset_items:
        assert ds['audio_target'].shape[-1] == ds['audio_mix'].shape[-1]
        mix_lengths.append(ds['audio_mix'].shape[-1])
        ref_lengths.append(ds['audio_ref'].shape[-1])
        target_lengths.append(ds['audio_target'].shape[-1])
        speakers.append(ds["speaker"])

    batch_mix = torch.zeros(len(mix_lengths), max(mix_lengths))
    batch_ref = torch.zeros(len(ref_lengths), max(ref_lengths))
    batch_target = torch.zeros(len(target_lengths), max(target_lengths))

    for i, ds in enumerate(dataset_items):
        batch_mix[i, :mix_lengths[i]] = ds['audio_mix']
        batch_ref[i, :ref_lengths[i]] = ds['audio_ref']
        batch_target[i, :target_lengths[i]] = ds['audio_target']

    mix_lengths = torch.tensor(mix_lengths).long()
    ref_lengths = torch.tensor(ref_lengths).long()
    target_lengths = torch.tensor(target_lengths).long()
    speakers = torch.tensor(speakers).long()

    return {
        'mix': batch_mix,
        'ref': batch_ref,
        'target': batch_target,
        'mix_lengths': mix_lengths,
        'ref_lengths': ref_lengths,
        'target_lengths': target_lengths,
        'speaker': speakers
    }
