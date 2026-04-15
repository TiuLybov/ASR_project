import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """
    spectrograms = [item["spectrogram"].squeeze(0).T for item in dataset_items]
    spectrogram_lengths = torch.tensor([s.shape[0] for s in spectrograms])
    spectrograms_padded = pad_sequence(spectrograms, batch_first=True)
    # (B, T, n_feats) -> (B, n_feats, T)
    spectrograms_padded = spectrograms_padded.permute(0, 2, 1)

    text_encoded = [item["text_encoded"].squeeze(0) for item in dataset_items]
    text_encoded_lengths = torch.tensor([t.shape[0] for t in text_encoded])
    text_encoded_padded = pad_sequence(text_encoded, batch_first=True, padding_value=0)

    texts = [item["text"] for item in dataset_items]
    audio_paths = [item["audio_path"] for item in dataset_items]

    batch = {
        "spectrogram": spectrograms_padded,
        "spectrogram_length": spectrogram_lengths,
        "text_encoded": text_encoded_padded.long(),
        "text_encoded_length": text_encoded_lengths,
        "text": texts,
        "audio_path": audio_paths,
    }

    return batch
