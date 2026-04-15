import re
from string import ascii_lowercase

import torch


class CTCTextEncoder:
    """
    Text encoder for LAS model.
    Despite the name (kept for template compatibility), supports
    SOS/EOS tokens and beam search decoding for attention-based models.
    """

    EMPTY_TOK = ""
    SOS_TOK = "<sos>"
    EOS_TOK = "<eos>"
    PAD_TOK = "<pad>"

    def __init__(self, alphabet=None, **kwargs):
        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.PAD_TOK, self.SOS_TOK, self.EOS_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        self.pad_id = self.char2ind[self.PAD_TOK]
        self.sos_id = self.char2ind[self.SOS_TOK]
        self.eos_id = self.char2ind[self.EOS_TOK]

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            encoded = [self.char2ind[char] for char in text]
            if len(encoded) == 0:
                encoded = [self.eos_id]
            return torch.Tensor(encoded).unsqueeze(0)
        except KeyError:
            unknown_chars = set(
                [char for char in text if char not in self.char2ind]
            )
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Decode indices into text. Stops at EOS if present.
        Skips SOS, PAD tokens.
        """
        result = []
        for ind in inds:
            ind = int(ind)
            if ind == self.eos_id:
                break
            if ind in (self.sos_id, self.pad_id):
                continue
            result.append(self.ind2char[ind])
        return "".join(result).strip()

    def ctc_decode(self, inds) -> str:
        """
        For compatibility with existing metrics.
        Just calls decode() since we don't use CTC.
        """
        return self.decode(inds)

    def beam_search_decode(self, log_probs, beam_size=5):
        """
        Simple beam search over a sequence of log_probs.

        Args:
            log_probs: Tensor of shape (T, vocab_size) — log probabilities
                at each decoding step.
            beam_size: number of beams.
        Returns:
            best_text: decoded string from the best beam.
        """
        beams = [(0.0, [])]

        for t in range(log_probs.shape[0]):
            next_beams = []
            for score, seq in beams:
                if len(seq) > 0 and seq[-1] == self.eos_id:
                    next_beams.append((score, seq))
                    continue
                probs_t = log_probs[t]
                topk_probs, topk_inds = torch.topk(probs_t, beam_size)
                for i in range(beam_size):
                    token = topk_inds[i].item()
                    new_score = score + topk_probs[i].item()
                    next_beams.append((new_score, seq + [token]))
            next_beams.sort(key=lambda x: x[0], reverse=True)
            beams = next_beams[:beam_size]

            if all(len(s) > 0 and s[-1] == self.eos_id for _, s in beams):
                break

        best_seq = beams[0][1]
        return self.decode(best_seq)

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
