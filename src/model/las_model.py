import random

import torch
import torch.nn as nn
import torch.nn.functional as F


class Listener(nn.Module):
    """
    Pyramidal BiLSTM encoder.
    Each pBLSTM layer reduces time dimension by 2x
    by concatenating consecutive frames.
    """

    def __init__(self, input_dim, hidden_dim, num_pyramid_layers=3):
        super().__init__()
        self.base_lstm = nn.LSTM(
            input_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.pyramid_layers = nn.ModuleList()
        for _ in range(num_pyramid_layers):
            self.pyramid_layers.append(
                nn.LSTM(
                    hidden_dim * 4,  # 2 (bidir) * 2 (concat pairs)
                    hidden_dim,
                    batch_first=True,
                    bidirectional=True,
                )
            )
        self.output_dim = hidden_dim * 2

    def forward(self, x, lengths):
        """
        Args:
            x: (B, T, input_dim)
            lengths: (B,)
        Returns:
            output: (B, T', hidden*2)
            lengths: (B,)
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False
        )
        output, _ = self.base_lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        for plstm in self.pyramid_layers:
            output, lengths = self._reduce_time(output, lengths)
            packed = nn.utils.rnn.pack_padded_sequence(
                output,
                lengths.cpu().clamp(min=1),
                batch_first=True,
                enforce_sorted=False,
            )
            output, _ = plstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, lengths

    def _reduce_time(self, x, lengths):
        B, T, D = x.shape
        if T % 2 != 0:
            x = x[:, :-1, :]
            T = T - 1
        x = x.contiguous().view(B, T // 2, D * 2)
        lengths = (lengths // 2).clamp(min=1)
        return x, lengths


class Attention(nn.Module):
    """
    Content-based attention (Bahdanau-style).
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_out, decoder_hidden, encoder_mask=None):
        """
        Args:
            encoder_out: (B, T, encoder_dim)
            decoder_hidden: (B, decoder_dim)
            encoder_mask: (B, T) — True where padded
        Returns:
            context: (B, encoder_dim)
            attn_weights: (B, T)
        """
        enc_proj = self.encoder_proj(encoder_out)  # (B, T, attn_dim)
        dec_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)  # (B, 1, attn_dim)
        energy = self.v(torch.tanh(enc_proj + dec_proj)).squeeze(-1)  # (B, T)

        if encoder_mask is not None:
            energy = energy.masked_fill(encoder_mask, float("-inf"))

        attn_weights = F.softmax(energy, dim=-1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_out).squeeze(1)
        return context, attn_weights


class Speller(nn.Module):
    """
    LSTM decoder with attention.
    """

    def __init__(
        self, vocab_size, embed_dim, decoder_dim, encoder_dim, attention_dim, num_layers=2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.num_layers = num_layers
        if num_layers > 1:
            self.extra_layers = nn.ModuleList(
                [nn.LSTMCell(decoder_dim, decoder_dim) for _ in range(num_layers - 1)]
            )
        self.fc = nn.Linear(decoder_dim + encoder_dim, vocab_size)
        self.decoder_dim = decoder_dim

    def forward(
        self,
        encoder_out,
        encoder_lengths,
        targets=None,
        max_len=500,
        teacher_forcing_ratio=0.9,
        sos_id=1,
        eos_id=2,
    ):
        """
        Args:
            encoder_out: (B, T, encoder_dim)
            encoder_lengths: (B,)
            targets: (B, S) — target token ids (without SOS prepended)
            max_len: max decoding length
            teacher_forcing_ratio: probability of using ground truth as input
            sos_id, eos_id: special token ids
        Returns:
            outputs: (B, max_len, vocab_size) — logits
        """
        B = encoder_out.size(0)
        device = encoder_out.device

        # build mask for encoder padding
        max_enc_len = encoder_out.size(1)
        encoder_lengths = encoder_lengths.to(device)
        encoder_mask = (
            torch.arange(max_enc_len, device=device).unsqueeze(0)
            >= encoder_lengths.unsqueeze(1)
        )

        # init decoder states
        h = [torch.zeros(B, self.decoder_dim, device=device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.decoder_dim, device=device) for _ in range(self.num_layers)]

        # determine decode length
        if targets is not None:
            decode_len = targets.size(1)
        else:
            decode_len = max_len

        outputs = []
        input_token = torch.full((B,), sos_id, dtype=torch.long, device=device)

        for t in range(decode_len):
            embed = self.embedding(input_token)  # (B, embed_dim)
            context, _ = self.attention(encoder_out, h[0], encoder_mask)
            lstm_input = torch.cat([embed, context], dim=-1)

            h[0], c[0] = self.lstm(lstm_input, (h[0], c[0]))
            for i in range(self.num_layers - 1):
                h[i + 1], c[i + 1] = self.extra_layers[i](h[i], (h[i + 1], c[i + 1]))

            out_concat = torch.cat([h[-1], context], dim=-1)
            logits = self.fc(out_concat)  # (B, vocab_size)
            outputs.append(logits)

            if targets is not None and random.random() < teacher_forcing_ratio:
                input_token = targets[:, t]
            else:
                input_token = logits.argmax(dim=-1)

        outputs = torch.stack(outputs, dim=1)  # (B, S, vocab_size)
        return outputs


class LASModel(nn.Module):
    """
    Listen, Attend and Spell model.
    """

    def __init__(
        self,
        n_feats,
        n_tokens,
        encoder_hidden=256,
        num_pyramid_layers=3,
        decoder_embed_dim=128,
        decoder_hidden=512,
        attention_dim=128,
        decoder_num_layers=2,
    ):
        super().__init__()
        self.encoder = Listener(n_feats, encoder_hidden, num_pyramid_layers)
        self.decoder = Speller(
            vocab_size=n_tokens,
            embed_dim=decoder_embed_dim,
            decoder_dim=decoder_hidden,
            encoder_dim=self.encoder.output_dim,
            attention_dim=attention_dim,
            num_layers=decoder_num_layers,
        )

    def forward(
        self,
        spectrogram,
        spectrogram_length,
        text_encoded=None,
        text_encoded_length=None,
        teacher_forcing_ratio=0.9,
        **batch,
    ):
        """
        Args:
            spectrogram: (B, n_feats, T)
            spectrogram_length: (B,)
            text_encoded: (B, S) — target ids
            text_encoded_length: (B,)
        Returns:
            dict with log_probs (B, S, vocab_size) and other info
        """
        x = spectrogram.transpose(1, 2)  # (B, T, n_feats)
        encoder_out, encoder_lengths = self.encoder(x, spectrogram_length)

        logits = self.decoder(
            encoder_out,
            encoder_lengths,
            targets=text_encoded,
            teacher_forcing_ratio=teacher_forcing_ratio,
            sos_id=1,
            eos_id=2,
        )

        log_probs = F.log_softmax(logits, dim=-1)

        return {
            "log_probs": log_probs,
            "logits": logits,
            "encoder_out": encoder_out,
            "encoder_lengths": encoder_lengths,
        }

    def decode(self, spectrogram, spectrogram_length, max_len=500):
        """
        Greedy decode without teacher forcing (for inference).
        """
        x = spectrogram.transpose(1, 2)
        encoder_out, encoder_lengths = self.encoder(x, spectrogram_length)
        logits = self.decoder(
            encoder_out,
            encoder_lengths,
            targets=None,
            max_len=max_len,
            teacher_forcing_ratio=0.0,
            sos_id=1,
            eos_id=2,
        )
        log_probs = F.log_softmax(logits, dim=-1)
        return {"log_probs": log_probs, "logits": logits}

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info
