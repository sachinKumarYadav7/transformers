import torch 
import torch.nn as nn
from torch.utils.data import Dataset

# Special tokens, shared with train.py so the tokenizer and the dataset agree.
SPECIAL_TOKENS = ["[UNK]", "[PAD]", "[SOS]", "[EOS]"]


class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        self.seq_len = seq_len

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]

        # Transform the text into tokens
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # get_ds() already filters out over-long pairs; truncate as a safety net so a
        # stray long sentence cannot kill a training run several hours in.
        enc_input_tokens = enc_input_tokens[: self.seq_len - 2]   # room for [SOS] and [EOS]
        dec_input_tokens = dec_input_tokens[: self.seq_len - 1]   # room for one of them

        # Add sos, eos and padding to each sentence
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2  # We will add <s> and </s>
        # We will only add <s>, and </s> only on the label
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                self.pad_token.repeat(enc_num_padding_tokens),
            ],
            dim=0,
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.pad_token.repeat(dec_num_padding_tokens),
            ],
            dim=0,
        )   
        # Add only </s> token
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                self.pad_token.repeat(dec_num_padding_tokens),
            ],
            dim=0,
        )   

        assert encoder_input.shape[0] == self.seq_len, f"Encoder input length is {encoder_input.shape[0]}, expected {self.seq_len}"
        assert decoder_input.shape[0] == self.seq_len, f"Decoder input length is {decoder_input.shape[0]}, expected {self.seq_len}"
        assert label.shape[0] == self.seq_len, f"Label length is {label.shape[0]}, expected {self.seq_len}" 

        return {
            "encoder_input": encoder_input,   # (seq_len,)
            "decoder_input": decoder_input,   # (seq_len,)
            # (1, 1, seq_len) so it broadcasts against attention scores of (batch, h, seq_len, seq_len)
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            # (1, seq_len, seq_len): hide padding AND anything ahead of the current position
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & casual_mask(decoder_input.size(0)),
            "label": label,                   # (seq_len,)
            "src_text": src_text,
            "tgt_text": tgt_text,
        }


def casual_mask(size):
    """Causal mask for the decoder: 1 where a position may be attended to, 0 where it must be hidden."""
    mask = torch.tril(torch.ones((1, size, size), dtype=torch.int))  # (1, size, size)
    return mask  # 1 == keep (at or before the current position), 0 == hide the future
