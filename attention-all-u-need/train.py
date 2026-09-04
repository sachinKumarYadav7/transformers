import os

# Fall back to CPU for any op without a Metal kernel instead of hard-erroring mid-epoch.
# Must be set before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from model import build_transformer
from dataset import BilingualDataset, casual_mask, SPECIAL_TOKENS
from config  import get_config, get_weights_file_path, latest_weights_file_path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn

import warnings
from tqdm import tqdm
from pathlib import Path


# hugging face library
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


import torchmetrics
from torch.utils.tensorboard import SummaryWriter

def greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(encoder_input, encoder_mask)

    # Intialize the decoder input with sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)

    while True:
        if decoder_input.size(1) >= max_len:
            break

        # buid mask for target
        decoder_mask = casual_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

        # output calculate
        out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=-1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1
        )
        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)

    

def run_validation(model, val_dataloader, src_tokenizer, tgt_tokenizer, seq_len, device, log_fn, global_step, writer, num_examples=2):
    model.eval()
    count = 0

    source_text = []
    excepted = []  
    predicted = []

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().strip().split()
            console_width = int(console_width)
    except Exception as e:
        console_width = 80  # default width if the command fails

    with torch.no_grad():
        for batch in val_dataloader:
            count += 1

            encoder_input = batch['encoder_input'].to(device) # (b, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (b, 1, 1, seq_len)

            # check that batch size is 1
            assert encoder_input.size(0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, src_tokenizer, tgt_tokenizer, seq_len, device)


            src_text = batch["src_text"][0]
            tgt_text = batch["tgt_text"][0]
            predicted_text = tgt_tokenizer.decode(model_out.tolist())

            source_text.append(src_text)
            excepted.append(tgt_text)
            predicted.append(predicted_text)


            # print the source, target, model output
            log_fn("-" * console_width)
            log_fn(f"Source: {src_text}")
            log_fn(f"Target: {tgt_text}")
            log_fn(f"Predicted: {predicted_text}")
            
            if count == num_examples:
                log_fn("-" * console_width)
                break

    if writer:
        # Evaluate the character error rate
        # Compute the char error rate (CER)
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, excepted)
        writer.add_scalar("validation/cer", cer, global_step)

        # compute word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, excepted)
        writer.add_scalar("validation/wer", wer, global_step)

        # calculate BLEU score
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, [[t] for t in excepted])
        writer.add_scalar("validation/bleu", bleu, global_step)
        writer.flush()



def get_all_sentences(ds, lang):
    for item in ds:
        yield item['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if tokenizer_path.exists():
        return Tokenizer.from_file(str(tokenizer_path))

    tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=SPECIAL_TOKENS, min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
    return tokenizer



def get_ds(config):
    raw_ds = load_dataset(config['datasource'])

    src_lang, tgt_lang = config["lang_src"], config["lang_tgt"]
    # The tokenizer is WordLevel over a Whitespace pre-tokenizer, so a whitespace word
    # count is exactly the token count -- cheap enough to filter 1.6M rows with.
    max_words = config["seq_len"] - 2

    def short_enough(example):
        t = example['translation']
        return 0 < len(t[src_lang].split()) <= max_words and 0 < len(t[tgt_lang].split()) <= max_words

    def prepare(split, max_samples):
        split = split.filter(short_enough)
        if max_samples is not None and len(split) > max_samples:
            split = split.shuffle(seed=42).select(range(max_samples))
        return split

    train_raw = prepare(raw_ds["train"], config["max_train_samples"])
    val_raw = prepare(raw_ds["validation"], config["max_val_samples"])

    # build tokenizers for source and target languages
    src_tokenizer = get_or_build_tokenizer(config, train_raw, src_lang)
    tgt_tokenizer = get_or_build_tokenizer(config, train_raw, tgt_lang)

    train_ds = BilingualDataset(train_raw, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, config["seq_len"])
    val_ds = BilingualDataset(val_raw, src_tokenizer, tgt_tokenizer, src_lang, tgt_lang, config["seq_len"])

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")
    print(f"Source vocab size: {src_tokenizer.get_vocab_size()}")
    print(f"Target vocab size: {tgt_tokenizer.get_vocab_size()}")

    # calculate the maximum sequence lengths for source and target languages
    max_len_src = 0
    max_len_tgt = 0
    for item in train_raw:
        src_ids = src_tokenizer.encode(item['translation'][src_lang]).ids
        tgt_ids = tgt_tokenizer.encode(item['translation'][tgt_lang]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Maximum source sequence length: {max_len_src}")
    print(f"Maximum target sequence length: {max_len_tgt}")

    train_dataloader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        persistent_workers=config["num_workers"] > 0,
    )
    # greedy_decode runs one sentence at a time, so validation must use batch_size=1
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)

    return train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer


def get_model(config, vocab_size_src, vocab_size_tgt):
    return build_transformer(
        vocab_size_src=vocab_size_src,
        vocab_size_tgt=vocab_size_tgt,
        max_seq_len=config["seq_len"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"],
    )


def train_model(config):
    # 1. Determine the device: CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    # 2. Print device details
    if device.type == 'cuda':
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"Device memory: {torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB")
    elif device.type == 'mps':
        print("Device name: Apple Silicon (Metal Performance Shaders - MPS)")
    else:
        print("Running on CPU. Consider using a GPU or Apple Silicon machine for faster training.")

    # Make sure the weights folder exist
    Path(get_weights_file_path(config, "0")).parent.mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, src_tokenizer, tgt_tokenizer = get_ds(config)
    model = get_model(config, src_tokenizer.get_vocab_size(), tgt_tokenizer.get_vocab_size()).to(device)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # TensorBoard
    writer = SummaryWriter(f"{config['experiment_name']}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # If the user specified a model to preload before training, load it here
    intial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename and os.path.exists(model_filename):
        checkpoint = torch.load(model_filename, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # resume on the epoch after the one that was saved
        intial_epoch = checkpoint.get('epoch', -1) + 1
        global_step = checkpoint.get('global_step', 0)
        print(f"Loaded model from {model_filename}, starting at epoch {intial_epoch}, global step {global_step}")
    else:
        print("No pretrained model found, starting from scratch.")

    # labels are target-language, so ignore the *target* tokenizer's pad id
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tgt_tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(intial_epoch, config["num_epochs"]):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")

        for batch in batch_iterator:
            encoder_input = batch['encoder_input'].to(device)   # (B, seq_len)
            decoder_input = batch['decoder_input'].to(device)   # (B, seq_len)
            encoder_mask = batch['encoder_mask'].to(device)     # (B, 1, 1, seq_len)
            decoder_mask = batch['decoder_mask'].to(device)     # (B, 1, seq_len, seq_len)

            # Run the tensor through the encoder, decoder and projection layer
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)
            logits = model.project(decoder_output)  #(B, seq_len, vocab_size)

            # Compare the output with labels 
            label = batch['label'].to(device) #(B, seq_len)

            # compute the loss using a simple cross entropy
            loss = loss_fn(logits.view(-1, logits.size(-1)), label.view(-1))
            batch_iterator.set_postfix(loss=f"{loss.item():6.3f}")
        
            # Log the loss
            writer.add_scalar("Loss/train", loss.item(), global_step)

            loss.backward()

            # keep a bad batch from blowing the model up
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        writer.flush()
        run_validation(
            model, val_dataloader, src_tokenizer, tgt_tokenizer, config['seq_len'], device,
            lambda msg: batch_iterator.write(msg), global_step, writer,
            num_examples=config["num_validation_examples"],
        )

        # Save the model checkpoint after each epoch (zero-padded so sorting stays correct past epoch 9)
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, model_filename)
        print(f"Saved model checkpoint to {model_filename}")



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
