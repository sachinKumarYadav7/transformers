from pathlib import Path

def get_config():

    return {
        "batch_size": 32,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 128,
        "d_model": 512,
        "num_layers": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,

        # Hugging Face dataset
        "datasource": "cfilt/iitb-english-hindi",
        "lang_src": "en",
        "lang_tgt": "hi",

        # Keep the run feasible on a laptop GPU. Set to None to use the full split.
        "max_train_samples": 150_000,
        "max_val_samples": 500,
        "num_validation_examples": 2,
        "num_workers": 2,

        "model_folder": "weights",
        "model_basename": "tmodel_",

        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

def _model_folder(config):
    # datasource contains a "/", which would otherwise create a nested directory
    datasource = config['datasource'].replace('/', '_')
    return f"{datasource}_{config['model_folder']}"

def get_weights_file_path(config, epoch: str):
    model_folder = _model_folder(config)
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

def latest_weights_file_path(config):
    model_folder = _model_folder(config)
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])
