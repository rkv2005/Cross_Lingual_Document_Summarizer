import os
from huggingface_hub import hf_hub_download

MODELS = [
    {
        "repo_id": "facebook/mbart-large-50-many-to-many-mmt",
        "local_dir": "models/mbartlarge50mmt",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "sentencepiece.bpe.model",
            "special_tokens_map.json",
            "tokenizer_config.json"
        ]
    },
    {
        "repo_id": "facebook/bart-large-cnn",
        "local_dir": "models/bartlargecnn",
        "files": [
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",         # This file exists for BART-large-cnn
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json"
        ]
    }
]

def download_and_verify(repo_id, local_dir, files):
    os.makedirs(local_dir, exist_ok=True)
    downloaded = []
    missing = []
    for filename in files:
        try:
            print(f"Downloading {filename} for {repo_id} ...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir,
                force_download=True
            )
            downloaded.append(filename)
        except Exception as e:
            # Only print a short message for missing files (404s are normal for some files)
            if "404" in str(e) or "Entry Not Found" in str(e):
                missing.append(filename)
            else:
                print(f"Error downloading {filename}: {e}")
    return downloaded, missing

if __name__ == "__main__":
    for model in MODELS:
        print(f"\n--- Downloading files for {model['repo_id']} ---")
        downloaded, missing = download_and_verify(model["repo_id"], model["local_dir"], model["files"])
        print(f"\nDownloaded for {model['repo_id']}:")
        for f in downloaded:
            print(f"  ✔ {f}")
        if missing:
            print(f"Missing for {model['repo_id']} (not present in repo, usually OK):")
            for f in missing:
                print(f"  ✖ {f}")
        else:
            print("All essential files were downloaded for this model.")

    print("\nDownload complete! Your ./models/ folder is ready for local use.")
