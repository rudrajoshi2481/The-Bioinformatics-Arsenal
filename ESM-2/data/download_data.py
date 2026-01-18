# download_uniref.py
import requests
from tqdm import tqdm
import gzip
import shutil
from pathlib import Path


def download_uniref50():
    # Set the target directory
    target_dir = Path("/data/joshi/utils/ESM_junk")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://ftp.uniprot.org/pub/databases/uniprot/uniref/uniref50/uniref50.fasta.gz"
    output_file = target_dir / "uniref50.fasta.gz"
    final_file = target_dir / "uniref50.fasta"
    
    print(f"Downloading UniRef50 to {target_dir}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_file, 'wb') as file:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
                pbar.update(len(chunk))
    
    print("\nDecompressing...")
    with gzip.open(output_file, 'rb') as f_in:
        with open(final_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    # Optional: Remove the .gz file after decompression to save space
    # output_file.unlink()
    
    print(f"Download complete! File saved to: {final_file}")


if __name__ == "__main__":
    download_uniref50()
