import os
import urllib.request
from tqdm import tqdm

def download_model(url, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.basename(url)
    output_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(output_path):
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get('Content-Length', 0))
            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                with open(output_path, 'wb') as f:
                    while True:
                        buffer = response.read(8192)
                        if not buffer:
                            break
                        f.write(buffer)
                        progress.update(len(buffer))
    return output_path

# Example usage
