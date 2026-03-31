import os
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

BUCKET_URL = "https://s3.amazonaws.com/openneuro.org/"
PREFIX = "ds004504/"
OUTPUT_DIR = r"d:\al project\ADFTD"

# Disable warnings for verify=False
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import time

def download_file(key, s3_size, retries=3):
    url = f"{BUCKET_URL}{key}"
    dest = os.path.join(OUTPUT_DIR, key[len(PREFIX):])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    
    # Size check
    if os.path.exists(dest):
        local_size = os.path.getsize(dest)
        if local_size == s3_size:
            # print(f"Skipping {key}, already synced.")
            return True
        else:
            print(f"Size mismatch for {key}: Local {local_size} vs S3 {s3_size}. Re-downloading...")

    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60, verify=False) as r:
                r.raise_for_status()
                with open(dest, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        f.write(chunk)
            # Final check
            if os.path.getsize(dest) == s3_size:
                print(f"SUCCESS: {key} downloaded.")
                return True
        except Exception as e:
            print(f"FAILED {key} on attempt {attempt+1}: {e}")
            if attempt < retries - 1:
                time.sleep(2**attempt)
    return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print(f"Fetching file list for {PREFIX}...")
    params = {'prefix': PREFIX}
    response = requests.get(BUCKET_URL, params=params, timeout=30, verify=False)
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    ns = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
    
    tasks = []
    for contents in root.findall('s3:Contents', ns):
        key = contents.find('s3:Key', ns).text
        size = int(contents.find('s3:Size', ns).text)
        if any(key.endswith(ext) for ext in ['.set', '.fdt', '.json', '.tsv']):
            tasks.append((key, size))
            
    print(f"Found {len(tasks)} files to sync.")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda x: download_file(*x), tasks))
        
    success_count = sum(1 for r in results if r)
    print(f"\nSync complete: {success_count}/{len(tasks)} files synced.")

if __name__ == "__main__":
    main()
