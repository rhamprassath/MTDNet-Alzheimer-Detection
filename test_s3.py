import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

url = "https://s3.amazonaws.com/openneuro.org/ds004504/derivatives/sub-001/eeg/sub-001_task-eyesclosed_eeg.set"
dest = r"d:\al project\test_download.set"

print(f"Testing download from {url}...")
try:
    with requests.get(url, stream=True, timeout=30, verify=False) as r:
        print(f"Status Code: {r.status_code}")
        print(f"Headers: {r.headers}")
        r.raise_for_status()
        with open(dest, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print("Download successful!")
except Exception as e:
    print(f"Download failed: {e}")
