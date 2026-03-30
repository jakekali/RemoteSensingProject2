# local_model_testing/download_from_drive.py
# Download GEE-exported CSV files from Google Drive into data/
# Uses the existing EE OAuth credentials (which include Drive scope).
#
# Run:
#   cd D:\remote_project_2
#   venv\Scripts\python local_model_testing/download_from_drive.py

import sys, os, io, json

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# ── Load EE OAuth credentials (includes Drive scope) ─────────────────────────
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

CRED_PATH = os.path.expanduser('~/.config/earthengine/credentials')
with open(CRED_PATH) as f:
    cred_data = json.load(f)

creds = Credentials(
    token=None,
    refresh_token=cred_data['refresh_token'],
    token_uri='https://oauth2.googleapis.com/token',
    client_id=cred_data['client_id'],
    client_secret=cred_data['client_secret'],
    scopes=cred_data['scopes'],
)
creds.refresh(Request())
print("[drive] OAuth credentials loaded & refreshed")

service = build('drive', 'v3', credentials=creds)

# ── Find GEE_exports folder ───────────────────────────────────────────────────
GEE_FOLDER_ID = '1joJKki09Fo4P7MKMLkzx8YgC9EfRz2Ct'

# ── Files to download ─────────────────────────────────────────────────────────
# Add any filename prefixes you want here (partial match, case-insensitive)
TARGET_PREFIXES = [
    'boost_alfalfa_oats_2022',
    'boost_alfalfa_oats_2023',
    'test_2024_mclean',
    'test_2024_renville',
    'training_2022_v4_7class',
    'training_full_v4_7class',
]


def list_folder(folder_id):
    """Return list of {id, name, size} dicts in a Drive folder."""
    results = []
    page_token = None
    while True:
        resp = service.files().list(
            q=f"'{folder_id}' in parents and trashed=false",
            fields="nextPageToken, files(id, name, size, mimeType)",
            pageToken=page_token,
            pageSize=100,
        ).execute()
        results.extend(resp.get('files', []))
        page_token = resp.get('nextPageToken')
        if not page_token:
            break
    return results


def download_file(file_id, file_name, dest_dir):
    dest_path = os.path.join(dest_dir, file_name)
    if os.path.exists(dest_path):
        size_mb = os.path.getsize(dest_path) / 1e6
        print(f"  [SKIP] {file_name}  (already exists, {size_mb:.1f} MB)")
        return dest_path

    print(f"  [DL]   {file_name} ...", end='', flush=True)
    request = service.files().get_media(fileId=file_id)
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    buf.seek(0)
    with open(dest_path, 'wb') as f:
        f.write(buf.read())
    size_mb = os.path.getsize(dest_path) / 1e6
    print(f" done ({size_mb:.1f} MB)")
    return dest_path


# ── Main ─────────────────────────────────────────────────────────────────────
print(f"\nListing files in GEE_exports folder ({GEE_FOLDER_ID})...")
files = list_folder(GEE_FOLDER_ID)
print(f"  Found {len(files)} file(s):")
for f in files:
    sz = f.get('size', '?')
    sz_str = f"{int(sz)/1e6:.1f} MB" if sz != '?' else '?'
    print(f"    {f['name']}  ({sz_str})")

print(f"\nDownloading target files to {DATA_DIR}/ ...")
downloaded = []
for f in files:
    name_lower = f['name'].lower()
    if any(p.lower() in name_lower for p in TARGET_PREFIXES):
        dest = download_file(f['id'], f['name'], DATA_DIR)
        downloaded.append(dest)

if not downloaded:
    print("\n  No matching files found in Drive folder.")
    print("  Make sure GEE export tasks have completed first.")
    print("  Target prefixes:", TARGET_PREFIXES)
else:
    print(f"\nDone. {len(downloaded)} file(s) saved to {DATA_DIR}/")
    for d in downloaded:
        print(f"  {os.path.basename(d)}")
