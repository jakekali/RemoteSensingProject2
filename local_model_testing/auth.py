# local_model_testing/auth.py
# GEE authentication for the local model testing workspace.
#
# Supports two auth methods (priority order):
#   1. Service account JSON key  — set GOOGLE_APPLICATION_CREDENTIALS env var
#                                   OR drop file at SERVICE_ACCOUNT_KEY_PATH below
#   2. Fallback to saved OAuth token (~/.config/earthengine/credentials)
#
# Usage:
#   from auth import init_ee, EE_PROJECT
#   init_ee()   # call once at top of any script

import os
import ee

# ── Config ────────────────────────────────────────────────────────────────────
EE_PROJECT = 'theta-grid-99720'

# Path to service account JSON key file.
# Override with env var:  set GOOGLE_APPLICATION_CREDENTIALS=C:/path/to/key.json
SERVICE_ACCOUNT_KEY_PATH = os.path.join(
    os.path.dirname(__file__), 'auth_keys', 'theta-grid-99720-ea12c2bea3c1.json'
)

# ── Init ──────────────────────────────────────────────────────────────────────
def init_ee(project: str = None, verbose: bool = True, force_oauth: bool = False) -> None:
    """
    Initialize GEE with service account credentials if available,
    otherwise fall back to the saved OAuth token.
    Set force_oauth=True or pass '--oauth' in CLI args to skip service account.
    Pass '--project PROJECT_ID' in CLI args to override the project.
    """
    import sys
    
    # 1. Determine Project (CLI arg > Function arg > Default)
    cli_project = None
    if '--project' in sys.argv:
        idx = sys.argv.index('--project')
        if idx + 1 < len(sys.argv):
            cli_project = sys.argv[idx + 1]
    
    actual_project = cli_project or project or EE_PROJECT

    # 2. Determine Auth Method
    cli_oauth = '--oauth' in sys.argv
    force = force_oauth or cli_oauth

    # Check env var first, then hardcoded path
    key_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', SERVICE_ACCOUNT_KEY_PATH)

    if os.path.exists(key_path) and not force:
        credentials = ee.ServiceAccountCredentials(
            email=_get_service_account_email(key_path),
            key_file=key_path,
        )
        ee.Initialize(credentials=credentials, project=actual_project)
        if verbose:
            print(f"[auth] GEE init via service account: {key_path}")
    else:
        # OAuth fallback — uses ~/.config/earthengine/credentials
        ee.Initialize(project=actual_project)
        if verbose:
            print(f"[auth] GEE init via User OAuth (fallback or forced)")

    if verbose:
        print(f"[auth] Project: {actual_project}")


def _get_service_account_email(key_path: str) -> str:
    """Extract the client_email field from a service account JSON key file."""
    import json
    with open(key_path) as f:
        data = json.load(f)
    return data['client_email']


# ── Setup helper (run once to test) ──────────────────────────────────────────
def check_auth():
    """Quick sanity check — prints EE project and a simple API call."""
    init_ee()
    result = ee.Number(42).add(1).getInfo()
    print(f"[auth] Sanity check: 42 + 1 = {result}  (expected 43)")
    print("[auth] All good!")


if __name__ == '__main__':
    # Drop your service account JSON at gee_creds.json (next to this file),
    # then run:  ..\venv\Scripts\python auth.py
    check_auth()
