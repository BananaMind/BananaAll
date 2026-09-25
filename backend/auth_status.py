import json
import sys
from huggingface_hub import HfApi, get_token

try:
    token = get_token()
    if not token:
        print(json.dumps({"loggedIn": False, "message": "No local Hugging Face login"}))
    else:
        user = HfApi().whoami(token=token)
        print(json.dumps({"loggedIn": True, "username": user.get("name") or user.get("fullname") or "Signed in"}))
except Exception as exc:
    print(json.dumps({"loggedIn": False, "message": str(exc).splitlines()[0][:160]}))
