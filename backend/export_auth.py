"""Include a saved HF token only when a selected repository requires one."""
import json
import sys

from huggingface_hub import HfApi, get_token


def access_error(error):
    response = getattr(error, "response", None)
    return getattr(response, "status_code", None) in (401, 403, 404)


def inspect_repositories(repositories):
    api = HfApi()
    token = None
    protected = []
    seen = set()
    for repository in repositories:
        repo_id = repository["id"]
        repo_type = repository["type"]
        key = (repo_type, repo_id)
        if key in seen:
            continue
        seen.add(key)
        label = f"{repo_type} {repo_id}"
        try:
            info = api.repo_info(repo_id=repo_id, repo_type=repo_type, token=False, timeout=10)
        except Exception as error:
            if not access_error(error):
                raise RuntimeError(f"Could not check Hugging Face access for {label}. Check your connection and try again.") from None
            if token is None:
                token = get_token()
            if not token:
                raise RuntimeError(f"Cannot access Hugging Face {label}. Check the ID or sign in with hf auth login before exporting.") from None
            try:
                api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token, timeout=10)
            except Exception as auth_error:
                if access_error(auth_error):
                    raise RuntimeError(f"Cannot access Hugging Face {label} with your saved token. Check the ID and repository access.") from None
                raise RuntimeError(f"Could not check Hugging Face access for {label}. Check your connection and try again.") from None
            protected.append(label)
        else:
            if getattr(info, "private", False) or getattr(info, "gated", False):
                protected.append(label)
    if protected and token is None:
        token = get_token()
    return {"token": token if protected else None, "tokenNeeded": bool(protected)}


if __name__ == "__main__":
    try:
        print(json.dumps(inspect_repositories(json.loads(sys.argv[1]))))
    except Exception as error:
        raise SystemExit(str(error)) from None
