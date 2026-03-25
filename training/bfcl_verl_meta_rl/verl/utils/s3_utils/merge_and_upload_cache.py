import argparse
import json
import os
import tempfile
import subprocess


def download_from_s3(s3_uri, local_path):
    result = subprocess.run(["aws", "s3", "cp", s3_uri, local_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"⚠️ Warning: Failed to download from S3: {s3_uri}. Proceeding with empty cache.")
        return None
    return local_path


def upload_to_s3(local_path, s3_uri):
    subprocess.check_call(["aws", "s3", "cp", local_path, s3_uri])
    print(f"✅ Uploaded merged cache to {s3_uri}")


def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_file", type=str, required=True, help="Path to local cache file (e.g., /workspace/cache/serper_search_cache.json)")
    parser.add_argument("--s3_uri", type=str, required=True, help="Target S3 URI (e.g., s3://shopqa-users/WebThinker/cache/serper_search_cache.json)")
    args = parser.parse_args()

    # Prepare local download path
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        s3_local_path = tmp.name

    s3_cache = {}
    if download_from_s3(args.s3_uri, s3_local_path):
        s3_cache = load_json(s3_local_path)

    local_cache = load_json(args.cache_file)

    local_cache.update(s3_cache)

    # Save merged to a new temp file
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as merged_tmp:
        save_json(local_cache, merged_tmp.name)
        upload_to_s3(merged_tmp.name, args.s3_uri)


if __name__ == "__main__":
    main()
