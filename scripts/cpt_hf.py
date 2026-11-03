"""D-128 VERA-CPT transfer through private Hugging Face repositories (docs/08 §14; user choice 2026-11-03).

  data repo  (dataset, private)  corpus/<name>/{x.f32, recordings.parquet, header.json}      server -> Colab
                                 evalcache/<name>/{x.f32, ref.f32, meta.parquet, header.json}  server -> Colab
  model repo (model, private)    cpt/<run>/{vera_cpt.pt, metrics.json, history.jsonl, config.yaml}   Colab -> server
                                 runs/<run folder>/...                                              Colab -> server

Every push writes <prefix>/MANIFEST.json (relative path -> sha256, bytes); every pull checks every file against it.

usage: python scripts/cpt_hf.py push-data  --repo loopin51/vera-cpt-data  --what corpus|evalcache [--names a,b]
       python scripts/cpt_hf.py pull-data  --repo loopin51/vera-cpt-data  --what corpus|evalcache --dest DIR [--names a,b]
       python scripts/cpt_hf.py push-dir   --repo loopin51/vera-cpt --src DIR --prefix cpt/run1|runs
       python scripts/cpt_hf.py pull-dir   --repo loopin51/vera-cpt --prefix cpt/run1|runs --dest DIR
"""
import os, sys, json, hashlib, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHAT = {'corpus': ('data/cache/cpt', ['sparrkulee', 'alice', 'ds004408', 'things_eeg1', 'things_eeg2'],
                   ['x.f32', 'recordings.parquet', 'header.json']),
        'evalcache': ('data/cache/nc', ['physionet_mi', 'faced'], ['x.f32', 'ref.f32', 'meta.parquet', 'header.json'])}


def sha256(path, buf=1 << 24):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(buf), b''): h.update(b)
    return h.hexdigest()


def manifest(base, files):
    return {os.path.relpath(f, base): dict(sha256=sha256(f), bytes=os.path.getsize(f)) for f in files}


def api():
    from huggingface_hub import HfApi
    return HfApi()


def push(repo, repo_type, base, files, prefix, message, one_commit=False):
    """upload files (paths under base) to <prefix>/<relative path>, then <prefix>/MANIFEST.json. Large data files go one
    per commit (resumable per file); run folders (many small files) in one commit."""
    a = api(); a.create_repo(repo, repo_type=repo_type, private=True, exist_ok=True)
    assert a.repo_info(repo, repo_type=repo_type).private, f'{repo} is not private'
    man = manifest(base, files); mp = os.path.join(base, '.MANIFEST.json'); json.dump(man, open(mp, 'w'), indent=1)
    if one_commit:
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(path_in_repo=f'{prefix}/{rel}', path_or_fileobj=os.path.join(base, rel)) for rel in man]
        ops.append(CommitOperationAdd(path_in_repo=f'{prefix}/MANIFEST.json', path_or_fileobj=mp))
        a.create_commit(repo, operations=ops, repo_type=repo_type, commit_message=message); os.remove(mp); return man
    for rel in man:
        print(f'upload {rel} {man[rel]["bytes"] / 2 ** 30:.2f} GB', flush=True)
        a.upload_file(path_or_fileobj=os.path.join(base, rel), path_in_repo=f'{prefix}/{rel}', repo_id=repo, repo_type=repo_type,
                      commit_message=f'{message}: {rel}')
    a.upload_file(path_or_fileobj=mp, path_in_repo=f'{prefix}/MANIFEST.json', repo_id=repo, repo_type=repo_type, commit_message=f'{message}: manifest')
    os.remove(mp); return man


def pull(repo, repo_type, prefix, dest, only=None):
    """download <prefix>/** into dest and verify every file against <prefix>/MANIFEST.json."""
    from huggingface_hub import hf_hub_download, snapshot_download
    mp = hf_hub_download(repo, f'{prefix}/MANIFEST.json', repo_type=repo_type, local_dir=dest)
    man = json.load(open(mp)); rels = [r for r in man if only is None or r.split('/')[0] in only]
    snapshot_download(repo, repo_type=repo_type, local_dir=dest, allow_patterns=[f'{prefix}/{r}' for r in rels], max_workers=8)
    bad = [r for r in rels if sha256(os.path.join(dest, prefix, r)) != man[r]['sha256']]
    assert not bad, f'sha256 mismatch: {bad}'
    print(json.dumps(dict(prefix=prefix, files=len(rels), gb=round(sum(man[r]['bytes'] for r in rels) / 2 ** 30, 2), sha256='all match')), flush=True)
    return os.path.join(dest, prefix)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['push-data', 'pull-data', 'push-dir', 'pull-dir'])
    ap.add_argument('--repo', required=True); ap.add_argument('--what', choices=list(WHAT)); ap.add_argument('--names')
    ap.add_argument('--dest'); ap.add_argument('--src'); ap.add_argument('--prefix'); a = ap.parse_args()
    if a.cmd in ('push-data', 'pull-data'):
        sub, default, keep = WHAT[a.what]; names = a.names.split(',') if a.names else default
        if a.cmd == 'push-data':
            base = os.path.join(ROOT, sub); files = [os.path.join(base, n, f) for n in names for f in keep if os.path.exists(os.path.join(base, n, f))]
            for n in names: assert os.path.exists(os.path.join(base, n, 'header.json')), f'missing {n}'
            push(a.repo, 'dataset', base, files, a.what, f'D-128 VERA-CPT {a.what}')
        else:
            pull(a.repo, 'dataset', a.what, a.dest, only=set(names))
    elif a.cmd == 'push-dir':
        files = [os.path.join(dp, f) for dp, _, fs in os.walk(a.src) for f in fs if not f.endswith(('.tmp', 'state.pt'))]
        push(a.repo, 'model', a.src, files, a.prefix, f'D-128 VERA-CPT {a.prefix}', one_commit=True)
    else:
        pull(a.repo, 'model', a.prefix, a.dest)


if __name__ == '__main__':
    main()
