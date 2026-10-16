"""D-110: download the rest of UGR-MINDVOICE (OSF 6sh5d, public, no account) to the server HDD with OSF sha256 checks.

Run only after D-110 is committed, detached:
  setsid nohup env/bin/python -I scripts/download_ugr_mindvoice.py > /mnt/data/eeg_datasets/ugr_mindvoice/fetch_rest.log 2>&1 < /dev/null &
Walks the OSF v2 file tree, writes the full manifest (path, size, md5, sha256, download link) to <root>/manifest_osf.json and
fetches what is missing: EDF files to <root>/eeg/<sub>/<ses>/eeg/, Materials media (wav, jpg, png) to <root>/meta/files/Materials/.
Files already present (meta, session audio) are only checked against the OSF sha256. A partial file is resumed with curl -C -;
a file whose sha256 does not match is deleted and fetched again (3 tries). The EDF files are never opened here (D-110 seal).
Ends with the line ALLDONE when every wanted file matches. Re-running resumes.
"""
import os, sys, json, time, hashlib, subprocess, urllib.request
ROOT = '/mnt/data/eeg_datasets/ugr_mindvoice'
API = 'https://api.osf.io/v2/nodes/6sh5d/files/osfstorage/'


def get(url):
    for i in range(5):
        try: return json.load(urllib.request.urlopen(url, timeout=120))
        except Exception as e: print('retry', url, e, flush=True); time.sleep(10 * (i + 1))
    raise RuntimeError(url)


def walk(url, out):
    while url:
        d = get(url)
        for x in d['data']:
            a = x['attributes']
            if a['kind'] == 'folder': walk(x['relationships']['files']['links']['related']['href'], out)
            else: out.append(dict(path=a['materialized_path'], size=a['size'], md5=a['extra']['hashes']['md5'],
                                  sha256=a['extra']['hashes']['sha256'], download=x['links']['download']))
        url = d['links'].get('next')
    return out


def local(p):
    if p.startswith('/Data/') and p.endswith('.edf'): return os.path.join(ROOT, 'eeg', p[len('/Data/'):])
    if p.endswith('/audio/anonymized.wav'): return os.path.join(ROOT, 'audio', p[len('/Data/'):])
    return os.path.join(ROOT, 'meta/files', p.lstrip('/'))


def digest(f, algo):
    h = hashlib.new(algo)
    with open(f, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 22), b''): h.update(b)
    return h.hexdigest()


def verify(out, f):
    """size and the OSF sha256 (md5 if OSF gives no sha256); a file with neither hash fails."""
    if not (os.path.exists(out) and os.path.getsize(out) == f['size']): return False
    if f['sha256']: return digest(out, 'sha256') == f['sha256']
    if f['md5']: return digest(out, 'md5') == f['md5']
    return False


def main():
    files = walk(API, []); json.dump(files, open(os.path.join(ROOT, 'manifest_osf.json'), 'w'), indent=1)
    print('files', len(files), 'GB %.2f' % (sum(f['size'] for f in files) / 1e9), flush=True)
    want = [f for f in files if f['path'].endswith('.edf') or (f['path'].startswith('/Materials/') and f['path'].lower().endswith(('.wav', '.jpg', '.png')))]
    have = [f for f in files if f not in want and os.path.exists(local(f['path']))]
    bad = 0
    for f in have:
        ok = verify(local(f['path']), f)
        print('check', 'OK' if ok else 'MISMATCH', f['path'], flush=True); bad += not ok
    for f in want:
        out = local(f['path']); os.makedirs(os.path.dirname(out), exist_ok=True)
        for k in range(3):
            if not (os.path.exists(out) and os.path.getsize(out) == f['size']):
                subprocess.run(['curl', '-sSL', '--retry', '5', '-C', '-', '-o', out, f['download']], check=False)
            if verify(out, f):
                print('got', f['path'], f['size'], 'hash OK', flush=True); break
            print('sha256/size mismatch, refetch', f['path'], flush=True)
            if os.path.exists(out): os.remove(out)
        else:
            print('FAILED', f['path'], flush=True); bad += 1
    print('ALLDONE' if bad == 0 else f'INCOMPLETE {bad}', flush=True)


if __name__ == '__main__':
    main()
