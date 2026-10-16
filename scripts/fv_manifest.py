"""Build the F-V download manifest (D-111 decision 9) from the public HF tree API and the figshare API.

Reads only API metadata; never opens a downloaded file. Writes a TSV with one line per file:
dest_relpath, url, size, algo, expected_hash, group. Lines are ordered so that the sealed test files and the
sub-01 ses-01 training file come first (smoke preprocessing can start early), then images, then the rest.
Run on the server from the project folder: env/bin/python -I scripts/fv_manifest.py results/fv_data
"""
import json, os, sys, urllib.parse, urllib.request

HF_REPO = 'gasparyanartur/things-eeg2'
FIGSHARE_ID = 30227503


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def hf_rows():
    tree = get('https://huggingface.co/api/datasets/%s/tree/main?recursive=true' % HF_REPO)
    rows = []
    for x in tree:
        if x['type'] != 'file' or x['path'] == '.gitattributes':
            continue
        p = x['path']
        url = 'https://huggingface.co/datasets/%s/resolve/main/%s' % (HF_REPO, urllib.parse.quote(p))
        if 'lfs' in x:
            algo, h = 'sha256', x['lfs']['oid']
        else:
            algo, h = 'git-blob-sha1', x['oid']  # small non-LFS files: checked by size only
        if p.endswith('raw_eeg_test.npy'):
            g, key = 'test', (0, p)
        elif p == 'raw-eeg/sub-01/ses-01/raw_eeg_training.npy':
            g, key = 'train', (1, p)
        elif p.startswith('imgs/'):
            g, key = 'image', (2, p)
        elif p.startswith('raw-eeg/'):
            g, key = 'train', (4, p)
        else:
            g, key = 'meta', (3, p)
        rows.append((key, ('things_eeg2/' + p, url, x['size'], algo, h, g)))
    return [r for _, r in sorted(rows)]


def figshare_rows():
    files = get('https://api.figshare.com/v2/articles/%d/files?page_size=100' % FIGSHARE_ID)
    rows = []
    for x in files:
        assert x['supplied_md5'] == x['computed_md5'], x['name']
        assert '/' not in x['name'] and not x['name'].startswith('.'), x['name']
        rows.append(('gao_imagery/' + x['name'], x['download_url'], x['size'], 'md5', x['computed_md5'], 'gao'))
    return sorted(rows, key=lambda r: (not r[0].endswith('.zip'), r[0]))


out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rows = hf_rows() + figshare_rows()
assert len({r[0] for r in rows}) == len(rows)
with open(os.path.join(out, 'manifest.tsv'), 'w') as f:
    for r in rows:
        f.write('\t'.join(map(str, r)) + '\n')
for g in ('test', 'train', 'image', 'meta', 'gao'):
    s = [r for r in rows if r[5] == g]
    print(g, len(s), sum(r[2] for r in s))
print('total', len(rows), sum(r[2] for r in rows))
