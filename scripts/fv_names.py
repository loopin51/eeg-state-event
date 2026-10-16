"""Build the F-V concept name table configs/fv_names.tsv (D-112 section 2).

Inputs (paths only, nothing is executed from them):
  --meta     THINGS concepts-metadata_things.tsv (1,854 rows; official source OSF jum2f, a copy may be used
             while OSF is unreachable; its sha256 is written into the output header comment of the report json)
  --train    text file, one THINGS-EEG2 training image folder name per line (e.g. 00001_aardvark)
  --test     text file, one test image folder name per line (e.g. 00010_baton4)
  --out      configs/fv_names.tsv
  --report   results/fv/names_report.json

Rule (registered before any embedding or EEG value is computed):
  concept      = folder name without the 5-digit prefix (THINGS uniqueID with spaces as '_')
  display_name = THINGS 'Word' of that uniqueID
  prompt_name  = display_name, except when the same Word occurs more than once among the 1,854 concepts;
                 then '{Word}, {gloss}' where gloss is the THINGS definition with parenthesized parts removed, cut at the first ';',
                 and shortened to at most 60 characters at a word boundary.
  category     = 'Top-down Category (manual selection)' (empty if none)
  things_id    = THINGS 1,854 number (rank of the '_'-joined uniqueID in sorted order, 5 digits) + '_' + concept; rows are written in this order.
                 Checked against THINGS-EEG2 image_metadata *_img_concepts_THINGS when available (--image-meta).
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter

DEF = 'Definition (from WordNet, Google, or Wikipedia)'
CAT = 'Top-down Category (manual selection)'


def gloss(text, limit=60):
    g = re.sub(r'\s*\([^)]*\)', '', text)
    g = re.sub(r'\s+', ' ', g.split(';', 1)[0]).strip().rstrip('.,: ')
    if len(g) <= limit:
        return g
    cut = g[:limit + 1].rsplit(' ', 1)[0]
    return cut.rstrip('.,: ')


def norm(s):
    return s.strip().replace(' ', '_')


def main():
    ap = argparse.ArgumentParser()
    for k in ('meta', 'train', 'test', 'out', 'report'):
        ap.add_argument('--' + k, required=True)
    ap.add_argument('--image-meta', help='image_meta.json written by scripts/fv_meta.py (optional check)')
    a = ap.parse_args()

    raw = open(a.meta, 'rb').read()
    rows = list(csv.DictReader(raw.decode('utf-8', errors='strict').splitlines(), delimiter='\t'))
    assert len(rows) == 1854, len(rows)
    by_id = {norm(r['uniqueID']): r for r in rows}
    order = {c: i + 1 for i, c in enumerate(sorted(by_id))}  # THINGS-EEG2 1,854 numbering: sorted uniqueID with '_' for ' '
    assert len(by_id) == 1854
    word_n = Counter(r['Word'].strip() for r in rows)

    out, seen = [], set()
    for split, path in (('train', a.train), ('test', a.test)):
        folders = [l.strip() for l in open(path) if l.strip()]
        for f in sorted(folders):
            m = re.fullmatch(r'(\d{5})_(.+)', f)
            assert m, f
            cid = m.group(2)
            assert cid in by_id, f'no THINGS uniqueID for folder {f}'
            assert cid not in seen, f'concept {cid} appears twice'
            seen.add(cid)
            r = by_id[cid]
            w = r['Word'].strip()
            p = w if word_n[w] == 1 else f'{w}, {gloss(r[DEF])}'
            out.append((split, f, cid, w, p, r[CAT].strip(), f'{order[cid]:05d}_{cid}'))
    n_tr = sum(o[0] == 'train' for o in out)
    n_te = sum(o[0] == 'test' for o in out)
    assert (n_tr, n_te) == (1654, 200), (n_tr, n_te)
    assert len(seen) == 1854
    out.sort(key=lambda o: order[o[2]])

    with open(a.out, 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh, delimiter='\t', lineterminator='\n')
        w.writerow(['split', 'folder', 'concept', 'display_name', 'prompt_name', 'category', 'things_id'])
        w.writerows(out)
    chk = None
    if a.image_meta:
        im = json.load(open(a.image_meta))
        want = {}
        for sp in ('train', 'test'):
            for loc, th in zip(im[f'{sp}_img_concepts'], im[f'{sp}_img_concepts_THINGS']):
                want[(sp, loc)] = th
        got = {(o[0], o[1]): o[6] for o in out}
        bad = [k for k in got if want.get(k) != got[k]]
        assert not bad, bad[:5]
        chk = len(got)
    rep = {
        'image_meta_checked': chk,
        'meta_sha256': hashlib.sha256(raw).hexdigest(),
        'names_sha256': hashlib.sha256(open(a.out, 'rb').read()).hexdigest(),
        'n_train': n_tr, 'n_test': n_te,
        'homonym_words': sorted(x for x, n in word_n.items() if n > 1),
        'homonym_concepts': [(o[0], o[2], o[4]) for o in out if o[3] != o[4]],
        'n_no_category': sum(o[5] == '' for o in out),
    }
    json.dump(rep, open(a.report, 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: rep[k] for k in ('names_sha256', 'n_train', 'n_test', 'n_no_category')}))
    print('homonym concepts:', len(rep['homonym_concepts']))


if __name__ == '__main__':
    sys.exit(main())
