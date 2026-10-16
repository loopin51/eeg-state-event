#!/usr/bin/env bash
# F-V data download (D-111 decision 9). Not a computation: run detached on the server with
#   setsid nohup bash scripts/fv_download.sh results/fv_data/manifest.tsv /mnt/data/eeg_datasets results/fv_data 3 \
#     > results/fv_data/download.log 2>&1 < /dev/null &
# Downloaded files are untrusted data: this script only checks size, sha256 (and md5 where figshare gives one),
# never opens or executes them. Each verified file is made read-only. Test files (raw_eeg_test.npy) are written
# to the seal list the moment they verify. Rerunning skips files already in done.tsv and resumes .part files.
set -u
MAN=$1 ROOT=$2 OUT=$3 NP=${4:-3}
SEAL=results/things_eeg2/seal_test_sha256.tsv
FILES=results/things_eeg2/files_sha256.tsv
GAO=results/gao_imagery/files_sha256.tsv
mkdir -p "$OUT" results/things_eeg2 results/gao_imagery
[ -s "$SEAL" ] || printf 'path\tbytes\tsha256\n' > "$SEAL"
[ -s "$FILES" ] || printf 'path\tbytes\tsha256\tgroup\n' > "$FILES"
[ -s "$GAO" ] || printf 'path\tbytes\tsha256\tmd5\n' > "$GAO"
[ -s "$OUT/done.tsv" ] || printf 'path\tbytes\tsha256\talgo\texpected\tmatch\tgroup\tfinished\telapsed_s\n' > "$OUT/done.tsv"
export MAN ROOT OUT SEAL FILES GAO

fetch() {
  local rel url size algo exp group dest got rc sha md5 ok t0 a
  IFS=$'\t' read -r rel url size algo exp group < <(sed -n "${1}p" "$MAN")
  awk -F'\t' -v r="$rel" '$1==r{f=1} END{exit !f}' "$OUT/done.tsv" && return 0
  dest="$ROOT/$rel"
  mkdir -p "$(dirname "$dest")"
  t0=$(date +%s)
  echo "$(date '+%F %T') start $rel $size"
  got=0
  for a in 1 2 3 4 5 6 7 8; do
    curl -sS -L --fail --retry 5 --retry-delay 30 --retry-all-errors --connect-timeout 60 -C - -o "$dest.part" "$url"
    rc=$?
    got=$(stat -c %s "$dest.part" 2>/dev/null || echo 0)
    [ "$got" = "$size" ] && break
    echo "$(date '+%F %T') retry $a rc=$rc got=$got $rel"
    if [ "$got" -gt "$size" ] || [ "$rc" = 33 ] || [ "$rc" = 36 ]; then rm -f "$dest.part"; fi
    sleep 60
  done
  if [ "$got" != "$size" ]; then echo "$(date '+%F %T') FAIL size $rel got=$got want=$size"; return 1; fi
  sha=$(sha256sum "$dest.part" | cut -d' ' -f1)
  md5=-
  case $algo in
    sha256) [ "$sha" = "$exp" ] && ok=yes || ok=no ;;
    md5) md5=$(md5sum "$dest.part" | cut -d' ' -f1); [ "$md5" = "$exp" ] && ok=yes || ok=no ;;
    *) ok=size_only ;;
  esac
  if [ "$ok" = no ]; then mv "$dest.part" "$dest.bad"; echo "$(date '+%F %T') FAIL hash $rel"; return 1; fi
  mv "$dest.part" "$dest"
  chmod a-w "$dest"
  (
    flock 9
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$rel" "$size" "$sha" "$algo" "$exp" "$ok" "$group" "$(date '+%F %T')" "$(( $(date +%s) - t0 ))" >> "$OUT/done.tsv"
    case $group in
      test) printf '%s\t%s\t%s\n' "$rel" "$size" "$sha" >> "$SEAL" ;;
      gao) printf '%s\t%s\t%s\t%s\n' "$rel" "$size" "$sha" "$md5" >> "$GAO" ;;
      *) printf '%s\t%s\t%s\t%s\n' "$rel" "$size" "$sha" "$group" >> "$FILES" ;;
    esac
  ) 9>"$OUT/.lock"
  echo "$(date '+%F %T') done $rel $(( $(date +%s) - t0 )) s match=$ok"
}
export -f fetch

N=$(wc -l < "$MAN")
seq 1 "$N" | xargs -P "$NP" -I{} bash -c 'fetch "$@"' _ {}
echo "$(date '+%F %T') all workers finished"
