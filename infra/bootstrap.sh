#!/usr/bin/env bash
# Make an ephemeral pod reusable: put every cache on the network volume, fix
# ownership, and check out the code. Idempotent -- safe to re-run on every pod.
#
# Without this, each pod start re-downloads gigabytes of Omniverse assets and
# recompiles shader caches on paid GPU time (README §8.3). Caches are relocated
# by symlinking the home paths at the volume, which works on providers that give
# you one volume mount and no control over bind mounts.
set -euo pipefail

VOL="${IDTB_VOL:-/workspace}"
REPO="${IDTB_REPO:-https://github.com/quastAI/lejepa_identifiability.git}"
REF="${IDTB_REF:-main}"
CHECKOUT="$VOL/repo"

# Recent Isaac Sim containers run as a non-root user; a volume owned by root
# fails later with permission errors that look like Isaac bugs.
[[ -d "$VOL" ]] || { echo "no volume at $VOL -- attach it first" >&2; exit 1; }
if [[ ! -O "$VOL" ]]; then
  echo "==> taking ownership of $VOL"
  chown -R "$(id -u):$(id -g)" "$VOL" 2>/dev/null \
    || sudo chown -R "$(id -u):$(id -g)" "$VOL"
fi

# home path -> directory on the volume
link_cache() {
  local home_path="$1" target="$VOL/cache/$2"
  mkdir -p "$target"
  if [[ -L "$home_path" ]]; then
    [[ "$(readlink -f "$home_path")" == "$(readlink -f "$target")" ]] \
      && { echo "    $home_path (already linked)"; return; }
    rm "$home_path"
  elif [[ -d "$home_path" ]]; then
    # preserve anything the image shipped or a previous run downloaded
    cp -an "$home_path/." "$target/" 2>/dev/null || true
    rm -rf "$home_path"
  fi
  mkdir -p "$(dirname "$home_path")"
  ln -s "$target" "$home_path"
  echo "    $home_path -> $target"
}

echo "==> relocating caches onto $VOL"
link_cache "$HOME/.cache/ov" ov              # Kit extension + asset cache
link_cache "$HOME/.cache/nvidia" nvidia      # GL shader cache
link_cache "$HOME/.cache/pip" pip
link_cache "$HOME/.nv" nv                    # CUDA compute cache
link_cache "$HOME/.local/share/ov" ov-data   # Omniverse app data
link_cache "$HOME/.nvidia-omniverse" omniverse

echo "==> data directories"
mkdir -p "$VOL/data" "$VOL/logs"
printf '    %s\n' "$VOL/data" "$VOL/logs"

echo "==> code at $CHECKOUT"
if [[ -d "$CHECKOUT/.git" ]]; then
  git -C "$CHECKOUT" fetch --depth 1 origin "$REF"
  git -C "$CHECKOUT" checkout -q FETCH_HEAD
else
  git clone --depth 1 --branch "$REF" "$REPO" "$CHECKOUT"
fi
echo "    $(git -C "$CHECKOUT" log -1 --oneline)"

cat <<EOF

Bootstrap complete.

  code      $CHECKOUT
  datasets  $VOL/data
  caches    $VOL/cache   (survives pod restarts -- verify by restarting once
                          and confirming Isaac does not re-download assets)

Next: run a shipped Isaac Lab tutorial and look at the PNG before running any
of our code, so "environment broken" and "our blind code broken" stay separable.
EOF
