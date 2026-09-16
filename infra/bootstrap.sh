#!/usr/bin/env bash
# Make an ephemeral pod reusable: put every cache on the network volume, fix
# ownership, and check out the code. Idempotent -- safe to re-run on every pod.
#
# Without this, each pod start re-downloads gigabytes of Omniverse assets and
# recompiles shader caches on paid GPU time (README §8.3). Caches are relocated
# by symlinking their real paths at the volume, which works on providers that
# give you one volume mount and no control over individual bind mounts.
#
# Run this INSIDE the nvcr.io/nvidia/isaac-lab container, not on the bare pod --
# the caches it relocates belong to that image.
set -euo pipefail

# The image sets HOME=/root while the container runs as uid 1000, so every $HOME
# cache path below resolves into a directory this user cannot write. Measured on
# the pod: the relocation dies partway through with Permission denied, leaving
# some caches moved and some not -- the worst outcome, because the run looks
# half-done rather than failed. The password database is the truth about where
# this user's home is; the inherited environment is not.
#
# Only consulted when HOME is actually unusable. An inherited HOME that works is
# also the one Isaac will use at runtime, and relocating a different path than
# the one Kit reads would persist nothing while appearing to succeed.
if [[ ! -w "${HOME:-/nonexistent}" ]]; then
  passwd_home="$(getent passwd "$(id -un)" 2>/dev/null | cut -d: -f6 || true)"
  if [[ -n "$passwd_home" && "$passwd_home" != "${HOME:-}" ]]; then
    echo "==> HOME=${HOME:-<unset>} is not writable by $(id -un); using $passwd_home from the password database"
    export HOME="$passwd_home"
  fi
fi

# Not /workspace: the vendor image already owns that path (see the guard below).
VOL="${IDTB_VOL:-/idtb}"
REPO="${IDTB_REPO:-https://github.com/quastAI/lejepa_identifiability.git}"
REF="${IDTB_REF:-main}"
CHECKOUT="$VOL/repo"

# Isaac Lab v3.0.0-beta2 unpacks itself into /workspace/isaaclab (docker/.env.base,
# DOCKER_ISAACLAB_PATH). A network volume mounted at /workspace shadows it, and
# the symptom is a missing isaaclab.sh rather than anything mentioning mounts.
if [[ "$VOL" == "/workspace" || "$VOL" == /workspace/* ]]; then
  cat >&2 <<'MSG'
refusing to use /workspace as the network volume.

nvcr.io/nvidia/isaac-lab:3.0.0-beta2 unpacks Isaac Lab into /workspace/isaaclab.
A volume mounted over /workspace hides it and ./isaaclab.sh simply disappears.

The mount path is a pod setting, not a volume setting: recreate the pod with the
volume mounted at /idtb -- the volume's contents are untouched -- then re-run
this script. Nothing else needs to change.
MSG
  exit 1
fi

# The isaac-lab 3.0 image ends on `USER isaaclab` (uid/gid 1000) and ships no
# sudo, while a provider volume arrives root-owned. uid 1000 then cannot write it
# and Kit fails indirectly -- PermissionError on logs/, or omni.datastore lock
# errors under kit/cache -- which reads as an Isaac bug. Fail here, with the fix,
# rather than there.
[[ -d "$VOL" ]] || { echo "no volume at $VOL -- attach it first" >&2; exit 1; }
if [[ ! -O "$VOL" ]]; then
  echo "==> taking ownership of $VOL"
  if ! chown -R "$(id -u):$(id -g)" "$VOL" 2>/dev/null; then
    cat >&2 <<MSG

cannot take ownership of $VOL as $(id -un) (uid $(id -u)).

The container runs as a non-root user and has no sudo, so it cannot fix this
itself. From a root shell on the pod, once per volume:

    chown -R $(id -u):$(id -g) $VOL

then re-run this script.
MSG
    exit 1
  fi
fi

# real path -> directory on the volume
link_cache() {
  local real_path="$1" target="$VOL/cache/$2"
  # Paths under $ISAACSIM_ROOT_PATH only exist inside the Isaac image; skip them
  # cleanly on a bare pod rather than creating a dangling symlink in /.
  if [[ "$real_path" != "$HOME"/* && ! -d "$(dirname "$real_path")" ]]; then
    echo "    $real_path (skipped -- not in this image)"
    return
  fi
  mkdir -p "$target"
  if [[ -L "$real_path" ]]; then
    [[ "$(readlink -f "$real_path")" == "$(readlink -f "$target")" ]] \
      && { echo "    $real_path (already linked)"; return; }
    rm "$real_path"
  elif [[ -d "$real_path" ]]; then
    # preserve anything the image shipped or a previous run downloaded
    cp -an "$real_path/." "$target/" 2>/dev/null || true
    rm -rf "$real_path"
  fi
  mkdir -p "$(dirname "$real_path")"
  ln -s "$target" "$real_path"
  echo "    $real_path -> $target"
}

# This list is the one in Isaac Lab's own docker-compose.yaml
# (x-default-isaac-lab-volumes), which is what the vendor persists between runs.
# tests/test_infra_scripts.py pins it so the two cannot drift apart silently.
SIM_ROOT="${ISAACSIM_ROOT_PATH:-/isaac-sim}"

echo "==> relocating caches onto $VOL"
link_cache "$SIM_ROOT/kit/cache" kit            # Kit extension cache -- the big one
link_cache "$SIM_ROOT/kit/data" kit-data        # Kit runtime data -- new in Isaac Lab 3.0
link_cache "$HOME/.cache/ov" ov                 # Omniverse asset cache
link_cache "$HOME/.cache/nvidia" nvidia         # GL shader cache (GLCache)
link_cache "$HOME/.cache/pip" pip
link_cache "$HOME/.nv" nv                       # CUDA compute cache (ComputeCache)
link_cache "$HOME/.local/share/ov" ov-data      # Omniverse app data
link_cache "$HOME/.nvidia-omniverse" omniverse  # logs

echo "==> data directories"
mkdir -p "$VOL/data" "$VOL/logs"
printf '    %s\n' "$VOL/data" "$VOL/logs"

# Written to the volume so it survives the pod and is one `source` away on every
# start. OMNI_KIT_ALLOW_ROOT only when the uid really is root: the 2.3.2 image
# needs it, the 3.0 image (uid 1000) does not, and asserting it unconditionally
# would be cargo-culting a fact that flipped between image generations.
echo "==> environment at $VOL/env.sh"
{
  echo "# source this on every pod start"
  echo "export ISAACLAB_PATH=\"\${ISAACLAB_PATH:-/workspace/isaaclab}\""
  echo "export IDTB_VOL=\"$VOL\""
  # Isaac has to agree with us about HOME, or it reads caches at a path we never
  # relocated and re-downloads everything while the symlinks sit there unused.
  echo "export HOME=\"$HOME\""
  if [[ "$(id -u)" == "0" ]]; then
    echo "export OMNI_KIT_ALLOW_ROOT=1  # uid is root; Kit refuses to start without it"
  else
    echo "# OMNI_KIT_ALLOW_ROOT not set: running as uid $(id -u), not root"
  fi
} > "$VOL/env.sh"
sed 's/^/    /' "$VOL/env.sh"


# Docker's embedded DNS proxy (127.0.0.11) is sometimes not forwarding yet just
# after the container boots: hostname lookups fail while raw IP connectivity is
# fine, and it clears itself within seconds. Measured on the pod, and a known
# provider quirk rather than anything about this image. Under set -e a single
# hit kills the whole bootstrap, after the caches have been relocated -- so this
# is the one step that retries. It is the same failure preflight.sh reports as
# curl exit 6; if it is still failing after 5 attempts it is not a race.
git_retry() {
  local n=0
  until "$@"; do
    n=$((n + 1))
    if ((n >= 5)); then
      echo "git failed after 5 attempts -- check DNS: getent hosts github.com" >&2
      exit 1
    fi
    echo "    git failed, retrying in ${IDTB_GIT_RETRY_SLEEP:-5}s ($n/5)..."
    sleep "${IDTB_GIT_RETRY_SLEEP:-5}"
  done
}

echo "==> code at $CHECKOUT"
if [[ -d "$CHECKOUT/.git" ]]; then
  git_retry git -C "$CHECKOUT" fetch --depth 1 origin "$REF"
  git -C "$CHECKOUT" checkout -q FETCH_HEAD
else
  git_retry git clone --depth 1 --branch "$REF" "$REPO" "$CHECKOUT"
fi
echo "    $(git -C "$CHECKOUT" log -1 --oneline)"

cat <<EOF

Bootstrap complete.

  code      $CHECKOUT
  datasets  $VOL/data
  caches    $VOL/cache   (survives pod restarts -- verify by restarting once
                          and confirming Isaac does not re-download assets)
  env       source $VOL/env.sh

Next: source the env file, then run a shipped Isaac Lab tutorial and look at the
PNG before running any of our code, so "environment broken" and "our blind code
broken" stay separable.

  source $VOL/env.sh
  cd \${ISAACLAB_PATH:-/workspace/isaaclab}
  ./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py --headless
EOF
