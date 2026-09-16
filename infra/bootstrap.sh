#!/usr/bin/env bash
# Make an ephemeral pod reusable: put every cache on the network volume and fix
# ownership. Idempotent, and needs no network -- safe to re-run on every pod.
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

# This script lives in the repo, so the repo is already here by the time it runs.
# It used to clone one as its last step, which was circular -- and it cloned to a
# second location, so the checkout you edited and the checkout Isaac ran were
# different directories. It reports what is here and never fetches.
CHECKOUT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

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

# Parent directories we were not allowed to modify, newline-separated. Collected
# rather than fatal: aborting mid-list leaves some caches relocated and some not,
# which looks half-done rather than failed -- the same trap as the HOME problem
# above. A string, not an array, because the dev machine under test runs bash 3.2
# where an empty array expansion is itself an error under set -u.
blocked=""

# real path -> directory on the volume
link_cache() {
  local real_path="$1" target="$VOL/cache/$2"
  local parent="${real_path%/*}"
  # Paths under $ISAACSIM_ROOT_PATH only exist inside the Isaac image; skip them
  # cleanly on a bare pod rather than creating a dangling symlink in /.
  if [[ "$real_path" != "$HOME"/* && ! -d "$parent" ]]; then
    echo "    $real_path (skipped -- not in this image)"
    return
  fi
  if [[ -L "$real_path" && "$(readlink -f "$real_path")" == "$(readlink -f "$target")" ]]; then
    echo "    $real_path (already linked)"
    return
  fi
  # Replacing a path with a symlink is a write to its *parent*, so the image
  # leaving /isaac-sim/kit root-owned blocks this however readable the cache
  # itself is -- confirmed on the pod: /isaac-sim/kit/cache is ubuntu:ubuntu and
  # writable, /isaac-sim/kit is not. Root is not reachable to fix it either: `su`
  # and `sudo` both fail from inside this container (no sudo binary at all), so
  # this is the accepted state of the image, not a transient error to retry.
  # Some parents do not exist yet and get created, so the permission that
  # matters is on the nearest ancestor that does exist.
  local probe="$parent"
  while [[ ! -e "$probe" ]]; do
    probe="${probe%/*}"
    [[ -n "$probe" ]] || probe="/"
  done
  if [[ ! -w "$probe" ]]; then
    blocked="${blocked}${probe}"$'\n'
    echo "    $real_path (BLOCKED -- $probe is not writable by $(id -un))"
    return
  fi
  mkdir -p "$target"
  if [[ -L "$real_path" ]]; then
    rm "$real_path"
  elif [[ -d "$real_path" ]]; then
    # preserve anything the image shipped or a previous run downloaded
    cp -an "$real_path/." "$target/" 2>/dev/null || true
    rm -rf "$real_path"
  fi
  mkdir -p "$parent"
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


echo "==> code at $CHECKOUT"
if [[ -d "$CHECKOUT/.git" ]]; then
  echo "    $(git -C "$CHECKOUT" log -1 --oneline 2>/dev/null || echo '<git cannot read it>')"
else
  echo "    not a git checkout -- 'git pull' will not update it"
fi
# A checkout outside the volume lives in the container's own filesystem and is
# gone on the next pod start, along with anything edited on the pod.
if [[ "$CHECKOUT" != "$VOL" && "$CHECKOUT" != "$VOL"/* ]]; then
  echo "    WARNING: not on $VOL -- this checkout does not survive a pod restart"
fi

cat <<EOF

Bootstrap done.

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

# Last and loud, but not fatal: there is nothing to fix here and re-run for.
# Root is unreachable from inside the container, and the cache directories
# themselves are already correctly owned (measured: /isaac-sim/kit/cache is
# ubuntu:ubuntu) -- Isaac writes into them at runtime exactly as the vendor
# intended. Only the parent blocks *relocating* them, so what's actually lost is
# persistence: those writes live in the container's own ephemeral layer instead
# of on $VOL, and do not survive a pod restart or recreate. Surfaced every run
# so a slow cold start later has an answer instead of looking like a mystery.
if [[ -n "$blocked" ]]; then
  blocked_dirs=$(printf '%s' "$blocked" | sort -u)
  cat <<MSG

NOTE: $(printf '%s\n' "$blocked_dirs" | wc -l | tr -d ' ') path(s) are owned by root inside this image and cannot be relocated
onto $VOL from here (no root shell reachable in this container). Their caches
will not survive a pod restart or recreate -- accepted, see README §8.3:

$(printf '%s\n' "$blocked_dirs" | sed 's|^|    |')
MSG
fi
