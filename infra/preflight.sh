#!/usr/bin/env bash
# Read-only pod survey. Run this first on a fresh pod and paste the whole output.
#
# The driver reading decides the Isaac Sim release, not the other way round
# (README §8.2). Nothing here assumes or installs a version.
#
# Every check runs even if an earlier one fails -- one boot, one full report.
# Exit status is 1 if a hard rule is violated (see VERDICT), 0 otherwise.
set -uo pipefail

# Not /workspace: the vendor isaac-lab image unpacks Isaac Lab into
# /workspace/isaaclab, so a volume mounted there shadows it (README §8.3).
VOL="${IDTB_VOL:-/idtb}"
problems=()

section() { printf '\n=== %s ===\n' "$1"; }
fact() { printf '%-22s %s\n' "$1" "${2:-<unavailable>}"; }
fail() { problems+=("$1"); printf 'FAIL  %s\n' "$1"; }

section "host"
fact "date" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
fact "hostname" "$(hostname)"
fact "os" "$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
fact "kernel" "$(uname -r)"
fact "glibc" "$(ldd --version 2>/dev/null | head -1)"
fact "cpus" "$(nproc 2>/dev/null)"
fact "ram" "$(free -h 2>/dev/null | awk '/^Mem:/ {print $2}')"
fact "uid:gid" "$(id -u):$(id -g) ($(id -un))"
fact "python3" "$(python3 -V 2>&1)"

section "gpu"
if ! command -v nvidia-smi >/dev/null; then
  fail "nvidia-smi absent -- no GPU visible in this container"
else
  nvidia-smi --query-gpu=index,name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>&1 | sed 's/^/  /'
  gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
  # Isaac Sim requires RT cores. A100/H100/H200 have none and are unsupported
  # at any price (README §1). GB200/B200 datacenter parts are equally suspect.
  if grep -qiE 'A100|H100|H200|B200|GB[0-9]' <<<"$gpus"; then
    fail "allocated GPU has no RT cores: $gpus -- unsupported, do not proceed"
  fi
  fact "advertised part" "confirm the name above matches what was ordered"
fi

section "egress"
# Isaac streams assets from the Omniverse CDN on first run; a blocked egress
# looks like a mysteriously empty stage rather than a network error.
for url in https://nvcr.io/v2/ \
           https://omniverse-content-production.s3.us-west-2.amazonaws.com \
           https://pypi.org/simple/ \
           https://github.com; do
  code=$(curl -sS -m 15 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null || echo "000")
  fact "$url" "$code"
  [[ "$code" == "000" ]] && fail "no egress to $url"
done

section "isaac image"
# Only meaningful when this runs inside nvcr.io/nvidia/isaac-lab; on a bare pod
# these are all absent, which is fine -- it just means the image is not up yet.
SIM_ROOT="${ISAACSIM_ROOT_PATH:-/isaac-sim}"
LAB_PATH="${ISAACLAB_PATH:-/workspace/isaaclab}"
if [[ -d "$SIM_ROOT" ]]; then
  fact "isaac sim root" "$SIM_ROOT"
  fact "isaaclab.sh" "$([[ -x "$LAB_PATH/isaaclab.sh" ]] && echo "$LAB_PATH/isaaclab.sh" || echo "MISSING at $LAB_PATH")"
  [[ -x "$LAB_PATH/isaaclab.sh" ]] \
    || fail "$LAB_PATH/isaaclab.sh missing -- is a volume mounted over /workspace?"
  # Kit aborts on a root uid without this; the image runs as root by default.
  fact "OMNI_KIT_ALLOW_ROOT" "${OMNI_KIT_ALLOW_ROOT:-<unset>}"
  [[ "$(id -u)" == "0" && -z "${OMNI_KIT_ALLOW_ROOT:-}" ]] \
    && fail "running as root without OMNI_KIT_ALLOW_ROOT=1 -- Kit will refuse to start"
  fact "ACCEPT_EULA" "${ACCEPT_EULA:-<unset>}"
  [[ -z "${ACCEPT_EULA:-}" ]] \
    && fail "ACCEPT_EULA unset -- set it as a pod env var, the image requires it"
else
  fact "isaac sim root" "absent -- not running inside the isaac-lab image"
fi

section "volume ($VOL)"
if [[ "$VOL" == "/workspace" || "$VOL" == /workspace/* ]]; then
  fail "volume at $VOL shadows the image's /workspace/isaaclab -- recreate the pod with the volume mounted elsewhere (e.g. /idtb)"
elif [[ ! -d "$VOL" ]]; then
  fail "$VOL does not exist -- is the network volume attached?"
else
  fact "device" "$(df -h "$VOL" | awk 'NR==2 {print $1}')"
  fact "size / free" "$(df -h "$VOL" | awk 'NR==2 {print $2" / "$4}')"
  fact "fstype" "$(stat -f -c %T "$VOL" 2>/dev/null)"
  fact "owner" "$(stat -c '%U:%G (%u:%g)' "$VOL" 2>/dev/null)"
  probe="$VOL/.preflight.$$"
  if touch "$probe" 2>/dev/null; then
    rm -f "$probe"
    fact "writable" "yes"
  else
    fail "$VOL is not writable by $(id -un) -- run bootstrap.sh to fix ownership"
  fi
fi

section "VERDICT"
if ((${#problems[@]} == 0)); then
  echo "all checks passed -- record the driver version above, then resolve the"
  echo "Isaac Sim / Isaac Lab / Python versions from it (README §3.2)."
  exit 0
fi
printf '%d problem(s):\n' "${#problems[@]}"
printf '  - %s\n' "${problems[@]}"
exit 1
