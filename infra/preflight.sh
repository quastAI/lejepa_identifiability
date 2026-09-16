#!/usr/bin/env bash
# Read-only pod survey. Run this first on a fresh pod and paste the whole output.
#
# The versions are resolved now (README §3.1: Isaac Sim 6.0.0 / Isaac Lab
# 3.0.0-beta2 / Python 3.12), so this script's job has changed: it no longer asks
# what the driver allows, it checks that *this* host still clears what §3.1
# already decided. A pod recreate can land on a different host with an older
# driver, which is exactly how that decision silently stops holding.
#
# Every check runs even if an earlier one fails -- one boot, one full report.
# Exit status is 1 if a hard rule is violated (see VERDICT), 0 otherwise.
set -uo pipefail

# Not /workspace: the vendor isaac-lab image unpacks Isaac Lab into
# /workspace/isaaclab, so a volume mounted there shadows it (README §8.3).
VOL="${IDTB_VOL:-/idtb}"

# What §3.1 resolved. Kept overridable: the Decision Register is the record, this
# file is only a check against it, and re-resolving must not mean editing code.
MIN_DRIVER="${IDTB_MIN_DRIVER:-580.95.05}"   # Isaac Sim 6.0.0, Linux x86_64
BAD_DRIVER="${IDTB_BAD_DRIVER:-595}"         # IsaacSim #537: 595.x breaks CUDA detection
WANT_SIM="${IDTB_ISAACSIM_VERSION:-6.0.0}"   # tag 3.0.0-beta2; -post1 is 6.0.1

problems=()
warnings=()

section() { printf '\n=== %s ===\n' "$1"; }
fact() { printf '%-22s %s\n' "$1" "${2:-<unavailable>}"; }
fail() { problems+=("$1"); printf 'FAIL  %s\n' "$1"; }
# Reported, exit status unaffected: things a human must see before spending a
# session on this pod, but which are not grounds for the script to stop them.
warn() { warnings+=("$1"); printf 'WARN  %s\n' "$1"; }

# True when $1 < $2, compared field-wise as numbers. Not `sort -V`: this also
# runs on the macOS dev box under test, and 10# is load-bearing -- a driver field
# like .03 or .08 is otherwise read as octal and errors or compares wrong.
ver_lt() {
  local -a a b
  local i x y
  IFS=. read -ra a <<<"$1"
  IFS=. read -ra b <<<"$2"
  for ((i = 0; i < ${#a[@]} || i < ${#b[@]}; i++)); do
    x=${a[i]:-0}; y=${b[i]:-0}
    [[ $x =~ ^[0-9]+$ ]] || x=0
    [[ $y =~ ^[0-9]+$ ]] || y=0
    ((10#$x < 10#$y)) && return 0
    ((10#$x > 10#$y)) && return 1
  done
  return 1
}

section "host"
fact "date" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
fact "hostname" "$(hostname)"
fact "os" "$(. /etc/os-release 2>/dev/null && echo "$PRETTY_NAME")"
fact "kernel" "$(uname -r)"
fact "glibc" "$(ldd --version 2>/dev/null | head -1)"
fact "cpus" "$(nproc 2>/dev/null)"
fact "ram" "$(free -h 2>/dev/null | awk '/^Mem:/ {print $2}')"
fact "uid:gid" "$(id -u):$(id -g) ($(id -un))"
# Informational only, and absent in the vendor image by design -- Isaac ships its
# own interpreter (reported under "isaac image"). Guarded with command -v so a
# missing binary is a recorded fact, not a shell error printed inside the value.
fact "python3" "$(command -v python3 >/dev/null && python3 -V 2>&1 || echo '<absent -- fine: the image ships its own, see below>')"

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

  # nvidia-smi inside a container reports the *host* driver, which is a property
  # of the machine the pod landed on, not of anything we control from in here.
  #
  # $MIN_DRIVER is the version NVIDIA *tested* this release on, not a hard floor:
  # from 5.1.0 on, the requirements page says "Isaac Sim was tested on these
  # driver versions" and drops the minimum/recommended split. Kit's own refusal
  # is far lower (it rejects < 535.129), so a below-tested driver usually boots.
  # This is still a hard failure here, because what this project measures is
  # pixel-level render determinism (README §7.1) and an untested driver is
  # exactly the kind of uncontrolled variable that corrupts that measurement
  # quietly. Override IDTB_MIN_DRIVER to deliberately try anyway.
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null \
           | head -1 | tr -d '[:space:]')
  fact "driver vs README §3.1" "$driver  (Isaac Sim $WANT_SIM tested at $MIN_DRIVER)"
  if [[ -n "$driver" ]] && ver_lt "$driver" "$MIN_DRIVER"; then
    fail "driver $driver is below $MIN_DRIVER, the driver Isaac Sim $WANT_SIM was tested on (README §3.1). The driver belongs to the host, so no image tag changes it: recreate the pod filtering for CUDA 13.0, which is the 580 branch. Kit itself would probably start on $driver -- that is the trap, not the reassurance"
  fi
  # Newer is not safer: the 595 branch tests fine for 6.0.1/6.1.0 but is reported
  # to break CUDA detection where 580 works (IsaacSim #537). The target is the
  # 580 branch specifically -- CUDA 13.0, not 12.8 below it and not 13.2 above.
  if [[ -n "$driver" ]] && ! ver_lt "$driver" "$BAD_DRIVER"; then
    warn "driver $driver is on the $BAD_DRIVER branch or newer, which Isaac Sim $WANT_SIM was not tested against and which IsaacSim #537 reports breaking CUDA detection -- if Isaac cannot see the GPU, this is the first suspect"
  fi
fi

section "egress"
# Isaac streams assets from the Omniverse CDN on first run; a blocked egress
# looks like a mysteriously empty stage rather than a network error.
#
# curl's own exit status is what decides this, NOT the printed code: on a failed
# connection curl still writes "000" for %{http_code} *and* exits non-zero, so an
# `|| echo 000` fallback appends a second one and the string becomes "000000" --
# which then compares unequal to "000" and a total egress blackout reports as
# green. That happened on a real pod. The exit code is also the only thing that
# distinguishes the causes: 6 = DNS, 7 = connection refused, 28 = timeout,
# 35/60 = TLS.
if ! command -v curl >/dev/null; then
  fail "curl absent -- egress unverified"
else
  for url in https://nvcr.io/v2/ \
             https://omniverse-content-production.s3.us-west-2.amazonaws.com \
             https://pypi.org/simple/ \
             https://github.com; do
    code=$(curl -sS -m 15 -o /dev/null -w '%{http_code}' "$url" 2>/dev/null)
    rc=$?
    if ((rc != 0)) || [[ -z "$code" || "$code" == "000" ]]; then
      fact "$url" "UNREACHABLE (curl exit $rc, http ${code:-none})"
      fail "no egress to $url (curl exit $rc)"
    else
      fact "$url" "$code"
    fi
  done
fi

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

  # Which image is actually running. The tag is not visible from inside the
  # container, and 3.0.0-beta2 vs 3.0.0-beta2-post1 is Isaac Sim 6.0.0 vs 6.0.1
  # -- four characters, different driver floor, and the wrong one fails at a
  # layer that never mentions drivers (README §8.3). Probed, never assumed:
  # these paths are what the image is believed to ship, and an absent one is
  # reported as absent rather than treated as a failure.
  sim_version=""
  [[ -r "$SIM_ROOT/VERSION" ]] && sim_version=$(tr -d '[:space:]' < "$SIM_ROOT/VERSION")
  fact "isaac sim version" "${sim_version:-<no $SIM_ROOT/VERSION -- report what the image does ship>}"
  if [[ -n "$sim_version" && "$sim_version" != "$WANT_SIM"* ]]; then
    fail "image is Isaac Sim $sim_version, but §3.1 resolved $WANT_SIM -- wrong tag? 3.0.0-beta2 is $WANT_SIM, 3.0.0-beta2-post1 is 6.0.1"
  fi

  lab_version=""
  [[ -r "$LAB_PATH/VERSION" ]] && lab_version=$(tr -d '[:space:]' < "$LAB_PATH/VERSION")
  [[ -z "$lab_version" && -d "$LAB_PATH/.git" ]] \
    && lab_version=$(git -C "$LAB_PATH" describe --tags --always 2>/dev/null)
  fact "isaac lab version" "${lab_version:-<not readable from $LAB_PATH>}"
  [[ -r "$LAB_PATH/docker/.env.base" ]] \
    && fact "  .env.base sim ver" "$(grep -m1 '^ISAACSIM_VERSION' "$LAB_PATH/docker/.env.base" 2>/dev/null)"

  # §3.1 records Python 3.12 as forced by the Isaac Sim release. This is where
  # that gets measured: the interpreter Isaac actually runs, not the host's.
  for py in "$SIM_ROOT/kit/python/bin/python3" "$SIM_ROOT/python.sh"; do
    if [[ -x "$py" ]]; then
      fact "bundled python" "$py -> $("$py" -V 2>&1 | tail -1)"
      break
    fi
  done

  # Kit aborts on a root uid without this; the 2.3.2 image runs as root, 3.0 does not.
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
if ((${#warnings[@]} > 0)); then
  printf '%d warning(s), not blocking:\n' "${#warnings[@]}"
  printf '  - %s\n' "${warnings[@]}"
fi
if ((${#problems[@]} == 0)); then
  echo "all checks passed -- record the survey in README §8.1, then run"
  echo "infra/bootstrap.sh and the shipped Isaac Lab tutorial before our code."
  exit 0
fi
printf '%d problem(s):\n' "${#problems[@]}"
printf '  - %s\n' "${problems[@]}"
exit 1
