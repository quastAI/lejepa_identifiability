"""The pod scripts are the only code that runs before anything is debuggable.

They execute on a rented GPU nobody has logged into, so a mistake in them costs a
pod session rather than a test run. Everything here is hermetic: a fake ``HOME``,
a fake Isaac root and a local git repo, so nothing touches the dev machine.

The constants below are transcribed from Isaac Lab **v3.0.0-beta2**'s own
``docker/docker-compose.yaml`` and ``docker/.env.base`` — the release behind
``nvcr.io/nvidia/isaac-lab:3.0.0-beta2`` (README §3.1). They are what makes the
vendor image's layout a checked fact rather than a remembered one.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

INFRA = Path(__file__).resolve().parent.parent / "infra"
BOOTSTRAP = INFRA / "bootstrap.sh"
PREFLIGHT = INFRA / "preflight.sh"

# docker/.env.base: DOCKER_ISAACLAB_PATH. The reason /workspace is unusable as a
# volume mount point -- a volume there hides Isaac Lab itself.
VENDOR_ISAACLAB_PATH = "/workspace/isaaclab"

# docker-compose.yaml, x-default-isaac-lab-volumes. Caches the vendor persists
# between runs; each must be relocated onto our volume or it is re-downloaded or
# recompiled on paid GPU time. Relocating a parent directory counts.
VENDOR_CACHES = [
    "/isaac-sim/kit/cache",
    "/isaac-sim/kit/data",  # added in Isaac Lab 3.0; absent from the 2.3.2 list
    "$HOME/.cache/ov",
    "$HOME/.cache/pip",
    "$HOME/.cache/nvidia/GLCache",
    "$HOME/.nv/ComputeCache",
    "$HOME/.nvidia-omniverse/logs",
    "$HOME/.local/share/ov/data",
]
# Deliberately not relocated: /isaac-sim/kit/logs/Kit/Isaac-Sim and
# $HOME/Documents are outputs, not caches -- nothing is recomputed by losing them.


def run(script, env=None, **kw):
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env={**os.environ, **(env or {})},
        **kw,
    )


# --- stubs -------------------------------------------------------------------
# The survey reports on a machine we do not have, so the interesting paths can
# only be tested by standing in for the two binaries it reads the pod through.
# Both are driven by env vars, so one stub covers every case.

CURL_STUB = """#!/usr/bin/env bash
# curl writes %{http_code} even when the transfer never happened: "000" plus a
# non-zero exit. That combination is the whole point of these tests.
printf '%s' "${STUB_HTTP_CODE:-000}"
exit "${STUB_CURL_RC:-0}"
"""

NVIDIA_SMI_STUB = """#!/usr/bin/env bash
q=""
for a in "$@"; do case "$a" in --query-gpu=*) q="${a#--query-gpu=}" ;; esac; done
case "$q" in
  name) echo "${STUB_GPU:-NVIDIA GeForce RTX 4090}" ;;
  driver_version) echo "${STUB_DRIVER:-580.178.04}" ;;
  *) echo "0, ${STUB_GPU:-NVIDIA GeForce RTX 4090}, ${STUB_DRIVER:-580.178.04}, 24564 MiB, 8.9" ;;
esac
"""


@pytest.fixture
def stub_bin(tmp_path):
    """A PATH prefix holding fake ``curl`` and ``nvidia-smi``.

    Prepended rather than replacing PATH: the survey legitimately shells out to a
    dozen other tools, and stubbing those would be testing the stubs.
    """
    d = tmp_path / "stub-bin"
    d.mkdir()
    for name, body in (("curl", CURL_STUB), ("nvidia-smi", NVIDIA_SMI_STUB)):
        p = d / name
        p.write_text(body)
        p.chmod(0o755)
    return {"PATH": f"{d}:{os.environ['PATH']}"}


def test_preflight_reports_a_blackout_as_a_failure(stub_bin):
    """The regression that made a real pod report green with zero egress.

    curl prints ``000`` *and* exits non-zero, so an ``|| echo 000`` fallback
    concatenated a second one; the resulting ``000000`` compared unequal to
    ``000`` and every endpoint being unreachable passed silently. Exit status,
    not the printed code, is what decides this.
    """
    r = run(PREFLIGHT, {**stub_bin, "STUB_CURL_RC": "6", "STUB_HTTP_CODE": "000"})
    assert "000000" not in r.stdout, "the concatenation bug is back"
    assert r.stdout.count("FAIL  no egress to") == 4, "a blackout was not reported per endpoint"
    assert "curl exit 6" in r.stdout, "the cause (6 = DNS) is not surfaced"
    assert r.returncode == 1


def test_preflight_accepts_a_reachable_registry(stub_bin):
    """401 from nvcr.io is reachable-and-unauthenticated, not a failure."""
    r = run(PREFLIGHT, {**stub_bin, "STUB_CURL_RC": "0", "STUB_HTTP_CODE": "401"})
    assert "no egress" not in r.stdout


@pytest.mark.parametrize(
    ("driver", "clears"),
    [
        ("580.178.04", True),  # the host the stack was resolved on
        ("570.195.03", False),  # a recreated pod landed here -- older branch
        ("580.95.05", True),  # exactly at the floor
        ("580.95.04", False),
        # Leading zeros must not be read as octal: 08 and 09 are invalid octal
        # and would abort the comparison rather than merely compare wrong.
        ("580.95.08", True),
        ("580.09.05", False),
    ],
)
def test_preflight_checks_the_driver_against_the_resolved_release(stub_bin, driver, clears):
    """A pod recreate can land on an older host driver, silently invalidating §3.1.

    The image tag cannot fix it -- the driver belongs to the machine -- so the
    survey has to say so rather than leave it to a Kit-level error later.
    """
    r = run(PREFLIGHT, {**stub_bin, "STUB_DRIVER": driver, "IDTB_MIN_DRIVER": "580.95.05"})
    below = f"driver {driver} is below" in r.stdout
    assert below != clears, r.stdout
    assert driver in r.stdout, "the measured driver is not reported at all"


def test_preflight_flags_a_known_bug_on_the_595_branch_without_blocking(stub_bin):
    """595 is the tested branch for 6.0.1 (README §3.1) -- and also where
    IsaacSim #537 reports CUDA detection breaking, where 580 worked. Both are
    true at once, so this is a heads-up alongside a clean driver check, not a
    version-mismatch warning; it must not touch the exit status.
    """
    r = run(
        PREFLIGHT,
        {**stub_bin, "STUB_DRIVER": "595.79", "STUB_HTTP_CODE": "200", "IDTB_VOL": "/nonexistent"},
    )
    assert "WARN" in r.stdout
    assert "#537" in r.stdout
    assert "not blocking" in r.stdout
    assert "driver 595.79 is below" not in r.stdout, "595.79 clears the 595.58.03 floor"
    # The only failure here is the absent volume -- the warning added none.
    assert "1 problem(s)" in r.stdout


def test_preflight_default_driver_floor_matches_the_resolved_release(stub_bin):
    """README §3.1: Isaac Sim 6.0.1, tested at 595.58.03 -- without an override,
    the script's own baked-in default must match what the Decision Register
    actually says, or the two silently drift apart.
    """
    r = run(PREFLIGHT, {**stub_bin, "STUB_DRIVER": "580.178.04"})
    assert "driver 580.178.04 is below 595.58.03" in r.stdout
    assert "Isaac Sim 6.0.1" in r.stdout


@pytest.fixture
def fake_image(tmp_path):
    """A stand-in for the vendor image's layout: Isaac root + Isaac Lab checkout."""
    sim = tmp_path / "isaac-sim"
    lab = tmp_path / "isaaclab"
    (sim / "kit" / "python" / "bin").mkdir(parents=True)
    lab.mkdir()
    sh = lab / "isaaclab.sh"
    sh.write_text("#!/usr/bin/env bash\n")
    sh.chmod(0o755)
    return {
        "sim": sim,
        "env": {
            "ISAACSIM_ROOT_PATH": str(sim),
            "ISAACLAB_PATH": str(lab),
            "ACCEPT_EULA": "Y",
        },
    }


@pytest.mark.parametrize(
    ("version", "accepted"),
    [("6.0.1", True), ("6.0.1-rc.10+release.1234", True), ("6.0.0", False), ("5.1.0", False)],
)
def test_preflight_catches_the_wrong_image_tag(stub_bin, fake_image, version, accepted):
    """``3.0.0-beta2`` and ``3.0.0-beta2-post1`` differ by four characters.

    They are Isaac Sim 6.0.0 and 6.0.1, which have different tested drivers, and
    the wrong one pulls cleanly and fails somewhere that never mentions drivers.
    We're deliberately on ``-post1`` (6.0.1, README §3.4.1), not plain ``beta2``.
    The tag is invisible from inside the container, so the version file is the
    only way to know which one is running.
    """
    (fake_image["sim"] / "VERSION").write_text(version + "\n")
    r = run(PREFLIGHT, {**stub_bin, **fake_image["env"], "IDTB_ISAACSIM_VERSION": "6.0.1"})
    assert (f"image is Isaac Sim {version}" in r.stdout) != accepted, r.stdout
    assert version in r.stdout, "the measured Isaac Sim version is not reported"


def test_preflight_default_target_matches_the_resolved_release(stub_bin, fake_image):
    """Without an override, the script's own baked-in target must be 6.0.1 --
    the README §3.1 decision -- not the 6.0.0 it was reversed from.
    """
    (fake_image["sim"] / "VERSION").write_text("6.0.0\n")
    r = run(PREFLIGHT, {**stub_bin, **fake_image["env"]})
    assert "image is Isaac Sim 6.0.0" in r.stdout
    assert "3.0.0-beta2-post1 is 6.0.1" in r.stdout


def test_preflight_survives_an_image_that_ships_no_version_file(stub_bin, fake_image):
    """Everything about the image's layout is written blind; absence is a fact.

    A probe that guessed wrong must report what it could not find, never fail --
    otherwise the survey blocks on its own assumption.
    """
    r = run(PREFLIGHT, {**stub_bin, **fake_image["env"]})
    assert "isaac sim version" in r.stdout
    assert "wrong tag" not in r.stdout
    assert "=== VERDICT" in r.stdout


def test_preflight_does_not_let_a_missing_binary_corrupt_a_fact_line():
    """The vendor image has no ``python3`` on PATH -- it ships its own.

    Unguarded, the shell's own "command not found" lands *inside* the reported
    value, so the survey line reads as a script error rather than a fact.
    """
    body = PREFLIGHT.read_text()
    assert "command -v python3" in body


@pytest.mark.parametrize("script", [BOOTSTRAP, PREFLIGHT], ids=lambda p: p.name)
def test_syntax_is_valid(script):
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0


@pytest.mark.parametrize("script", [BOOTSTRAP, PREFLIGHT], ids=lambda p: p.name)
def test_is_executable(script):
    assert os.access(script, os.X_OK), f"{script.name} is not chmod +x"


@pytest.mark.parametrize("vol", ["/workspace", "/workspace/idtb"])
def test_bootstrap_refuses_to_mount_over_isaaclab(vol):
    """A volume at /workspace shadows the image's Isaac Lab install.

    The symptom is a missing ``isaaclab.sh``, which reads as a broken image
    rather than a mount problem, so the script has to name the cause itself.
    """
    r = run(BOOTSTRAP, {"IDTB_VOL": vol})
    assert r.returncode == 1
    assert "isaaclab" in r.stderr
    # It must refuse before doing anything, not after relocating half the caches.
    assert "relocating" not in r.stdout


def test_preflight_flags_the_workspace_collision():
    r = run(PREFLIGHT, {"IDTB_VOL": "/workspace"})
    assert "FAIL" in r.stdout
    assert "shadows" in r.stdout
    assert r.returncode == 1


def test_preflight_never_fails_fast():
    """One boot, one full report: a failing check must not skip later sections."""
    r = run(PREFLIGHT, {"IDTB_VOL": "/nonexistent-volume"})
    for section in ("host", "gpu", "egress", "isaac image", "volume", "VERDICT"):
        assert f"=== {section}" in r.stdout, f"{section} missing after an earlier failure"


def test_bootstrap_relocates_every_vendor_cache():
    body = BOOTSTRAP.read_text()
    linked = [line.split('"')[1] for line in body.splitlines() if line.startswith("link_cache ")]
    assert linked, "no link_cache calls found"
    for cache in VENDOR_CACHES:
        # $SIM_ROOT defaults to /isaac-sim; compare on the resolved form.
        covered = any(
            cache.startswith(entry.replace("$SIM_ROOT", "/isaac-sim")) for entry in linked
        )
        assert covered, f"{cache} is persisted by the vendor image but not relocated"


@pytest.fixture
def pod(tmp_path):
    """A sandboxed stand-in for a fresh pod: fake home, fake Isaac root, volume."""
    home = tmp_path / "home"
    (home / ".cache").mkdir(parents=True)
    sim_root = tmp_path / "isaac-sim"
    (sim_root / "kit" / "cache").mkdir(parents=True)
    (sim_root / "kit" / "cache" / "shipped.bin").write_text("from the image")
    vol = tmp_path / "vol"
    vol.mkdir()

    return {
        "tmp": tmp_path,
        "vol": vol,
        "home": home,
        "env": {
            "HOME": str(home),
            "IDTB_VOL": str(vol),
            "ISAACSIM_ROOT_PATH": str(sim_root),
        },
    }


@pytest.mark.skipif(shutil.which("git") is None, reason="git required")
def test_bootstrap_end_to_end_and_idempotent(pod):
    first = run(BOOTSTRAP, pod["env"])
    assert first.returncode == 0, first.stderr

    vol, home = pod["vol"], pod["home"]

    # Caches are symlinks pointing onto the volume, not copies left behind.
    ov = home / ".cache" / "ov"
    assert ov.is_symlink() and ov.resolve() == (vol / "cache" / "ov").resolve()

    # Whatever the image shipped survives the relocation -- otherwise the first
    # bootstrap silently throws away a prebuilt cache.
    assert (vol / "cache" / "kit" / "shipped.bin").read_text() == "from the image"

    # OMNI_KIT_ALLOW_ROOT is conditional on the uid, because the fact flipped
    # between image generations: isaac-lab 2.3.2 runs as root and needs it, 3.0
    # runs as uid 1000 and does not. Asserting it unconditionally would pin the
    # wrong one of those.
    env_sh = (vol / "env.sh").read_text()
    exported = "export OMNI_KIT_ALLOW_ROOT=1" in env_sh
    assert exported == (os.geteuid() == 0), env_sh
    if not exported:
        assert "not root" in env_sh, "silently omitted it instead of saying why"
    assert f'export IDTB_VOL="{vol}"' in env_sh
    assert (vol / "data").is_dir()
    assert (vol / "logs").is_dir()

    # It never clones: the script ships inside the repo, so a checkout it could
    # clone into is one you already have. Cloning made a second copy, and the
    # one you edited was then not the one Isaac ran.
    assert not (vol / "repo").exists(), "bootstrap created a second checkout"

    # Re-run: every pod start runs this, so a second run must be a no-op.
    second = run(BOOTSTRAP, pod["env"])
    assert second.returncode == 0, second.stderr
    assert "already linked" in second.stdout
    assert ov.resolve() == (vol / "cache" / "ov").resolve()
    assert (vol / "cache" / "kit" / "shipped.bin").exists()


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anything -- nothing to detect")
def test_bootstrap_survives_a_root_owned_isaac_tree(pod):
    """The image leaves ``/isaac-sim/kit`` root-owned while running as uid 1000,
    and root is not reachable to fix it: measured on the pod, both ``su`` and
    ``sudo`` fail from inside the container (no sudo binary at all). So this is
    the image's accepted, permanent state, not a transient error — the run must
    report it plainly and still finish successfully, every time, rather than
    fail on something nothing can act on.

    Replacing a path with a symlink is a write to its *parent*, so the cache
    itself being writable is irrelevant to the relocation — but it does mean
    Isaac's own runtime writes into it are unaffected; only cross-restart
    persistence of that one cache is lost. Every other cache must still be
    relocated, and the one that was not must still be impossible to miss.
    """
    sim_root = pod["tmp"] / "isaac-sim"
    (sim_root / "kit").chmod(0o555)
    try:
        r = run(BOOTSTRAP, pod["env"])
    finally:
        (sim_root / "kit").chmod(0o755)  # or tmp_path cleanup fails

    assert "BLOCKED" in r.stdout, "silently skipped a cache the volume exists to hold"
    # Not fatal where it happens: the rest of the list still gets done.
    assert (pod["home"] / ".cache" / "ov").is_symlink()
    assert (pod["vol"] / "data").is_dir()
    # Nothing to retry, so this is a NOTE, not a failure -- the run still exits 0.
    assert r.returncode == 0
    assert "NOTE" in r.stdout
    assert str(sim_root / "kit") in r.stdout
    assert "chown" not in r.stdout + r.stderr, "points at a fix that's confirmed unreachable"


def test_bootstrap_skips_isaac_paths_outside_the_image(pod):
    """On a bare pod /isaac-sim is absent; that must not leave a dangling link."""
    absent = pod["tmp"] / "absent-sim-root"
    r = run(BOOTSTRAP, {**pod["env"], "ISAACSIM_ROOT_PATH": str(absent)})
    assert r.returncode == 0, r.stderr
    assert "skipped" in r.stdout
    assert not absent.exists(), "created a dangling symlink outside the image"


@pytest.mark.skipif(os.geteuid() == 0, reason="root can write anything -- nothing to detect")
def test_bootstrap_uses_the_password_database_when_home_is_unwritable(pod, tmp_path):
    """The image sets ``HOME=/root`` while running as uid 1000.

    Every ``$HOME`` cache path then resolves somewhere this user cannot write,
    and under ``set -e`` the run dies *partway* — some caches relocated, some
    not, which looks half-done rather than failed. ``getent`` is stubbed here
    because the real one would point at the developer's own home directory.
    """
    real_home = tmp_path / "real-home"
    (real_home / ".cache").mkdir(parents=True)
    unwritable = tmp_path / "fake-root"
    unwritable.mkdir()
    unwritable.chmod(0o500)

    stub = tmp_path / "stub-bin"
    stub.mkdir()
    getent = stub / "getent"
    getent.write_text(f"#!/usr/bin/env bash\necho 'u:x:1000:1000::{real_home}:/bin/bash'\n")
    getent.chmod(0o755)

    r = run(
        BOOTSTRAP,
        {
            **pod["env"],
            "HOME": str(unwritable),
            "PATH": f"{stub}:{os.environ['PATH']}",
        },
    )
    assert r.returncode == 0, r.stderr
    assert "not writable" in r.stdout, "silently used the broken HOME"
    assert (real_home / ".cache" / "ov").is_symlink(), "caches did not follow the corrected HOME"
    assert not (unwritable / ".cache").exists()
    # Isaac must agree with us about HOME, or it reads caches at a path we never
    # relocated and re-downloads everything while the symlinks sit unused.
    assert f'export HOME="{real_home}"' in (pod["vol"] / "env.sh").read_text()


def test_bootstrap_keeps_a_working_home(pod):
    """A usable HOME is also the one Isaac will read; do not second-guess it."""
    r = run(BOOTSTRAP, pod["env"])
    assert r.returncode == 0, r.stderr
    assert "not writable" not in r.stdout


def test_bootstrap_needs_no_network_at_all(pod, tmp_path):
    """Nothing it does is allowed to depend on the network.

    The repo has to be present for this script to exist, so fetching one was
    circular; with ``git`` itself broken the run must still complete, because a
    pod's DNS is briefly dead right after boot and the caches are already moved
    by the time the last step runs.
    """
    stub = tmp_path / "stub-bin"
    stub.mkdir()
    git = stub / "git"
    git.write_text("#!/usr/bin/env bash\necho 'could not resolve host' >&2\nexit 128\n")
    git.chmod(0o755)

    r = run(BOOTSTRAP, {**pod["env"], "PATH": f"{stub}:{os.environ['PATH']}"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert (pod["vol"] / "cache").is_dir(), "a broken git stopped the cache relocation"


def test_bootstrap_reports_a_checkout_that_will_not_survive_a_restart(pod):
    """A checkout in the container's own filesystem is gone on the next start.

    Including whatever was edited on the pod. The script runs from inside the
    repo, so it knows where that is and is the only thing positioned to say so.
    """
    r = run(BOOTSTRAP, pod["env"])
    assert str(BOOTSTRAP.parent.parent) in r.stdout, "does not report which checkout it is in"
    # The repo under test is on the dev machine, not on the fake volume.
    assert "does not survive a pod restart" in r.stdout


def test_bootstrap_handles_both_uid_regimes():
    """Both branches must exist -- the test suite only ever exercises one."""
    body = BOOTSTRAP.read_text()
    assert 'if [[ "$(id -u)" == "0" ]]; then' in body, "no root branch"
    assert "export OMNI_KIT_ALLOW_ROOT=1" in body, "root branch does not set it"
    # And the no-sudo case must stop rather than press on into a Kit-level error.
    assert "cannot take ownership" in body
