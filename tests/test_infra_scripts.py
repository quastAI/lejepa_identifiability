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
    linked = [
        line.split('"')[1]
        for line in body.splitlines()
        if line.startswith("link_cache ")
    ]
    assert linked, "no link_cache calls found"
    for cache in VENDOR_CACHES:
        # $SIM_ROOT defaults to /isaac-sim; compare on the resolved form.
        covered = any(
            cache.startswith(entry.replace("$SIM_ROOT", "/isaac-sim")) for entry in linked
        )
        assert covered, f"{cache} is persisted by the vendor image but not relocated"


@pytest.fixture
def pod(tmp_path):
    """A sandboxed stand-in for a fresh pod: fake home, Isaac root and origin."""
    home = tmp_path / "home"
    (home / ".cache").mkdir(parents=True)
    sim_root = tmp_path / "isaac-sim"
    (sim_root / "kit" / "cache").mkdir(parents=True)
    (sim_root / "kit" / "cache" / "shipped.bin").write_text("from the image")
    vol = tmp_path / "vol"
    vol.mkdir()

    origin = tmp_path / "origin"
    origin.mkdir()
    git = ["git", "-C", str(origin)]
    subprocess.run([*git, "init", "-q", "-b", "main"], check=True)
    (origin / "README.md").write_text("x")
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run(
        [*git, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        check=True,
    )

    return {
        "tmp": tmp_path,
        "vol": vol,
        "home": home,
        "env": {
            "HOME": str(home),
            "IDTB_VOL": str(vol),
            "ISAACSIM_ROOT_PATH": str(sim_root),
            "IDTB_REPO": str(origin),
            "IDTB_REF": "main",
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
    assert (vol / "repo" / ".git").is_dir()
    assert (vol / "data").is_dir() and (vol / "logs").is_dir()

    # Re-run: every pod start runs this, so a second run must be a no-op.
    second = run(BOOTSTRAP, pod["env"])
    assert second.returncode == 0, second.stderr
    assert "already linked" in second.stdout
    assert ov.resolve() == (vol / "cache" / "ov").resolve()
    assert (vol / "cache" / "kit" / "shipped.bin").exists()


def test_bootstrap_skips_isaac_paths_outside_the_image(pod):
    """On a bare pod /isaac-sim is absent; that must not leave a dangling link."""
    absent = pod["tmp"] / "absent-sim-root"
    r = run(BOOTSTRAP, {**pod["env"], "ISAACSIM_ROOT_PATH": str(absent)})
    assert r.returncode == 0, r.stderr
    assert "skipped" in r.stdout
    assert not absent.exists(), "created a dangling symlink outside the image"


def test_bootstrap_handles_both_uid_regimes():
    """Both branches must exist -- the test suite only ever exercises one."""
    body = BOOTSTRAP.read_text()
    assert 'if [[ "$(id -u)" == "0" ]]; then' in body, "no root branch"
    assert "export OMNI_KIT_ALLOW_ROOT=1" in body, "root branch does not set it"
    # And the no-sudo case must stop rather than press on into a Kit-level error.
    assert "cannot take ownership" in body
