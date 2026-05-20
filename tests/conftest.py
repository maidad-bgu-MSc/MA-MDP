import os
import sys


def _setup_sumo_env():
    """Try to locate SUMO and set SUMO_HOME before any test module is collected."""
    try:
        import sumo
        sumo_home = os.path.abspath(os.path.dirname(sumo.__file__))
        os.environ["SUMO_HOME"] = sumo_home

        python_dir = os.path.dirname(sys.executable)
        scripts_dir = os.path.join(python_dir, "Scripts")
        paths = [os.path.join(sumo_home, "bin"), scripts_dir]
        current_path = os.environ.get("PATH", "")
        os.environ["PATH"] = os.pathsep.join(paths) + os.pathsep + current_path
        return True
    except ImportError:
        pass

    if os.environ.get("SUMO_HOME"):
        return True

    return False


_SUMO_AVAILABLE = _setup_sumo_env()

# Prevent collection of SUMO-dependent test files when SUMO is not installed.
# This avoids ImportError during collection rather than at test runtime.
collect_ignore = []
if not _SUMO_AVAILABLE:
    collect_ignore = [
        "test_1x4_dynamics.py",
        "test_action_handshake.py",
        "test_imports.py",  # requires sumo_rl and simulator.env_setup at import time
    ]