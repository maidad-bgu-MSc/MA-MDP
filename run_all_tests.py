import os
import sys
import subprocess

def install_pytest_if_missing():
    """Auto-installs pytest using pip if it is not present in the current environment."""
    try:
        import pytest
        print("pytest is already installed.")
    except ImportError:
        print("pytest not found. Auto-installing pytest via pip...")
        cmd = [sys.executable, "-m", "pip", "install", "pytest"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("Failed to auto-install pytest. Please run 'pip install pytest' manually.")
            print(result.stderr)
            sys.exit(1)
        print("pytest installed successfully!")

def run_tests():
    """Runs the entire pytest test suite programmatically."""
    import pytest
    print("\n" + "=" * 80)
    print("Launching ATLC MA-MDP Automated Testing Suite...")
    print("=" * 80 + "\n")
    
    # Run pytest on the tests directory, profiling durations and suppressing warnings
    exit_code = pytest.main(["tests/", "-v", "--durations=0", "-W", "ignore"])
    sys.exit(exit_code)

if __name__ == "__main__":
    install_pytest_if_missing()
    run_tests()
