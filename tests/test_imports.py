import importlib
import sys
import pytest

def test_syntax_and_imports():
    """Asserts that all core codebase files can be imported cleanly and have correct syntax."""
    modules_to_test = [
        "simulator.env_setup",
        "simulator.generate_1x4_wave",
        "train",
        "evaluate",
        "plot_results",
        "marl_algorithms",
        "watch_agents"
    ]
    
    for mod_name in modules_to_test:
        try:
            importlib.import_module(mod_name)
            print(f"Successfully imported {mod_name}")
        except Exception as e:
            pytest.fail(f"Failed to import {mod_name} with exception: {e}")

def test_dependencies():
    """Validates that all external reinforcement learning libraries are correctly installed."""
    required_libs = [
        "sumo_rl",
        "pettingzoo",
        "tianshou",
        "gymnasium",
        "torch",
        "matplotlib",
        "pandas",
        "numpy"
    ]
    for lib in required_libs:
        try:
            importlib.import_module(lib)
        except ImportError:
            pytest.fail(f"Critical dependency '{lib}' is missing from the environment!")
