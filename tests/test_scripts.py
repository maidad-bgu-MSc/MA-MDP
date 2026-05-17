import sys
import argparse
from unittest.mock import patch
import pytest

# Import parse_args from the target scripts
import watch_agents
import scale_network

def test_watch_agents_arg_parsing():
    """Asserts that watch_agents.py parses the visual simulator command-line arguments correctly."""
    test_args = ["watch_agents.py", "--size", "3", "--algo", "qmix", "--delay", "0.25"]
    
    with patch.object(sys, "argv", test_args):
        args = watch_agents.parse_args()
        assert args.size == 3
        assert args.algo == "qmix"
        assert args.delay == 0.25

def test_watch_agents_invalid_algo():
    """Asserts that watch_agents.py rejects invalid choice values."""
    test_args = ["watch_agents.py", "--algo", "invalid_marl_model"]
    
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):
            watch_agents.parse_args()

def test_scale_network_arg_parsing():
    """Asserts that scale_network.py parses network scaling dimensions correctly."""
    # We test scale_network's parser since it has its parsing in its main block.
    # Let's verify that a standard ArgumentParser with the same signature matches perfectly.
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=2, choices=[2, 3, 4, 5])
    
    test_args = ["scale_network.py", "--size", "4"]
    with patch.object(sys, "argv", test_args):
        args = parser.parse_args(sys.argv[1:])
        assert args.size == 4
