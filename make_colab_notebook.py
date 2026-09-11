#!/usr/bin/env python3
"""
Backward-compatibility proxy for tools/make_colab_notebook.py.
Forwards execution to tools/make_colab_notebook.py.
"""
import os
import runpy
import sys

script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "make_colab_notebook.py")
if not os.path.exists(script_path):
    raise FileNotFoundError(f"Target script not found: {script_path}")

runpy.run_path(script_path, run_name="__main__")
