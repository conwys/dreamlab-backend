import sys
import os

tests_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.dirname(tests_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"DEBUG: Added {project_root} to sys.path for tests.")
