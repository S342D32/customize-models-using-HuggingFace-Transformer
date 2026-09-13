"""Pytest configuration for the Custom ResNet project.

Ensures the project root is importable so that flat imports like
``from configuration_resnet import ResnetConfig`` work correctly.
"""

import sys
from pathlib import Path

# Insert project root at the front of sys.path so that flat module
# imports (configuration_resnet, modeling_resnet, etc.) resolve.
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
