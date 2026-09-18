"""Put the repository root on ``sys.path``.

Importing this module first lets every script in this directory run from a
clean checkout without installing the package, while keeping all remaining
imports at the top of the file as PEP 8 requires.  If ``bayu`` is installed
(``pip install -e .``), importing this module is harmless.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
