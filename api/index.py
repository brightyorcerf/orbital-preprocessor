import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ground.dashboard import main

app = main