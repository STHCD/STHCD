import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.schedule_registry import list_schedule_aliases


if __name__ == "__main__":
    for alias in sorted(list_schedule_aliases()):
        print(alias)
