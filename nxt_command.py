"""
This script reads NXT command from STDIN, tweaks it usefully, and run it
"""

import fileinput
import os

lines = [line.strip() for line in fileinput.input()]  # default: STDIN
cmd = " ".join(lines)

print("Running Command:")
print(cmd)

os.system(cmd)
