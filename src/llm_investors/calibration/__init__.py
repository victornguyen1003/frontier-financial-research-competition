"""Calibration of the Study 1 v3 environment (`calibrate`).

Reads the licensed TEJ file and the hand-collected attribute data once and writes
results/calibration/{prices,attributes}.json (committed). The price generator and the persona
builder read only those JSON files, never the raw data.
"""
