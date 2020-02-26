"""Project-wide constants."""

V_COLS = ["V%d" % i for i in range(1, 29)]
FEATURES = ["Time"] + V_COLS + ["Amount"]
TARGET = "Class"
