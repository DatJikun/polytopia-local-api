"""Known Polytopia HUD / button coordinates for 1920x1200 fullscreen.

These are empirically measured on this Cloud Agent display (DISPLAY=:1).
Escape opens Settings — never send it. Close overlays with BACK, not Escape.
"""

# HUD strip used for OCR
HUD_CROP = (700, 0, 1250, 80)
# Bottom-left unit / tile panel
UNIT_CROP = (0, 1000, 760, 1200)

# Confirm / train (Polytopia blue)
DO_IT = (1117, 681)  # RGB often (0, 153, 255)
TRAIN = (1110, 790)  # darker blue (0, 122, 204) on taller modals
RESEARCH = (1082, 715)

# Overlays
BACK = (38, 42)  # white circle, black chevron (tech tree / settings)

# Bottom bar (window 1920x1200). End Turn needs a firm click, sometimes twice.
SETTINGS = (880, 1124)
GAME_STATS = (1000, 1124)
TECH_TREE = (1092, 1124)
END_TURN = (1111, 1130)

# City level-up: Workshop (left), not Explorer
WORKSHOP = (894, 655)
