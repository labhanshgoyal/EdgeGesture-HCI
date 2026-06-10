"""
Configuration file for Space Mission HCI System.
Contains all constants, gesture mappings, and UI settings.
"""
#Gestures
GESTURE_LABELS = [
    "Grab",
    "Tap",
    "Expand",
    "Pinch",
    "Rotation CW",
    "Rotation CCW",
    "Swipe Right",
    "Swipe Left",
    "Swipe Up",
    "Swipe Down",
    "Swipe X",
    "Swipe V",
    "Swipe +",
    "Shake",
]

#Commands Mapping
GESTURE_TO_COMMAND = {
    "Grab":         "CONFIRM / EXECUTE",
    "Tap":          "SELECT ITEM",
    "Expand":       "ZOOM IN / OPEN PANEL",
    "Pinch":        "ZOOM OUT / CLOSE PANEL",
    "Rotation CW":  "INCREASE VALUE",
    "Rotation CCW": "DECREASE VALUE",
    "Swipe Right":  "NEXT PANEL",
    "Swipe Left":   "PREVIOUS PANEL",
    "Swipe Up":     "SCROLL UP / BOOST",
    "Swipe Down":   "SCROLL DOWN / REDUCE",
    "Swipe X":      "CANCEL / REJECT",
    "Swipe V":      "MARK COMPLETE",
    "Swipe +":      "ADD NEW ITEM",
    "Shake":        "EMERGENCY STOP",
}

SPACECRAFT_PANELS = [
    "Navigation",
    "Life Support",
    "Communications",
    "Power Systems",
    "Diagnostics",
]

COLORS = {
    "bg_primary":    "#0a0a1a", #black
    "bg_secondary":  "#1a1a2e", #dark blue
    "accent_blue":   "#0f3460", #blue
    "accent_red":    "#e94560", #red
    "accent_green":  "#00d4aa", #teal
    "accent_yellow": "#f5c842", #yellow
    "text_primary":  "#e0e0e0", #white
    "text_secondary":"#a0a0b0", #grey
    "warning":       "#ff6b35", #orange
    "success":       "#00e676", #green
    "danger":        "#ff1744", #red
}

MEDIAPIPE_CONFIG = {
    "max_num_hands":        1,
    "min_detection_confidence": 0.7,
    "min_tracking_confidence":  0.5,
}

ADVERSARIAL_DEFAULTS = {
    "lighting_intensity":    0.0,
    "noise_intensity":       0.0,
    "latency_ms":            0,
    "glove_blur_radius":     0,
    "microgravity_jitter":   0.0,
}