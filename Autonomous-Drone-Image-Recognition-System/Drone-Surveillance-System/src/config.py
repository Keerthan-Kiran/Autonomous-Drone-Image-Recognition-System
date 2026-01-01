
# --- MODEL SETTINGS ---
# Using YOLOv8 Medium as requested for high accuracy/balance
MODEL_NAME = "yolov8m.pt" 

# Inference Resolution:
# Standard YOLO is 640px. We use 1280px to see small cars/people clearly.
INFERENCE_SIZE = 1280
CONF_THRESHOLD = 0.25

# --- LOCKING LOGIC ---
# If an object covers >5% of the screen, we lock it.
LOCK_SIZE_THRESHOLD = 0.05

# How many frames to buffer before the AI learns what is "Normal"
ANOMALY_BUFFER_SIZE = 50

# How long (in frames) to keep a target locked
LOCK_DURATION_FRAMES = 60
