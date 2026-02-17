# ============================================================
# EXPERIMENT CONTROL PANEL
# ============================================================

CNN_MODE = "2D"          # "1D" or "2D"

# ----- Input Embeddings -----
USE_WEIGHTED = True
USE_DNA2VEC  = False

# ----- Decision Vector (ONLY ONE allowed) -----
USE_DECISION = "soft"  # "soft", "hard", or None

# ----- Training -----
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-2
PATIENCE = 5
DROPOUT = 0.5
WEIGHT=1e-3
