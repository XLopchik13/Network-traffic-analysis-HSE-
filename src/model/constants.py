"""Constants for the salary prediction model."""

# --- GBR model (salary regression) ---
MODEL_FILENAME = "salary_model.pkl"
RESOURCES_DIR_NAME = "resources"

DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 6
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42

# Salary range considered valid for training (rubles).
# Values outside this range are treated as data entry errors.
SALARY_MIN = 5_000
SALARY_MAX = 500_000

# --- Neural network (FCN) ---
PYTORCH_MODEL_FILENAME = "fcn_model.pt"

# Wider architecture + no dropout: BN alone regularises sufficiently for
# 38K samples with 36 features.  High dropout (0.3) caused underfitting.
NN_HIDDEN_DIMS = [512, 256, 128, 64]
NN_DROPOUT_RATE = 0.0
NN_LEARNING_RATE = 5e-4
NN_WEIGHT_DECAY = 1e-4
NN_BATCH_SIZE = 512
NN_EPOCHS = 300
NN_PATIENCE = 30          # early-stopping patience (epochs without improvement)
NN_LR_PATIENCE = 15       # epochs before reducing learning rate
NN_LR_FACTOR = 0.5        # learning rate reduction factor

# --- MLflow ---
MLFLOW_TRACKING_URI = "http://kamnsv.com:55000/"
MLFLOW_EXPERIMENT_NAME = "LIne Regression HH"
MLFLOW_MODEL_NAME = "petrov_nikita_nikolaevich_fcn"
