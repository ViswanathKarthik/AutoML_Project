"""Configuration settings for AutoML project"""

# Feature Selection
MI_THRESHOLD = 0.01
CORRELATION_THRESHOLD = 0.95

CATEGORICAL_THRESHOLD = 20

MODEL_TOP_K = 30

RANDOM_STATE = 42
# Model Training
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Text/Categorical Features
TEXT_FEATURES = ['name', 'title', 'description', 'text']  # Features likely to be text
ID_FEATURES = ['id', 'customer_id', 'user_id', 'product_id']  # ID features
ADDRESS_FEATURES = ['address', 'location', 'city', 'state', 'country', 'postal_code']

# Output Settings
OUTPUT_FORMATS = ['json', 'csv', 'html', 'pdf']
REPORT_DIR = 'output/reports'
METRICS_DIR = 'output/metrics'

# Models to try
CLASSIFICATION_MODELS = [
    'logistic_regression',
    'random_forest',
    'gradient_boosting',
    'svm'
]

REGRESSION_MODELS = [
    'linear_regression',
    'ridge_regression',
    'random_forest',
    'gradient_boosting'
]
