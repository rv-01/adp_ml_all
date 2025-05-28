# Advanced Adaptive Data Quality Framework
## Complete Technical Documentation

### Version: 2.0.0
### Last Updated: December 2024

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Theoretical Foundation](#theoretical-foundation)
3. [System Architecture](#system-architecture)
4. [Algorithm Deep Dive](#algorithm-deep-dive)
5. [Installation Guide](#installation-guide)
6. [User Manual](#user-manual)
7. [Configuration Reference](#configuration-reference)
8. [API Documentation](#api-documentation)
9. [Advanced Features](#advanced-features)
10. [Performance Tuning](#performance-tuning)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Extension Development](#extension-development)
13. [Best Practices](#best-practices)
14. [Appendices](#appendices)

---

## Executive Summary

### What is Adaptive Data Quality?

Traditional data quality systems rely on fixed, rule-based approaches that check for predefined violations like "age must be between 0 and 120" or "email must contain @". While effective for obvious errors, these systems have fundamental limitations:

- **Static Rules**: Cannot adapt to evolving data patterns
- **Limited Coverage**: Miss subtle anomalies that don't violate explicit rules
- **High Maintenance**: Require constant rule updates as data changes
- **No Learning**: Cannot improve accuracy based on experience

The Advanced Adaptive Data Quality Framework represents a paradigm shift toward **intelligent, self-improving data quality** that combines:

- **Machine Learning**: Learns what "normal" data looks like automatically
- **Ensemble Methods**: Uses multiple algorithms for comprehensive coverage
- **Human-AI Collaboration**: Incorporates domain expertise through feedback
- **Adaptive Learning**: Continuously improves based on real-world performance

### Key Benefits

**For Data Scientists:**
- Catches subtle anomalies traditional rules miss
- Provides explainable results for model debugging
- Reduces false positives through adaptive learning
- Enables proactive data quality monitoring

**For Data Engineers:**
- Modular architecture for easy integration
- Comprehensive audit trail for compliance
- Scalable ensemble approach
- Real-time adaptation to data drift

**For Business Users:**
- Clear explanations for every quality flag
- Prioritized issues based on confidence levels
- Actionable recommendations for resolution
- Continuous improvement without technical intervention

### Business Impact

Organizations using adaptive data quality typically see:
- **40-60% reduction** in false positive alerts
- **25-35% improvement** in anomaly detection accuracy
- **50-70% reduction** in manual rule maintenance
- **Significant cost savings** from prevented data quality issues

---

## Theoretical Foundation

### Ensemble Learning Theory

The framework implements **heterogeneous ensemble learning**, combining multiple diverse algorithms to achieve superior performance than any single method. This approach is based on several key principles:

#### Diversity Principle
Different algorithms excel at detecting different types of anomalies:
- **Tree-based methods** (Isolation Forest) → General outliers and feature interactions
- **Density-based methods** (LOF, DBSCAN) → Local anomalies and clustering violations
- **Boundary-based methods** (One-Class SVM) → Complex non-linear patterns
- **Reconstruction-based methods** (Autoencoder) → High-dimensional pattern violations
- **Statistical methods** (Z-score, IQR) → Distribution-based outliers
- **Temporal methods** (Time-series analysis) → Seasonal and trend anomalies

#### Weighted Voting
Each algorithm contributes to the final decision based on its historical performance:

```
Final_Score = Σ(Algorithm_i_Score × Weight_i)
```

Where weights are dynamically adjusted based on feedback:

```
New_Weight_i = (1 - α) × Old_Weight_i + α × Performance_i
```

#### Consensus Thresholding
A record is flagged only when multiple algorithms agree, reducing false positives:

```
Flag_Record = (Ensemble_Score > θ₁) AND (Consensus_Rate > θ₂)
```

### Adaptive Learning Theory

The system implements **online learning** with **exponential forgetting**, giving more weight to recent performance:

#### Performance Tracking
Each algorithm's accuracy is tracked using a sliding window:

```
Accuracy_i(t) = Correct_Predictions_i / Total_Predictions_i (last N samples)
```

#### Weight Update Rule
Algorithm weights are updated using exponential moving average:

```
w_i(t+1) = (1-λ)w_i(t) + λ × RelativePerformance_i(t)
```

Where λ is the learning rate and RelativePerformance is normalized accuracy.

#### Threshold Adaptation
Detection thresholds adapt based on feedback patterns:

```
If FalsePositiveRate > τ₁: Increase thresholds (more conservative)
If FalseNegativeRate > τ₂: Decrease thresholds (more sensitive)
```

### Statistical Foundation

#### Population Stability Index (PSI)
Used for drift detection:

```
PSI = Σ[(P_current - P_reference) × ln(P_current / P_reference)]
```

Where P represents probability distributions of feature values.

#### Kolmogorov-Smirnov Test
Tests whether two samples come from the same distribution:

```
D = max|F₁(x) - F₂(x)|
```

Used to detect significant changes in data distributions.

---

## System Architecture

### High-Level Architecture

The framework follows a modular, microservices-inspired architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface                        │
│                  (Flask Application)                    │
├─────────────────────────────────────────────────────────┤
│                    API Layer                           │
│              (REST Endpoints)                          │
├─────────────────────────────────────────────────────────┤
│   Data Ingestion  │  Ensemble Engine  │  Explainability │
│                   │                   │                 │
├───────────────────┼───────────────────┼─────────────────┤
│      Audit Store  │  Configuration    │  Human Feedback │
│                   │   Management      │     Loop        │
├─────────────────────────────────────────────────────────┤
│              External Dependencies                      │
│    scikit-learn, TensorFlow, pandas, Flask            │
└─────────────────────────────────────────────────────────┘
```

### Core Components

#### 1. Data Ingestion Module (`src/data_ingestion/`)

**Purpose**: Standardize data loading and preprocessing across different sources.

**Key Classes**:
- `DataLoader`: Main interface for data loading
- `FileProcessor`: Handles CSV/Parquet files
- `DatabaseConnector`: Database integration (future)

**Features**:
- Automatic file format detection
- Robust encoding handling
- Type inference and conversion
- Missing value standardization
- Metadata enrichment

**Example Usage**:
```python
loader = DataLoader()
df = loader.load_from_upload(file_object)
# Returns preprocessed DataFrame with standardized columns
```

#### 2. Ensemble Detection Module (`src/advanced_ml/`)

**Purpose**: Core intelligence of the system - runs multiple algorithms and combines results.

**Key Classes**:
- `AdvancedEnsembleDetector`: Main ensemble orchestrator
- `DriftDetector`: Monitors data distribution changes
- `TemporalAnomalyDetector`: Specialized time-series analysis

**Algorithm Pipeline**:
1. **Feature Engineering**: Creates algorithm-specific representations
2. **Parallel Execution**: Runs all 7 algorithms simultaneously
3. **Score Normalization**: Converts algorithm outputs to 0-1 probabilities
4. **Weighted Voting**: Combines scores using dynamic weights
5. **Consensus Check**: Validates agreement between algorithms

#### 3. Explainability Module (`src/explainability/`)

**Purpose**: Generate human-readable explanations for all quality flags.

**Key Classes**:
- `ExplainabilityEngine`: Main explanation generator
- `FeatureContributor`: Calculates feature importance
- `ActionRecommender`: Suggests remediation actions

**Explanation Types**:
- **Root Cause**: Which features triggered the anomaly
- **Confidence**: How certain the system is about the flag
- **Context**: How this compares to normal patterns
- **Actions**: Specific steps to investigate or resolve

#### 4. Audit Module (`src/audit/`)

**Purpose**: Complete governance and compliance tracking.

**Key Classes**:
- `AuditStore`: Database interface for all tracking
- `SessionManager`: Manages review workflows
- `MetricsCalculator`: Performance and trend analysis

**Tracked Information**:
- Every quality check run with full results
- All human feedback decisions with timestamps
- Algorithm performance over time
- System configuration changes
- Data drift events and responses

### Data Flow Architecture

```
Raw Data → Preprocessing → Feature Engineering → Ensemble Detection
                                                       ↓
Dashboard ← Audit Store ← Human Review ← Explainability Engine
    ↑                         ↓
Performance Metrics    Adaptive Learning
```

**Detailed Flow**:
1. **Data Input**: User uploads CSV/Parquet file
2. **Preprocessing**: Standardization, type detection, metadata addition
3. **Feature Engineering**: Create multiple representations for different algorithms
4. **Ensemble Detection**: 7 algorithms analyze data in parallel
5. **Weighted Voting**: Combine results using learned weights
6. **Explainability**: Generate human-readable explanations
7. **Human Review**: User provides feedback on flagged records
8. **Adaptive Learning**: System updates weights and thresholds
9. **Audit Trail**: All actions logged for compliance
10. **Dashboard Update**: Real-time metrics and visualizations

---

## Algorithm Deep Dive

### 1. Isolation Forest (Tree-Based)

**Purpose**: Detects anomalies by isolating observations through random splits.

**Theory**: Anomalies are easier to isolate than normal points because they're few and different. The algorithm builds random trees that split the data - anomalous points require fewer splits to isolate.

**Parameters**:
```python
IsolationForest(
    n_estimators=200,      # Number of trees
    contamination=0.1,     # Expected anomaly rate
    max_features=0.8,      # Feature sampling rate
    bootstrap=True,        # Sample with replacement
    random_state=42        # Reproducibility
)
```

**Strengths**:
- Excellent for general outliers
- Handles mixed data types well
- Computationally efficient
- Good feature interaction detection

**Weaknesses**:
- Less effective for local anomalies
- Sensitive to feature scaling
- May miss subtle patterns

**Score Interpretation**:
- Raw scores are negative (more negative = more anomalous)
- Converted to probability: `p = 1 / (1 + exp(2 × score))`

### 2. Local Outlier Factor (Density-Based)

**Purpose**: Identifies points with unusual local density compared to their neighbors.

**Theory**: Compares the local density of a point to the local densities of its neighbors. Points with substantially lower density than neighbors are considered anomalies.

**Parameters**:
```python
LocalOutlierFactor(
    n_neighbors=20,        # Number of neighbors to consider
    contamination=0.1,     # Expected anomaly rate
    metric='minkowski',    # Distance metric
    novelty=True          # Allow prediction on new data
)
```

**Mathematical Foundation**:
```
LOF(p) = Σ(LRD(o) / LRD(p)) / |N_k(p)|
```
Where LRD is Local Reachability Density and N_k(p) are k-nearest neighbors.

**Strengths**:
- Excellent for local anomalies
- Adapts to varying densities
- Good for clustering violations
- Robust to global outliers

**Weaknesses**:
- Computationally expensive O(n²)
- Sensitive to parameter choice
- Struggles with high dimensions

### 3. One-Class SVM (Boundary-Based)

**Purpose**: Learns a decision boundary around normal data using support vector machines.

**Theory**: Maps data to high-dimensional space using kernel functions and finds the optimal hyperplane that separates normal data from the origin with maximum margin.

**Parameters**:
```python
OneClassSVM(
    kernel='rbf',          # Radial basis function kernel
    gamma='scale',         # Kernel coefficient
    nu=0.1                # Upper bound on anomaly fraction
)
```

**Kernel Function (RBF)**:
```
K(x, y) = exp(-γ ||x - y||²)
```

**Strengths**:
- Handles complex non-linear patterns
- Memory efficient (uses support vectors)
- Good theoretical foundation
- Robust to outliers during training

**Weaknesses**:
- Requires parameter tuning
- Computationally expensive for large datasets
- Black box (limited interpretability)

### 4. Deep Autoencoder (Neural Network)

**Purpose**: Uses neural networks to learn compressed representations of normal data.

**Theory**: An autoencoder learns to reconstruct input data through a bottleneck layer. High reconstruction error indicates anomalous patterns the network hasn't learned.

**Architecture**:
```python
# Encoder: Gradually compress information
input_layer → Dense(64) → Dense(32) → Dense(16)
                ↓
# Decoder: Reconstruct original data  
Dense(16) → Dense(32) → Dense(64) → output_layer
```

**Loss Function**:
```
Loss = MSE(X, X_reconstructed) = Σ(x_i - x̂_i)² / n
```

**Training Process**:
1. Feed normal data through encoder-decoder
2. Calculate reconstruction error
3. Backpropagate to minimize error
4. Repeat for multiple epochs

**Strengths**:
- Captures complex multi-dimensional patterns
- Learns hierarchical representations
- Good for high-dimensional data
- Can detect subtle anomalies

**Weaknesses**:
- Requires substantial training data
- Black box (limited interpretability)
- Computationally intensive
- Hyperparameter sensitive

### 5. Statistical Methods (Classical)

**Purpose**: Detect outliers using traditional statistical measures.

**Methods Implemented**:

#### Z-Score Detection
```python
z_score = |x - μ| / σ
# Flag if z_score > 3 (99.7% confidence interval)
```

#### IQR Method
```python
Q1 = 25th percentile
Q3 = 75th percentile
IQR = Q3 - Q1
outlier = x < (Q1 - 1.5×IQR) OR x > (Q3 + 1.5×IQR)
```

**Strengths**:
- Highly interpretable
- Fast computation
- Well-understood theory
- Good for obvious outliers

**Weaknesses**:
- Assumes normal distribution
- Only detects univariate outliers
- Sensitive to extreme values
- Limited to numerical data

### 6. Density Clustering (DBSCAN)

**Purpose**: Identifies points that don't belong to any dense cluster.

**Theory**: Groups points into clusters based on density. Points that cannot be assigned to any cluster (noise points) are considered anomalies.

**Parameters**:
```python
DBSCAN(
    eps=0.5,              # Maximum distance between neighbors
    min_samples=5         # Minimum points to form cluster
)
```

**Algorithm Steps**:
1. For each point, find all neighbors within eps distance
2. If point has min_samples neighbors, start new cluster
3. Recursively add density-connected points to cluster
4. Points not in any cluster are noise (anomalies)

**Automatic eps Selection**:
```python
# Use k-distance graph knee point
distances = k_nearest_neighbors(data, k=5)
eps = np.percentile(distances, 90)
```

**Strengths**:
- No assumption about cluster shapes
- Automatically determines cluster count
- Robust to outliers
- Good for irregular patterns

**Weaknesses**:
- Sensitive to parameters
- Struggles with varying densities
- Difficulty with high dimensions
- Can miss isolated anomalies

### 7. Temporal Anomaly Detection

**Purpose**: Detects anomalies in time-series patterns and temporal relationships.

**Methods**:

#### Moving Average Detection
```python
rolling_mean = data.rolling(window=20).mean()
rolling_std = data.rolling(window=20).std()
z_score = |data - rolling_mean| / rolling_std
anomaly = z_score > 3
```

#### Seasonal Decomposition
```python
# Decompose time series into trend, seasonal, residual
trend, seasonal, residual = seasonal_decompose(data)
# Detect anomalies in residual component
```

#### Change Point Detection
- Identifies sudden changes in statistical properties
- Uses CUSUM (Cumulative Sum) algorithms
- Detects regime changes and level shifts

**Strengths**:
- Specialized for temporal patterns
- Handles seasonality and trends
- Good for time-series data
- Can detect subtle temporal shifts

**Weaknesses**:
- Requires temporal ordering
- Limited to time-series data
- Parameter sensitive
- May miss non-temporal anomalies

---

## Installation Guide

### System Requirements

**Minimum Requirements**:
- Python 3.8+
- 4GB RAM
- 2GB disk space
- Modern web browser

**Recommended Requirements**:
- Python 3.9+
- 8GB+ RAM
- 10GB+ disk space
- Multi-core CPU for faster processing

**Operating System Support**:
- Windows 10/11
- macOS 10.14+
- Linux (Ubuntu 18.04+, CentOS 7+)

### Step-by-Step Installation

#### 1. Environment Setup

```bash
# Create project directory
mkdir adaptive_data_quality
cd adaptive_data_quality

# Create virtual environment
python -m venv adq_env

# Activate virtual environment
# Windows:
adq_env\Scripts\activate
# macOS/Linux:
source adq_env/bin/activate
```

#### 2. Install Dependencies

Create `requirements.txt`:
```text
Flask==2.3.3
pandas==2.1.0
scikit-learn==1.3.0
numpy==1.24.3
plotly==5.15.0
scipy==1.11.0
tensorflow==2.13.0
statsmodels==0.14.0
pyod==1.1.0
Werkzeug==2.3.7
Jinja2==3.1.2
pyarrow==13.0.0
```

Install dependencies:
```bash
pip install -r requirements.txt
```

#### 3. Project Structure Setup

```bash
# Create directory structure
mkdir -p src/{data_ingestion,advanced_ml,explainability,audit}
mkdir -p {templates,static,sample_data,model_cache}

# Create __init__.py files
touch src/__init__.py
touch src/data_ingestion/__init__.py
touch src/advanced_ml/__init__.py  
touch src/explainability/__init__.py
touch src/audit/__init__.py
```

#### 4. Copy Framework Files

Copy all the provided Python files into their respective directories:
- `config.py` → root directory
- `app.py` → root directory
- Module files → corresponding `src/` subdirectories
- HTML templates → `templates/` directory

#### 5. Generate Sample Data

```bash
python generate_sample_data.py
```

#### 6. Initialize Database

```bash
python -c "
from src.audit.store import AuditStore
from config import AdvancedConfig
store = AuditStore(AdvancedConfig.AUDIT_DB_PATH)
store.initialize_db()
print('Database initialized successfully')
"
```

#### 7. Test Installation

```bash
python app.py
```

Expected output:
```
🚀 Advanced Adaptive Data Quality Framework
📊 Ensemble Algorithms: 7
🧠 Machine Learning: Multi-Algorithm Voting System
🔄 Adaptive Learning: Real-time weight adjustment

🌐 Starting Flask server...
   Open http://localhost:5000 in your browser
```

### Docker Installation (Alternative)

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t adaptive-dq .
docker run -p 5000:5000 adaptive-dq
```

### Troubleshooting Installation

**Common Issues**:

1. **TensorFlow Installation Errors**:
   ```bash
   # Try specific version
   pip install tensorflow==2.13.0 --no-cache-dir
   ```

2. **Permission Errors on Windows**:
   ```bash
   # Run as administrator or use --user flag
   pip install --user -r requirements.txt
   ```

3. **Memory Issues**:
   ```bash
   # Reduce TensorFlow memory usage
   export TF_FORCE_GPU_ALLOW_GROWTH=true
   ```

4. **Port 5000 in Use**:
   ```python
   # Change port in app.py
   app.run(debug=True, host='0.0.0.0', port=5001)
   ```

---

## User Manual

### Getting Started

#### 1. Accessing the System

1. Start the application: `python app.py`
2. Open browser to `http://localhost:5000`
3. You'll see the main dashboard with system overview

#### 2. Understanding the Dashboard

The dashboard provides real-time insights into system performance:

**Key Metrics**:
- **Ensemble Runs**: Total number of quality assessments performed
- **Records Processed**: Total data points analyzed
- **Anomalies Found**: Total anomalies detected across all runs
- **Active Algorithms**: Number of detection algorithms currently running

**Algorithm Weights Chart**:
- Shows current influence of each algorithm
- Updates in real-time as system learns
- Hover for detailed weight information

**System Status Panel**:
- Learning sample count (need 10+ for adaptation)
- System health indicators
- Quick access to data upload

### Data Upload and Analysis

#### 1. Preparing Your Data

**Supported Formats**:
- CSV files (any encoding)
- Parquet files
- Maximum file size: 16MB

**Data Requirements**:
- At least 50+ records recommended
- Mixed data types (numerical, categorical, dates) work best
- Column headers should be descriptive

**Example Good Data Structure**:
```csv
customer_id,transaction_amount,transaction_date,customer_age,account_balance,email
1001,245.50,2024-01-15,34,15000.00,john@example.com
1002,89.25,2024-01-15,28,8500.50,jane@example.com
```

#### 2. Upload Process

1. **Click "Upload Data"** in navigation menu
2. **Select your file** using the file picker
3. **Click "Run Ensemble Analysis"**
4. **Wait for processing** (typically 10-60 seconds depending on data size)

**Processing Steps** (automatic):
- File validation and loading
- Data type detection and standardization
- Feature engineering for different algorithms
- Parallel execution of 7 detection algorithms
- Weighted ensemble voting and consensus checking
- Explanation generation for flagged records

#### 3. Understanding Results

After processing, you'll be redirected to the review page showing:

**Summary Statistics**:
- Total anomalies found
- Breakdown by confidence level (High/Medium/Low)

**Individual Anomaly Details**:
- **Record Index**: Position in original dataset
- **Confidence Level**: System certainty about the anomaly
- **Ensemble Score**: Weighted probability (0-1)
- **Detection Summary**: Why it was flagged
- **Analysis Details**: Specific reasons from algorithms
- **Recommended Actions**: Suggested next steps

### Providing Feedback

Feedback is crucial for system learning and accuracy improvement.

#### 1. Feedback Options

For each flagged record, you can choose:

**✅ Confirm Anomaly**:
- This IS a real data quality issue
- Increases confidence in algorithms that flagged it
- Helps system learn your quality standards

**❌ False Positive**:
- This is actually normal data
- Reduces influence of algorithms that incorrectly flagged it
- Improves precision over time

**🚨 Critical Issue**:
- This requires immediate attention
- Highest priority for your team
- Helps system learn severity levels

#### 2. Providing Effective Feedback

**Best Practices**:
- Review at least 10-20 flagged records per session
- Be consistent in your decision criteria
- Add comments for complex cases
- Focus on borderline cases (medium confidence)

**Example Feedback Scenarios**:

```
Scenario 1: Transaction amount of $50,000
- If normal for your business → "False Positive"
- If suspicious for this customer → "Confirm Anomaly"
- If requires urgent investigation → "Critical Issue"

Scenario 2: Missing email address
- If acceptable in your process → "False Positive"  
- If violates data requirements → "Confirm Anomaly"

Scenario 3: Age of 25 for senior discount
- Clear data entry error → "Confirm Anomaly"
- System integration issue → "Critical Issue"
```

#### 3. Tracking Learning Progress

Monitor system improvement through:
- **Algorithm weights** changing based on performance
- **Precision/Recall metrics** improving over time
- **Fewer false positives** in subsequent runs
- **Better anomaly prioritization**

### Advanced Usage

#### 1. Interpreting Algorithm Votes

Each algorithm contributes to the final decision:

**Tree-Based (Isolation Forest)**:
- Good for: General outliers, feature interactions
- High vote: Record has unusual feature combinations

**Density-Based (LOF, DBSCAN)**:
- Good for: Local anomalies, clustering violations
- High vote: Record doesn't fit neighborhood patterns

**Boundary-Based (One-Class SVM)**:
- Good for: Complex non-linear patterns
- High vote: Record outside learned decision boundary

**Neural (Autoencoder)**:
- Good for: Complex multi-dimensional patterns
- High vote: High reconstruction error

**Statistical (Z-score, IQR)**:
- Good for: Obvious numerical outliers
- High vote: Values far from distribution center

**Temporal**:
- Good for: Time-series anomalies
- High vote: Unusual temporal patterns

#### 2. Custom Thresholds

You can influence system sensitivity through consistent feedback:

**For Higher Precision** (fewer false positives):
- Consistently mark borderline cases as "False Positive"
- System will increase confidence thresholds

**For Higher Recall** (catch more anomalies):
- Mark missed anomalies as "Critical Issues"
- System will decrease confidence thresholds

#### 3. Data Drift Monitoring

The system automatically monitors for data drift:
- **Population Stability Index** changes
- **Distribution shifts** in key features
- **Performance degradation** of algorithms

When drift is detected:
- Dashboard shows drift alerts
- Algorithm weights may be reset
- Recommendation for model retraining

---

## Configuration Reference

### Basic Configuration (`config.py`)

```python
class AdvancedConfig:
    # Flask Application Settings
    SECRET_KEY = 'your-secret-key'
    DEBUG = True  # Set to False in production
    
    # Database Configuration
    AUDIT_DB_PATH = 'advanced_audit.db'
    MODEL_CACHE_PATH = 'model_cache/'
    
    # File Upload Settings
    MAX_UPLOAD_SIZE = 16 * 1024 * 1024  # 16MB
    ALLOWED_EXTENSIONS = {'csv', 'parquet'}
    
    # Ensemble Algorithm Configuration
    ENSEMBLE_ALGORITHMS = [
        'isolation_forest',
        'local_outlier_factor', 
        'one_class_svm',
        'autoencoder',
        'statistical_outlier',
        'density_clustering',
        'temporal_anomaly'
    ]
    
    # Initial Algorithm Weights
    INITIAL_WEIGHTS = {
        'isolation_forest': 0.15,
        'local_outlier_factor': 0.20,
        'one_class_svm': 0.15,
        'autoencoder': 0.25,
        'statistical_outlier': 0.10,
        'density_clustering': 0.10,
        'temporal_anomaly': 0.05
    }
    
    # Adaptive Learning Parameters
    FEEDBACK_LEARNING_RATE = 0.01
    MIN_FEEDBACK_SAMPLES = 10
    ENSEMBLE_UPDATE_FREQUENCY = 100
    
    # Detection Thresholds
    HIGH_CONFIDENCE_THRESHOLD = 0.8
    MEDIUM_CONFIDENCE_THRESHOLD = 0.6
    CONSENSUS_THRESHOLD = 0.7
    
    # Drift Detection
    DRIFT_DETECTION_WINDOW = 1000
    DRIFT_THRESHOLD = 0.05
    PSI_THRESHOLD = 0.2
```

### Algorithm-Specific Parameters

#### Isolation Forest
```python
ISOLATION_FOREST_PARAMS = {
    'n_estimators': 200,      # Number of trees
    'contamination': 0.1,     # Expected anomaly rate
    'max_samples': 'auto',    # Samples per tree
    'max_features': 0.8,      # Feature sampling rate
    'bootstrap': True,        # Sample with replacement
    'random_state': 42        # Reproducibility
}
```

#### Local Outlier Factor
```python
LOF_PARAMS = {
    'n_neighbors': 20,        # Neighbors to consider
    'contamination': 0.1,     # Expected anomaly rate
    'metric': 'minkowski',    # Distance metric
    'p': 2,                   # Minkowski parameter
    'novelty': True          # Enable prediction mode
}
```

#### One-Class SVM
```python
SVM_PARAMS = {
    'kernel': 'rbf',          # Kernel type
    'gamma': 'scale',         # Kernel coefficient
    'nu': 0.1,               # Anomaly fraction upper bound
    'shrinking': True,        # Use shrinking heuristic
    'cache_size': 200         # Kernel cache size (MB)
}
```

#### Deep Autoencoder
```python
AUTOENCODER_PARAMS = {
    'encoding_dims': [64, 32, 16],  # Layer sizes
    'dropout_rate': 0.1,            # Dropout for regularization
    'learning_rate': 0.001,         # Adam optimizer rate
    'epochs': 50,                   # Training epochs
    'batch_size': 32,               # Mini-batch size
    'validation_split': 0.1         # Validation data fraction
}
```

### Environment Variables

Set these for production deployment:

```bash
# Security
export SECRET_KEY="your-production-secret-key"
export FLASK_ENV="production"

# Database
export AUDIT_DB_PATH="/data/audit.db"
export MODEL_CACHE_PATH="/data/models/"

# Performance
export FLASK_WORKERS=4
export MAX_UPLOAD_SIZE=33554432  # 32MB

# Logging
export LOG_LEVEL="INFO"
export LOG_FILE="/logs/adq.log"
```

### Advanced Configuration

#### Custom Algorithm Weights
```python
# Set weights based on your domain knowledge
DOMAIN_SPECIFIC_WEIGHTS = {
    # For financial data - emphasize statistical methods
    'financial': {
        'statistical_outlier': 0.30,
        'isolation_forest': 0.25,
        'local_outlier_factor': 0.20,
        'autoencoder': 0.15,
        'one_class_svm': 0.10
    },
    
    # For IoT sensor data - emphasize temporal methods
    'iot_sensors': {
        'temporal_anomaly': 0.35,
        'autoencoder': 0.25,
        'statistical_outlier': 0.20,
        'isolation_forest': 0.20
    },
    
    # For customer data - balanced approach
    'customer': {
        'local_outlier_factor': 0.25,
        'autoencoder': 0.20,
        'isolation_forest': 0.20,
        'statistical_outlier': 0.15,
        'one_class_svm': 0.10,
        'density_clustering': 0.10
    }
}
```

#### Learning Rate Schedules
```python
# Adaptive learning rate based on feedback quality
def adaptive_learning_rate(feedback_count, accuracy):
    base_rate = 0.01
    if accuracy > 0.9:
        return base_rate * 1.5  # Learn faster when doing well
    elif accuracy < 0.7:
        return base_rate * 0.5  # Learn slower when struggling
    return base_rate
```

---

## API Documentation

### REST Endpoints

#### 1. Data Upload and Processing

**POST** `/upload`
- **Purpose**: Upload data file and run ensemble analysis
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file`: Data file (CSV or Parquet)
- **Response**: Redirect to review page or error message

**Example**:
```bash
curl -X POST \
  -F "file=@dataset.csv" \
  http://localhost:5000/upload
```

#### 2. Feedback Submission

**POST** `/api/feedback`
- **Purpose**: Submit human feedback on flagged records
- **Content-Type**: `application/json`
- **Request Body**:
```json
{
  "session_id": "uuid-string",
  "record_id": 123,
  "action": "confirm|dismiss|escalate",
  "comment": "Optional explanation"
}
```
- **Response**:
```json
{
  "status": "success",
  "updated_weights": {
    "isolation_forest": 0.16,
    "local_outlier_factor": 0.21,
    ...
  }
}
```

#### 3. Performance Metrics

**GET** `/api/algorithm-performance`
- **Purpose**: Get current algorithm performance and weights
- **Response**:
```json
{
  "algorithm_weights": {
    "isolation_forest": 0.15,
    "local_outlier_factor": 0.20,
    ...
  },
  "algorithm_performance": {
    "isolation_forest": [1.0, 0.8, 1.0, ...],
    ...
  },
  "ensemble_metrics": {
    "accuracy": 0.87,
    "precision": 0.89,
    "recall": 0.85,
    "f1_score": 0.87
  },
  "total_feedback": 156
}
```

#### 4. Quality Trends

**GET** `/api/quality-trends`
- **Purpose**: Get historical quality metrics for dashboard
- **Response**:
```json
{
  "dates": ["2024-01-01", "2024-01-02", ...],
  "runs": [5, 3, 8, ...],
  "records": [1000, 500, 2000, ...],
  "flagged": [45, 12, 89, ...],
  "flag_rates": [4.5, 2.4, 4.45, ...]
}
```

### Python API

#### Core Classes

##### EnsembleDetector
```python
from src.advanced_ml.ensemble_detector import AdvancedEnsembleDetector
from config import AdvancedConfig

# Initialize detector
detector = AdvancedEnsembleDetector(AdvancedConfig)

# Run detection
results = detector.detect_anomalies(dataframe)

# Access results
anomaly_indices = results['flagged_indices']
confidence_levels = results['confidence_levels']
algorithm_votes = results['algorithm_votes']
```

##### DataLoader
```python
from src.data_ingestion.loaders import DataLoader

# Initialize loader
loader = DataLoader()

# Load from file
df = loader.load_from_upload(file_object)

# Load from database (future feature)
df = loader.load_from_database(connection_string, query)
```

##### AuditStore
```python
from src.audit.store import AuditStore

# Initialize store
store = AuditStore('audit.db')
store.initialize_db()

# Record quality check
session_id = store.record_advanced_quality_check(
    dataset_name='data.csv',
    total_records=1000,
    results=detection_results,
    timestamp=datetime.now()
)

# Record feedback
store.record_human_feedback(
    session_id=session_id,
    record_id=123,
    action='confirm',
    comment='Clear data error',
    timestamp=datetime.now()
)
```

#### Custom Algorithm Integration

```python
class CustomAnomalyDetector:
    def __init__(self, **params):
        self.params = params
        
    def fit_predict(self, X):
        # Implement your algorithm
        predictions = your_algorithm(X)
        return predictions  # -1 for anomaly, 1 for normal
        
    def score_samples(self, X):
        # Return anomaly scores
        scores = your_scoring_function(X)
        return scores

# Register custom algorithm
detector.register_algorithm('custom_method', CustomAnomalyDetector())
```

---

## Advanced Features

### 1. Data Drift Detection

The system continuously monitors for changes in data distribution that could affect model performance.

#### Statistical Methods

**Population Stability Index (PSI)**:
- Measures distribution changes between reference and current data
- PSI > 0.2 indicates significant drift
- Calculated per feature and aggregated

**Kolmogorov-Smirnov Test**:
- Tests whether two samples come from same distribution
- p-value < 0.05 indicates significant difference
- More sensitive than PSI for subtle changes

**Implementation**:
```python
class DriftDetector:
    def detect_drift(self, reference_data, current_data):
        drift_results = {}
        
        for column in common_columns:
            # PSI calculation
            psi_score = self._calculate_psi(
                reference_data[column], 
                current_data[column]
            )
            
            # KS test
            ks_stat, p_value = ks_2samp(
                reference_data[column], 
                current_data[column]
            )
            
            drift_results[column] = {
                'psi_score': psi_score,
                'ks_statistic': ks_stat,
                'ks_p_value': p_value,
                'drift_detected': psi_score > 0.2 or p_value < 0.05
            }
            
        return drift_results
```

#### Automated Response

When drift is detected:
1. **Alert Generation**: Dashboard shows drift warnings
2. **Weight Adjustment**: Reduce confidence in affected algorithms
3. **Retraining Recommendation**: Suggest model updates
4. **Threshold Adaptation**: Temporarily relax detection thresholds

### 2. Temporal Anomaly Detection

Specialized detection for time-series data with seasonal patterns.

#### Methods Implemented

**Seasonal Decomposition**:
```python
def detect_seasonal_anomalies(self, timeseries_data):
    # Decompose into trend, seasonal, residual components
    decomposition = seasonal_decompose(
        timeseries_data, 
        model='additive',
        period=detect_seasonality(timeseries_data)
    )
    
    # Detect anomalies in residual component
    residual_anomalies = detect_outliers(decomposition.resid)
    
    return residual_anomalies
```

**Change Point Detection**:
```python
def detect_change_points(self, timeseries_data):
    # CUSUM algorithm for change point detection
    cusum_pos = np.cumsum(timeseries_data - np.mean(timeseries_data))
    cusum_neg = np.cumsum(np.mean(timeseries_data) - timeseries_data)
    
    # Detect significant changes
    threshold = 3 * np.std(timeseries_data)
    change_points = np.where(
        (cusum_pos > threshold) | (cusum_neg > threshold)
    )[0]
    
    return change_points
```

**Moving Window Anomalies**:
```python
def detect_moving_anomalies(self, timeseries_data, window_size=20):
    rolling_stats = timeseries_data.rolling(window=window_size)
    rolling_mean = rolling_stats.mean()
    rolling_std = rolling_stats.std()
    
    # Z-score based on local statistics
    z_scores = np.abs((timeseries_data - rolling_mean) / rolling_std)
    anomalies = z_scores > 3
    
    return anomalies
```

### 3. Ensemble Weight Optimization

Advanced methods for optimizing algorithm weights beyond simple performance tracking.

#### Multi-Objective Optimization

```python
def optimize_weights_pareto(self, objectives):
    """
    Optimize weights considering multiple objectives:
    - Precision (minimize false positives)
    - Recall (minimize false negatives)  
    - Efficiency (minimize computation time)
    - Interpretability (favor explainable algorithms)
    """
    
    from scipy.optimize import minimize
    
    def objective_function(weights):
        # Simulate ensemble performance with given weights
        precision, recall, efficiency, interpretability = \
            simulate_performance(weights)
            
        # Multi-objective scoring
        score = (
            0.4 * precision + 
            0.3 * recall + 
            0.2 * efficiency + 
            0.1 * interpretability
        )
        
        return -score  # Minimize negative score
    
    # Constraint: weights sum to 1
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    # Optimize
    result = minimize(
        objective_function,
        x0=current_weights,
        method='SLSQP',
        constraints=constraint,
        bounds=[(0, 1) for _ in range(len(algorithms))]
    )
    
    return result.x
```

#### Bayesian Weight Updates

```python
def bayesian_weight_update(self, prior_weights, feedback_data):
    """
    Update weights using Bayesian inference
    """
    
    # Prior distribution (Dirichlet)
    alpha_prior = prior_weights * 100  # Convert to concentration parameters
    
    # Likelihood from feedback
    for algorithm, feedback in feedback_data.items():
        successes = sum(feedback)
        failures = len(feedback) - successes
        
        # Update posterior
        alpha_prior[algorithm] += successes
        # Beta distribution conjugate to Dirichlet
    
    # Sample from posterior
    posterior_weights = np.random.dirichlet(alpha_prior)
    
    return posterior_weights
```

### 4. Explainable AI Features

Advanced explanation methods for complex ensemble decisions.

#### SHAP Integration

```python
import shap

class SHAPExplainer:
    def __init__(self, ensemble_detector):
        self.detector = ensemble_detector
        self.explainers = {}
        
    def explain_prediction(self, record_index, data):
        explanations = {}
        
        for algorithm_name, algorithm in self.detector.algorithms.items():
            if hasattr(algorithm, 'decision_function'):
                # Create SHAP explainer
                explainer = shap.Explainer(algorithm, data)
                shap_values = explainer(data.iloc[[record_index]])
                
                explanations[algorithm_name] = {
                    'shap_values': shap_values.values[0],
                    'feature_names': data.columns.tolist(),
                    'base_value': shap_values.base_values[0]
                }
                
        return explanations
```

#### Counterfactual Explanations

```python
def generate_counterfactuals(self, anomalous_record, normal_data):
    """
    Generate counterfactual explanations:
    "If field X was Y instead of Z, this would be normal"
    """
    
    counterfactuals = []
    
    for feature in anomalous_record.index:
        if anomalous_record[feature] != normal_data[feature].mode()[0]:
            # Try changing to most common value
            modified_record = anomalous_record.copy()
            modified_record[feature] = normal_data[feature].mode()[0]
            
            # Check if this makes it normal
            new_score = self.detector.predict_single(modified_record)
            
            if new_score < 0.5:  # Would be classified as normal
                counterfactuals.append({
                    'feature': feature,
                    'original_value': anomalous_record[feature],
                    'suggested_value': normal_data[feature].mode()[0],
                    'impact': anomalous_record.anomaly_score - new_score
                })
    
    return sorted(counterfactuals, key=lambda x: x['impact'], reverse=True)
```

#### Natural Language Explanations

```python
class NLGExplainer:
    def generate_explanation(self, detection_result):
        """
        Generate natural language explanation
        """
        
        record = detection_result['record_data']
        confidence = detection_result['confidence']
        algorithms = detection_result['algorithm_votes']
        
        # Base explanation
        explanation = f"This record was flagged with {confidence} confidence. "
        
        # Algorithm consensus
        flagging_algorithms = [alg for alg, score in algorithms.items() if score > 0.5]
        explanation += f"{len(flagging_algorithms)} out of 7 algorithms detected anomalies. "
        
        # Specific reasons
        if 'statistical_outlier' in flagging_algorithms:
            outlier_features = self._find_statistical_outliers(record)
            explanation += f"Statistical analysis found outliers in {', '.join(outlier_features)}. "
            
        if 'local_outlier_factor' in flagging_algorithms:
            explanation += "This record doesn't match the pattern of similar records. "
            
        if 'temporal_anomaly' in flagging_algorithms:
            explanation += "Unusual timing patterns were detected. "
            
        # Recommended actions
        if confidence == 'high':
            explanation += "Immediate investigation is recommended."
        elif confidence == 'medium':
            explanation += "Review this record when possible."
        else:
            explanation += "Monitor for similar patterns."
            
        return explanation
```

### 5. Model Persistence and Versioning

Advanced model management for production deployment.

#### Model Serialization

```python
import joblib
import pickle
from datetime import datetime

class ModelManager:
    def __init__(self, cache_path):
        self.cache_path = cache_path
        
    def save_ensemble(self, ensemble_detector, version=None):
        """
        Save ensemble model with versioning
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        model_data = {
            'algorithms': ensemble_detector.algorithms,
            'weights': ensemble_detector.algorithm_weights,
            'scalers': ensemble_detector.scalers,
            'performance_history': ensemble_detector.algorithm_performance,
            'feedback_history': ensemble_detector.feedback_history,
            'version': version,
            'timestamp': datetime.now()
        }
        
        filepath = f"{self.cache_path}/ensemble_v{version}.pkl"
        joblib.dump(model_data, filepath)
        
        return filepath
        
    def load_ensemble(self, version='latest'):
        """
        Load ensemble model by version
        """
        if version == 'latest':
            # Find most recent version
            import glob
            model_files = glob.glob(f"{self.cache_path}/ensemble_v*.pkl")
            if not model_files:
                return None
            filepath = max(model_files, key=os.path.getctime)
        else:
            filepath = f"{self.cache_path}/ensemble_v{version}.pkl"
            
        model_data = joblib.load(filepath)
        return model_data
```

#### A/B Testing Framework

```python
class ABTestManager:
    def __init__(self):
        self.test_configs = {}
        self.results = {}
        
    def create_test(self, test_name, config_a, config_b, split_ratio=0.5):
        """
        Create A/B test comparing two configurations
        """
        self.test_configs[test_name] = {
            'config_a': config_a,
            'config_b': config_b,
            'split_ratio': split_ratio,
            'start_time': datetime.now(),
            'results_a': [],
            'results_b': []
        }
        
    def assign_configuration(self, test_name, user_id):
        """
        Assign user to test group
        """
        import hashlib
        
        # Consistent assignment based on user ID
        hash_value = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
        
        test_config = self.test_configs[test_name]
        if (hash_value % 100) / 100 < test_config['split_ratio']:
            return 'a', test_config['config_a']
        else:
            return 'b', test_config['config_b']
            
    def record_result(self, test_name, group, metric_value):
        """
        Record A/B test result
        """
        if group == 'a':
            self.test_configs[test_name]['results_a'].append(metric_value)
        else:
            self.test_configs[test_name]['results_b'].append(metric_value)
            
    def analyze_test(self, test_name):
        """
        Analyze A/B test statistical significance
        """
        from scipy.stats import ttest_ind
        
        results_a = self.test_configs[test_name]['results_a']
        results_b = self.test_configs[test_name]['results_b']
        
        if len(results_a) < 30 or len(results_b) < 30:
            return {'status': 'insufficient_data'}
            
        t_stat, p_value = ttest_ind(results_a, results_b)
        
        return {
            'status': 'complete',
            'mean_a': np.mean(results_a),
            'mean_b': np.mean(results_b),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'winner': 'a' if np.mean(results_a) > np.mean(results_b) else 'b'
        }
```

---

## Performance Tuning

### 1. Algorithm Optimization

#### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

class ParallelEnsemble:
    def __init__(self, n_workers=None):
        self.n_workers = n_workers or multiprocessing.cpu_count()
        
    def detect_anomalies_parallel(self, data):
        """
        Run algorithms in parallel for faster processing
        """
        
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all algorithm tasks
            futures = {
                executor.submit(self._run_algorithm, 'isolation_forest', data): 'isolation_forest',
                executor.submit(self._run_algorithm, 'local_outlier_factor', data): 'local_outlier_factor',
                executor.submit(self._run_algorithm, 'one_class_svm', data): 'one_class_svm',
                executor.submit(self._run_algorithm, 'autoencoder', data): 'autoencoder',
                executor.submit(self._run_algorithm, 'statistical', data): 'statistical',
                executor.submit(self._run_algorithm, 'clustering', data): 'clustering',
                executor.submit(self._run_algorithm, 'temporal', data): 'temporal'
            }
            
            # Collect results
            results = {}
            for future in futures:
                algorithm_name = futures[future]
                try:
                    results[algorithm_name] = future.result(timeout=300)  # 5 minute timeout
                except Exception as e:
                    print(f"Algorithm {algorithm_name} failed: {e}")
                    results[algorithm_name] = self._empty_result()
                    
        return self._combine_results(results)
```

#### Memory Optimization

```python
class MemoryEfficientEnsemble:
    def __init__(self):
        self.streaming_batch_size = 1000
        
    def process_large_dataset(self, data_iterator):
        """
        Process data in batches to handle large datasets
        """
        
        all_results = []
        
        for batch in self._batch_iterator(data_iterator, self.streaming_batch_size):
            # Process batch
            batch_results = self.detect_anomalies(batch)
            
            # Store only essential results
            essential_results = {
                'anomaly_indices': batch_results['anomaly_indices'],
                'confidence_levels': batch_results['confidence_levels'],
                'batch_offset': len(all_results) * self.streaming_batch_size
            }
            
            all_results.append(essential_results)
            
            # Clear memory
            del batch_results
            import gc
            gc.collect()
            
        return self._merge_batch_results(all_results)
        
    def _batch_iterator(self, data, batch_size):
        """
        Yield data in batches
        """
        for i in range(0, len(data), batch_size):
            yield data.iloc[i:i+batch_size]
```

#### Algorithm Selection

```python
class AdaptiveAlgorithmSelector:
    def __init__(self):
        self.algorithm_costs = {
            'isolation_forest': 1.0,
            'local_outlier_factor': 3.0,  # Most expensive
            'one_class_svm': 2.5,
            'autoencoder': 2.0,
            'statistical': 0.5,           # Cheapest
            'clustering': 2.0,
            'temporal': 1.5
        }
        
    def select_algorithms(self, data_size, time_budget, accuracy_requirement):
        """
        Select optimal subset of algorithms based on constraints
        """
        
        # Estimate runtime for each algorithm
        estimated_times = {}
        for alg, cost in self.algorithm_costs.items():
            estimated_times[alg] = cost * (data_size ** 0.8) / 1000  # Empirical scaling
            
        # Greedy selection based on accuracy/time ratio
        selected_algorithms = []
        total_time = 0
        
        # Sort algorithms by efficiency (accuracy per unit time)
        efficiency_scores = self._calculate_efficiency_scores()
        sorted_algorithms = sorted(
            efficiency_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        for alg_name, efficiency in sorted_algorithms:
            if total_time + estimated_times[alg_name] <= time_budget:
                selected_algorithms.append(alg_name)
                total_time += estimated_times[alg_name]
                
                # Check if we meet accuracy requirement
                if self._estimate_ensemble_accuracy(selected_algorithms) >= accuracy_requirement:
                    break
                    
        return selected_algorithms
```

### 2. Database Optimization

#### Indexing Strategy

```sql
-- Create indexes for common queries
CREATE INDEX idx_quality_checks_timestamp ON quality_checks(timestamp);
CREATE INDEX idx_quality_checks_dataset ON quality_checks(dataset_name);
CREATE INDEX idx_human_feedback_session ON human_feedback(session_id);
CREATE INDEX idx_human_feedback_timestamp ON human_feedback(timestamp);

-- Composite indexes for complex queries
CREATE INDEX idx_quality_checks_date_dataset ON quality_checks(DATE(timestamp), dataset_name);
CREATE INDEX idx_feedback_session_action ON human_feedback(session_id, action);
```

#### Connection Pooling

```python
import sqlite3
from contextlib import contextmanager
import threading

class DatabasePool:
    def __init__(self, db_path, max_connections=10):
        self.db_path = db_path
        self.max_connections = max_connections
        self.pool = []
        self.pool_lock = threading.Lock()
        
    @contextmanager
    def get_connection(self):
        with self.pool_lock:
            if self.pool:
                conn = self.pool.pop()
            else:
                conn = sqlite3.connect(self.db_path)
                conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
                conn.execute("PRAGMA synchronous=NORMAL")  # Faster writes
                
        try:
            yield conn
        finally:
            with self.pool_lock:
                if len(self.pool) < self.max_connections:
                    self.pool.append(conn)
                else:
                    conn.close()
```

#### Query Optimization

```python
class OptimizedAuditStore:
    def get_quality_stats_optimized(self, days=30):
        """
        Optimized version using single query with window functions
        """
        
        query = """
        WITH daily_stats AS (
            SELECT 
                DATE(timestamp) as date,
                COUNT(*) as runs,
                SUM(total_records) as records,
                SUM(flagged_records) as flagged,
                AVG(CAST(flagged_records AS FLOAT) / total_records) as flag_rate
            FROM quality_checks 
            WHERE timestamp > datetime('now', '-{} days')
            GROUP BY DATE(timestamp)
        ),
        overall_stats AS (
            SELECT 
                SUM(runs) as total_runs,
                SUM(records) as total_records,
                SUM(flagged) as total_flagged,
                AVG(flag_rate) as avg_flag_rate
            FROM daily_stats
        )
        SELECT * FROM overall_stats
        """.format(days)
        
        with self.get_connection() as conn:
            result = conn.execute(query).fetchone()
            
        return {
            'total_runs': result[0] or 0,
            'total_records_processed': result[1] or 0,
            'total_flagged': result[2] or 0,
            'avg_flag_rate': round((result[3] or 0) * 100, 2)
        }
```

### 3. Web Application Optimization

#### Caching Strategy

```python
from functools import lru_cache
import time

class CachedDashboard:
    def __init__(self):
        self.cache_timeout = 300  # 5 minutes
        self.last_cache_update = {}
        
    @lru_cache(maxsize=128)
    def get_cached_stats(self, cache_key):
        """
        Cache expensive dashboard calculations
        """
        current_time = time.time()
        
        if (cache_key in self.last_cache_update and 
            current_time - self.last_cache_update[cache_key] < self.cache_timeout):
            return self._get_from_cache(cache_key)
            
        # Expensive calculation
        stats = self._calculate_fresh_stats()
        
        # Update cache
        self.last_cache_update[cache_key] = current_time
        self._save_to_cache(cache_key, stats)
        
        return stats
```

#### Async Processing

```python
from celery import Celery
import redis

# Configure Celery for background tasks
celery_app = Celery('adaptive_dq')
celery_app.config_from_object({
    'broker_url': 'redis://localhost:6379/0',
    'result_backend': 'redis://localhost:6379/0'
})

@celery_app.task
def process_large_dataset_async(file_path, session_id):
    """
    Process large datasets asynchronously
    """
    
    # Load data
    loader = DataLoader()
    df = loader.load_from_file(file_path)
    
    # Run detection
    detector = AdvancedEnsembleDetector(AdvancedConfig)
    results = detector.detect_anomalies(df)
    
    # Store results
    audit_store = AuditStore()
    audit_store.store_async_results(session_id, results)
    
    # Notify completion
    return {'status': 'complete', 'session_id': session_id}

# Flask integration
@app.route('/upload-async', methods=['POST'])
def upload_async():
    # Save uploaded file
    file_path = save_uploaded_file(request.files['file'])
    
    # Start async processing
    task = process_large_dataset_async.delay(file_path, session_id)
    
    return jsonify({
        'status': 'processing',
        'task_id': task.id,
        'session_id': session_id
    })
```

### 4. Resource Monitoring

```python
import psutil
import logging

class ResourceMonitor:
    def __init__(self):
        self.logger = logging.getLogger('resource_monitor')
        
    def monitor_system_resources(self):
        """
        Monitor system resources during processing
        """
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        metrics = {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'timestamp': time.time()
        }
        
        # Log warnings for high resource usage
        if cpu_percent > 80:
            self.logger.warning(f"High CPU usage: {cpu_percent}%")
            
        if memory.percent > 85:
            self.logger.warning(f"High memory usage: {memory.percent}%")
            
        return metrics
        
    def optimize_based_on_resources(self):
        """
        Adjust algorithm parameters based on available resources
        """
        
        memory = psutil.virtual_memory()
        
        if memory.percent > 70:
            # Reduce memory-intensive algorithm parameters
            return {
                'isolation_forest': {'n_estimators': 100},  # Reduce from 200
                'autoencoder': {'batch_size': 16},          # Reduce from 32
                'enable_parallel': False                     # Disable parallelization
            }
        else:
            return {
                'isolation_forest': {'n_estimators': 200},
                'autoencoder': {'batch_size': 32},
                'enable_parallel': True
            }
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. Installation Problems

**Issue**: TensorFlow installation fails
```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.13.0
```

**Solutions**:
```bash
# Try different Python version
python --version  # Ensure 3.8-3.11

# Install CPU-only version
pip install tensorflow-cpu==2.13.0

# Use conda instead of pip
conda install tensorflow=2.13.0

# Install from source (last resort)
pip install --no-binary=tensorflow tensorflow==2.13.0
```

**Issue**: scikit-learn version conflicts
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solutions**:
```bash
# Create fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # or fresh_env\Scripts\activate on Windows

# Install packages one by one
pip install scikit-learn==1.3.0
pip install pandas==2.1.0
# ... continue with other packages

# Use pip-tools for dependency resolution
pip install pip-tools
pip-compile requirements.in  # Create requirements.in first
pip-sync
```

#### 2. Data Loading Issues

**Issue**: CSV encoding errors
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0
```

**Solutions**:
```python
# Add encoding detection to DataLoader
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read())
    return result['encoding']

# In load_from_upload method:
encoding = detect_encoding(file_obj)
df = pd.read_csv(file_obj, encoding=encoding)
```

**Issue**: Memory errors with large files
```
MemoryError: Unable to allocate 8.00 GiB for an array
```

**Solutions**:
```python
# Implement chunked processing
def load_large_csv(file_path, chunk_size=10000):
    chunks = []
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        # Process chunk
        processed_chunk = preprocess_chunk(chunk)
        chunks.append(processed_chunk)
        
        # Control memory usage
        if len(chunks) > 10:  # Process in batches
            yield pd.concat(chunks)
            chunks = []
    
    if chunks:
        yield pd.concat(chunks)
```

#### 3. Algorithm Performance Issues

**Issue**: Isolation Forest very slow
```python
# Symptom: Processing takes > 5 minutes for 10k records
```

**Solutions**:
```python
# Optimize parameters
IsolationForest(
    n_estimators=100,     # Reduce from 200
    max_samples=1000,     # Limit sample size
    max_features=0.5,     # Reduce feature sampling
    n_jobs=-1            # Use all CPU cores
)

# Pre-filter data
def prefilter_data(df):
    # Remove obvious normal cases
    filtered_df = df[df['suspicious_score'] > threshold]
    return filtered_df
```

**Issue**: One-Class SVM memory error
```
sklearn.exceptions.MemoryError: Failed to allocate memory for SVM
```

**Solutions**:
```python
# Reduce dimensionality first
from sklearn.decomposition import PCA

pca = PCA(n_components=min(50, n_features))
reduced_features = pca.fit_transform(scaled_features)

# Use linear kernel for large datasets
OneClassSVM(kernel='linear', nu=0.1)

# Sample data if too large
if len(data) > 10000:
    sample_indices = np.random.choice(len(data), 10000, replace=False)
    sample_data = data.iloc[sample_indices]
```

#### 4. Database Issues

**Issue**: SQLite database locked
```
sqlite3.OperationalError: database is locked
```

**Solutions**:
```python
# Use WAL mode for better concurrency
conn.execute("PRAGMA journal_mode=WAL")

# Implement retry logic
import time
import random

def execute_with_retry(conn, query, max_retries=5):
    for attempt in range(max_retries):
        try:
            return conn.execute(query)
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                time.sleep(random.uniform(0.1, 0.5))  # Random backoff
                continue
            raise

# Use connection pooling
class ConnectionPool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool = queue.Queue(maxsize=pool_size)
        for _ in range(pool_size):
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA journal_mode=WAL")
            self.pool.put(conn)
```

**Issue**: Database corruption
```
sqlite3.DatabaseError: database disk image is malformed
```

**Solutions**:
```bash
# Check database integrity
sqlite3 audit_store.db "PRAGMA integrity_check;"

# Repair database
sqlite3 audit_store.db ".dump" > backup.sql
rm audit_store.db
sqlite3 audit_store.db < backup.sql

# Prevent corruption
# 1. Regular backups
# 2. Use WAL mode
# 3. Proper connection handling
```

#### 5. Web Interface Issues

**Issue**: Flask development server crashes
```
OSError: [Errno 98] Address already in use
```

**Solutions**:
```bash
# Find process using port 5000
lsof -i :5000  # macOS/Linux
netstat -ano | findstr :5000  # Windows

# Kill process
kill -9 [PID]  # macOS/Linux
taskkill /PID [PID] /F  # Windows

# Use different port
app.run(port=5001)

# Production deployment
gunicorn app:app --bind 0.0.0.0:5000 --workers 4
```

**Issue**: Static files not loading
```
404 Not Found: /static/css/style.css
```

**Solutions**:
```python
# Ensure static folder exists
import os
os.makedirs('static/css', exist_ok=True)

# Check Flask configuration
app = Flask(__name__)
print(f"Static folder: {app.static_folder}")
print(f"Static URL path: {app.static_url_path}")

# Debug static file serving
@app.route('/debug-static')
def debug_static():
    import os
    static_files = []
    for root, dirs, files in os.walk(app.static_folder):
        for file in files:
            static_files.append(os.path.join(root, file))
    return {'static_files': static_files}
```

#### 6. Performance Issues

**Issue**: Dashboard loads slowly
```
# Symptoms: Page takes >10 seconds to load
```

**Solutions**:
```python
# Add caching
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/')
@cache.cached(timeout=300)  # Cache for 5 minutes
def dashboard():
    stats = get_dashboard_stats()
    return render_template('dashboard.html', stats=stats)

# Optimize database queries
def get_quality_stats_fast(self):
    # Use single query instead of multiple
    query = """
    SELECT 
        COUNT(*) as total_runs,
        SUM(total_records) as total_records,
        SUM(flagged_records) as total_flagged,
        AVG(CAST(flagged_records AS FLOAT) / total_records) as avg_rate,
        MAX(timestamp) as last_run
    FROM quality_checks 
    WHERE timestamp > datetime('now', '-30 days')
    """
    
    result = conn.execute(query).fetchone()
    # Process single result instead of multiple queries
```

**Issue**: Algorithm ensemble too slow
```
# Symptoms: Processing 1000 records takes >5 minutes
```

**Solutions**:
```python
# Profile the code
import cProfile
import pstats

def profile_ensemble():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run your code
    results = ensemble_detector.detect_anomalies(data)
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)  # Top 10 slowest functions

# Optimize bottlenecks
# Common issues and fixes:
# 1. Feature engineering too complex → Simplify transformations
# 2. Too many features → Use feature selection
# 3. Algorithms not optimized → Tune parameters
# 4. No parallelization → Use concurrent.futures
```

### Debugging Tools

#### 1. Logging Configuration

```python
import logging
import sys

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('adaptive_dq.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set specific log levels
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    return logging.getLogger('adaptive_dq')

# Usage in modules
logger = setup_logging()

def detect_anomalies(self, df):
    logger.info(f"Starting anomaly detection on {len(df)} records")
    
    try:
        results = self._run_ensemble(df)
        logger.info(f"Detection complete. Found {len(results['flagged_indices'])} anomalies")
        return results
    except Exception as e:
        logger.error(f"Detection failed: {str(e)}", exc_info=True)
        raise
```

#### 2. Performance Monitoring

```python
import time
from functools import wraps

def monitor_performance(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info(f"{func.__name__} completed in {end_time - start_time:.2f}s, "
                       f"memory change: {memory_after - memory_before:.1f}MB")
            
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed after {time.time() - start_time:.2f}s: {e}")
            raise
            
    return wrapper

# Usage
@monitor_performance
def detect_anomalies(self, df):
    # Your code here
    pass
```

#### 3. Data Validation

```python
def validate_input_data(df):
    """
    Comprehensive input data validation
    """
    issues = []
    
    # Check basic properties
    if df.empty:
        issues.append("DataFrame is empty")
        
    if len(df) < 10:
        issues.append(f"Too few records: {len(df)} (minimum 10 recommended)")
        
    # Check for all-null columns
    null_columns = df.columns[df.isnull().all()].tolist()
    if null_columns:
        issues.append(f"Columns with all null values: {null_columns}")
        
    # Check for single-value columns
    single_value_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            single_value_cols.append(col)
    if single_value_cols:
        issues.append(f"Columns with single value: {single_value_cols}")
        
    # Check data types
    object_cols_high_cardinality = []
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > len(df) * 0.8:  # High cardinality
            object_cols_high_cardinality.append(col)
    if object_cols_high_cardinality:
        issues.append(f"High cardinality text columns: {object_cols_high_cardinality}")
        
    return issues

# Usage in DataLoader
def load_from_upload(self, file_obj):
    df = pd.read_csv(file_obj)
    
    # Validate data
    issues = validate_input_data(df)
    if issues:
        logger.warning(f"Data quality issues detected: {issues}")
        # Optionally, raise exception for critical issues
        
    return self._preprocess_dataframe(df)
```

---

## Extension Development

### Creating Custom Algorithms

#### 1. Algorithm Interface

```python
from abc import ABC, abstractmethod
import numpy as np

class AnomalyDetectorInterface(ABC):
    """
    Interface that all anomaly detection algorithms must implement
    """
    
    @abstractmethod
    def __init__(self, **params):
        """Initialize algorithm with parameters"""
        pass
    
    @abstractmethod
    def fit(self, X):
        """Train the algorithm on normal data"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Predict anomalies (1 for normal, -1 for anomaly)"""
        pass
    
    @abstractmethod
    def score_samples(self, X):
        """Return anomaly scores for samples"""
        pass
    
    @abstractmethod
    def get_params(self):
        """Return algorithm parameters for reproducibility"""
        pass
    
    @abstractmethod
    def set_params(self, **params):
        """Set algorithm parameters"""
        pass
```

#### 2. Example Custom Algorithm

```python
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class KNNAnomalyDetector(AnomalyDetectorInterface):
    """
    Custom anomaly detector based on k-nearest neighbors distance
    """
    
    def __init__(self, n_neighbors=5, contamination=0.1, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.metric = metric
        self.scaler = StandardScaler()
        self.knn_model = None
        self.threshold = None
        
    def fit(self, X):
        """
        Fit the k-NN model and determine anomaly threshold
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit k-NN model
        self.knn_model = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric
        )
        self.knn_model.fit(X_scaled)
        
        # Calculate distances for threshold determination
        distances, _ = self.knn_model.kneighbors(X_scaled)
        avg_distances = np.mean(distances[:, 1:], axis=1)  # Exclude self
        
        # Set threshold based on contamination rate
        self.threshold = np.percentile(avg_distances, (1 - self.contamination) * 100)
        
        return self
    
    def predict(self, X):
        """
        Predict anomalies based on distance threshold
        """
        scores = self.score_samples(X)
        predictions = np.where(scores > self.threshold, -1, 1)
        return predictions
    
    def score_samples(self, X):
        """
        Return average distance to k nearest neighbors as anomaly score
        """
        if self.knn_model is None:
            raise ValueError("Model must be fitted before scoring")
            
        X_scaled = self.scaler.transform(X)
        distances, _ = self.knn_model.kneighbors(X_scaled)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        
        return avg_distances
    
    def get_params(self):
        return {
            'n_neighbors': self.n_neighbors,
            'contamination': self.contamination,
            'metric': self.metric
        }
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self
```

#### 3. Registering Custom Algorithm

```python
class ExtendedEnsembleDetector(AdvancedEnsembleDetector):
    """
    Extended ensemble detector with custom algorithm support
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.custom_algorithms = {}
        
    def register_custom_algorithm(self, name, algorithm_class, weight=0.1, **algorithm_params):
        """
        Register a custom anomaly detection algorithm
        
        Args:
            name: Unique name for the algorithm
            algorithm_class: Class implementing AnomalyDetectorInterface
            weight: Initial weight in ensemble (0-1)
            **algorithm_params: Parameters to pass to algorithm constructor
        """
        
        # Validate algorithm implements interface
        if not issubclass(algorithm_class, AnomalyDetectorInterface):
            raise ValueError("Algorithm must implement AnomalyDetectorInterface")
            
        # Create algorithm instance
        algorithm_instance = algorithm_class(**algorithm_params)
        
        # Register in ensemble
        self.algorithms[name] = algorithm_instance
        self.algorithm_weights[name] = weight
        self.algorithm_performance[name] = []
        
        # Normalize weights
        self._normalize_weights()
        
        logger.info(f"Registered custom algorithm: {name}")
        
    def _normalize_weights(self):
        """Ensure all weights sum to 1.0"""
        total_weight = sum(self.algorithm_weights.values())
        if total_weight > 0:
            for alg_name in self.algorithm_weights:
                self.algorithm_weights[alg_name] /= total_weight

# Usage example
detector = ExtendedEnsembleDetector(AdvancedConfig)

# Register custom k-NN algorithm
detector.register_custom_algorithm(
    name='knn_distance',
    algorithm_class=KNNAnomalyDetector,
    weight=0.15,
    n_neighbors=10,
    contamination=0.1,
    metric='manhattan'
)
```

### Custom Feature Engineering

#### 1. Feature Engineering Interface

```python
class FeatureEngineerInterface(ABC):
    """
    Interface for custom feature engineering modules
    """
    
    @abstractmethod
    def transform_features(self, df):
        """
        Transform input DataFrame into features suitable for algorithms
        
        Args:
            df: Input pandas DataFrame
            
        Returns:
            Tuple of (features_df, feature_info)
        """
        pass
    
    @abstractmethod
    def get_feature_names(self):
        """Return list of generated feature names"""
        pass
```

#### 2. Domain-Specific Feature Engineering

```python
class FinancialFeatureEngineer(FeatureEngineerInterface):
    """
    Specialized feature engineering for financial data
    """
    
    def __init__(self):
        self.feature_names = []
        
    def transform_features(self, df):
        """
        Create financial domain-specific features
        """
        features_df = pd.DataFrame(index=df.index)
        feature_info = {'domain': 'financial', 'feature_types': {}}
        
        # Transaction amount features
        if 'transaction_amount' in df.columns:
            amt = df['transaction_amount']
            
            # Amount-based features
            features_df['amount_raw'] = amt
            features_df['amount_log'] = np.log1p(amt.abs())
            features_df['amount_zscore'] = (amt - amt.mean()) / amt.std()
            
            # Amount categorization
            features_df['amount_category'] = pd.cut(
                amt, 
                bins=[0, 100, 1000, 10000, float('inf')],
                labels=['small', 'medium', 'large', 'very_large']
            ).astype(str)
            
            # Velocity features (if timestamps available)
            if 'transaction_date' in df.columns:
                df_sorted = df.sort_values('transaction_date')
                features_df['amount_velocity'] = df_sorted['transaction_amount'].rolling(
                    window=5, min_periods=1).mean()
                
        # Account balance features
        if 'account_balance' in df.columns:
            balance = df['account_balance']
            
            # Balance health indicators
            features_df['balance_to_amount_ratio'] = balance / (df['transaction_amount'] + 1e-8)
            features_df['low_balance_flag'] = (balance < 1000).astype(int)
            
            # Balance percentile
            features_df['balance_percentile'] = balance.rank(pct=True)
            
        # Customer risk scoring
        if all(col in df.columns for col in ['customer_age', 'account_balance', 'transaction_amount']):
            # Simple risk score based on age, balance, and transaction size
            age_score = (df['customer_age'] - 18) / 47  # Normalize 18-65
            balance_score = np.log1p(df['account_balance']) / 10
            amount_score = np.log1p(df['transaction_amount']) / 5
            
            features_df['customer_risk_score'] = (
                0.3 * age_score + 
                0.4 * balance_score + 
                0.3 * amount_score
            )
            
        # Store feature information
        self.feature_names = features_df.columns.tolist()
        feature_info['feature_types'] = {
            col: str(features_df[col].dtype) for col in features_df.columns
        }
        
        return features_df.fillna(0), feature_info
    
    def get_feature_names(self):
        return self.feature_names
```

#### 3. Integrating Custom Feature Engineering

```python
class CustomizableEnsembleDetector(ExtendedEnsembleDetector):
    """
    Ensemble detector with pluggable feature engineering
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.feature_engineers = []
        
    def add_feature_engineer(self, engineer):
        """
        Add custom feature engineering module
        """
        if not isinstance(engineer, FeatureEngineerInterface):
            raise ValueError("Engineer must implement FeatureEngineerInterface")
            
        self.feature_engineers.append(engineer)
        logger.info(f"Added feature engineer: {engineer.__class__.__name__}")
        
    def _prepare_advanced_features(self, df):
        """
        Override to use custom feature engineering
        """
        all_features = []
        combined_feature_info = {'feature_types': {}}
        
        # Run base feature engineering
        base_features, base_info = super()._prepare_advanced_features(df)
        all_features.append(base_features)
        combined_feature_info['feature_types'].update(base_info['feature_types'])
        
        # Run custom feature engineers
        for engineer in self.feature_engineers:
            try:
                custom_features, custom_info = engineer.transform_features(df)
                all_features.append(custom_features)
                combined_feature_info['feature_types'].update(custom_info['feature_types'])
                
                logger.info(f"Generated {len(custom_features.columns)} features using {engineer.__class__.__name__}")
                
            except Exception as e:
                logger.error(f"Feature engineer {engineer.__class__.__name__} failed: {e}")
                continue
        
        # Combine all features
        if len(all_features) > 1:
            combined_features = pd.concat(all_features, axis=1)
        else:
            combined_features = all_features[0]
            
        # Remove duplicate columns
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
        
        return combined_features, combined_feature_info

# Usage
detector = CustomizableEnsembleDetector(AdvancedConfig)

# Add domain-specific feature engineering
financial_engineer = FinancialFeatureEngineer()
detector.add_feature_engineer(financial_engineer)

# Add custom algorithm
detector.register_custom_algorithm('knn_distance', KNNAnomalyDetector, weight=0.1)
```

### Plugin Architecture

#### 1. Plugin Manager

```python
import importlib
import inspect
from pathlib import Path

class PluginManager:
    """
    Manager for loading and registering plugins
    """
    
    def __init__(self, plugin_directory='plugins'):
        self.plugin_directory = Path(plugin_directory)
        self.loaded_plugins = {}
        
    def discover_plugins(self):
        """
        Automatically discover plugins in plugin directory
        """
        if not self.plugin_directory.exists():
            return []
            
        plugins = []
        
        for plugin_file in self.plugin_directory.glob('*.py'):
            if plugin_file.name.startswith('__'):
                continue
                
            try:
                plugin_name = plugin_file.stem
                spec = importlib.util.spec_from_file_location(plugin_name, plugin_file)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for plugin classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (hasattr(obj, 'PLUGIN_TYPE') and 
                        hasattr(obj, 'PLUGIN_NAME') and
                        hasattr(obj, 'PLUGIN_VERSION')):
                        
                        plugins.append({
                            'name': obj.PLUGIN_NAME,
                            'type': obj.PLUGIN_TYPE,
                            'version': obj.PLUGIN_VERSION,
                            'class': obj,
                            'module': module
                        })
                        
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file}: {e}")
                
        return plugins
    
    def load_plugin(self, plugin_info):
        """
        Load and initialize a plugin
        """
        try:
            plugin_instance = plugin_info['class']()
            
            self.loaded_plugins[plugin_info['name']] = {
                'instance': plugin_instance,
                'info': plugin_info
            }
            
            logger.info(f"Loaded plugin: {plugin_info['name']} v{plugin_info['version']}")
            return plugin_instance
            
        except Exception as e:
            logger.error(f"Failed to initialize plugin {plugin_info['name']}: {e}")
            return None
    
    def get_plugins_by_type(self, plugin_type):
        """
        Get all loaded plugins of a specific type
        """
        return [
            plugin['instance'] 
            for plugin in self.loaded_plugins.values() 
            if plugin['info']['type'] == plugin_type
        ]
```

#### 2. Example Plugin

```python
# plugins/fraud_detection_plugin.py

from src.advanced_ml.ensemble_detector import AnomalyDetectorInterface
from src.data_ingestion.loaders import FeatureEngineerInterface
import numpy as np
import pandas as pd

class FraudDetectionAlgorithm(AnomalyDetectorInterface):
    """
    Specialized algorithm for fraud detection
    """
    
    PLUGIN_TYPE = 'algorithm'
    PLUGIN_NAME = 'fraud_detector'
    PLUGIN_VERSION = '1.0.0'
    
    def __init__(self, velocity_threshold=5, amount_threshold=10000):
        self.velocity_threshold = velocity_threshold
        self.amount_threshold = amount_threshold
        self.baseline_patterns = None
        
    def fit(self, X):
        """
        Learn normal transaction patterns
        """
        # Calculate baseline statistics
        self.baseline_patterns = {
            'avg_amount': X['transaction_amount'].mean(),
            'std_amount': X['transaction_amount'].std(),
            'avg_frequency': self._calculate_frequency(X),
            'common_merchants': X['merchant_category'].value_counts().head(10).index.tolist()
        }
        return self
        
    def predict(self, X):
        """
        Detect fraudulent transactions
        """
        scores = self.score_samples(X)
        threshold = np.percentile(scores, 90)  # Top 10% as anomalies
        return np.where(scores > threshold, -1, 1)
        
    def score_samples(self, X):
        """
        Calculate fraud risk scores
        """
        scores = np.zeros(len(X))
        
        for i, row in X.iterrows():
            risk_factors = []
            
            # High amount transactions
            if row['transaction_amount'] > self.amount_threshold:
                risk_factors.append(0.3)
                
            # Unusual merchant category
            if row['merchant_category'] not in self.baseline_patterns['common_merchants']:
                risk_factors.append(0.2)
                
            # Off-hours transactions
            if 'transaction_hour' in row and (row['transaction_hour'] < 6 or row['transaction_hour'] > 22):
                risk_factors.append(0.2)
                
            # High velocity (if customer_id available)
            if 'customer_velocity' in row and row['customer_velocity'] > self.velocity_threshold:
                risk_factors.append(0.4)
                
            scores[i] = sum(risk_factors)
            
        return scores
    
    def _calculate_frequency(self, X):
        """Calculate transaction frequency per customer"""
        if 'customer_id' in X.columns and 'transaction_date' in X.columns:
            return X.groupby('customer_id').size().mean()
        return 1.0

class FraudFeatureEngineer(FeatureEngineerInterface):
    """
    Feature engineering specialized for fraud detection
    """
    
    PLUGIN_TYPE = 'feature_engineer'
    PLUGIN_NAME = 'fraud_features'
    PLUGIN_VERSION = '1.0.0'
    
    def transform_features(self, df):
        """
        Create fraud-specific features
        """
        features_df = pd.DataFrame(index=df.index)
        
        # Time-based features
        if 'transaction_date' in df.columns:
            dt = pd.to_datetime(df['transaction_date'])
            features_df['transaction_hour'] = dt.dt.hour
            features_df['is_weekend'] = dt.dt.dayofweek.isin([5, 6]).astype(int)
            features_df['is_night'] = ((dt.dt.hour < 6) | (dt.dt.hour > 22)).astype(int)
            
        # Customer velocity features
        if 'customer_id' in df.columns and 'transaction_date' in df.columns:
            df_sorted = df.sort_values(['customer_id', 'transaction_date'])
            
            # Transactions per customer per day
            daily_counts = df_sorted.groupby([
                'customer_id', 
                df_sorted['transaction_date'].dt.date
            ]).size().reset_index(name='daily_tx_count')
            
            # Average velocity per customer
            avg_velocity = daily_counts.groupby('customer_id')['daily_tx_count'].mean()
            features_df['customer_velocity'] = df['customer_id'].map(avg_velocity).fillna(1)
            
        # Amount deviation features
        if 'transaction_amount' in df.columns and 'customer_id' in df.columns:
            customer_avg_amount = df.groupby('customer_id')['transaction_amount'].mean()
            customer_std_amount = df.groupby('customer_id')['transaction_amount'].std()
            
            features_df['amount_deviation'] = abs(
                df['transaction_amount'] - df['customer_id'].map(customer_avg_amount)
            ) / (df['customer_id'].map(customer_std_amount) + 1e-8)
            
        # Merchant risk features
        if 'merchant_category' in df.columns:
            merchant_risk_scores = {
                'online': 0.7,
                'cash_advance': 0.9,
                'gambling': 0.8,
                'gas': 0.2,
                'grocery': 0.1
            }
            
            features_df['merchant_risk'] = df['merchant_category'].map(
                merchant_risk_scores
            ).fillna(0.5)
            
        feature_info = {
            'domain': 'fraud_detection',
            'feature_types': {col: str(features_df[col].dtype) for col in features_df.columns}
        }
        
        return features_df.fillna(0), feature_info
        
    def get_feature_names(self):
        return self.feature_names
```

#### 3. Plugin Integration

```python
class PluginEnabledDetector(CustomizableEnsembleDetector):
    """
    Ensemble detector with full plugin support
    """
    
    def __init__(self, config, plugin_directory='plugins'):
        super().__init__(config)
        self.plugin_manager = PluginManager(plugin_directory)
        self.load_plugins()
        
    def load_plugins(self):
        """
        Discover and load all available plugins
        """
        plugins = self.plugin_manager.discover_plugins()
        
        for plugin_info in plugins:
            plugin_instance = self.plugin_manager.load_plugin(plugin_info)
            
            if plugin_instance is None:
                continue
                
            # Register based on plugin type
            if plugin_info['type'] == 'algorithm':
                self.algorithms[plugin_info['name']] = plugin_instance
                self.algorithm_weights[plugin_info['name']] = 0.1  # Default weight
                self.algorithm_performance[plugin_info['name']] = []
                
            elif plugin_info['type'] == 'feature_engineer':
                self.add_feature_engineer(plugin_instance)
                
        # Normalize weights after adding new algorithms
        self._normalize_weights()
        
    def list_available_plugins(self):
        """
        List all loaded plugins with their information
        """
        plugin_info = []
        
        for name, plugin_data in self.plugin_manager.loaded_plugins.items():
            info = plugin_data['info']
            plugin_info.append({
                'name': info['name'],
                'type': info['type'],
                'version': info['version'],
                'status': 'active'
            })
            
        return plugin_info

# Usage
detector = PluginEnabledDetector(AdvancedConfig, plugin_directory='plugins')

# List loaded plugins
plugins = detector.list_available_plugins()
for plugin in plugins:
    print(f"Loaded: {plugin['name']} v{plugin['version']} ({plugin['type']})")
```

### REST API Extensions

#### 1. Plugin Management API

```python
@app.route('/api/plugins', methods=['GET'])
def list_plugins():
    """
    List all available plugins
    """
    try:
        plugins = ensemble_detector.list_available_plugins()
        return jsonify({
            'status': 'success',
            'plugins': plugins
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/plugins/<plugin_name>/enable', methods=['POST'])
def enable_plugin(plugin_name):
    """
    Enable a specific plugin
    """
    try:
        # Implementation would depend on your plugin architecture
        success = ensemble_detector.enable_plugin(plugin_name)
        
        if success:
            return jsonify({'status': 'success', 'message': f'Plugin {plugin_name} enabled'})
        else:
            return jsonify({'status': 'error', 'message': f'Failed to enable plugin {plugin_name}'}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/api/algorithms/custom', methods=['POST'])
def register_custom_algorithm():
    """
    Register a custom algorithm via API
    """
    try:
        data = request.get_json()
        
        # Validate request
        required_fields = ['name', 'algorithm_type', 'parameters']
        if not all(field in data for field in required_fields):
            return jsonify({'status': 'error', 'message': 'Missing required fields'}), 400
            
        # Register algorithm (implementation depends on your needs)
        success = ensemble_detector.register_algorithm_from_config(
            name=data['name'],
            algorithm_type=data['algorithm_type'],
            parameters=data['parameters']
        )
        
        if success:
            return jsonify({'status': 'success', 'message': f'Algorithm {data["name"]} registered'})
        else:
            return jsonify({'status': 'error', 'message': 'Failed to register algorithm'}), 400
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
```

---

## Best Practices

### 1. Data Quality Best Practices

#### Data Preparation

**Clean Data Guidelines**:
```python
def prepare_quality_data(df):
    """
    Best practices for preparing data for quality analysis
    """
    
    # 1. Remove completely empty rows/columns
    df = df.dropna(how='all').dropna(axis=1, how='all')
    
    # 2. Handle extreme duplicates (>95% identical)
    duplicate_rate = df.duplicated().sum() / len(df)
    if duplicate_rate > 0.95:
        logger.warning(f"High duplicate rate: {duplicate_rate:.2%}")
        df = df.drop_duplicates()
        
    # 3. Remove single-value columns (no information)
    single_value_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if single_value_cols:
        logger.info(f"Removing single-value columns: {single_value_cols}")
        df = df.drop(columns=single_value_cols)
        
    # 4. Handle extreme high-cardinality categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > len(df) * 0.9:  # >90% unique values
            logger.warning(f"High cardinality column {col}: {df[col].nunique()} unique values")
            # Consider hashing or grouping rare categories
            
    # 5. Validate date columns
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                pd.to_datetime(df[col], errors='coerce')
            except:
                logger.warning(f"Column {col} appears to be date but conversion failed")
                
    return df
```

**Feature Selection**:
```python
def select_quality_features(df, max_features=50):
    """
    Select most informative features for quality detection
    """
    
    features_to_keep = []
    
    # Always keep numerical features (good for statistical methods)
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
    features_to_keep.extend(numerical_features[:20])  # Top 20 numerical
    
    # Keep datetime features (important for temporal analysis)
    datetime_features = df.select_dtypes(include=['datetime64']).columns.tolist()
    features_to_keep.extend(datetime_features)
    
    # Select categorical features with optimal cardinality
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    optimal_categorical = [
        col for col in categorical_features 
        if 2 <= df[col].nunique() <= len(df) * 0.1  # 2 to 10% unique values
    ]
    features_to_keep.extend(optimal_categorical[:10])  # Top 10 categorical
    
    # Remove duplicates and limit total
    features_to_keep = list(set(features_to_keep))[:max_features]
    
    return df[features_to_keep]
```

#### Sampling Strategies

```python
def smart_sampling(df, target_size=10000, strategy='stratified'):
    """
    Intelligent sampling for large datasets
    """
    
    if len(df) <= target_size:
        return df
        
    if strategy == 'stratified':
        # Stratified sampling based on key categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            # Use first categorical column for stratification
            strat_col = categorical_cols[0]
            
            # Calculate sample size per category
            category_counts = df[strat_col].value_counts()
            sample_sizes = (category_counts / len(df) * target_size).astype(int)
            
            sampled_dfs = []
            for category, sample_size in sample_sizes.items():
                if sample_size > 0:
                    category_df = df[df[strat_col] == category]
                    sampled_df = category_df.sample(
                        n=min(sample_size, len(category_df)), 
                        random_state=42
                    )
                    sampled_dfs.append(sampled_df)
                    
            return pd.concat(sampled_dfs)
            
    elif strategy == 'systematic':
        # Systematic sampling
        step = len(df) // target_size
        indices = range(0, len(df), step)
        return df.iloc[list(indices)[:target_size]]
        
    else:  # random
        return df.sample(n=target_size, random_state=42)
```

### 2. Algorithm Configuration Best Practices

#### Domain-Specific Configurations

```python
# Financial Services Configuration
FINANCIAL_CONFIG = {
    'algorithm_weights': {
        'statistical_outlier': 0.25,    # High weight for obvious violations
        'isolation_forest': 0.20,       # Good for fraud patterns
        'local_outlier_factor': 0.20,   # Customer behavior analysis
        'autoencoder': 0.15,            # Complex fraud schemes
        'one_class_svm': 0.10,          # Regulatory compliance
        'density_clustering': 0.05,     # Account clustering
        'temporal_anomaly': 0.05        # Time-based patterns
    },
    'thresholds': {
        'high_confidence': 0.85,        # Conservative for financial data
        'medium_confidence': 0.70,
        'consensus': 0.80               # High consensus required
    },
    'learning_rate': 0.005             # Slow learning for stability
}

# IoT/Sensor Data Configuration  
IOT_CONFIG = {
    'algorithm_weights': {
        'temporal_anomaly': 0.30,       # Primary focus on time patterns
        'autoencoder': 0.25,            # Sensor correlation patterns
        'statistical_outlier': 0.20,    # Sensor reading outliers
        'isolation_forest': 0.15,       # General device anomalies
        'local_outlier_factor': 0.05,   # Less relevant for sensors
        'one_class_svm': 0.03,          # Minimal for IoT
        'density_clustering': 0.02      # Minimal for IoT
    },
    'thresholds': {
        'high_confidence': 0.75,        # More sensitive for IoT
        'medium_confidence': 0.50,
        'consensus': 0.60               # Lower consensus OK
    },
    'learning_rate': 0.02              # Faster adaptation to sensor changes
}

# Customer Data Configuration
CUSTOMER_CONFIG = {
    'algorithm_weights': {
        'local_outlier_factor': 0.25,   # Customer segmentation
        'autoencoder': 0.20,            # Complex customer patterns
        'isolation_forest': 0.20,       # General customer outliers
        'statistical_outlier': 0.15,    # Demographic outliers
        'one_class_svm': 0.10,          # Customer boundary detection
        'density_clustering': 0.05,     # Customer clustering
        'temporal_anomaly': 0.05        # Purchase timing
    },
    'thresholds': {
        'high_confidence': 0.80,
        'medium_confidence': 0.60,
        'consensus': 0.70
    },
    'learning_rate': 0.01
}
```

#### Performance-Based Configuration

```python
def optimize_configuration_for_performance(system_resources, data_characteristics):
    """
    Optimize configuration based on system capabilities and data properties
    """
    
    config = {}
    
    # Adjust based on available memory
    memory_gb = system_resources['memory_gb']
    
    if memory_gb < 4:
        # Low memory configuration
        config['algorithms'] = [
            'statistical_outlier',      # Lightweight
            'isolation_forest',         # Efficient
            'density_clustering'        # Moderate memory
        ]
        config['max_features'] = 20
        config['batch_size'] = 1000
        
    elif memory_gb < 8:
        # Medium memory configuration
        config['algorithms'] = [
            'statistical_outlier',
            'isolation_forest', 
            'local_outlier_factor',
            'one_class_svm',
            'density_clustering'
        ]
        config['max_features'] = 50
        config['batch_size'] = 5000
        
    else:
        # High memory configuration
        config['algorithms'] = [
            'statistical_outlier',
            'isolation_forest',
            'local_outlier_factor', 
            'one_class_svm',
            'autoencoder',
            'density_clustering',
            'temporal_anomaly'
        ]
        config['max_features'] = 100
        config['batch_size'] = 10000
        
    # Adjust based on data size
    data_size = data_characteristics['num_records']
    
    if data_size > 100000:
        # Large dataset optimizations
        config['parallel_processing'] = True
        config['streaming_mode'] = True
        config['sample_for_training'] = True
        config['sample_size'] = 50000
        
    # Adjust based on data complexity
    if data_characteristics['num_features'] > 100:
        config['feature_selection'] = True
        config['dimensionality_reduction'] = True
        config['pca_components'] = 50
        
    return config
```

### 3. Production Deployment Best Practices

#### Environment Configuration

```python
# production_config.py
import os

class ProductionConfig:
    """
    Production-ready configuration
    """
    
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY')
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable must be set")
        
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL', 'postgresql://user:pass@localhost/adq')
    DATABASE_POOL_SIZE = int(os.environ.get('DB_POOL_SIZE', 10))
    
    # Performance
    WORKERS = int(os.environ.get('WORKERS', 4))
    WORKER_CONNECTIONS = int(os.environ.get('WORKER_CONNECTIONS', 1000))
    MAX_REQUESTS = int(os.environ.get('MAX_REQUESTS', 1000))
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', '/var/log/adq/app.log')
    
    # Monitoring
    METRICS_ENABLED = os.environ.get('METRICS_ENABLED', 'true').lower() == 'true'
    HEALTH_CHECK_INTERVAL = int(os.environ.get('HEALTH_CHECK_INTERVAL', 60))
    
    # ML Model Settings
    MODEL_CACHE_SIZE = int(os.environ.get('MODEL_CACHE_SIZE', 5))
    MODEL_REFRESH_INTERVAL = int(os.environ.get('MODEL_REFRESH_HOURS', 24)) * 3600
    
    # Rate Limiting
    RATE_LIMIT_ENABLED = True
    RATE_LIMIT_PER_MINUTE = int(os.environ.get('RATE_LIMIT', 100))
```

#### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Start application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "app:app"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "5000:5000"
    environment:
      - SECRET_KEY=${SECRET_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/adq
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - db
      - redis
    volumes:
      - ./logs:/var/log/adq
      - ./models:/app/model_cache
    restart: unless-stopped
    
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=adq
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped
    
  redis:
    image: redis:6-alpine
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
    restart: unless-stopped

volumes:
  postgres_data:
```

#### Monitoring and Alerting

```python
import logging
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Metrics collection
REQUEST_COUNT = Counter('adq_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('adq_request_duration_seconds', 'Request duration')
ACTIVE_SESSIONS = Gauge('adq_active_sessions', 'Active user sessions')
ALGORITHM_PERFORMANCE = Gauge('adq_algorithm_accuracy', 'Algorithm accuracy', ['algorithm'])

class ProductionMonitoring:
    def __init__(self):
        self.logger = logging.getLogger('production_monitor')
        
    def setup_logging(self):
        """
        Configure production logging
        """
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler('/var/log/adq/app.log')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for errors only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(simple_formatter)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
    def monitor_request(self, func):
        """
        Decorator to monitor request performance
        """
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                # Execute request
                result = func(*args, **kwargs)
                
                # Record success metrics
                REQUEST_COUNT.labels(method='POST', endpoint=func.__name__).inc()
                REQUEST_DURATION.observe(time.time() - start_time)
                
                return result
                
            except Exception as e:
                # Record error metrics
                REQUEST_COUNT.labels(method='POST', endpoint=f'{func.__name__}_error').inc()
                self.logger.error(f"Request failed: {str(e)}", exc_info=True)
                raise
                
        return wrapper
        
    def health_check(self):
        """
        Comprehensive health check
        """
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        # Database connectivity
        try:
            # Test database connection
            audit_store.get_quality_stats()
            health_status['checks']['database'] = 'healthy'
        except Exception as e:
            health_status['checks']['database'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'unhealthy'
            
        # ML models status
        try:
            # Test ensemble detector
            ensemble_detector.algorithm_weights
            health_status['checks']['ml_models'] = 'healthy'
        except Exception as e:
            health_status['checks']['ml_models'] = f'unhealthy: {str(e)}'
            health_status['status'] = 'unhealthy'
            
        # Memory usage
        import psutil
        memory_percent = psutil.virtual_memory().percent
        health_status['checks']['memory_usage'] = f'{memory_percent:.1f}%'
        
        if memory_percent > 90:
            health_status['status'] = 'unhealthy'
            
        return health_status

# Flask integration
monitoring = ProductionMonitoring()

@app.route('/health')
def health_check():
    """Health check endpoint for load balancers"""
    return jsonify(monitoring.health_check())

@app.route('/metrics')
def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(), 200, {'Content-Type': 'text/plain; charset=utf-8'}
```

### 4. Security Best Practices

#### Input Validation

```python
from flask import request
import bleach
import pandas as pd

class SecurityValidator:
    """
    Security validation for user inputs
    """
    
    def __init__(self):
        self.max_file_size = 50 * 1024 * 1024  # 50MB
        self.allowed_extensions = {'.csv', '.parquet'}
        self.max_records = 1000000  # 1M records max
        
    def validate_file_upload(self, file):
        """
        Validate uploaded file for security
        """
        errors = []
        
        # Check file size
        file.seek(0, 2)  # Seek to end
        file_size = file.tell()
        file.seek(0)     # Reset to beginning
        
        if file_size > self.max_file_size:
            errors.append(f"File too large: {file_size} bytes (max: {self.max_file_size})")
            
        # Check file extension
        filename = file.filename.lower()
        if not any(filename.endswith(ext) for ext in self.allowed_extensions):
            errors.append(f"Invalid file type. Allowed: {self.allowed_extensions}")
            
        # Basic malware check (simple heuristics)
        if self._contains_suspicious_content(file):
            errors.append("File contains suspicious content")
            
        return errors
        
    def validate_dataframe(self, df):
        """
        Validate DataFrame content
        """
        errors = []
        
        # Check record count
        if len(df) > self.max_records:
            errors.append(f"Too many records: {len(df)} (max: {self.max_records})")
            
        # Check for SQL injection attempts in string columns
        string_columns = df.select_dtypes(include=['object']).columns
        for col in string_columns:
            if df[col].astype(str).str.contains(
                r'(union|select|insert|delete|drop|create|alter)', 
                case=False, 
                na=False
            ).any():
                errors.append(f"Suspicious SQL-like content in column: {col}")
                
        # Check for script injection
        for col in string_columns:
            if df[col].astype(str).str.contains(
                r'<script|javascript:|vbscript:', 
                case=False, 
                na=False
            ).any():
                errors.append(f"Suspicious script content in column: {col}")
                
        return errors
        
    def sanitize_feedback(self, feedback_text):
        """
        Sanitize user feedback to prevent XSS
        """
        if not feedback_text:
            return ""
            
        # Remove HTML tags and scripts
        cleaned = bleach.clean(
            feedback_text, 
            tags=[],  # No HTML tags allowed
            strip=True
        )
        
        # Limit length
        return cleaned[:500]  # Max 500 characters
        
    def _contains_suspicious_content(self, file):
        """
        Basic check for suspicious file content
        """
        # Read first 1KB to check for suspicious patterns
        chunk = file.read(1024)
        file.seek(0)  # Reset
        
        if isinstance(chunk, bytes):
            chunk = chunk.decode('utf-8', errors='ignore')
            
        # Check for suspicious patterns
        suspicious_patterns = [
            '<?php',
            '<script',
            'eval(',
            'exec(',
            'system(',
            '__import__'
        ]
        
        return any(pattern in chunk.lower() for pattern in suspicious_patterns)

# Flask integration
validator = SecurityValidator()

@app.before_request
def security_headers():
    """Add security headers to all responses"""
    pass

@app.after_request
def add_security_headers(response):
    """Add security headers"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

@app.route('/upload', methods=['POST'])
def secure_upload():
    """Secure file upload with validation"""
    
    # Validate file
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'No file provided'}), 400
        
    file_errors = validator.validate_file_upload(file)
    if file_errors:
        return jsonify({'error': 'File validation failed', 'details': file_errors}), 400
        
    try:
        # Load and validate DataFrame
        df = data_loader.load_from_upload(file)
        df_errors = validator.validate_dataframe(df)
        
        if df_errors:
            return jsonify({'error': 'Data validation failed', 'details': df_errors}), 400
            
        # Continue with normal processing...
        
    except Exception as e:
        logger.error(f"Upload processing failed: {str(e)}")
        return jsonify({'error': 'Processing failed'}), 500
```

---

## Appendices

### Appendix A: Mathematical Foundations

#### A.1 Ensemble Scoring Mathematics

The ensemble scoring system combines multiple algorithm outputs using weighted voting:

**Basic Ensemble Score**:
```
S_ensemble(x) = Σ(i=1 to n) w_i × S_i(x)
```

Where:
- `S_i(x)` = Score from algorithm i for sample x
- `w_i` = Weight of algorithm i
- `Σw_i = 1` (weights sum to 1)

**Confidence Calculation**:
```
Confidence(x) = 1 - exp(-α × S_ensemble(x))
```

Where α is a scaling parameter (typically 2.0).

**Consensus Score**:
```
Consensus(x) = |{i : S_i(x) > θ}| / n
```

Where θ is the individual algorithm threshold (typically 0.5).

#### A.2 Adaptive Learning Mathematics

**Weight Update Rule (Exponential Moving Average)**:
```
w_i(t+1) = (1-λ) × w_i(t) + λ × P_i(t)
```

Where:
- `λ` = Learning rate (0 < λ < 1)
- `P_i(t)` = Performance of algorithm i at time t
- `w_i(t)` = Current weight of algorithm i

**Performance Calculation**:
```
P_i(t) = Σ(j=t-k to t) f_j × e^(-(t-j)/τ) / Σ(j=t-k to t) e^(-(t-j)/τ)
```

Where:
- `f_j` = Feedback score at time j (1 for correct, 0 for incorrect)
- `τ` = Time decay constant
- `k` = Window size for performance calculation

#### A.3 Statistical Tests for Drift Detection

**Population Stability Index (PSI)**:
```
PSI = Σ(i=1 to n) (P_curr,i - P_ref,i) × ln(P_curr,i / P_ref,i)
```

Where:
- `P_curr,i` = Proportion in bin i for current data
- `P_ref,i` = Proportion in bin i for reference data

**Kolmogorov-Smirnov Test Statistic**:
```
D_n,m = sup_x |F_n(x) - G_m(x)|
```

Where:
- `F_n(x)` = Empirical distribution function of reference sample
- `G_m(x)` = Empirical distribution function of current sample

### Appendix B: Algorithm Parameter Reference

#### B.1 Isolation Forest Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_estimators | int | 200 | Number of trees in the forest |
| contamination | float | 0.1 | Expected proportion of outliers |
| max_samples | int/float | 'auto' | Number of samples to draw per tree |
| max_features | int/float | 1.0 | Number of features to draw per tree |
| bootstrap | bool | False | Whether samples are drawn with replacement |
| random_state | int | None | Random seed for reproducibility |

**Tuning Guidelines**:
- Increase `n_estimators` for more stable results (diminishing returns after 200)
- Adjust `contamination` based on expected anomaly rate in your domain
- Use `max_samples=256` for large datasets (>100k records) for efficiency
- Set `bootstrap=True` for small datasets to increase diversity

#### B.2 Local Outlier Factor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| n_neighbors | int | 20 | Number of neighbors for density estimation |
| algorithm | str | 'auto' | Algorithm for nearest neighbors search |
| leaf_size | int | 30 | Leaf size for tree algorithms |
| metric | str | 'minkowski' | Distance metric |
| p | int | 2 | Parameter for Minkowski metric |
| contamination | float | 0.1 | Expected proportion of outliers |

**Tuning Guidelines**:
- Use `n_neighbors=5-50` depending on dataset size and local structure
- For high-dimensional data, consider `metric='cosine'` or `metric='manhattan'`
- Increase `n_neighbors` for smoother decision boundaries
- Decrease `n_neighbors` to capture fine-grained local patterns

#### B.3 One-Class SVM Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| kernel | str | 'rbf' | Kernel type |
| degree | int | 3 | Degree for polynomial kernel |
| gamma | str/float | 'scale' | Kernel coefficient |
| coef0 | float | 0.0 | Independent term in kernel |
| tol | float | 1e-3 | Tolerance for stopping criterion |
| nu | float | 0.5 | Upper bound on fraction of training errors |
| shrinking | bool | True | Whether to use shrinking heuristic |
| cache_size | float | 200 | Kernel cache size (MB) |

**Tuning Guidelines**:
- Use `kernel='rbf'` for most cases, `kernel='linear'` for high-dimensional data
- Adjust `nu` to expected outlier fraction (similar to contamination parameter)
- Use `gamma='scale'` or `gamma='auto'` for automatic gamma selection
- Increase `cache_size` for better performance on large datasets

#### B.4 Deep Autoencoder Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| encoding_dims | list | [64,32,16] | Dimensions of encoding layers |
| learning_rate | float | 0.001 | Adam optimizer learning rate |
| epochs | int | 50 | Number of training epochs |
| batch_size | int | 32 | Training batch size |
| validation_split | float | 0.1 | Fraction of data for validation |
| dropout_rate | float | 0.1 | Dropout rate for regularization |
| activation | str | 'relu' | Activation function |
| loss | str | 'mse' | Loss function |

**Tuning Guidelines**:
- Adjust `encoding_dims` based on input dimensionality (bottleneck should be 10-20% of input)
- Increase `epochs` for complex data, decrease for simple patterns
- Use larger `batch_size` for stable training, smaller for noisy gradients
- Increase `dropout_rate` if overfitting is observed

### Appendix C: Performance Benchmarks

#### C.1 Algorithm Performance Comparison

| Algorithm | Time Complexity | Space Complexity | Scalability | Interpretability |
|-----------|----------------|------------------|-------------|------------------|
| Isolation Forest | O(n log n) | O(n) | Excellent | Medium |
| Local Outlier Factor | O(n²) | O(n) | Poor | High |
| One-Class SVM | O(n²) to O(n³) | O(n) | Poor | Low |
| Deep Autoencoder | O(n × epochs) | O(n) | Good | Low |
| Statistical Methods | O(n) | O(1) | Excellent | High |
| DBSCAN | O(n log n) | O(n) | Good | High |
| Temporal Analysis | O(n) | O(w) | Excellent | High |

#### C.2 Memory Usage Guidelines

| Dataset Size | Recommended RAM | Notes |
|--------------|----------------|-------|
| < 10K records | 4GB | All algorithms work well |
| 10K - 100K | 8GB | Avoid LOF for >50K records |
| 100K - 1M | 16GB | Use sampling for SVM and LOF |
| 1M - 10M | 32GB+ | Streaming/batch processing required |
| > 10M | 64GB+ | Distributed processing recommended |

#### C.3 Processing Time Estimates

**Single-threaded performance on Intel i7 CPU**:

| Dataset Size | Isolation Forest | LOF | One-Class SVM | Autoencoder | Statistical |
|--------------|------------------|-----|---------------|-------------|-------------|
| 1K records | 0.1s | 0.5s | 1.0s | 5s | 0.01s |
| 10K records | 1s | 50s | 100s | 30s | 0.1s |
| 100K records | 10s | 5000s* | 10000s* | 300s | 1s |

*Not recommended for this dataset size

### Appendix D: Error Codes and Messages

#### D.1 System Error Codes

| Code | Category | Message | Resolution |
|------|----------|---------|-----------|
| ADQ-001 | File Upload | File size exceeds maximum limit | Reduce file size or increase limit |
| ADQ-002 | File Upload | Unsupported file format | Convert to CSV or Parquet |
| ADQ-003 | Data Processing | Insufficient data for analysis | Provide at least 50 records |
| ADQ-004 | Data Processing | Too many missing values | Clean data or adjust thresholds |
| ADQ-005 | ML Processing | Algorithm initialization failed | Check memory and parameters |
| ADQ-006 | ML Processing | Feature engineering failed | Validate data types and structure |
| ADQ-007 | Database | Connection failed | Check database connectivity |
| ADQ-008 | Database | Query timeout | Optimize query or increase timeout |
| ADQ-009 | Authentication | Invalid session | Re-authenticate |
| ADQ-010 | Configuration | Invalid parameter value | Check configuration documentation |

#### D.2 Algorithm-Specific Errors

**Isolation Forest**:
- `IF-001`: Contamination parameter out of range (0,1)
- `IF-002`: Insufficient features for tree building
- `IF-003`: Memory allocation failed for forest

**Local Outlier Factor**:
- `LOF-001`: Number of neighbors exceeds dataset size
- `LOF-002`: Distance metric not supported
- `LOF-003`: Numerical overflow in distance calculation

**Autoencoder**:
- `AE-001`: Network architecture invalid
- `AE-002`: Training convergence failed
- `AE-003`: GPU memory insufficient

### Appendix E: Integration Examples

#### E.1 Apache Airflow Integration

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd

def run_data_quality_check(**context):
    """
    Airflow task for automated data quality checking
    """
    
    # Load data from data warehouse
    query = """
    SELECT * FROM production.customer_transactions 
    WHERE created_date >= CURRENT_DATE - INTERVAL '1 day'
    """
    
    df = pd.read_sql(query, connection_string)
    
    # Run quality check
    detector = AdvancedEnsembleDetector(AdvancedConfig)
    results = detector.detect_anomalies(df)
    
    # Store results
    audit_store = AuditStore()
    session_id = audit_store.record_quality_check(
        dataset_name=f"daily_transactions_{context['ds']}",
        total_records=len(df),
        flagged_records=len(results['flagged_indices']),
        results=results,
        timestamp=datetime.now()
    )
    
    # Alert if high anomaly rate
    anomaly_rate = len(results['flagged_indices']) / len(df)
    if anomaly_rate > 0.05:  # More than 5% anomalies
        send_alert(f"High anomaly rate detected: {anomaly_rate:.2%}")
    
    return session_id

# DAG definition
dag = DAG(
    'daily_data_quality_check',
    default_args={
        'owner': 'data_team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 1,
        'retry_delay': timedelta(minutes=5)
    },
    description='Daily automated data quality checking',
    schedule_interval='0 6 * * *',  # Run at 6 AM daily
    catchup=False
)

quality_check_task = PythonOperator(
    task_id='run_quality_check',
    python_callable=run_data_quality_check,
    dag=dag
)
```

#### E.2 Kafka Streaming Integration

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import pandas as pd

class StreamingQualityChecker:
    """
    Real-time data quality checking with Kafka
    """
    
    def __init__(self, input_topic, output_topic, bootstrap_servers):
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
        self.detector = AdvancedEnsembleDetector(AdvancedConfig)
        self.batch_size = 1000
        self.batch_buffer = []
        
    def process_stream(self):
        """
        Process streaming data in micro-batches
        """
        
        for message in self.consumer:
            record = message.value
            self.batch_buffer.append(record)
            
            # Process when batch is full
            if len(self.batch_buffer) >= self.batch_size:
                self._process_batch()
                self.batch_buffer = []
                
    def _process_batch(self):
        """
        Process a batch of records
        """
        
        # Convert to DataFrame
        df = pd.DataFrame(self.batch_buffer)
        
        # Run quality check
        results = self.detector.detect_anomalies(df)
        
        # Send alerts for anomalies
        for idx in results['flagged_indices']:
            anomaly_record = {
                'original_record': self.batch_buffer[idx],
                'anomaly_score': results['ensemble_probabilities'][idx],
                'confidence_level': results['confidence_levels'].get(idx, 'low'),
                'timestamp': datetime.now().isoformat()
            }
            
            self.producer.send(self.output_topic, anomaly_record)

# Usage
checker = StreamingQualityChecker(
    input_topic='raw_data',
    output_topic='data_quality_alerts',
    bootstrap_servers=['localhost:9092']
)

checker.process_stream()
```

### Appendix F: Glossary

**Adaptive Learning**: The system's ability to improve its performance based on feedback and experience, automatically adjusting algorithm weights and thresholds.

**Anomaly Score**: A numerical value (typically 0-1) indicating how anomalous a record is, with higher scores indicating greater likelihood of being an anomaly.

**Autoencoder**: A neural network architecture that learns to reconstruct input data through a compressed representation, useful for detecting complex patterns.

**Consensus Threshold**: The minimum agreement level required between multiple algorithms before flagging a record as anomalous.

**Contamination Rate**: The expected proportion of anomalies in a dataset, used as a parameter for many anomaly detection algorithms.

**Data Drift**: The phenomenon where the statistical properties of data change over time, potentially affecting model performance.

**Ensemble Method**: A technique that combines multiple algorithms to achieve better performance than any single algorithm alone.

**Feature Engineering**: The process of creating new features from raw data to improve algorithm performance.

**Human-in-the-Loop**: A system design that incorporates human feedback to improve automated decision-making.

**Isolation Forest**: A tree-based anomaly detection algorithm that isolates anomalies by randomly selecting features and split values.

**Local Outlier Factor (LOF)**: A density-based anomaly detection algorithm that identifies points with unusual local density compared to their neighbors.

**One-Class SVM**: A support vector machine variant that learns a decision boundary around normal data points.

**Population Stability Index (PSI)**: A statistical measure used to detect changes in data distribution over time.

**Precision**: The proportion of flagged records that are actually anomalies (true positives / (true positives + false positives)).

**Recall**: The proportion of actual anomalies that are correctly flagged (true positives / (true positives + false negatives)).

**Temporal Anomaly**: An anomaly that can only be detected by considering the time dimension of the data.

**Weighted Voting**: A method of combining algorithm outputs where each algorithm's contribution is multiplied by its performance-based weight.

---

## Conclusion

The Advanced Adaptive Data Quality Framework represents a significant evolution in data quality management, moving beyond static rule-based systems to intelligent, self-improving solutions. By combining multiple complementary algorithms, incorporating human expertise, and continuously adapting to new patterns, this framework provides a robust foundation for maintaining high-quality data in modern environments.

The comprehensive documentation provided here should enable organizations to:
- Understand the theoretical foundations of adaptive data quality
- Successfully implement and deploy the framework
- Customize the system for their specific domains and requirements
- Integrate with existing data infrastructure and workflows
- Monitor and maintain the system in production environments

As data complexity continues to grow and quality requirements become more stringent, adaptive approaches like this framework will become essential tools for data professionals. The combination of machine learning, human intelligence, and software engineering best practices demonstrated here provides a template for building production-ready, intelligent data quality systems.

For questions, support, or contributions to this framework, please refer to the project repository and community resources.

---

*Last updated: December 2024*
*Version: 2.0.0*
*Framework License: MIT*