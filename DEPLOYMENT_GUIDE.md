# CATIA Deployment & Operations Guide

## Deployment Options

### 1. Local Development

**Setup:**
```bash
cd CATIA
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Run:**
```bash
python main.py
```

**Output:**
- `outputs/catia_report.json` - Comprehensive report
- `outputs/loss_exceedance_curve.html` - Interactive visualization
- `outputs/risk_distribution.html` - Risk histogram
- `outputs/return_period_curve.html` - Return period analysis
- `outputs/mitigation_comparison.html` - Strategy comparison

---

### 2. Docker Containerization

**Build Image:**
```bash
docker build -t catia:latest .
```

**Run Container:**
```bash
# Single run
docker run --rm -v $(pwd)/outputs:/app/outputs catia:latest

# Interactive
docker run -it --rm -v $(pwd)/outputs:/app/outputs catia:latest bash
```

**Docker Compose (Optional):**
```yaml
version: '3.8'
services:
  catia:
    build: .
    volumes:
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    environment:
      - CATIA_MOCK_DATA=true
    ports:
      - "8050:8050"
```

**Run with Compose:**
```bash
docker-compose up
```

---

### 3. AWS Lambda Deployment

**Prerequisites:**
- AWS CLI configured
- IAM role with Lambda permissions

**Package:**
```bash
# Create deployment package
mkdir lambda_package
pip install -r requirements.txt -t lambda_package/
cp -r src lambda_package/
cp config.py main.py lambda_package/
cd lambda_package && zip -r ../lambda_function.zip . && cd ..
```

**Deploy:**
```bash
aws lambda create-function \
  --function-name catia-analysis \
  --runtime python3.11 \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-role \
  --handler main.run_catia_analysis \
  --zip-file fileb://lambda_function.zip \
  --timeout 300 \
  --memory-size 1024
```

**Invoke:**
```bash
aws lambda invoke \
  --function-name catia-analysis \
  --payload '{"region": "US_Gulf_Coast", "use_mock_data": true}' \
  response.json
```

**Update:**
```bash
aws lambda update-function-code \
  --function-name catia-analysis \
  --zip-file fileb://lambda_function.zip
```

---

### 4. Google Cloud Functions

**Prerequisites:**
- Google Cloud SDK installed
- Project configured

**Deploy:**
```bash
gcloud functions deploy catia-analysis \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --entry-point run_catia_analysis \
  --memory 1024MB \
  --timeout 300s
```

**Invoke:**
```bash
gcloud functions call catia-analysis \
  --data '{"region":"US_Gulf_Coast","use_mock_data":true}'
```

---

### 5. Azure Functions

**Prerequisites:**
- Azure CLI installed
- Storage account created

**Deploy:**
```bash
# Create function app
az functionapp create \
  --resource-group myResourceGroup \
  --consumption-plan-location eastus \
  --runtime python \
  --runtime-version 3.11 \
  --functions-version 4 \
  --name catia-analysis

# Deploy code
func azure functionapp publish catia-analysis
```

---

### 6. Kubernetes Deployment

**Create Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: catia-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: catia
  template:
    metadata:
      labels:
        app: catia
    spec:
      containers:
      - name: catia
        image: catia:latest
        ports:
        - containerPort: 8050
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        volumeMounts:
        - name: outputs
          mountPath: /app/outputs
      volumes:
      - name: outputs
        emptyDir: {}
```

**Deploy:**
```bash
kubectl apply -f deployment.yaml
kubectl expose deployment catia-deployment --type=LoadBalancer --port=8050
```

---

## Configuration Management

### Environment Variables

```bash
# Mock data mode
export CATIA_MOCK_DATA=true

# API credentials
export NOAA_API_KEY=your_key
export ECMWF_API_KEY=your_key
export ECMWF_API_URL=your_url

# Output directory
export CATIA_OUTPUT_DIR=/path/to/outputs

# Logging
export CATIA_LOG_LEVEL=INFO
```

### Configuration File

Edit `config.py` for:
- API endpoints
- ML hyperparameters
- Simulation parameters
- Risk metrics thresholds
- Mitigation strategies
- Output settings

---

## Monitoring & Logging

### Local Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/catia.log'),
        logging.StreamHandler()
    ]
)
```

### Cloud Logging

**AWS CloudWatch:**
```bash
# View logs
aws logs tail /aws/lambda/catia-analysis --follow
```

**Google Cloud Logging:**
```bash
gcloud functions logs read catia-analysis --limit 50
```

**Azure Monitor:**
```bash
az monitor log-analytics query \
  --workspace myWorkspace \
  --analytics-query "traces | where message contains 'CATIA'"
```

---

## Performance Optimization

### Parallel Processing

```python
from multiprocessing import Pool

def run_simulation(iteration):
    simulator = FinancialImpactSimulator(0.5, {'mu': 15, 'sigma': 2})
    return simulator.simulate_annual_losses(1)

with Pool(processes=4) as pool:
    results = pool.map(run_simulation, range(10000))
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def get_climate_data(region, date):
    # Cached results
    pass
```

### Database Connection Pooling

```python
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://user:password@localhost/catia',
    pool_size=20,
    max_overflow=40
)
```

---

## Monitoring & Alerts

### Health Checks

```python
def health_check():
    try:
        # Test data acquisition
        data = fetch_all_data("test_region", use_mock=True)
        
        # Test model
        predictor = RiskPredictor()
        
        # Test simulation
        simulator = FinancialImpactSimulator(0.5, {'mu': 15, 'sigma': 2})
        
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

### Metrics Collection

```python
import time

def track_execution_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"{func.__name__} took {duration:.2f}s")
        return result
    return wrapper

@track_execution_time
def run_analysis():
    # Analysis code
    pass
```

---

## Backup & Recovery

### Data Backup

```bash
# Backup outputs
tar -czf catia_backup_$(date +%Y%m%d).tar.gz outputs/

# Backup models
cp models/risk_model.pkl models/risk_model_backup_$(date +%Y%m%d).pkl
```

### Model Versioning

```python
import pickle
from datetime import datetime

def save_model_version(model, version_name=None):
    if version_name is None:
        version_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    path = f"models/risk_model_{version_name}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    return path
```

---

## Scaling Considerations

### Horizontal Scaling
- Deploy multiple instances
- Use load balancer
- Distribute data fetching
- Parallelize simulations

### Vertical Scaling
- Increase memory allocation
- Increase CPU allocation
- Optimize algorithms
- Use faster hardware

### Database Scaling
- Read replicas for queries
- Write master for updates
- Caching layer (Redis)
- Partitioning for large datasets

---

## Security Best Practices

### API Credentials

```bash
# Use environment variables (never hardcode)
import os
api_key = os.getenv('NOAA_API_KEY')

# Use secrets manager
from aws_secretsmanager_caching import SecretCache
cache = SecretCache()
secret = cache.get_secret_string('catia/noaa-api-key')
```

### Data Encryption

```bash
# Encrypt sensitive data
openssl enc -aes-256-cbc -in data.csv -out data.csv.enc

# Decrypt
openssl enc -d -aes-256-cbc -in data.csv.enc -out data.csv
```

### Access Control

```python
# Implement authentication
from functools import wraps

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not verify_token(token):
            return {"error": "Unauthorized"}, 401
        return f(*args, **kwargs)
    return decorated
```

---

## Troubleshooting

### Common Issues

**Issue:** Out of memory
- **Solution:** Reduce `monte_carlo_iterations`, use streaming

**Issue:** Slow API calls
- **Solution:** Implement caching, use parallel requests

**Issue:** Model drift
- **Solution:** Retrain monthly, monitor loss ratio

**Issue:** Deployment failures
- **Solution:** Check logs, verify dependencies, test locally

---

## Maintenance Schedule

### Daily
- Monitor logs for errors
- Check health endpoints
- Verify data freshness

### Weekly
- Review performance metrics
- Check for model drift
- Validate data quality

### Monthly
- Retrain models with new data
- Update dependencies
- Backup all data and models

### Quarterly
- Full system audit
- Performance optimization
- Security review

---

## Support & Documentation

- **README.md:** Overview
- **QUICK_START.md:** Getting started
- **IMPLEMENTATION_GUIDE.md:** Detailed setup
- **API_REFERENCE.md:** API documentation
- **ACTUARIAL_METHODOLOGY.md:** Mathematical foundations

