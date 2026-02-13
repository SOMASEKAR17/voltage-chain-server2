# Voltage Chain Battery Prediction API - Route Documentation

## Overview

The Voltage Chain Battery Prediction API provides endpoints for predicting battery health, remaining useful life (RUL), and receiving recommendations based on simplified battery metrics.

**Base URL:** `http://localhost:8000/api`

---

## Routes

### 1. Health Check

**Endpoint:** [GET `/api/health`]

**Description:** Check if the API is running and healthy.

**Method:** `GET`

**Parameters:** None

**Request Example:**

```bash
curl -X GET "http://localhost:8000/api/health"
```

**Response (200 OK):**

```json
{
  "status": "ok",
  "message": "Battery Prediction API is healthy ✅"
}
```

---

### 2. Battery RUL Prediction (Simplified)

**Endpoint:** [POST `/api/predict-rul`]

**Description:** Predict battery Remaining Useful Life (RUL) based on essential battery parameters. This simplified interface accepts only the core metrics needed for reliable predictions.

**Method:** `POST`

**Content-Type:** `application/json`

**Request Body Parameters:**

| Parameter             | Type    | Required | Example | Description                             |
| --------------------- | ------- | -------- | ------- | --------------------------------------- |
| `current_capacity`    | float   | ✅ Yes   | 2.0     | Current capacity in Ahr (Ampere-hours)  |
| `initial_capacity`    | float   | ✅ Yes   | 3.0     | Initial/Rated capacity in Ahr           |
| `ambient_temperature` | float   | ❌ No    | 25      | Ambient temperature in °C (default: 25) |
| `cycle_count`         | integer | ✅ Yes   | 40      | Total number of charge/discharge cycles |
| `age_days`            | float   | ✅ Yes   | 300     | Battery age in days                     |

**Request Example:**

```bash
curl -X POST "http://localhost:8000/api/predict-rul" \
  -H "Content-Type: application/json" \
  -d '{
    "current_capacity": 2.0,
    "initial_capacity": 3.0,
    "ambient_temperature": 29,
    "cycle_count": 40,
    "age_days": 300
  }'
```

**Response (200 OK):**

```json
{
  "success": true,
  "battery_metrics": {
    "initial_capacity_ahr": 3.0,
    "current_capacity_ahr": 2.0,
    "cycle_count": 40,
    "age_days": 300,
    "ambient_temperature_c": 29
  },
  "health_analysis": {
    "soh_percentage": 66.7,
    "health_status": "POOR",
    "health_description": "Significant degradation, limited lifespan",
    "degradation_factor_percent": 33.3
  },
  "rul_prediction": {
    "rul_cycles": 51,
    "estimated_days_to_eol": 385,
    "estimated_time_to_eol": "385 days (~12.8 months)"
  },
  "recommendations": [
    "Limit to non-critical backup applications only",
    "Schedule replacement soon",
    "Avoid high-power draw applications"
  ]
}
```

**Response (400 Bad Request):**

```json
{
  "detail": "Prediction error: Current capacity must be less than or equal to initial capacity"
}
```

**Response (422 Unprocessable Entity):**

```json
{
  "detail": [
    {
      "loc": ["body", "current_capacity"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

### 3. Get Battery Health Status

**Endpoint:** [GET `/api/health-status/{soh_percentage}`]

**Description:** Get battery health status and description for a specific State of Health percentage value.

**Method:** `GET`

**Path Parameters:**
| Parameter | Type | Required | Range | Description |
|-----------|------|----------|-------|-------------|
| `soh_percentage` | float | ✅ Yes | 0-100 | State of Health as percentage (e.g., 66.7 for 66.7%) |

**Request Example:**

```bash
curl -X GET "http://localhost:8000/api/health-status/66.7"
```

**Response (200 OK):**

```json
{
  "soh_percentage": 66.7,
  "health_status": "POOR",
  "health_description": "Significant degradation, limited lifespan"
}
```

**Response (400 Bad Request):**

```json
{
  "detail": "SoH percentage must be between 0 and 100"
}
```

**Health Status Categories:**

| Status    | SoH Range | Description                               | Degradation Level |
| --------- | --------- | ----------------------------------------- | ----------------- |
| EXCELLENT | 95-100%   | New or near-new condition                 | 0-5%              |
| GOOD      | 85-94%    | Minimal degradation, normal operation     | 6-15%             |
| FAIR      | 70-84%    | Noticeable degradation, approaching EOL   | 16-30%            |
| POOR      | 60-69%    | Significant degradation, limited lifespan | 31-40%            |
| CRITICAL  | < 60%     | Near end-of-life, replacement recommended | > 40%             |

---

### 4. Test Prediction (Example Data)

**Endpoint:** [GET `/api/test-prediction`]

**Description:** Test the prediction endpoint with pre-configured example battery data. This endpoint simulates real-world battery analysis.

**Method:** `GET`

**Parameters:** None

**Example Data:**

- Current Capacity: 2.0 Ahr
- Initial Capacity: 3.0 Ahr
- Cycle Count: 40 cycles
- Age: 300 days
- Ambient Temperature: 29°C

**Request Example:**

```bash
curl -X GET "http://localhost:8000/api/test-prediction"
```

**Response (200 OK):**

```json
{
  "success": true,
  "battery_metrics": {
    "initial_capacity_ahr": 3.0,
    "current_capacity_ahr": 2.0,
    "cycle_count": 40,
    "age_days": 300,
    "ambient_temperature_c": 29
  },
  "health_analysis": {
    "soh_percentage": 66.7,
    "health_status": "POOR",
    "health_description": "Significant degradation, limited lifespan",
    "degradation_factor_percent": 33.3
  },
  "rul_prediction": {
    "rul_cycles": 51,
    "estimated_days_to_eol": 385,
    "estimated_time_to_eol": "385 days (~12.8 months)"
  },
  "recommendations": [
    "Limit to non-critical backup applications only",
    "Schedule replacement soon",
    "Avoid high-power draw applications"
  ]
}
```

---

## Response Fields Explanation

### battery_metrics

Contains the input battery parameters in a structured format:

- `initial_capacity_ahr`: Initial/rated capacity in Ampere-hours
- `current_capacity_ahr`: Current measured capacity in Ampere-hours
- `cycle_count`: Total number of charge/discharge cycles
- `age_days`: Battery age in days
- `ambient_temperature_c`: Ambient temperature in Celsius

### health_analysis

Contains the battery health assessment:

- `soh_percentage`: State of Health as a percentage (0-100%)
- `health_status`: Categorical health status (EXCELLENT, GOOD, FAIR, POOR, CRITICAL)
- `health_description`: Human-readable health description
- `degradation_factor_percent`: Percentage of capacity lost due to degradation

### rul_prediction

Contains the remaining useful life predictions:

- `rul_cycles`: Predicted remaining cycles until reaching End of Life (70% SoH)
- `estimated_days_to_eol`: Estimated days until End of Life
- `estimated_time_to_eol`: Human-readable format (e.g., "385 days (~12.8 months)")

### recommendations

Array of actionable recommendations based on battery health:

- Usage limitations
- Maintenance suggestions
- Replacement timeline guidance

---

## Health Status Recommendations

### EXCELLENT (SoH ≥ 95%)

**Degradation:** 0-5%

- Primary power source for all applications including critical ones
- No action needed - battery in excellent condition
- Suitable for mission-critical operations

### GOOD (SoH 85-94%)

**Degradation:** 6-15%

- Suitable for primary power applications
- Continue normal operation
- Monitor capacity periodically

### FAIR (SoH 70-84%)

**Degradation:** 16-30%

- Acceptable for non-critical applications only
- Plan for replacement within next cycle
- Monitor battery health closely

### POOR (SoH 60-69%)

**Degradation:** 31-40%

- Limit to non-critical backup applications only
- Schedule replacement soon
- Avoid high-power draw applications

### CRITICAL (SoH < 60%)

**Degradation:** > 40%

- Emergency use only - replacement urgent
- Avoid critical applications
- Plan replacement immediately

---

## Error Responses

### 400 Bad Request

Returned when required parameters are missing, invalid, or prediction fails.

```json
{
  "detail": "Prediction error: <error message>"
}
```

**Common scenarios:**

- Current capacity greater than initial capacity
- Negative cycle count or age
- Missing required fields

### 404 Not Found

Returned when endpoint doesn't exist.

### 422 Unprocessable Entity

Returned when request body has validation errors.

```json
{
  "detail": [
    {
      "loc": ["body", "parameter_name"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

---

## Model Details

- **Algorithm:** XGBoost Regressor
- **Target:** Remaining Useful Life (RUL) in cycles
- **Training R² Score:** ~0.92
- **End of Life Threshold:** 70% SoH (when predicted RUL reaches 0)
- **Features:** Automatically derived from basic battery parameters

---

## Example Integration

### Python

```python
import requests

url = "http://localhost:8000/api/predict-rul"
payload = {
    "current_capacity": 2.0,
    "initial_capacity": 3.0,
    "ambient_temperature": 29,
    "cycle_count": 40,
    "age_days": 300
}

response = requests.post(url, json=payload)
result = response.json()

if result['success']:
    metrics = result['battery_metrics']
    health = result['health_analysis']
    rul = result['rul_prediction']

    print(f"Current Capacity: {metrics['current_capacity_ahr']} Ahr")
    print(f"Health Status: {health['health_status']}")
    print(f"SoH: {health['soh_percentage']}%")
    print(f"RUL: {rul['rul_cycles']} cycles")
    print(f"Time to EOL: {rul['estimated_time_to_eol']}")
    print(f"Recommendations: {', '.join(result['recommendations'])}")
```

### JavaScript/Node.js

```javascript
const url = "http://localhost:8000/api/predict-rul";
const payload = {
  current_capacity: 2.0,
  initial_capacity: 3.0,
  ambient_temperature: 29,
  cycle_count: 40,
  age_days: 300,
};

fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
})
  .then((res) => res.json())
  .then((data) => {
    if (data.success) {
      console.log(`Health Status: ${data.health_analysis.health_status}`);
      console.log(`SoH: ${data.health_analysis.soh_percentage}%`);
      console.log(`RUL: ${data.rul_prediction.rul_cycles} cycles`);
      console.log(`Time to EOL: ${data.rul_prediction.estimated_time_to_eol}`);
    }
  });
```

### cURL

```bash
# Quick test with example data
curl -X GET "http://localhost:8000/api/test-prediction"

# With custom parameters
curl -X POST "http://localhost:8000/api/predict-rul" \
  -H "Content-Type: application/json" \
  -d '{
    "current_capacity": 2.0,
    "initial_capacity": 3.0,
    "ambient_temperature": 29,
    "cycle_count": 40,
    "age_days": 300
  }'

# Check health status
curl -X GET "http://localhost:8000/api/health-status/66.7"
```

---

## API Documentation UI

Access the interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## Deployment Checklist

- [ ] All required model files present (`web3_battery_rul.json`, `feature_scaler.pkl`, `model_metadata.json`)
- [ ] FastAPI server running on correct port (default: 8000)
- [ ] Virtual environment activated with all dependencies installed
- [ ] CORS enabled if frontend is on different origin
- [ ] Error logging configured

---

_Last Updated: February 14, 2026_
_API Version: 2.0 - Simplified Input Interface_
