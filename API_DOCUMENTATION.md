# Voltage Chain Battery Prediction API - Route Documentation

## Overview

The Voltage Chain Battery Prediction API provides endpoints for predicting battery health, remaining useful life (RUL), and receiving recommendations based on battery metrics and cycle features.

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

### 2. Battery RUL Prediction

**Endpoint:** [POST `/api/predict-rul`]

**Description:** Predict battery Remaining Useful Life (RUL) based on comprehensive battery parameters and cycle features.

**Method:** `POST`

**Content-Type:** `application/json`

**Request Body Parameters:**

#### Cycle Features

| Parameter           | Type  | Required | Default | Description                            |
| ------------------- | ----- | -------- | ------- | -------------------------------------- |
| `cycle_duration`    | float | ✅ Yes   | -       | Duration of discharge cycle in seconds |
| `measurement_count` | int   | ❌ No    | 1000    | Number of measurements in cycle        |

#### Voltage Features

| Parameter       | Type  | Required | Default | Description                   |
| --------------- | ----- | -------- | ------- | ----------------------------- |
| `voltage_mean`  | float | ❌ No    | 3.8     | Mean voltage during cycle (V) |
| `voltage_std`   | float | ❌ No    | 0.1     | Voltage standard deviation    |
| `voltage_min`   | float | ❌ No    | 3.5     | Minimum voltage (V)           |
| `voltage_max`   | float | ❌ No    | 4.2     | Maximum voltage (V)           |
| `voltage_range` | float | ❌ No    | 0.7     | Voltage range (max-min)       |
| `voltage_drop`  | float | ❌ No    | 0.5     | Voltage drop during cycle     |

#### Current Features

| Parameter      | Type  | Required | Default | Description                |
| -------------- | ----- | -------- | ------- | -------------------------- |
| `current_mean` | float | ❌ No    | -2.0    | Mean current (A)           |
| `current_std`  | float | ❌ No    | 0.1     | Current standard deviation |
| `current_min`  | float | ❌ No    | -2.2    | Minimum current (A)        |
| `current_max`  | float | ❌ No    | -1.8    | Maximum current (A)        |

#### Temperature Features

| Parameter    | Type  | Required | Default | Description                    |
| ------------ | ----- | -------- | ------- | ------------------------------ |
| `temp_mean`  | float | ❌ No    | 25.0    | Mean temperature (°C)          |
| `temp_std`   | float | ❌ No    | 2.0     | Temperature standard deviation |
| `temp_min`   | float | ❌ No    | 23.0    | Minimum temperature (°C)       |
| `temp_max`   | float | ❌ No    | 28.0    | Maximum temperature (°C)       |
| `temp_range` | float | ❌ No    | 5.0     | Temperature range (°C)         |

#### Power Features

| Parameter    | Type  | Required | Default | Description       |
| ------------ | ----- | -------- | ------- | ----------------- |
| `power_mean` | float | ❌ No    | 7.6     | Mean power (W)    |
| `power_max`  | float | ❌ No    | 9.2     | Maximum power (W) |

#### Battery Health Metrics

| Parameter             | Type  | Required | Default | Description                       |
| --------------------- | ----- | -------- | ------- | --------------------------------- |
| `Capacity`            | float | ✅ Yes   | -       | Current capacity (Ah)             |
| `ambient_temperature` | float | ❌ No    | 25.0    | Ambient temperature (°C)          |
| `cycle_count`         | int   | ✅ Yes   | -       | Number of charge/discharge cycles |
| `age_days`            | float | ✅ Yes   | -       | Battery age in days               |
| `initial_capacity`    | float | ✅ Yes   | -       | Initial capacity (Ah)             |
| `SoH`                 | float | ✅ Yes   | -       | State of Health (0.0 - 1.0)       |

#### Degradation Metrics

| Parameter                        | Type  | Required | Default | Description                             |
| -------------------------------- | ----- | -------- | ------- | --------------------------------------- |
| `capacity_degradation`           | float | ✅ Yes   | -       | Capacity degradation level (0-1)        |
| `capacity_slope`                 | float | ✅ Yes   | -       | Rate of capacity loss per cycle         |
| `normalized_degradation_rate`    | float | ✅ Yes   | -       | Normalized degradation rate             |
| `capacity_gradient_pct`          | float | ✅ Yes   | -       | Capacity gradient percentage per cycle  |
| `cycle_utilization`              | float | ✅ Yes   | -       | Cycle utilization ratio (0-1)           |
| `degradation_acceleration`       | float | ✅ Yes   | -       | Rate of acceleration of degradation     |
| `daily_degradation_rate`         | float | ✅ Yes   | -       | Degradation rate per day                |
| `extrapolated_rul_from_gradient` | float | ✅ Yes   | -       | RUL extrapolated from capacity gradient |

**Request Example:**

```bash
curl -X POST "http://localhost:8000/api/predict-rul" \
  -H "Content-Type: application/json" \
  -d '{
    "cycle_duration": 3600.0,
    "measurement_count": 1000,
    "voltage_mean": 3.8,
    "voltage_std": 0.1,
    "voltage_min": 3.5,
    "voltage_max": 4.2,
    "voltage_range": 0.7,
    "voltage_drop": 0.5,
    "current_mean": -2.0,
    "current_std": 0.1,
    "current_min": -2.2,
    "current_max": -1.8,
    "temp_mean": 25.0,
    "temp_std": 2.0,
    "temp_min": 23.0,
    "temp_max": 28.0,
    "temp_range": 5.0,
    "power_mean": 7.6,
    "power_max": 9.2,
    "Capacity": 3.0,
    "ambient_temperature": 25.0,
    "cycle_count": 500,
    "age_days": 365,
    "initial_capacity": 3.4,
    "SoH": 0.88,
    "capacity_degradation": 0.12,
    "capacity_slope": -0.0067,
    "normalized_degradation_rate": -0.002,
    "capacity_gradient_pct": 0.12,
    "cycle_utilization": 0.833,
    "degradation_acceleration": 0.00024,
    "daily_degradation_rate": 0.000329,
    "extrapolated_rul_from_gradient": 667
  }'
```

**Response (200 OK):**

```json
{
  "success": true,
  "rul_cycles": 667.5,
  "soh": 0.88,
  "health_status": "GOOD",
  "health_description": "Minimal degradation, normal operation",
  "recommendations": [
    "Suitable for primary power applications",
    "Continue normal operation",
    "Monitor capacity periodically"
  ],
  "model_r2_score": 0.92
}
```

**Response (400 Bad Request):**

```json
{
  "detail": "Prediction error: Missing required parameters"
}
```

---

### 3. Get Battery Health Status

**Endpoint:** [GET `/api/health-status/{soh}`]

**Description:** Get battery health status and description for a specific State of Health value.

**Method:** `GET`

**Path Parameters:**
| Parameter | Type | Required | Range | Description |
|-----------|------|----------|-------|-------------|
| `soh` | float | ✅ Yes | 0.0-1.0 | State of Health value |

**Request Example:**

```bash
curl -X GET "http://localhost:8000/api/health-status/0.88"
```

**Response (200 OK):**

```json
{
  "soh": 0.88,
  "health_status": "GOOD",
  "description": "Minimal degradation, normal operation"
}
```

**Response (400 Bad Request):**

```json
{
  "detail": "SoH must be between 0 and 1"
}
```

**Health Status Categories:**

| Status    | SoH Range   | Description                               |
| --------- | ----------- | ----------------------------------------- |
| EXCELLENT | ≥ 0.95      | New or near-new condition                 |
| GOOD      | 0.85 - 0.94 | Minimal degradation, normal operation     |
| FAIR      | 0.70 - 0.84 | Noticeable degradation, approaching EOL   |
| POOR      | 0.60 - 0.69 | Significant degradation, limited lifespan |
| CRITICAL  | < 0.60      | Near end-of-life, replacement recommended |

---

### 4. Test Prediction (Example Data)

**Endpoint:** [GET `/api/test-prediction`]

**Description:** Test the prediction endpoint with pre-configured example battery data. Useful for testing and demonstration.

**Method:** `GET`

**Parameters:** None

**Request Example:**

```bash
curl -X GET "http://localhost:8000/api/test-prediction"
```

**Response (200 OK):**

```json
{
  "success": true,
  "rul_cycles": 667.5,
  "soh": 0.88,
  "health_status": "GOOD",
  "health_description": "Minimal degradation, normal operation",
  "recommendations": [
    "Suitable for primary power applications",
    "Continue normal operation",
    "Monitor capacity periodically"
  ],
  "model_r2_score": 0.92
}
```

---

## Health Status Recommendations

### EXCELLENT (SoH ≥ 0.95)

- Primary power source - all applications including critical ones
- No action needed - battery in excellent condition

### GOOD (SoH 0.85-0.94)

- Suitable for primary power applications
- Continue normal operation
- Monitor capacity periodically

### FAIR (SoH 0.70-0.84)

- Acceptable for non-critical applications
- Plan for replacement within next cycle
- Monitor battery health closely

### POOR (SoH 0.60-0.69)

- Limit to non-critical backup applications only
- Schedule replacement soon
- Avoid high-power draw applications

### CRITICAL (SoH < 0.60)

- Emergency use only - replacement urgent
- Avoid critical applications
- Plan replacement immediately

---

## Error Responses

### 400 Bad Request

Returned when required parameters are missing or invalid.

```json
{
  "detail": "Prediction error: <error message>"
}
```

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

## Notes

- All numeric values should be float or integer types
- The API uses an XGBoost model for RUL prediction
- Features are automatically scaled using a pre-trained StandardScaler
- The model was trained with an R² test score of ~0.92
- Predictions are in terms of remaining cycles until battery reaches 70% SoH (End of Life threshold)
- All requests should include proper Content-Type headers

---

## Example Integration

### Python

```python
import requests

url = "http://localhost:8000/api/predict-rul"
payload = {
    "cycle_duration": 3600.0,
    "Capacity": 3.0,
    "cycle_count": 500,
    "age_days": 365,
    "initial_capacity": 3.4,
    "SoH": 0.88,
    "capacity_degradation": 0.12,
    "capacity_slope": -0.0067,
    "normalized_degradation_rate": -0.002,
    "capacity_gradient_pct": 0.12,
    "cycle_utilization": 0.833,
    "degradation_acceleration": 0.00024,
    "daily_degradation_rate": 0.000329,
    "extrapolated_rul_from_gradient": 667
}

response = requests.post(url, json=payload)
result = response.json()
print(f"RUL: {result['rul_cycles']} cycles")
print(f"Health Status: {result['health_status']}")
```

### JavaScript/Node.js

```javascript
const url = "http://localhost:8000/api/predict-rul";
const payload = {
  cycle_duration: 3600.0,
  Capacity: 3.0,
  cycle_count: 500,
  age_days: 365,
  initial_capacity: 3.4,
  SoH: 0.88,
  capacity_degradation: 0.12,
  capacity_slope: -0.0067,
  normalized_degradation_rate: -0.002,
  capacity_gradient_pct: 0.12,
  cycle_utilization: 0.833,
  degradation_acceleration: 0.00024,
  daily_degradation_rate: 0.000329,
  extrapolated_rul_from_gradient: 667,
};

fetch(url, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
})
  .then((res) => res.json())
  .then((data) => {
    console.log(`RUL: ${data.rul_cycles} cycles`);
    console.log(`Health: ${data.health_status}`);
  });
```

---

## API Documentation UI

Access the interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

_Last Updated: February 14, 2026_
