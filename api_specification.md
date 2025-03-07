# Marketing Mix Model API Specification

## Base URL
`https://mmm-api.render.com` (production)
`http://localhost:5000` (development)

## Authentication
All endpoints require API key authentication via header:
`X-API-Key: your_api_key_here`

## Endpoints

### 1. Upload Data
**Endpoint:** `/api/upload`
**Method:** POST
**Content-Type:** multipart/form-data

**Request:**
- `file`: CSV file containing marketing data (required)

**Response:**
```json
{
  "success": true,
  "message": "File uploaded successfully",
  "data": {
    "file_id": "abc123",
    "columns": ["date", "tv_spend", "digital_spend", "search_spend", "sales"],
    "rows": 104,
    "preview": [
      {"date": "2022-01-01", "tv_spend": 1000, "digital_spend": 500, ...},
      {"date": "2022-01-08", "tv_spend": 1200, "digital_spend": 550, ...}
    ]
  }
}
```

### 2. Train Model
**Endpoint:** /api/model/train
**Method:** POST
**Content-Type:** application/json

**Request:**
```json
{
  "file_id": "abc123",
  "config": {
    "target_variable": "sales",
    "date_variable": "date",
    "media_variables": ["tv_spend", "digital_spend", "search_spend"],
    "control_variables": ["price_index", "competitor_price_index"],
    "model_type": "log-log",  
    "adstock_params": {
      "tv_spend": {"decay_rate": 0.8, "max_lag": 8},
      "digital_spend": {"decay_rate": 0.5, "max_lag": 4},
      "search_spend": {"decay_rate": 0.3, "max_lag": 2}
    }
  }
}
```
**Response:**
```json
{
  "success": true,
  "message": "Model trained successfully",
  "data": {
    "model_id": "model_xyz789",
    "performance": {
      "r_squared": 0.85,
      "adj_r_squared": 0.83,
      "rmse": 156.2,
      "mape": 5.7
    }
  }
}
```

### 3. Get Model Results
**Endpoint:** /api/model/results/{model_id}
**Method:** GET

**Response:**
```json
{
  "success": true,
  "message": "Model results retrieved successfully",
  "data": {
    "model_id": "model_xyz789",
    "performance": {
      "r_squared": 0.85,
      "adj_r_squared": 0.83,
      "rmse": 156.2,
      "mape": 5.7
    },
    "elasticities": {
      "tv_spend": 0.15,
      "digital_spend": 0.08,
      "search_spend": 0.05
    },
    "coefficients": {
      "tv_spend": 0.0023,
      "digital_spend": 0.0016,
      "search_spend": 0.0009,
      "price_index": -0.0456,
      "competitor_price_index": 0.0189
    },
    "actual_vs_predicted": [
      {"date": "2022-01-01", "actual": 12500, "predicted": 12350},
      {"date": "2022-01-08", "actual": 13200, "predicted": 13450}
    ]
  }
}
```

### 4. Get Model Visualizations
**Endpoint:** /api/model/visualizations/{model_id}
**Method:** GET
**Query Parameters:**
- `type`: Visualization type (optional, returns all if not specified)
- - Options: "actual_vs_predicted", "response_curves", "contributions"

**Response:**
```json
{
  "success": true,
  "message": "Visualizations retrieved successfully",
  "data": {
    "actual_vs_predicted": {
      "x": ["2022-01-01", "2022-01-08", "2022-01-15", "2022-01-22"],
      "y_actual": [12500, 13200, 13800, 12900],
      "y_predicted": [12350, 13450, 13600, 13100]
    },
    "response_curves": {
      "tv_spend": {
        "spend": [0, 1000, 2000, 3000, 4000, 5000],
        "response": [0, 1500, 2800, 3900, 4800, 5500]
      },
      "digital_spend": {
        "spend": [0, 500, 1000, 1500, 2000, 2500],
        "response": [0, 800, 1500, 2100, 2600, 3000]
      },
      "search_spend": {
        "spend": [0, 200, 400, 600, 800, 1000],
        "response": [0, 400, 750, 1050, 1300, 1500]
      }
    },
    "contributions": {
      "labels": ["TV", "Digital", "Search", "Price", "Competitor Price", "Baseline"],
      "values": [45, 25, 15, -8, 3, 20]
    }
  }
}
```

### 5. Optimize Budget
**Endpoint:** /api/model/optimize/{model_id}
**Method:** POST
**Content-Type:** application/json

**Request:**
```json
{
  "total_budget": 1000000,  
  "constraints": {
    "tv_spend": {
      "min": 300000,
      "max": 600000
    },
    "digital_spend": {
      "min": 100000,
      "max": 300000
    },
    "search_spend": {
      "min": 50000,
      "max": 150000
    }
  },
  "scenarios": [
    {"name": "Baseline", "budget_multiplier": 1.0},
    {"name": "Increased Budget", "budget_multiplier": 1.2},
    {"name": "Decreased Budget", "budget_multiplier": 0.8}
  ]
}
```

**Response:**
```json
{
  "success": true,
  "message": "Budget optimization completed successfully",
  "data": {
    "current_allocation": {
      "tv_spend": 500000,
      "digital_spend": 200000,
      "search_spend": 100000,
      "total": 800000
    },
    "optimized_allocation": {
      "tv_spend": 550000,
      "digital_spend": 150000,
      "search_spend": 100000,
      "total": 800000
    },
    "expected_lift": 3.5,
    "roi_improvement": 0.12,
    "scenarios": [
      {
        "name": "Baseline",
        "budget": 800000,
        "allocation": {
          "tv_spend": 550000,
          "digital_spend": 150000,
          "search_spend": 100000
        },
        "expected_lift": 3.5
      },
      {
        "name": "Increased Budget",
        "budget": 960000,
        "allocation": {
          "tv_spend": 600000,
          "digital_spend": 210000,
          "search_spend": 150000
        },
        "expected_lift": 5.2
      },
      {
        "name": "Decreased Budget",
        "budget": 640000,
        "allocation": {
          "tv_spend": 400000,
          "digital_spend": 140000,
          "search_spend": 100000
        },
        "expected_lift": 1.8
      }
    ]
  }
}
```

### 6. Diagnostic Results
**Endpoint:** /api/model/diagnostics/{model_id}
**Method:** GET

**Response:**
```json
{
  "success": true,
  "message": "Diagnostic results retrieved successfully",
  "data": {
    "residual_analysis": {
      "normality_test": {
        "statistic": 0.978,
        "p_value": 0.127,
        "is_normal": true
      },
      "heteroskedasticity_test": {
        "statistic": 2.43,
        "p_value": 0.093,
        "has_heteroskedasticity": false
      },
      "autocorrelation_test": {
        "durbin_watson": 1.95,
        "has_autocorrelation": false
      }
    },
    "stability_assessment": {
      "coefficient_stability": {
        "tv_spend": {"cv": 0.18, "is_stable": true},
        "digital_spend": {"cv": 0.23, "is_stable": true},
        "search_spend": {"cv": 0.29, "is_stable": true}
      }
    },
    "model_quality": {
      "issues": [],
      "recommendations": []
    }
  }
}
```

## Error Handling
All endpoints return appropriate HTTP status codes:

- 200: Success
- 400: Bad request (invalid parameters)
- 401: Unauthorized (invalid API key)
- 404: Resource not found
- 500: Server error

Error response format:
```json
{
  "success": false,
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "The uploaded file is not a valid CSV",
    "details": "Expected comma-separated values but found tab-separated values"
  }
}
```

## Data Requirements
CSV Format
The uploaded CSV should contain:

- A date column (weekly or daily data preferred)
- Marketing spend columns (one per channel)
- Sales or conversion metric column
- Optional control variables (price, competitor activity, etc.)

Minimum 30 rows recommended for reliable model results.