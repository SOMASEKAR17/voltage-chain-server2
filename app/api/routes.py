from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from app.services.battery_service import predict_battery_rul, get_health_status, get_recommendations

router = APIRouter(
    prefix="/api",
    tags=["Battery Prediction"]
)


class BatteryFeatures(BaseModel):
    """Battery parameters for RUL prediction"""
    cycle_duration: float = Field(..., description="Duration of discharge cycle in seconds")
    measurement_count: int = Field(default=1000, description="Number of measurements in cycle")
    
    # Voltage features
    voltage_mean: float = Field(default=3.8, description="Mean voltage during cycle")
    voltage_std: float = Field(default=0.1, description="Voltage standard deviation")
    voltage_min: float = Field(default=3.5, description="Minimum voltage")
    voltage_max: float = Field(default=4.2, description="Maximum voltage")
    voltage_range: float = Field(default=0.7, description="Voltage range (max-min)")
    voltage_drop: float = Field(default=0.5, description="Voltage drop")
    
    # Current features
    current_mean: float = Field(default=-2.0, description="Mean current")
    current_std: float = Field(default=0.1, description="Current standard deviation")
    current_min: float = Field(default=-2.2, description="Minimum current")
    current_max: float = Field(default=-1.8, description="Maximum current")
    
    # Temperature features
    temp_mean: float = Field(default=25.0, description="Mean temperature")
    temp_std: float = Field(default=2.0, description="Temperature standard deviation")
    temp_min: float = Field(default=23.0, description="Minimum temperature")
    temp_max: float = Field(default=28.0, description="Maximum temperature")
    temp_range: float = Field(default=5.0, description="Temperature range")
    
    # Power features
    power_mean: float = Field(default=7.6, description="Mean power")
    power_max: float = Field(default=9.2, description="Maximum power")
    
    # Battery health metrics
    Capacity: float = Field(..., description="Current capacity")
    ambient_temperature: float = Field(default=25.0, description="Ambient temperature")
    cycle_count: int = Field(..., description="Number of charge cycles")
    age_days: float = Field(..., description="Battery age in days")
    initial_capacity: float = Field(..., description="Initial capacity")
    SoH: float = Field(..., description="State of Health (0-1)")
    
    # Degradation metrics
    capacity_degradation: float = Field(description="Capacity degradation level")
    capacity_slope: float = Field(description="Capacity slope")
    normalized_degradation_rate: float = Field(description="Normalized degradation rate")
    capacity_gradient_pct: float = Field(description="Capacity gradient percentage")
    cycle_utilization: float = Field(description="Cycle utilization")
    degradation_acceleration: float = Field(description="Degradation acceleration")
    daily_degradation_rate: float = Field(description="Daily degradation rate")
    extrapolated_rul_from_gradient: float = Field(description="Extrapolated RUL from gradient")


class BatteryPredictionResponse(BaseModel):
    """Response from battery prediction"""
    success: bool
    rul_cycles: float = Field(description="Predicted Remaining Useful Life in cycles")
    soh: float = Field(description="State of Health (0-1)")
    health_status: str = Field(description="Health status category")
    health_description: str = Field(description="Description of health status")
    recommendations: List[str] = Field(description="Recommendations based on battery state")
    model_r2_score: float = Field(description="Model R² test score")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Battery Prediction API is healthy ✅"
    }


@router.post("/predict-rul", response_model=BatteryPredictionResponse)
async def predict_rul(battery_data: BatteryFeatures):
    """
    Predict battery Remaining Useful Life (RUL) based on battery metrics
    
    Args:
        battery_data: Battery parameters and cycle features
        
    Returns:
        BatteryPredictionResponse: Prediction results with health status and recommendations
    """
    try:
        # Convert to dictionary for prediction
        features_dict = battery_data.dict()
        
        # Get prediction
        rul_prediction, soh = predict_battery_rul(features_dict)
        
        # Get health status
        status, description = get_health_status(soh)
        
        # Get recommendations
        recommendations = get_recommendations(
            rul_prediction, 
            soh, 
            battery_data.cycle_count
        )
        
        return BatteryPredictionResponse(
            success=True,
            rul_cycles=rul_prediction,
            soh=soh,
            health_status=status,
            health_description=description,
            recommendations=recommendations,
            model_r2_score=metadata.get('r2_test', 0.0)
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@router.get("/health-status/{soh}")
async def get_battery_status(soh: float):
    """
    Get battery health status for a given State of Health value
    
    Args:
        soh: State of Health value (0-1)
        
    Returns:
        Dictionary with health status and description
    """
    if soh < 0 or soh > 1:
        raise HTTPException(status_code=400, detail="SoH must be between 0 and 1")
    
    status, description = get_health_status(soh)
    return {
        "soh": soh,
        "health_status": status,
        "description": description
    }


@router.get("/test-prediction")
async def test_prediction_endpoint():
    """Test endpoint with example battery data"""
    example_data = BatteryFeatures(
        cycle_duration=3600.0,
        measurement_count=1000,
        voltage_mean=3.8,
        voltage_std=0.1,
        voltage_min=3.5,
        voltage_max=4.2,
        voltage_range=0.7,
        voltage_drop=0.5,
        current_mean=-2.0,
        current_std=0.1,
        current_min=-2.2,
        current_max=-1.8,
        temp_mean=25.0,
        temp_std=2.0,
        temp_min=23.0,
        temp_max=28.0,
        temp_range=5.0,
        power_mean=7.6,
        power_max=9.2,
        Capacity=3.0,
        ambient_temperature=25.0,
        cycle_count=500,
        age_days=365,
        initial_capacity=3.4,
        SoH=0.88,
        capacity_degradation=0.12,
        capacity_slope=-0.0067,
        normalized_degradation_rate=-0.002,
        capacity_gradient_pct=0.12,
        cycle_utilization=0.833,
        degradation_acceleration=0.00024,
        daily_degradation_rate=0.000329,
        extrapolated_rul_from_gradient=667
    )
    
    return await predict_rul(example_data)


# Import metadata at module level for use in endpoints
from pathlib import Path
import json

metadata_path = Path(__file__).parent.parent.parent / "modelTraining" / "model_metadata.json"
with open(str(metadata_path), 'r') as f:
    metadata = json.load(f)