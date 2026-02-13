from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from app.services.battery_service import predict_battery_rul, get_health_status, get_recommendations
import math

router = APIRouter(
    prefix="/api",
    tags=["Battery Prediction"]
)


class SimpleBatteryInput(BaseModel):
    """Simplified battery parameters for RUL prediction"""
    current_capacity: float = Field(..., gt=0, description="Current capacity in Ahr (e.g., 2.0)")
    initial_capacity: float = Field(..., gt=0, description="Initial/Rated capacity in Ahr (e.g., 3.0)")
    ambient_temperature: float = Field(default=25.0, description="Ambient temperature in °C (e.g., 25)")
    cycle_count: int = Field(..., ge=0, description="Number of charge/discharge cycles (e.g., 50)")
    age_days: float = Field(..., ge=0, description="Battery age in days (e.g., 100)")


class BatteryPredictionResponse(BaseModel):
    """Response from battery prediction"""
    success: bool = Field(True, description="Whether prediction was successful")
    battery_metrics: Dict = Field(description="Battery input metrics")
    health_analysis: Dict = Field(description="Health status and metrics")
    rul_prediction: Dict = Field(description="RUL prediction results")
    recommendations: List[str] = Field(description="Recommendations based on battery state")


class BatteryHealthStatus(BaseModel):
    """Battery health status response"""
    soh_percentage: float = Field(description="State of Health as percentage (0-100)")
    health_status: str = Field(description="Health status category (EXCELLENT, GOOD, FAIR, POOR, CRITICAL)")
    health_description: str = Field(description="Description of health status")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Battery Prediction API is healthy ✅"
    }


@router.post("/predict-rul", response_model=BatteryPredictionResponse)
async def predict_rul(battery_data: SimpleBatteryInput):
    """
    Predict battery Remaining Useful Life (RUL) based on simplified battery metrics.
    
    This endpoint accepts basic battery parameters and returns comprehensive RUL predictions
    with health status and recommendations.
    
    Args:
        battery_data: Battery parameters (current capacity, initial capacity, temperature, cycles, age)
        
    Returns:
        BatteryPredictionResponse: Prediction results including RUL, health status, and recommendations
    """
    try:
        # Calculate State of Health (SoH)
        soh = battery_data.current_capacity / battery_data.initial_capacity
        soh_percentage = soh * 100
        
        # Calculate degradation factor
        degradation_factor = (1 - soh) * 100
        
        # Calculate degradation rate (percentage per cycle)
        if battery_data.cycle_count > 0:
            capacity_gradient_pct = (degradation_factor / battery_data.cycle_count)
        else:
            capacity_gradient_pct = 0.1
        
        # Build features dictionary for model prediction
        features_dict = {
            'Capacity': battery_data.current_capacity,
            'ambient_temperature': battery_data.ambient_temperature,
            'cycle_count': battery_data.cycle_count,
            'age_days': battery_data.age_days,
            'initial_capacity': battery_data.initial_capacity,
            'SoH': soh,
            'capacity_degradation': 1 - soh,
            'capacity_slope': -(battery_data.initial_capacity / (battery_data.cycle_count + 1)) if battery_data.cycle_count >= 0 else -0.01,
            'normalized_degradation_rate': -(1 - soh) / battery_data.initial_capacity if battery_data.initial_capacity > 0 else 0,
            'capacity_gradient_pct': capacity_gradient_pct,
            'cycle_utilization': battery_data.cycle_count / (battery_data.cycle_count + 100),
            'degradation_acceleration': ((1 - soh) ** 2) / max(battery_data.cycle_count, 1),
            'daily_degradation_rate': (1 - soh) / (battery_data.age_days + 1),
            'extrapolated_rul_from_gradient': (soh - 0.70) / (capacity_gradient_pct / 100) if capacity_gradient_pct > 0.0001 else 0,
            'cycle_duration': 3600.0,
            'measurement_count': 1000,
            'voltage_mean': 3.8,
            'voltage_std': 0.1,
            'voltage_min': 3.5,
            'voltage_max': 4.2,
            'voltage_range': 0.7,
            'voltage_drop': 0.5,
            'current_mean': -2.0,
            'current_std': 0.1,
            'current_min': -2.2,
            'current_max': -1.8,
            'temp_mean': battery_data.ambient_temperature,
            'temp_std': 2.0,
            'temp_min': battery_data.ambient_temperature - 2,
            'temp_max': battery_data.ambient_temperature + 2,
            'temp_range': 5.0,
            'power_mean': 7.6,
            'power_max': 9.2
        }
        
        # Get prediction from model
        rul_prediction, _ = predict_battery_rul(features_dict)
        rul_cycles = max(0, int(rul_prediction))
        
        # Get health status
        status, description = get_health_status(soh)
        
        # Calculate estimated time to EOL (in days)
        if battery_data.cycle_count > 0 and battery_data.age_days > 0:
            avg_cycles_per_day = battery_data.cycle_count / battery_data.age_days
            if avg_cycles_per_day > 0:
                estimated_days = rul_cycles / avg_cycles_per_day
            else:
                estimated_days = rul_cycles * (battery_data.age_days / max(battery_data.cycle_count, 1))
        else:
            estimated_days = rul_cycles
        
        months = estimated_days / 30.44  # Average days per month
        
        # Get recommendations
        recommendations = get_recommendations(rul_prediction, soh, battery_data.cycle_count)
        
        return BatteryPredictionResponse(
            success=True,
            battery_metrics={
                "initial_capacity_ahr": round(battery_data.initial_capacity, 2),
                "current_capacity_ahr": round(battery_data.current_capacity, 2),
                "cycle_count": battery_data.cycle_count,
                "age_days": battery_data.age_days,
                "ambient_temperature_c": battery_data.ambient_temperature
            },
            health_analysis={
                "soh_percentage": round(soh_percentage, 1),
                "health_status": status,
                "health_description": description,
                "degradation_factor_percent": round(degradation_factor, 1)
            },
            rul_prediction={
                "rul_cycles": rul_cycles,
                "estimated_days_to_eol": round(estimated_days, 0),
                "estimated_time_to_eol": f"{int(estimated_days)} days (~{months:.1f} months)"
            },
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@router.get("/health-status/{soh_percentage}")
async def get_battery_status(soh_percentage: float):
    """
    Get battery health status for a given State of Health percentage value.
    
    Args:
        soh_percentage: State of Health as percentage (0-100)
        
    Returns:
        BatteryHealthStatus: Health status details
    """
    if soh_percentage < 0 or soh_percentage > 100:
        raise HTTPException(status_code=400, detail="SoH percentage must be between 0 and 100")
    
    soh = soh_percentage / 100
    status, description = get_health_status(soh)
    
    return BatteryHealthStatus(
        soh_percentage=soh_percentage,
        health_status=status,
        health_description=description
    )


@router.get("/test-prediction")
async def test_prediction_endpoint():
    """
    Test endpoint with example battery data from the provided test case.
    
    Example values:
    - Current Capacity: 2 Ahr
    - Initial Capacity: 3 Ahr
    - Cycle Count: 40
    - Age: 300 days
    - Ambient Temperature: 29°C
    """
    example_data = SimpleBatteryInput(
        current_capacity=2.0,
        initial_capacity=3.0,
        ambient_temperature=29,
        cycle_count=40,
        age_days=300
    )
    
    return await predict_rul(example_data)


# Import metadata at module level for use in endpoints
from pathlib import Path
import json

metadata_path = Path(__file__).parent.parent.parent / "modelTraining" / "model_metadata.json"
with open(str(metadata_path), 'r') as f:
    metadata = json.load(f)