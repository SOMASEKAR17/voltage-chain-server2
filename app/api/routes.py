from fastapi import APIRouter

router = APIRouter(
    prefix="/api",
    tags=["Test"]
)


@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "API is healthy ✅"
    }


@router.get("/test")
async def test_route():
    return {
        "prediction": "dummy output",
        "success": True
    }