from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Optional, Any
from pathlib import Path
import logging
from app.services import generate_ml_ready_datasets as gen

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/datasets/generate-ml-ready-files",
    summary="Generate ML-ready CSVs",
    description=(
        "Run the internal dataset generator to produce "
        "data/global_training_ml_ready.csv and data/local_validation_ml_ready.csv, "
        "overwriting any existing files. Returns success or failure."
    ),
    response_model=Dict[str, Any],
    tags=["5. Model Training & Export"],
)
async def generate_ml_ready_files(
    max_global: Optional[int] = Query(None, description="Maximum number of global records (omit for all)"),
    max_local: Optional[int] = Query(None, description="Maximum number of local South African records (omit for all)"),
    batch_size: int = Query(100, description="Batch size for environmental data extraction"),
    verbose: bool = Query(False, description="Enable verbose logging output"),
) -> Dict[str, Any]:
    """
    Generate machine learning-ready datasets and write them to canonical CSV files.

    This endpoint runs the internal dataset generator, producing standardized CSV files for model training and validation. Existing files will be overwritten.

    Args:
        max_global (Optional[int]): Limit the number of global records (omit for all).
        max_local (Optional[int]): Limit the number of local South African records (omit for all).
        batch_size (int): Batch size for environmental data extraction.
        verbose (bool): Enable verbose logging for detailed output.

    Returns:
        Dict[str, Any]: Status, written file paths, and a success message.
    """
    try:
        await gen.run(
            max_global=max_global,
            max_local=max_local,
            batch_size=batch_size,
            verbose=verbose,
        )

        repo_root = Path(__file__).resolve().parents[2]
        data_dir = repo_root / "data"
        written_files = [
            str((data_dir / "global_training_ml_ready.csv").resolve()),
            str((data_dir / "local_validation_ml_ready.csv").resolve()),
        ]
        
        return {
            "status": "success",
            "written_files": written_files,
            "message": "ML-ready datasets generated and stored successfully",
        }
    except Exception as e:
        logger.error(f"Error generating ML-ready files: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")