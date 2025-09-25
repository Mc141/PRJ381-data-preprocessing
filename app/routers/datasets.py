from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Optional, Any
from pathlib import Path
import sys
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/datasets/generate-ml-ready-files",
    summary="Generate ML-ready CSVs without DB",
    description=(
        "Runs the internal generator to write "
        "data/global_training_ml_ready.csv and data/local_validation_ml_ready.csv, "
        "overwriting existing files. Returns success/failure only."
    ),
    response_model=Dict[str, Any],
    tags=["5. Model Training & Export"],
)
async def generate_ml_ready_files(
    max_global: Optional[int] = Query(None, description="Limit global records (omit for all)"),
    max_local: Optional[int] = Query(None, description="Limit local ZA records (omit for all)"),
    batch_size: int = Query(100, description="Batch size for environmental extraction"),
    verbose: bool = Query(False, description="Enable verbose logging"),
) -> Dict[str, Any]:
    """Generate ML-ready datasets to canonical files without using the database."""
    try:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
    # Import dataset generator service
        from app.services import generate_ml_ready_datasets as gen

        await gen.run(
            max_global=max_global,
            max_local=max_local,
            batch_size=batch_size,
            verbose=verbose,
        )

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