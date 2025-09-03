"""
GBIF Router - Global Biodiversity Data Endpoints
===============================================

FastAPI router for managing GBIF (Global Biodiversity Information Facility) occurrence data.
Replaces iNaturalist functionality with comprehensive global species occurrence management.

Endpoints:
    - GET /gbif/occurrences - Fetch global species occurrences
    - GET /gbif/occurrences/filtered - Fetch with quality filters
    - GET /gbif/species/{scientific_name} - Get species information
    - GET /gbif/db - Retrieve stored GBIF data
    - DELETE /gbif/db - Delete stored GBIF data
    - POST /gbif/enrich-environmental - Enrich with climate data
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Path
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime
from pydantic import BaseModel, Field

from app.services.gbif_fetcher import GBIFFetcher, fetch_pyracantha_global, fetch_pyracantha_south_africa, get_species_info
from app.services.worldclim_extractor import enrich_gbif_occurrences
from app.services.database import get_database

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Pydantic models for request/response
class SpeciesInfo(BaseModel):
    """Species information response model."""
    scientific_name: str = Field(..., description="Scientific name of the species")
    gbif_species_key: Optional[int] = Field(None, description="GBIF species key (taxon ID)")
    lookup_date: str = Field(..., description="Date when lookup was performed")

class OccurrenceFilters(BaseModel):
    """Filters for occurrence data requests."""
    quality_filters: bool = Field(True, description="Apply quality filters (coordinates, dates, etc.)")
    date_from: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    country: Optional[str] = Field(None, description="ISO country code (e.g., 'ZA' for South Africa)")
    coordinate_uncertainty_max: Optional[int] = Field(10000, description="Maximum coordinate uncertainty in meters")
    max_records: Optional[int] = Field(None, description="Maximum number of records to fetch")

class EnrichmentRequest(BaseModel):
    """Request model for environmental enrichment."""
    climate_variables: Optional[List[str]] = Field(None, description="Climate variables to extract (bio1, bio5, etc.)")
    include_elevation: bool = Field(True, description="Include SRTM elevation data")
    batch_size: int = Field(20, description="Batch size for processing")

# Progress tracking for long-running operations
progress_tracker = {}

async def progress_callback(operation_id: str, current: int, total: int, percentage: float):
    """Update progress for long-running operations."""
    progress_tracker[operation_id] = {
        "current": current,
        "total": total,
        "percentage": round(percentage, 1),
        "updated_at": datetime.utcnow().isoformat()
    }

@router.get("/gbif/occurrences", 
           summary="🌍 Fetch Global Species Occurrences",
           description="""
**STEP 2 of ML Pipeline** - Collect global biodiversity data

### 📋 Prerequisites:
1. **System Health**: Verify `/status/health` shows database connectivity
2. **Network Access**: Ensure GBIF API is accessible

### 🎯 What This Does:
- Fetches **~1,700+ global Pyracantha angustifolia occurrences** from GBIF
- Applies **quality filters** for coordinate accuracy and data reliability
- **Stores in MongoDB** for fast processing and reuse
- **Progress tracking** for long-running operations

### 🔍 Data Quality Filters:
- Valid coordinates (latitude/longitude)
- No coordinate uncertainty issues
- Recent observations (species presence confirmed)
- Taxonomically verified records

### ⚡ Performance:
- Fetch time: ~2-3 minutes for full dataset
- Storage: ~15MB in MongoDB
- Network usage: ~10MB download from GBIF

### 🔗 Next Steps:
1. Run `/worldclim/ensure-data` to download climate data
2. Then use `/datasets/merge-global` to create training dataset

### ⚠️ Important:
**ALWAYS use `store_in_db=true`** - Required for dataset creation pipeline!
           """,
           response_model=Dict[str, Any])
async def fetch_global_occurrences(
    background_tasks: BackgroundTasks,
    max_records: Optional[int] = Query(None, description="Maximum records to fetch (None for all ~40,000)"),
    store_in_db: bool = Query(True, description="Store results in MongoDB"),
    operation_id: Optional[str] = Query(None, description="Operation ID for progress tracking")
) -> Dict[str, Any]:
    """
    Fetch global Pyracantha angustifolia occurrences from GBIF API.
    
    This endpoint retrieves comprehensive global occurrence data, applying quality filters
    to ensure coordinate accuracy and data reliability for transfer learning applications.
    """
    try:
        # Generate operation ID if not provided
        if not operation_id:
            operation_id = f"global_fetch_{int(datetime.utcnow().timestamp())}"
        
        logger.info(f"Starting global GBIF fetch (operation: {operation_id})")
        
        # Set up progress tracking
        def progress_func(current, total, percentage):
            return progress_callback(operation_id, current, total, percentage)
        
        # Fetch data using complete scientific name with authority for taxonomic precision
        async with GBIFFetcher() as fetcher:
            occurrences = await fetcher.fetch_all_occurrences(
                "Pyracantha angustifolia (Franch.) C.K.Schneid.",
                quality_filters=True,
                coordinate_uncertainty_max=10000,
                max_records=max_records,
                progress_callback=progress_func
            )
        
        # Process records
        processed_occurrences = []
        for record in occurrences:
            processed = GBIFFetcher().process_occurrence_record(record)
            processed_occurrences.append(processed)
        
        # Store in database if requested
        if store_in_db and processed_occurrences:
            background_tasks.add_task(store_gbif_data, processed_occurrences, "global_occurrences")
        
        # Clean up progress tracking
        if operation_id in progress_tracker:
            del progress_tracker[operation_id]
        
        return {
            "status": "success",
            "operation_id": operation_id,
            "total_records": len(processed_occurrences),
            "species": "Pyracantha angustifolia (Franch.) C.K.Schneid.",
            "data_source": "GBIF",
            "quality_filtered": True,
            "fetch_timestamp": datetime.utcnow().isoformat(),
            "stored_in_db": store_in_db,
            "data": processed_occurrences[:10] if processed_occurrences else [],  # Return first 10 for preview
            "message": f"Fetched {len(processed_occurrences):,} global occurrences for Pyracantha angustifolia (Franch.) C.K.Schneid."
        }
        
    except Exception as e:
        logger.error(f"Error fetching global occurrences: {e}")
        if operation_id in progress_tracker:
            del progress_tracker[operation_id]
        raise HTTPException(status_code=500, detail=f"Failed to fetch global occurrences: {str(e)}")

@router.get("/gbif/occurrences/filtered",
           summary="Fetch Filtered Occurrences", 
           description="Fetch GBIF occurrences with custom filters",
           response_model=Dict[str, Any])
async def fetch_filtered_occurrences(
    background_tasks: BackgroundTasks,
    scientific_name: str = Query("Pyracantha angustifolia (Franch.) C.K.Schneid.", description="Scientific name with authority"),
    quality_filters: bool = Query(True, description="Apply quality filters"),
    date_from: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    country: Optional[str] = Query(None, description="ISO country code"),
    coordinate_uncertainty_max: Optional[int] = Query(10000, description="Max coordinate uncertainty (meters)"),
    max_records: Optional[int] = Query(None, description="Maximum records"),
    store_in_db: bool = Query(False, description="Store in database"),
    operation_id: Optional[str] = Query(None, description="Operation ID")
) -> Dict[str, Any]:
    """
    Fetch GBIF occurrence records with comprehensive filtering options.
    
    Supports:
    - Date range filtering
    - Geographic filtering by country
    - Quality filtering for coordinates and data reliability
    - Customizable record limits
    """
    try:
        if not operation_id:
            operation_id = f"filtered_fetch_{int(datetime.utcnow().timestamp())}"
        
        logger.info(f"Fetching filtered occurrences for '{scientific_name}' (operation: {operation_id})")
        
        # Progress tracking
        def progress_func(current, total, percentage):
            return progress_callback(operation_id, current, total, percentage)
        
        # Fetch filtered data
        async with GBIFFetcher() as fetcher:
            occurrences = await fetcher.fetch_all_occurrences(
                scientific_name=scientific_name,
                quality_filters=quality_filters,
                date_from=date_from,
                date_to=date_to,
                country=country,
                coordinate_uncertainty_max=coordinate_uncertainty_max,
                max_records=max_records,
                progress_callback=progress_func
            )
        
        # Process records
        processed_occurrences = [
            GBIFFetcher().process_occurrence_record(record) 
            for record in occurrences
        ]
        
        # Store if requested
        if store_in_db and processed_occurrences:
            collection_name = f"filtered_occurrences_{country or 'global'}"
            background_tasks.add_task(store_gbif_data, processed_occurrences, collection_name)
        
        # Build filter summary
        filters_applied = {
            "scientific_name": scientific_name,
            "quality_filters": quality_filters,
            "date_range": f"{date_from} to {date_to}" if date_from or date_to else None,
            "country": country,
            "max_coordinate_uncertainty": coordinate_uncertainty_max,
            "max_records": max_records
        }
        
        # Clean up progress
        if operation_id in progress_tracker:
            del progress_tracker[operation_id]
        
        return {
            "status": "success",
            "operation_id": operation_id,
            "total_records": len(processed_occurrences),
            "filters_applied": filters_applied,
            "data_source": "GBIF",
            "fetch_timestamp": datetime.utcnow().isoformat(),
            "stored_in_db": store_in_db,
            "data": processed_occurrences[:10],  # Preview
            "message": f"Fetched {len(processed_occurrences):,} filtered occurrences"
        }
        
    except Exception as e:
        logger.error(f"Error fetching filtered occurrences: {e}")
        if operation_id in progress_tracker:
            del progress_tracker[operation_id]
        raise HTTPException(status_code=500, detail=f"Failed to fetch filtered occurrences: {str(e)}")

@router.get("/gbif/species/{scientific_name}",
           summary="Get Species Information",
           description="Get GBIF species information and taxon key",
           response_model=SpeciesInfo)
async def get_species_information(
    scientific_name: str = Path(..., description="Scientific name of the species")
) -> SpeciesInfo:
    """
    Get species information from GBIF including taxon key for further queries.
    """
    try:
        species_info = await get_species_info(scientific_name)
        
        if not species_info:
            raise HTTPException(status_code=404, detail=f"Species '{scientific_name}' not found in GBIF")
        
        return SpeciesInfo(**species_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting species info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get species information: {str(e)}")

@router.get("/gbif/db",
           summary="Retrieve Stored GBIF Data",
           description="Retrieve GBIF occurrence data from MongoDB",
           response_model=Dict[str, Any])
async def get_stored_gbif_data(
    collection: str = Query("global_occurrences", description="Collection name"),
    limit: int = Query(1000, description="Maximum records to return"),
    skip: int = Query(0, description="Records to skip (pagination)"),
    country_filter: Optional[str] = Query(None, description="Filter by country code")
) -> Dict[str, Any]:
    """
    Retrieve stored GBIF occurrence data from MongoDB with pagination and filtering.
    """
    try:
        db = get_database()
        gbif_collection = db[collection]
        
        # Build query
        query = {}
        if country_filter:
            query["country_code"] = country_filter
        
        # Get total count
        total_count = gbif_collection.count_documents(query)
        
        # Fetch data with pagination
        cursor = gbif_collection.find(query).skip(skip).limit(limit)
        occurrences = list(cursor)
        
        # Convert ObjectId to string for JSON serialization
        for occurrence in occurrences:
            if "_id" in occurrence:
                occurrence["_id"] = str(occurrence["_id"])
        
        return {
            "status": "success",
            "collection": collection,
            "total_count": total_count,
            "returned_count": len(occurrences),
            "skip": skip,
            "limit": limit,
            "country_filter": country_filter,
            "data": occurrences,
            "retrieved_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stored GBIF data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve data: {str(e)}")

@router.delete("/gbif/db",
              summary="Delete Stored GBIF Data",
              description="Delete GBIF occurrence data from MongoDB",
              response_model=Dict[str, Any])
async def delete_stored_gbif_data(
    collection: str = Query("global_occurrences", description="Collection to delete"),
    confirm: bool = Query(False, description="Confirm deletion")
) -> Dict[str, Any]:
    """
    Delete stored GBIF occurrence data from MongoDB.
    Requires confirmation to prevent accidental deletion.
    """
    if not confirm:
        raise HTTPException(status_code=400, detail="Deletion requires confirmation (confirm=true)")
    
    try:
        db = get_database()
        gbif_collection = db[collection]
        
        # Count documents before deletion
        count_before = gbif_collection.count_documents({})
        
        # Delete all documents
        result = gbif_collection.delete_many({})
        
        logger.info(f"Deleted {result.deleted_count} documents from collection '{collection}'")
        
        return {
            "status": "success",
            "collection": collection,
            "documents_deleted": result.deleted_count,
            "documents_before": count_before,
            "deleted_at": datetime.utcnow().isoformat(),
            "message": f"Successfully deleted {result.deleted_count} GBIF records"
        }
        
    except Exception as e:
        logger.error(f"Error deleting GBIF data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete data: {str(e)}")

@router.post("/gbif/enrich-environmental",
            summary="Enrich with Environmental Data",
            description="Enrich GBIF records with WorldClim environmental variables",
            response_model=Dict[str, Any])
async def enrich_with_environmental_data(
    background_tasks: BackgroundTasks,
    enrichment_request: EnrichmentRequest,
    collection: str = Query("global_occurrences", description="Source collection"),
    target_collection: str = Query("enriched_occurrences", description="Target collection for enriched data"),
    operation_id: Optional[str] = Query(None, description="Operation ID")
) -> Dict[str, Any]:
    """
    Enrich stored GBIF occurrence records with environmental/climate data from WorldClim.
    
    This operation can take significant time for large datasets and runs in the background.
    """
    try:
        if not operation_id:
            operation_id = f"enrichment_{int(datetime.utcnow().timestamp())}"
        
        # Get stored occurrences
        db = get_database()
        source_collection = db[collection]
        
        occurrences = list(source_collection.find({}))
        if not occurrences:
            raise HTTPException(status_code=404, detail=f"No data found in collection '{collection}'")
        
        # Remove MongoDB ObjectId for processing
        for occ in occurrences:
            if "_id" in occ:
                del occ["_id"]
        
        logger.info(f"Starting environmental enrichment for {len(occurrences):,} records (operation: {operation_id})")
        
        # Run enrichment in background
        background_tasks.add_task(
            run_environmental_enrichment,
            occurrences,
            enrichment_request.climate_variables,
            enrichment_request.include_elevation,
            enrichment_request.batch_size,
            target_collection,
            operation_id
        )
        
        return {
            "status": "processing",
            "operation_id": operation_id,
            "source_collection": collection,
            "target_collection": target_collection,
            "total_records": len(occurrences),
            "climate_variables": enrichment_request.climate_variables,
            "batch_size": enrichment_request.batch_size,
            "started_at": datetime.utcnow().isoformat(),
            "message": f"Environmental enrichment started for {len(occurrences):,} records. Check progress with operation_id."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting environmental enrichment: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start enrichment: {str(e)}")

@router.get("/gbif/progress/{operation_id}",
           summary="Check Operation Progress",
           description="Check progress of long-running GBIF operations",
           response_model=Dict[str, Any])
async def check_operation_progress(
    operation_id: str = Path(..., description="Operation ID to check")
) -> Dict[str, Any]:
    """
    Check the progress of long-running GBIF operations like data fetching or enrichment.
    """
    if operation_id not in progress_tracker:
        raise HTTPException(status_code=404, detail=f"Operation '{operation_id}' not found or completed")
    
    progress_info = progress_tracker[operation_id]
    
    return {
        "operation_id": operation_id,
        "status": "in_progress",
        **progress_info
    }

# Background task functions
async def store_gbif_data(occurrences: List[Dict[str, Any]], collection_name: str):
    """Background task to store GBIF data in MongoDB."""
    try:
        db = get_database()
        collection = db[collection_name]
        
        # Add storage timestamp
        for occ in occurrences:
            occ["stored_at"] = datetime.utcnow().isoformat()
        
        # Insert data
        result = collection.insert_many(occurrences)
        logger.info(f"Stored {len(result.inserted_ids)} GBIF records in collection '{collection_name}'")
        
    except Exception as e:
        logger.error(f"Error storing GBIF data: {e}")

async def run_environmental_enrichment(occurrences: List[Dict[str, Any]], 
                                     climate_variables: Optional[List[str]],
                                     include_elevation: bool,
                                     batch_size: int,
                                     target_collection: str,
                                     operation_id: str):
    """Background task to run environmental enrichment."""
    try:
        # Progress callback
        def progress_func(current, total, percentage):
            return progress_callback(operation_id, current, total, percentage)
        
        # Run enrichment
        enriched_data = await enrich_gbif_occurrences(
            occurrences,
            climate_variables,
            include_elevation,
            progress_func
        )
        
        # Store enriched data
        db = get_database()
        collection = db[target_collection]
        
        # Add enrichment metadata
        for record in enriched_data:
            record["enrichment_completed_at"] = datetime.utcnow().isoformat()
            record["enrichment_operation_id"] = operation_id
        
        # Insert enriched data
        result = collection.insert_many(enriched_data)
        logger.info(f"Stored {len(result.inserted_ids)} enriched records in '{target_collection}'")
        
        # Clean up progress tracking
        if operation_id in progress_tracker:
            del progress_tracker[operation_id]
        
    except Exception as e:
        logger.error(f"Error in environmental enrichment: {e}")
        if operation_id in progress_tracker:
            del progress_tracker[operation_id]
