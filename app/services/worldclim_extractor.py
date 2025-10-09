"""
WorldClim Data Extractor

Extracts WorldClim v2.1 bioclimate data from GeoTIFF rasters with batch processing
and caching. Returns real data only (None when unavailable).
"""

import aiohttp
import rasterio
from rasterio.sample import sample_gen
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class WorldClimDataError(Exception):
    """Custom exception for WorldClim data errors."""
    pass


class WorldClimExtractor:
    """
    WorldClim data extractor for real raster files.
    
    Extracts WorldClim v2.1 bioclimate data from local GeoTIFF files with
    caching and batch processing support.
    """
    
    def __init__(self, data_dir: Optional[Path] = None, resolution: str = "10m"):
        """
        Initialize the WorldClim data extractor.
        
        Args:
            data_dir: Directory to store downloaded WorldClim files
            resolution: WorldClim resolution ("10m", "5m", "2.5m", "30s")
        """
        if data_dir:
            self.data_dir = data_dir
        else:
            # Get an absolute path relative to the project root
            project_root = Path(__file__).resolve().parent.parent.parent  # app/services -> app -> project_root
            self.data_dir = project_root / "data" / "worldclim"
            
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using WorldClim data directory: {self.data_dir.resolve()}")
        
        self.resolution = resolution
        self.session = None
        self.cache = {}  # In-memory cache for performance
        
        logger.info(f"WorldClim extractor initialized - Data dir: {self.data_dir}, Resolution: {self.resolution}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=5)
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers={"User-Agent": "PRJ381-WorldClim-Extractor/1.0"}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    def get_bio_dir(self) -> Path:
        """
        Get the directory path for WorldClim bio files.
        
        Returns:
            Path: Directory containing WorldClim bioclimate files
        """
        # Log the absolute path for debugging
        absolute_path = self.data_dir.resolve()
        logger.info(f"Using WorldClim data directory: {absolute_path}")
        return self.data_dir
    
    def check_files_exist(self) -> bool:
        """Check if all required WorldClim files exist in the data directory."""
        bio_dir = self.get_bio_dir()  # Use the same directory as used for downloads
        
        expected_files = [f"wc2.1_{self.resolution}_bio_{i}.tif" for i in range(1, 20)]

        for filename in expected_files:
            file_path = bio_dir / filename
            if not file_path.exists():
                logger.info(f"Missing file: {file_path}")
                return False

        logger.info("All WorldClim files present")
        return True
    
    async def extract_climate_data(self, latitude: float, longitude: float, 
                                 bio_variables: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Extract real climate data for a single coordinate.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees  
            bio_variables: List of bioclimate variables (e.g., ["bio1", "bio12"])
            
        Returns:
            Dictionary with extracted climate data or None values if unavailable
        """
        # Handle case when bio_variables is a comma-separated string
        if bio_variables and isinstance(bio_variables, str):
            bio_variables = [f"bio{var.strip()}" for var in bio_variables.split(',')]
            
        cache_key = f"{latitude:.3f},{longitude:.3f}:{','.join(sorted(bio_variables or []))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Default variables if none specified
        if bio_variables is None:
            bio_variables = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]
        
        # Check if data files exist
        if not self.check_files_exist():
            raise WorldClimDataError(
                f"WorldClim data files not found in {self.data_dir.resolve()}. "
                f"Please ensure all 19 bio*.tif files are present."
            )
        
        bio_dir = self.get_bio_dir()
        
        result = {
            "latitude": latitude,
            "longitude": longitude,
            "extraction_date": datetime.utcnow().isoformat(),
            "data_source": f"WorldClim_v2.1_real_data_{self.resolution}",
            "extraction_note": "Real data extracted from WorldClim GeoTIFF files"
        }
        
        coords = [(longitude, latitude)]
        
        try:
            for bio_var in bio_variables:
                # Handle case when bio_var might not have "bio" prefix
                processed_var = bio_var
                if not str(bio_var).startswith("bio"):
                    processed_var = f"bio{bio_var}"
                
                # Handle case when the variable could be a string like "1,4,5,6" instead of "bio1"
                try:
                    bio_num_str = processed_var.replace("bio", "")
                    # Check if it's a comma-separated list
                    if "," in bio_num_str:
                        # Just take the first value if it's a comma-separated list
                        bio_num_str = bio_num_str.split(",")[0]
                    bio_num = int(bio_num_str)
                    file_path = bio_dir / f"wc2.1_{self.resolution}_bio_{bio_num}.tif"
                except ValueError as e:
                    logger.error(f"Invalid bio variable format: {bio_var}, processed as {processed_var}: {e}")
                    result[bio_var] = None
                    continue
                
                try:
                    if file_path.exists():
                        with rasterio.open(file_path) as dataset:
                            values = list(sample_gen(dataset, coords))
                            
                            if values and len(values) > 0:
                                value = values[0][0]
                                
                                # Check for nodata values - convert to None (NaN), NO FAKE DATA
                                if dataset.nodata is not None and value == dataset.nodata:
                                    value = None  # Real nodata value - not dummy/fake data
                                elif np.isnan(value) or np.isinf(value):
                                    value = None  # Real missing data - not dummy/fake data
                                else:
                                    # Apply scaling if needed (WorldClim temperature is in °C * 10)
                                    if bio_num in [1, 4, 5, 6, 7, 8, 9, 10, 11]:  # Temperature variables
                                        value = value / 10.0  # Convert to actual °C
                                    
                                    # Convert numpy types to Python native types
                                    value = float(value) if isinstance(value, np.number) else value
                            else:
                                value = None  # Real missing data - NO dummy/fake values
                            
                            result[bio_var] = value
                    else:
                        logger.warning(f"File not found: {file_path}")
                        result[bio_var] = None  # Error = None/NaN, NOT fake data
                        
                except Exception as e:
                    logger.warning(f"Error extracting {bio_var} at {latitude}, {longitude}: {e}")
                    result[bio_var] = None  # Error = None/NaN, NOT fake data
            
            # Cache the result
            self.cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing coordinate {latitude}, {longitude}: {e}")
            # Return error result with None values - ERROR = NaN, NOT FAKE DATA!
            error_result = {
                "latitude": latitude,
                "longitude": longitude,
                "extraction_error": str(e),
                "extraction_date": datetime.utcnow().isoformat(),
                "data_source": f"WorldClim_v2.1_error_{self.resolution}",
                "extraction_note": "Real WorldClim data unavailable (error)"
            }
            
            for bio_var in bio_variables:
                error_result[bio_var] = None  # Error -> None/NaN, never dummy values
            
            return error_result
    
    async def extract_climate_batch(self, coordinates: List[Tuple[float, float]], 
                                  bio_variables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Extract climate data for multiple coordinates efficiently.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            bio_variables: List of bioclimate variables to extract
            
        Returns:
            List of climate data dictionaries
        """
        logger.info(f"Extracting climate data for {len(coordinates)} coordinates")
        
        # Handle case when bio_variables is a comma-separated string
        if bio_variables and isinstance(bio_variables, str):
            bio_variables = [f"bio{var.strip()}" for var in bio_variables.split(',')]
        
        if bio_variables is None:
            bio_variables = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]
        
        # Process coordinates in parallel
        tasks = [
            self.extract_climate_data(lat, lon, bio_variables)
            for lat, lon in coordinates
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                lat, lon = coordinates[i]
                logger.error(f"Error processing coordinate {lat}, {lon}: {result}")
                error_result = {
                    "latitude": lat,
                    "longitude": lon,
                    "extraction_error": str(result),
                    "extraction_date": datetime.utcnow().isoformat(),
                    "data_source": f"WorldClim_v2.1_error_{self.resolution}"
                }
                for bio_var in bio_variables:
                    error_result[bio_var] = None
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results


# Singleton instance for global use
_worldclim_extractor = None


def get_worldclim_extractor() -> WorldClimExtractor:
    """Get or create the global WorldClim extractor instance."""
    global _worldclim_extractor
    if _worldclim_extractor is None:
        _worldclim_extractor = WorldClimExtractor()
    return _worldclim_extractor
