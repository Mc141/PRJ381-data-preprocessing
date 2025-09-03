"""
WorldClim Data Extractor
========================

This module provides comprehensive WorldClim bioclimate data extraction functionality.
It handles downloading, caching, and extracting real climate variables from 
WorldClim v2.1 GeoTIFF raster files.

Features:
- Downloads WorldClim v2.1 raster files
- Extracts real climate data at specific coordinates
- Batch processing for multiple coordinates
- Caching for performance optimization
- Integration with GBIF occurrence data enrichment
- No fake data - returns NaN when data unavailable

Author: MC141
"""

import aiohttp
import aiofiles
import rasterio
from rasterio.sample import sample_gen
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime
import zipfile
import tempfile
import os

logger = logging.getLogger(__name__)

# Import elevation extractor for integrated enrichment
try:
    from .elevation_extractor import get_elevation_extractor
    ELEVATION_AVAILABLE = True
except ImportError:
    ELEVATION_AVAILABLE = False
    logger.warning("Elevation extractor not available")


class WorldClimDataError(Exception):
    """Custom exception for WorldClim data errors."""
    pass


def convert_numpy_types(data: Any) -> Any:
    """
    Convert numpy types to Python native types for MongoDB compatibility.
    
    Args:
        data: Any data structure that might contain numpy types
        
    Returns:
        Data with numpy types converted to Python native types
    """
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.number):
        return data.item()  # Convert numpy scalar to Python type
    elif hasattr(data, 'dtype') and np.issubdtype(data.dtype, np.floating):
        return float(data)
    elif hasattr(data, 'dtype') and np.issubdtype(data.dtype, np.integer):
        return int(data)
    else:
        return data


class WorldClimExtractor:
    """
    WorldClim data extractor that downloads and processes real raster files.
    
    This class provides comprehensive functionality for working with WorldClim v2.1
    bioclimate data, including downloading, caching, and extracting values at
    specific coordinates.
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
        
        # WorldClim download URLs for bioclimate variables
        self.base_url = "https://geodata.ucdavis.edu/climate/worldclim/2_1/base/"
        
        logger.info(f"WorldClim extractor initialized - Data dir: {self.data_dir}, Resolution: {self.resolution}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=5)
        timeout = aiohttp.ClientTimeout(total=300)  # 5 minutes for large downloads
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

    
    
    
    
    
    async def download_worldclim_data(self) -> Dict[str, Any]:
        """
        Download WorldClim bioclimate data if not already present.
        
        Returns:
            Status dictionary with download information
        """
        if self.check_files_exist():
            return {
                "status": "already_exists",
                "message": "WorldClim data already downloaded",
                "location": str(self.data_dir),
                "files_count": 19
            }
        
        bio_dir = self.get_bio_dir()
        zip_path = self.data_dir / f"wc2.1_{self.resolution}_bio.zip"
        download_url = f"{self.base_url}/wc2.1_{self.resolution}_bio.zip"
        
        try:
            if not self.session:
                async with self:
                    return await self._download_and_extract(download_url, zip_path, bio_dir)
            else:
                return await self._download_and_extract(download_url, zip_path, bio_dir)
                
        except Exception as e:
            error_msg = f"Failed to download WorldClim data: {e}"
            logger.error(error_msg)
            raise WorldClimDataError(error_msg)
    
    async def _download_and_extract(self, download_url: str, zip_path: Path, bio_dir: Path) -> Dict[str, Any]:
        """Download and extract WorldClim data."""
        logger.info(f"Downloading WorldClim data from {download_url}")
        logger.info(f"This may take several minutes (~900MB download)")
        
        start_time = datetime.utcnow()
        
        # Download zip file
        async with self.session.get(download_url) as response:
            if response.status != 200:
                raise WorldClimDataError(f"Download failed with status {response.status}")
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            async with aiofiles.open(zip_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(8192):
                    await f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        if downloaded % (1024 * 1024 * 10) == 0:  # Log every 10MB
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded / (1024*1024):.1f} MB)")
        
        # Extract zip file
        logger.info("Extracting WorldClim data...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        # Cleanup zip file
        zip_path.unlink()
        
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        
        if self.check_files_exist():
            logger.info(f"WorldClim download completed in {duration:.1f} seconds")
            return {
                "status": "downloaded",
                "message": "WorldClim data downloaded and extracted successfully",
                "location": str(self.data_dir),
                "files_count": 19,
                "download_time": duration,
                "file_size_mb": downloaded / (1024 * 1024)
            }
        else:
            missing_files = []
            expected_files = [f"wc2.1_{self.resolution}_bio_{i}.tif" for i in range(1, 20)]
            for filename in expected_files:
                file_path = self.data_dir / filename
                if not file_path.exists():
                    missing_files.append(str(filename))
            
            error_msg = f"Download completed but files are missing: {', '.join(missing_files)}"
            logger.error(f"Missing files after download: {missing_files}")
            logger.error(f"Expected in directory: {self.data_dir.resolve()}")
            raise WorldClimDataError(error_msg)
    
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
        cache_key = f"{latitude:.3f},{longitude:.3f}:{','.join(sorted(bio_variables or []))}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Default variables if none specified
        if bio_variables is None:
            bio_variables = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]
        
        # Ensure data is downloaded
        await self.download_worldclim_data()
        
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
                bio_num = int(bio_var.replace("bio", ""))
                file_path = bio_dir / f"wc2.1_{self.resolution}_bio_{bio_num}.tif"
                
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
                                    
                                    # Convert numpy types to Python native types for MongoDB compatibility
                                    if isinstance(value, np.number):
                                        value = value.item()
                            else:
                                value = None  # Real missing data - NO dummy/fake values
                            
                            result[bio_var] = value
                    else:
                        logger.warning(f"File not found: {file_path}")
                        result[bio_var] = None  # Error = None/NaN, NOT fake data
                        
                except Exception as e:
                    logger.warning(f"Error extracting {bio_var} at {latitude}, {longitude}: {e}")
                    result[bio_var] = None  # Error = None/NaN, NOT fake data
            
            # Convert any remaining numpy types
            result = convert_numpy_types(result)
            
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
    
    async def enrich_gbif_occurrences(self, occurrences: List[Dict[str, Any]],
                                    climate_variables: Optional[List[str]] = None,
                                    include_elevation: bool = True,
                                    progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Enrich GBIF occurrence records with real climate and elevation data.
        
        Args:
            occurrences: List of GBIF occurrence records
            climate_variables: Climate variables to extract
            include_elevation: Whether to include SRTM elevation data
            progress_callback: Progress callback function
            
        Returns:
            List of enriched occurrence records with real environmental data
        """
        logger.info(f"Enriching {len(occurrences):,} occurrences with real environmental data")
        
        if not climate_variables:
            climate_variables = ["bio1", "bio4", "bio5", "bio6", "bio12", "bio13", "bio14", "bio15"]
        
        # Extract coordinates from valid occurrences
        valid_occurrences = []
        coordinates = []
        
        for occurrence in occurrences:
            lat = occurrence.get("decimalLatitude")
            lon = occurrence.get("decimalLongitude")
            
            if lat is not None and lon is not None:
                valid_occurrences.append(occurrence)
                coordinates.append((lat, lon))
            else:
                # Add environmental_data as None for invalid coordinates
                occurrence["environmental_data"] = None
        
        if not coordinates:
            logger.warning("No valid coordinates found in occurrences")
            return occurrences
        
        logger.info(f"Processing {len(coordinates)} valid coordinates")
        
        # Extract climate data in batch
        logger.info("Extracting WorldClim climate data...")
        climate_results = await self.extract_climate_batch(coordinates, climate_variables)
        
        # Extract elevation data if requested and available
        elevation_results = []
        if include_elevation and ELEVATION_AVAILABLE:
            try:
                logger.info("Extracting SRTM elevation data...")
                elevation_extractor = get_elevation_extractor()
                async with elevation_extractor:
                    elevation_results = await elevation_extractor.extract_elevation_batch(coordinates)
            except Exception as e:
                logger.error(f"Error extracting elevation data: {e}")
                # Continue without elevation data
                elevation_results = []
        elif include_elevation and not ELEVATION_AVAILABLE:
            logger.warning("Elevation extraction requested but elevation service not available")
        
        # Merge environmental data back with occurrences
        for i, occurrence in enumerate(valid_occurrences):
            env_data = {}
            
            # Add climate data
            if i < len(climate_results):
                climate_data = climate_results[i]
                
                # Extract just the climate variables (remove metadata)
                for var in climate_variables:
                    env_data[var] = climate_data.get(var)
                
                # Add metadata
                occurrence["data_source"] = climate_data.get("data_source", "unknown")
                occurrence["extraction_date"] = climate_data.get("extraction_date")
            
            # Add elevation data if available
            if i < len(elevation_results):
                elevation_data = elevation_results[i]
                env_data["elevation"] = elevation_data.get("elevation")
                env_data["elevation_source"] = elevation_data.get("data_source", "unknown")
            
            occurrence["environmental_data"] = env_data
            
            if progress_callback:
                progress_callback(i + 1, len(valid_occurrences))
        
        data_sources = []
        if climate_results:
            data_sources.append("WorldClim climate")
        if elevation_results:
            data_sources.append("SRTM elevation")
            
        logger.info(f"Successfully enriched {len(valid_occurrences)} occurrences with {', '.join(data_sources)} data")
        return occurrences
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get the current status of the WorldClim extractor."""
        bio_dir = self.get_bio_dir()
        files_exist = self.check_files_exist()
        
        if files_exist:
            # Count actual files
            tif_files = list(bio_dir.glob("*.tif"))
            total_size = sum(f.stat().st_size for f in tif_files if f.exists())
            size_mb = total_size / (1024 * 1024)
        else:
            tif_files = []
            size_mb = 0
        
        extraction_capabilities = [
            "Single coordinate extraction",
            "Batch coordinate processing", 
            "GBIF occurrence enrichment",
            "Real data only (no fake values)"
        ]
        
        # Add elevation capability if available
        if ELEVATION_AVAILABLE:
            extraction_capabilities.append("SRTM elevation integration")
        
        return {
            "service": "WorldClim Data Extractor",
            "version": "v2.1",
            "resolution": self.resolution,
            "data_directory": str(self.data_dir),
            "files_downloaded": files_exist,
            "files_count": len(tif_files),
            "data_size_mb": round(size_mb, 2),
            "cache_size": len(self.cache),
            "variables_available": list(range(1, 20)) if files_exist else [],
            "elevation_service": "Available" if ELEVATION_AVAILABLE else "Not available",
            "extraction_capabilities": extraction_capabilities
        }


# Singleton instance for global use
_worldclim_extractor = None


def get_worldclim_extractor() -> WorldClimExtractor:
    """Get or create the global WorldClim extractor instance."""
    global _worldclim_extractor
    if _worldclim_extractor is None:
        _worldclim_extractor = WorldClimExtractor()
    return _worldclim_extractor


# Convenience functions for direct access
async def extract_climate_data(latitude: float, longitude: float, 
                             variables: Optional[List[str]] = None) -> Dict[str, Any]:
    """Extract climate data for a single coordinate."""
    extractor = get_worldclim_extractor()
    return await extractor.extract_climate_data(latitude, longitude, variables)


async def extract_climate_batch(coordinates: List[Tuple[float, float]], 
                               variables: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Extract climate data for multiple coordinates."""
    extractor = get_worldclim_extractor()
    return await extractor.extract_climate_batch(coordinates, variables)


async def enrich_gbif_occurrences(occurrences: List[Dict[str, Any]],
                                 climate_variables: Optional[List[str]] = None,
                                 include_elevation: bool = True,
                                 progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """Enrich GBIF occurrences with climate and elevation data."""
    extractor = get_worldclim_extractor()
    return await extractor.enrich_gbif_occurrences(occurrences, climate_variables, include_elevation, progress_callback)
