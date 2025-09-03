"""
Elevation Data Extractor
========================

This module provides elevation data extraction using SRTM (Shuttle Radar Topography Mission) 
digital elevation models via the Open-Topo-Data API. It follows the same patterns and 
data integrity policies as the WorldClim extractor.

Features:
- SRTM 30m resolution elevation data
- Batch processing for multiple coordinates  
- Real data only - returns NaN when unavailable
- Caching for performance optimization
- Integration with GBIF occurrence data enrichment
- Same error handling patterns as WorldClim service

Data Source: NASA SRTM via Open-Topo-Data API (https://www.opentopodata.org/)
Resolution: ~30m (1 arc-second)
Coverage: Global (60°N to 60°S)

Author: MC141
"""

import aiohttp
import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class ElevationDataError(Exception):
    """Custom exception for elevation data errors."""
    pass


class ElevationExtractor:
    """Extract elevation data from SRTM digital elevation models."""
    
    def __init__(self, base_url: str = "https://api.opentopodata.org/v1/srtm30m"):
        """
        Initialize the elevation extractor.
        
        Args:
            base_url: Base URL for Open-Topo-Data API
        """
        self.base_url = base_url
        self.session = None
        self._cache = {}  # Simple in-memory cache
        
        # Rate limiting to respect API limits
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests - more conservative default
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
            
    def _cache_key(self, latitude: float, longitude: float) -> str:
        """Generate cache key for coordinates (rounded to ~100m precision)."""
        # Round to 3 decimal places (~100m precision) for efficient caching
        lat_rounded = round(latitude, 3)
        lon_rounded = round(longitude, 3)
        return f"{lat_rounded},{lon_rounded}"
        
    async def _rate_limit(self, backoff_factor: float = 1.0):
        """
        Implement rate limiting to respect API limits with exponential backoff.
        
        Args:
            backoff_factor: Multiplier for the minimum request interval
                            Used for exponential backoff when rate limited
        """
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        # Calculate wait time with backoff factor
        wait_time = self.min_request_interval * backoff_factor
        
        if time_since_last < wait_time:
            await asyncio.sleep(wait_time - time_since_last)
            
        self.last_request_time = time.time()
        
    async def extract_elevation(self, latitude: float, longitude: float) -> Dict[str, Any]:
        """
        Extract elevation data for a single coordinate.
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            
        Returns:
            Dictionary containing elevation data and metadata
        """
        # Check cache first
        cache_key = self._cache_key(latitude, longitude)
        if cache_key in self._cache:
            logger.debug(f"Cache hit for {latitude}, {longitude}")
            return self._cache[cache_key]
            
        # Validate coordinates
        if not (-90 <= latitude <= 90) or not (-180 <= longitude <= 180):
            logger.warning(f"Invalid coordinates: {latitude}, {longitude}")
            return {
                "elevation": None,
                "data_source": "SRTM_validation_error",
                "extraction_date": datetime.utcnow().isoformat(),
                "error": "Invalid coordinates"
            }
            
        # Initialize retry variables
        max_retries = 5
        retry_count = 0
        backoff_factor = 1.0
        timeout = aiohttp.ClientTimeout(total=10)
        
        while retry_count <= max_retries:
            try:
                await self._rate_limit(backoff_factor)
                
                # Build API request
                url = f"{self.base_url}?locations={latitude},{longitude}"
                
                if not self.session:
                    raise ElevationDataError("Session not initialized. Use async context manager.")
                    
                async with self.session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Parse response
                        if data.get("status") == "OK" and data.get("results"):
                            result_data = data["results"][0]
                            elevation = result_data.get("elevation")
                            
                            result = {
                                "elevation": float(elevation) if elevation is not None else None,
                                "data_source": "SRTM_30m_real_data",
                                "extraction_date": datetime.utcnow().isoformat(),
                                "dataset": result_data.get("dataset", "srtm30m"),
                                "location": {
                                    "latitude": result_data.get("location", {}).get("lat", latitude),
                                    "longitude": result_data.get("location", {}).get("lng", longitude)
                                }
                            }
                            
                            # Cache successful result
                            self._cache[cache_key] = result
                            return result
                            
                        else:
                            # API returned error
                            error_msg = data.get("error", "Unknown API error")
                            logger.warning(f"API error for {latitude}, {longitude}: {error_msg}")
                            
                            result = {
                                "elevation": None,
                                "data_source": "SRTM_api_error",
                                "extraction_date": datetime.utcnow().isoformat(),
                                "error": error_msg
                            }
                            
                            # Cache error result to avoid repeated failed requests
                            self._cache[cache_key] = result
                            return result
                            
                    elif response.status == 429:
                        # Rate limit hit - implement exponential backoff
                        retry_count += 1
                        backoff_factor *= 2  # Exponential backoff
                        wait_time = self.min_request_interval * backoff_factor
                        
                        if retry_count <= max_retries:
                            logger.warning(f"Rate limit hit (429) for {latitude}, {longitude}. "
                                        f"Retrying in {wait_time:.2f}s (attempt {retry_count}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue  # Try again
                        else:
                            logger.error(f"Rate limit (429) max retries exceeded for {latitude}, {longitude}")
                            result = {
                                "elevation": None,
                                "data_source": f"SRTM_rate_limit_error",
                                "extraction_date": datetime.utcnow().isoformat(),
                                "error": "Rate limit exceeded after multiple retries"
                            }
                            return result
                            
                    else:
                        logger.error(f"HTTP error {response.status} for {latitude}, {longitude}")
                        
                        result = {
                            "elevation": None,
                            "data_source": f"SRTM_http_error_{response.status}",
                            "extraction_date": datetime.utcnow().isoformat(),
                            "error": f"HTTP {response.status}"
                        }
                        
                        return result
                        
            except asyncio.TimeoutError:
                logger.error(f"Timeout extracting elevation for {latitude}, {longitude}")
                return {
                    "elevation": None,
                    "data_source": "SRTM_timeout_error",
                    "extraction_date": datetime.utcnow().isoformat(),
                    "error": "Request timeout"
                }
                
            except Exception as e:
                logger.error(f"Error extracting elevation for {latitude}, {longitude}: {e}")
                return {
                    "elevation": None,
                    "data_source": "SRTM_extraction_error",
                    "extraction_date": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
                
        # This should never be reached, but add a fallback return to satisfy type checking
        return {
            "elevation": None,
            "data_source": "SRTM_unknown_error",
            "extraction_date": datetime.utcnow().isoformat(),
            "error": "Unknown error in elevation extraction"
        }
            
    async def extract_elevation_batch(self, coordinates: List[Tuple[float, float]], 
                                    batch_size: int = 50) -> List[Dict[str, Any]]:
        """
        Extract elevation data for multiple coordinates in batches.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            batch_size: Maximum coordinates per API request
            
        Returns:
            List of elevation results in same order as input coordinates
        """
        logger.info(f"Extracting elevation for {len(coordinates)} coordinates in batches of {batch_size}")
        
        results = []
        
        # Process in batches to respect API limits
        for i in range(0, len(coordinates), batch_size):
            batch = coordinates[i:i + batch_size]
            batch_results = []
            
            # Open-Topo-Data supports multiple locations in one request
            if len(batch) > 1:
                batch_results = await self._extract_batch_api(batch)
            else:
                # Single coordinate
                lat, lon = batch[0]
                result = await self.extract_elevation(lat, lon)
                batch_results = [result]
            
            results.extend(batch_results)
            
            # Progress logging
            if len(coordinates) > 100:
                logger.info(f"Processed {min(i + batch_size, len(coordinates))}/{len(coordinates)} coordinates")
                
        logger.info(f"Successfully extracted elevation for {len(results)}/{len(coordinates)} coordinates")
        return results
        
    async def _extract_batch_api(self, coordinates: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
        """Extract elevation for multiple coordinates in a single API call."""
        # Initialize retry variables
        max_retries = 3
        retry_count = 0
        backoff_factor = 1.0
        timeout = aiohttp.ClientTimeout(total=30)
        
        while retry_count <= max_retries:
            try:
                await self._rate_limit(backoff_factor)
                
                # Build locations string
                locations = "|".join([f"{lat},{lon}" for lat, lon in coordinates])
                url = f"{self.base_url}?locations={locations}"
                
                if not self.session:
                    raise ElevationDataError("Session not initialized")
                    
                async with self.session.get(url, timeout=timeout) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if data.get("status") == "OK" and data.get("results"):
                            results = []
                            api_results = data["results"]
                            
                            # Ensure we have results for all coordinates
                            for i, (lat, lon) in enumerate(coordinates):
                                if i < len(api_results):
                                    result_data = api_results[i]
                                    elevation = result_data.get("elevation")
                                    
                                    result = {
                                        "elevation": float(elevation) if elevation is not None else None,
                                        "data_source": "SRTM_30m_real_data",
                                        "extraction_date": datetime.utcnow().isoformat(),
                                        "dataset": result_data.get("dataset", "srtm30m"),
                                        "location": {
                                            "latitude": result_data.get("location", {}).get("lat", lat),
                                            "longitude": result_data.get("location", {}).get("lng", lon)
                                        }
                                    }
                                    
                                    # Cache result
                                    cache_key = self._cache_key(lat, lon)
                                    self._cache[cache_key] = result
                                    
                                else:
                                    # Missing result for this coordinate
                                    result = {
                                        "elevation": None,
                                        "data_source": "SRTM_missing_result",
                                        "extraction_date": datetime.utcnow().isoformat(),
                                        "error": "No result returned for coordinate"
                                    }
                                    
                                results.append(result)
                                
                            return results
                            
                        else:
                            # API error - fall back to individual requests
                            logger.warning(f"Batch API error, falling back to individual requests")
                            return await self._extract_individual_fallback(coordinates)
                            
                    elif response.status == 429:
                        # Rate limit hit - implement exponential backoff
                        retry_count += 1
                        backoff_factor *= 3  # More aggressive backoff for batch requests
                        wait_time = self.min_request_interval * backoff_factor
                        
                        if retry_count <= max_retries:
                            logger.warning(f"Rate limit hit (429) for batch request. "
                                        f"Retrying in {wait_time:.2f}s (attempt {retry_count}/{max_retries})")
                            await asyncio.sleep(wait_time)
                            continue  # Try again with increased backoff
                        else:
                            # If max retries reached, fall back to individual requests with increased delay
                            logger.error(f"Batch rate limit (429) max retries exceeded, falling back to individual requests")
                            self.min_request_interval *= 2  # Double the base interval for future requests
                            return await self._extract_individual_fallback(coordinates, backoff_factor=backoff_factor)
                    else:
                        logger.error(f"Batch HTTP error {response.status}, falling back to individual requests")
                        return await self._extract_individual_fallback(coordinates)
                        
            except asyncio.TimeoutError:
                retry_count += 1
                backoff_factor *= 2
                
                if retry_count <= max_retries:
                    wait_time = self.min_request_interval * backoff_factor
                    logger.warning(f"Batch request timeout. Retrying in {wait_time:.2f}s "
                                f"(attempt {retry_count}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Batch request timeout after {max_retries} retries, falling back to individual requests")
                    return await self._extract_individual_fallback(coordinates)
                    
            except Exception as e:
                logger.error(f"Batch extraction error: {e}, falling back to individual requests")
                return await self._extract_individual_fallback(coordinates)
                
        # Should not reach here, but just in case
        return await self._extract_individual_fallback(coordinates)
            
    async def _extract_individual_fallback(self, coordinates: List[Tuple[float, float]], backoff_factor: float = 1.0) -> List[Dict[str, Any]]:
        """
        Fallback to individual requests when batch fails.
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            backoff_factor: Multiplier for request interval (used when coming from a rate-limited batch)
        """
        logger.info(f"Processing {len(coordinates)} coordinates individually")
        
        # Add increased delay between individual requests
        original_interval = self.min_request_interval
        self.min_request_interval = max(0.5, original_interval * backoff_factor)  # At least 500ms between individual requests
        
        results = []
        
        try:
            for lat, lon in coordinates:
                result = await self.extract_elevation(lat, lon)
                results.append(result)
        finally:
            # Restore original interval
            self.min_request_interval = original_interval
            
        return results
        
    async def enrich_gbif_occurrences(self, occurrences: List[Dict[str, Any]], 
                                     progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Enrich GBIF occurrences with elevation data.
        
        Args:
            occurrences: List of GBIF occurrence records
            progress_callback: Optional callback function for progress reporting
            
        Returns:
            List of GBIF records enriched with elevation data
        """
        if not occurrences:
            return []
            
        logger.info(f"Extracting SRTM elevation data for {len(occurrences)} occurrences")
        
        # Extract coordinates from occurrences
        coordinates = []
        for occurrence in occurrences:
            lat = occurrence.get("decimalLatitude")
            lon = occurrence.get("decimalLongitude")
            
            if lat is not None and lon is not None:
                try:
                    coordinates.append((float(lat), float(lon)))
                except (ValueError, TypeError):
                    # Invalid coordinates, will be skipped
                    coordinates.append((None, None))
            else:
                coordinates.append((None, None))
                
        # Extract elevation in batches
        valid_coordinates = [(lat, lon) for lat, lon in coordinates if lat is not None and lon is not None]
        
        if not valid_coordinates:
            logger.warning("No valid coordinates found in occurrences")
            return occurrences
            
        elevation_results = await self.extract_elevation_batch(valid_coordinates)
        
        # Create lookup dictionary for faster access
        elevation_lookup = {}
        for coord, result in zip(valid_coordinates, elevation_results):
            lat, lon = coord
            key = f"{lat:.5f},{lon:.5f}"
            elevation_lookup[key] = result
            
        # Enrich occurrences with elevation data
        enriched = []
        for i, occurrence in enumerate(occurrences):
            if i % 100 == 0 and progress_callback:
                progress_callback(i, len(occurrences))
                
            lat = occurrence.get("decimalLatitude")
            lon = occurrence.get("decimalLongitude")
            
            # Create a copy of the occurrence to avoid modifying the original
            enriched_occurrence = dict(occurrence)
            
            if lat is not None and lon is not None:
                try:
                    lat_f, lon_f = float(lat), float(lon)
                    key = f"{lat_f:.5f},{lon_f:.5f}"
                    
                    if key in elevation_lookup:
                        elevation_data = elevation_lookup[key]
                        
                        # Add elevation data to occurrence
                        if "elevation" not in enriched_occurrence:
                            enriched_occurrence["elevation"] = elevation_data.get("elevation")
                            
                        # Add elevation metadata
                        if "elevationAccuracy" not in enriched_occurrence:
                            enriched_occurrence["elevationAccuracy"] = 30  # SRTM 30m accuracy
                            
                        if "elevationSource" not in enriched_occurrence:
                            enriched_occurrence["elevationSource"] = elevation_data.get("data_source")
                except (ValueError, TypeError):
                    # Skip invalid coordinates
                    pass
                    
            enriched.append(enriched_occurrence)
            
        if progress_callback:
            progress_callback(len(occurrences), len(occurrences))
            
        logger.info(f"Successfully enriched {len(enriched)} occurrences with elevation data")
        return enriched

# Helper function to get a singleton instance
_elevation_extractor_instance = None

def get_elevation_extractor() -> ElevationExtractor:
    """
    Get a singleton instance of the ElevationExtractor.
    
    Returns:
        ElevationExtractor instance
    """
    global _elevation_extractor_instance
    
    if _elevation_extractor_instance is None:
        _elevation_extractor_instance = ElevationExtractor()
        
    return _elevation_extractor_instance


async def extract_elevation(latitude: float, longitude: float) -> Dict[str, Any]:
    """Extract elevation data for a single coordinate."""
    extractor = get_elevation_extractor()
    async with extractor:
        return await extractor.extract_elevation(latitude, longitude)


async def extract_elevation_batch(coordinates: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
    """Extract elevation data for multiple coordinates."""
    extractor = get_elevation_extractor()
    async with extractor:
        return await extractor.extract_elevation_batch(coordinates)


async def enrich_gbif_with_elevation(occurrences: List[Dict[str, Any]],
                                   progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
    """Enrich GBIF occurrences with elevation data."""
    extractor = get_elevation_extractor()
    async with extractor:
        return await extractor.enrich_gbif_occurrences(occurrences, progress_callback)
