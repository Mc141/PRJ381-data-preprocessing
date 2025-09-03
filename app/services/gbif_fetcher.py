"""
GBIF Data Fetcher Service
========================

Asynchronous service for fetching and processing global occurrence data from GBIF API.
Supports large-scale data retrieval with quality filtering, pagination, and error handling.

GBIF API Documentation: https://www.gbif.org/developer/occurrence
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
from datetime import datetime, date
import time
from urllib.parse import urlencode

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GBIFError(Exception):
    """Custom exception for GBIF API errors."""
    pass

class GBIFFetcher:
    """
    Asynchronous GBIF occurrence data fetcher with comprehensive error handling.
    
    Supports:
    - Global species occurrence retrieval
    - Quality filtering (coordinate accuracy, date validation)
    - Batch processing for large datasets
    - Rate limiting and retry logic
    - Progress tracking for large downloads
    """
    
    def __init__(self):
        self.base_url = "https://api.gbif.org/v1"
        self.session = None
        self.rate_limit_delay = 0.1  # 100ms between requests
        self.max_retries = 3
        self.timeout = 30
        
    async def __aenter__(self):
        """Async context manager entry."""
        connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self.session = aiohttp.ClientSession(
            connector=connector, 
            timeout=timeout,
            headers={"User-Agent": "PRJ381-DataPreprocessing/1.0"}
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def get_species_key(self, scientific_name: str) -> Optional[int]:
        """
        Get GBIF species key for scientific name.
        
        Args:
            scientific_name: Scientific name with authority (e.g., "Pyracantha angustifolia (Franch.) C.K.Schneid.")
            
        Returns:
            Species key (taxon ID) or None if not found
        """
        url = f"{self.base_url}/species/match"
        params = {"name": scientific_name}
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # Check for exact or fuzzy match
                    if (data.get("matchType") in ["EXACT", "FUZZY"] and 
                        data.get("rank") == "SPECIES"):
                        species_key = data.get("speciesKey") or data.get("usageKey")
                        if species_key:
                            logger.info(f"Found GBIF species key {species_key} for '{scientific_name}' "
                                      f"(match: {data.get('matchType')}, confidence: {data.get('confidence')})")
                            return species_key
                    
                    logger.warning(f"No suitable match found for '{scientific_name}' "
                                 f"(matchType: {data.get('matchType')}, rank: {data.get('rank')})")
                    return None
                else:
                    logger.error(f"Species lookup failed: {response.status}")
                    return None
                    
        except Exception as e:
            logger.error(f"Error getting species key: {e}")
            return None
    
    async def get_occurrence_count(self, species_key: int, **filters) -> int:
        """
        Get total occurrence count for species with filters.
        
        Args:
            species_key: GBIF species key
            **filters: Additional filter parameters
            
        Returns:
            Total occurrence count
        """
        url = f"{self.base_url}/occurrence/search"
        params = {
            "taxonKey": species_key,  # Use taxonKey not speciesKey
            "limit": 0,  # Just get count
            **filters
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    count = data.get("count", 0)
                    logger.info(f"Found {count:,} occurrences for species key {species_key}")
                    return count
                else:
                    logger.error(f"Count request failed: {response.status}")
                    return 0
                    
        except Exception as e:
            logger.error(f"Error getting occurrence count: {e}")
            return 0
    
    async def fetch_occurrences_batch(self, species_key: int, offset: int = 0, 
                                    limit: int = 300, **filters) -> Dict[str, Any]:
        """
        Fetch a batch of occurrences with retry logic.
        
        Args:
            species_key: GBIF species key
            offset: Record offset for pagination
            limit: Number of records per batch (max 300)
            **filters: Additional filter parameters
            
        Returns:
            GBIF API response data
        """
        url = f"{self.base_url}/occurrence/search"
        params = {
            "taxonKey": species_key,  # Use taxonKey not speciesKey
            "offset": offset,
            "limit": min(limit, 300),  # GBIF max is 300
            **filters
        }
        
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.rate_limit_delay)
                
                async with self.session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        logger.debug(f"Fetched batch: offset={offset}, count={len(data.get('results', []))}")
                        return data
                    elif response.status == 429:  # Rate limited
                        wait_time = 2 ** attempt
                        logger.warning(f"Rate limited, waiting {wait_time}s (attempt {attempt + 1})")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"HTTP {response.status}: {await response.text()}")
                        if attempt == self.max_retries - 1:
                            raise GBIFError(f"HTTP {response.status}")
                        
            except asyncio.TimeoutError:
                logger.warning(f"Timeout on attempt {attempt + 1}")
                if attempt == self.max_retries - 1:
                    raise GBIFError("Request timeout")
            except Exception as e:
                logger.error(f"Request error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise GBIFError(f"Request failed: {e}")
        
        raise GBIFError("Max retries exceeded")
    
    async def fetch_all_occurrences(self, scientific_name: str, 
                                  quality_filters: bool = True,
                                  date_from: Optional[str] = None,
                                  date_to: Optional[str] = None,
                                  country: Optional[str] = None,
                                  coordinate_uncertainty_max: Optional[int] = None,
                                  batch_size: int = 300,
                                  max_records: Optional[int] = None,
                                  progress_callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        Fetch all occurrences for a species with comprehensive filtering.
        
        Args:
            scientific_name: Scientific name of species
            quality_filters: Apply quality filters (coordinates, dates, etc.)
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            country: ISO country code (e.g., 'ZA' for South Africa)
            coordinate_uncertainty_max: Max coordinate uncertainty in meters
            batch_size: Records per API request (max 300)
            max_records: Maximum total records to fetch
            progress_callback: Function to call with progress updates
            
        Returns:
            List of occurrence records
        """
        # Get species key
        species_key = await self.get_species_key(scientific_name)
        if not species_key:
            raise GBIFError(f"Species '{scientific_name}' not found in GBIF")
        
        # Build filters
        filters = {}
        
        if quality_filters:
            filters.update({
                "hasCoordinate": "true",
                "hasGeospatialIssue": "false",
                "occurrenceStatus": "PRESENT"
            })
        
        if date_from:
            filters["eventDate"] = f"{date_from},{date_to or '2024-12-31'}"
        
        if country:
            filters["country"] = country
            
        if coordinate_uncertainty_max:
            filters["coordinateUncertaintyInMeters"] = f"0,{coordinate_uncertainty_max}"
        
        # Get total count
        total_count = await self.get_occurrence_count(species_key, **filters)
        if total_count == 0:
            logger.warning("No occurrences found matching criteria")
            return []
        
        # Apply max_records limit
        if max_records:
            total_count = min(total_count, max_records)
        
        logger.info(f"Fetching {total_count:,} occurrences for '{scientific_name}'")
        
        # Fetch all batches
        all_occurrences = []
        offset = 0
        
        while offset < total_count:
            current_batch_size = min(batch_size, total_count - offset)
            
            try:
                batch_data = await self.fetch_occurrences_batch(
                    species_key, offset, current_batch_size, **filters
                )
                
                batch_results = batch_data.get("results", [])
                all_occurrences.extend(batch_results)
                
                offset += len(batch_results)
                
                # Progress callback
                if progress_callback:
                    progress = (offset / total_count) * 100
                    await progress_callback(offset, total_count, progress)
                
                logger.info(f"Progress: {offset:,}/{total_count:,} ({offset/total_count*100:.1f}%)")
                
                # Stop if we got fewer results than expected (end of data)
                if len(batch_results) < current_batch_size:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching batch at offset {offset}: {e}")
                break
        
        logger.info(f"Successfully fetched {len(all_occurrences):,} occurrences")
        return all_occurrences
    
    def process_occurrence_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and clean a GBIF occurrence record.
        
        Args:
            record: Raw GBIF occurrence record
            
        Returns:
            Cleaned occurrence record
        """
        try:
            # Extract core fields
            processed = {
                "gbif_id": record.get("key"),
                "scientific_name": record.get("scientificName"),
                "species": record.get("species"),
                "genus": record.get("genus"),
                "family": record.get("family"),
                
                # Location
                "latitude": record.get("decimalLatitude"),
                "longitude": record.get("decimalLongitude"),
                "coordinate_uncertainty": record.get("coordinateUncertaintyInMeters"),
                "elevation": record.get("elevation"),
                "depth": record.get("depth"),
                
                # Geography
                "country": record.get("country"),
                "country_code": record.get("countryCode"),
                "state_province": record.get("stateProvince"),
                "locality": record.get("locality"),
                
                # Temporal
                "event_date": record.get("eventDate"),
                "year": record.get("year"),
                "month": record.get("month"),
                "day": record.get("day"),
                
                # Data quality
                "basis_of_record": record.get("basisOfRecord"),
                "occurrence_status": record.get("occurrenceStatus"),
                "has_geospatial_issues": record.get("hasGeospatialIssue", False),
                
                # Data provenance
                "dataset_key": record.get("datasetKey"),
                "publisher": record.get("publisher"),
                "institution_code": record.get("institutionCode"),
                "collection_code": record.get("collectionCode"),
                
                # Processing metadata
                "gbif_fetch_date": datetime.utcnow().isoformat(),
                "data_source": "GBIF"
            }
            
            # Clean up None values
            return {k: v for k, v in processed.items() if v is not None}
            
        except Exception as e:
            logger.error(f"Error processing occurrence record: {e}")
            return record  # Return original if processing fails

# Convenience functions for common operations
async def fetch_pyracantha_global(max_records: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Fetch global Pyracantha angustifolia (Franch.) C.K.Schneid. occurrences from GBIF.
    
    Uses the complete scientific name with authority to get the specific taxon
    (~3,300 records) rather than the broader narrowleaf firethorn group (~19,500).
    
    Args:
        max_records: Maximum records to fetch (None for all ~3,300)
        
    Returns:
        List of processed occurrence records
    """
    async with GBIFFetcher() as fetcher:
        return await fetcher.fetch_all_occurrences(
            "Pyracantha angustifolia (Franch.) C.K.Schneid.",
            quality_filters=True,
            coordinate_uncertainty_max=10000,  # 10km max uncertainty
            max_records=max_records
        )

async def fetch_pyracantha_south_africa() -> List[Dict[str, Any]]:
    """
    Fetch South African Pyracantha angustifolia (Franch.) C.K.Schneid. occurrences for validation.
    
    Uses the complete scientific name with authority for taxonomic precision.
    
    Returns:
        List of South African occurrence records
    """
    async with GBIFFetcher() as fetcher:
        return await fetcher.fetch_all_occurrences(
            "Pyracantha angustifolia (Franch.) C.K.Schneid.",
            quality_filters=True,
            country="ZA",  # South Africa
            coordinate_uncertainty_max=5000,  # 5km max uncertainty for validation
        )

async def get_species_info(scientific_name: str) -> Optional[Dict[str, Any]]:
    """
    Get species information from GBIF.
    
    Args:
        scientific_name: Scientific name to look up
        
    Returns:
        Species information or None if not found
    """
    async with GBIFFetcher() as fetcher:
        species_key = await fetcher.get_species_key(scientific_name)
        if species_key:
            return {
                "scientific_name": scientific_name,
                "gbif_species_key": species_key,
                "lookup_date": datetime.utcnow().isoformat()
            }
        return None
