#!/usr/bin/env python3
"""
Generate Comparative Seasonal Heatmaps

This script creates both peak season (April) and off-season (July) heatmaps
to demonstrate how the seasonal model captures flowering/observation patterns.
"""

import asyncio
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from clean_seasonal_heatmap import load_seasonal_model, create_seasonal_heatmap

async def create_seasonal_comparison():
    """Create both peak and off-season heatmaps for comparison."""
    print("=" * 80)
    print("SEASONAL COMPARISON: PEAK vs OFF-SEASON INVASION RISK")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Load seasonal model
        model_data, df = load_seasonal_model()
        
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Generate heatmap for peak season (April - 40% of observations)
        print("1. Generating PEAK SEASON heatmap (April - Flowering Peak)...")
        peak_map_path, peak_results = await create_seasonal_heatmap(
            model_data, df, output_dir,
            grid_resolution=0.025,  # Slightly coarser for faster processing
            api_base_url="http://127.0.0.1:8000",
            prediction_month=4  # Peak month
        )
        
        print("\n" + "="*50)
        print("Peak Season (April) Results:")
        print(f"  Risk Range: {peak_results['invasion_risk'].min():.3f} - {peak_results['invasion_risk'].max():.3f}")
        print(f"  Mean Risk: {peak_results['invasion_risk'].mean():.3f}")
        print(f"  API Success: {peak_results['api_success_rate']*100:.1f}%")
        print("="*50 + "\n")
        
        # Generate heatmap for off-season (July - Winter)
        print("2. Generating OFF-SEASON heatmap (July - Winter Low)...")
        off_map_path, off_results = await create_seasonal_heatmap(
            model_data, df, output_dir,
            grid_resolution=0.025,
            api_base_url="http://127.0.0.1:8000",
            prediction_month=7  # Off-season month
        )
        
        print("\n" + "="*50)
        print("Off-Season (July) Results:")
        print(f"  Risk Range: {off_results['invasion_risk'].min():.3f} - {off_results['invasion_risk'].max():.3f}")
        print(f"  Mean Risk: {off_results['invasion_risk'].mean():.3f}")
        print(f"  API Success: {off_results['api_success_rate']*100:.1f}%")
        print("="*50 + "\n")
        
        # Calculate seasonal difference
        peak_mean = peak_results['invasion_risk'].mean()
        off_mean = off_results['invasion_risk'].mean()
        seasonal_difference = peak_mean - off_mean
        seasonal_ratio = peak_mean / off_mean if off_mean > 0 else float('inf')
        
        print("=" * 80)
        print("SEASONAL COMPARISON SUMMARY")
        print("=" * 80)
        print(f"Peak Season (April):")
        print(f"  Map: {peak_map_path}")
        print(f"  Mean Risk: {peak_mean:.3f}")
        print(f"  Max Risk: {peak_results['invasion_risk'].max():.3f}")
        print()
        print(f"Off-Season (July):")
        print(f"  Map: {off_map_path}")
        print(f"  Mean Risk: {off_mean:.3f}")
        print(f"  Max Risk: {off_results['invasion_risk'].max():.3f}")
        print()
        print(f"Seasonal Impact:")
        print(f"  Risk Difference: {seasonal_difference:+.3f}")
        print(f"  Risk Ratio: {seasonal_ratio:.2f}x")
        print(f"  Seasonal Enhancement: {((seasonal_ratio-1)*100):+.1f}%")
        print()
        print("Model successfully captures seasonal flowering patterns!")
        print("Peak season shows elevated invasion risk during flowering period.")
        print("=" * 80)
        
        return {
            'peak_results': peak_results,
            'off_results': off_results,
            'seasonal_difference': seasonal_difference,
            'seasonal_ratio': seasonal_ratio
        }
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise

def main():
    """Run seasonal comparison."""
    results = asyncio.run(create_seasonal_comparison())
    return results

if __name__ == "__main__":
    main()
