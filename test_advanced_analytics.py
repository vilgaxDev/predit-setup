"""
test_advanced_analytics.py
===========================
Test script for advanced analytics features.
"""

import pandas as pd
import sys
import os
import io

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from advanced_analytics import AdvancedHealthAnalytics

def test_risk_classification():
    """Test risk classification functionality."""
    print("="*60)
    print("TEST 1: Risk Classification")
    print("="*60)
    
    # Create sample data
    sample_data = {
        'Age': [20, 40, 17, 30, 25],
        'Parity': [2, 5, 1, 3, 0],
        'HIV Status (P/N/KP/U)': ['N', 'P', 'N', 'N', 'N'],
        'Gestation at Birth (Weeks)': [38, 36, 40, 28, 39],
        'Delivery Complications (codes)': ['', 'PPH', '', '', ''],
        'Mode_of_Delivery': ['SVD', 'CS', 'SVD', 'SVD', 'SVD'],
        'Birth Weight (in grams)': [3000, 2800, 3200, 2000, 3100],
        'No. of ANC Visits': [4, 3, 5, 2, 4],
        "Mother's Status after Delivery": ['Alive', 'Alive', 'Alive', 'Alive', 'Alive'],
        'Live Birth / FSB / MSB': ['Live Birth', 'Live Birth', 'Live Birth', 'Live Birth', 'Live Birth']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analytics
    analytics = AdvancedHealthAnalytics()
    
    # Classify risks
    results = analytics.classify_all_risks(df)
    
    print(f"\n✅ Risk classification completed for {len(df)} records")
    print("\nRisk Distribution:")
    print(results['risk_level'].value_counts())
    
    print("\nSample Risk Classification:")
    for idx, row in results.head(3).iterrows():
        print(f"\nRecord {idx + 1}:")
        print(f"  Risk Level: {row['risk_level']}")
        print(f"  Risk Factors: {row['risk_factors']}")
        print(f"  Priority: {row['priority']}")
        print(f"  Action Steps: {row['action_steps'][:100]}...")
    
    # Generate summary
    summary = analytics.generate_risk_summary(results)
    print(f"\n✅ Risk Summary Generated:")
    print(f"  Total: {summary['total_pregnancies']}")
    print(f"  High Risk: {summary['high_risk_count']} ({summary['high_risk_percentage']:.1f}%)")
    print(f"  Moderate Risk: {summary['moderate_risk_count']} ({summary['moderate_risk_percentage']:.1f}%)")
    print(f"  Low Risk: {summary['low_risk_count']} ({summary['low_risk_percentage']:.1f}%)")
    
    return True

def test_trend_analysis():
    """Test time-series trend analysis."""
    print("\n" + "="*60)
    print("TEST 2: Time-Series Trend Analysis")
    print("="*60)
    
    # Create sample data with dates
    import numpy as np
    from datetime import datetime, timedelta
    
    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=i*10) for i in range(30)]
    
    sample_data = {
        'Date of Delivery': dates,
        'Age': np.random.randint(18, 40, 30),
        'Birth Weight (in grams)': np.random.randint(2000, 4000, 30),
        'Gestation at Birth (Weeks)': np.random.randint(36, 42, 30),
        'No. of ANC Visits': np.random.randint(1, 8, 30)
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analytics
    analytics = AdvancedHealthAnalytics()
    
    # Analyze trends
    trends = analytics.analyze_time_series_trends(df, 'Date of Delivery', 'Age', 'monthly')
    
    print(f"\n✅ Trend analysis completed")
    print(f"  Number of periods analyzed: {len(trends)}")
    
    if trends:
        print("\nTrend Results:")
        for trend in trends[:5]:
            print(f"  {trend.period}: {trend.value:.2f} ({trend.trend})")
    
    return True

def test_mortality_analysis():
    """Test mortality root-cause analysis."""
    print("\n" + "="*60)
    print("TEST 3: Mortality Root-Cause Analysis")
    print("="*60)
    
    # Create sample data with mortality
    sample_data = {
        'Age': [25, 30, 35, 28, 32],
        'Parity': [1, 2, 3, 1, 2],
        'HIV Status (P/N/KP/U)': ['N', 'P', 'N', 'N', 'N'],
        'Gestation at Birth (Weeks)': [38, 36, 40, 28, 39],
        'Delivery Complications (codes)': ['', 'PPH', '', '', ''],
        'Mode_of_Delivery': ['SVD', 'CS', 'SVD', 'SVD', 'SVD'],
        'Birth Weight (in grams)': [3000, 2800, 3200, 2000, 3100],
        'No. of ANC Visits': [4, 3, 5, 2, 4],
        "Mother's Status after Delivery": ['Alive', 'Deceased', 'Alive', 'Alive', 'Alive'],
        'Live Birth / FSB / MSB': ['Live Birth', 'Live Birth', 'Live Birth', 'FSB', 'Live Birth']
    }
    
    df = pd.DataFrame(sample_data)
    
    # Initialize analytics
    analytics = AdvancedHealthAnalytics()
    
    # Analyze maternal mortality
    print("\nMaternal Mortality Analysis:")
    maternal_drivers = analytics.analyze_mortality_drivers(df, 'maternal')
    
    if maternal_drivers:
        print(f"  ✅ Found {len(maternal_drivers)} mortality drivers")
        for driver in maternal_drivers:
            print(f"    - {driver.cause}: {driver.frequency} cases ({driver.percentage:.1f}%)")
    else:
        print("  ℹ️  No maternal mortality drivers detected")
    
    # Analyze neonatal mortality
    print("\nNeonatal Mortality Analysis:")
    neonatal_drivers = analytics.analyze_mortality_drivers(df, 'neonatal')
    
    if neonatal_drivers:
        print(f"  ✅ Found {len(neonatal_drivers)} mortality drivers")
        for driver in neonatal_drivers:
            print(f"    - {driver.cause}: {driver.frequency} cases ({driver.percentage:.1f}%)")
    else:
        print("  ℹ️  No neonatal mortality drivers detected")
    
    return True

def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("ADVANCED ANALYTICS TEST SUITE")
    print("="*60)
    
    try:
        # Test 1: Risk Classification
        test1_passed = test_risk_classification()
        
        # Test 2: Trend Analysis
        test2_passed = test_trend_analysis()
        
        # Test 3: Mortality Analysis
        test3_passed = test_mortality_analysis()
        
        # Summary
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        print(f"Risk Classification: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
        print(f"Trend Analysis: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
        print(f"Mortality Analysis: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
        
        if all([test1_passed, test2_passed, test3_passed]):
            print("\n🎉 ALL TESTS PASSED!")
            return 0
        else:
            print("\n⚠️  SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
