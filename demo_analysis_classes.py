#!/usr/bin/env python3
"""
Demonstration script for the new class-based analysis structure.
This script shows how to use individual analysis classes directly.
"""

import numpy as np
from statistical_analyzer import (
    AnalysisBase, 
    MeanAnalysis, 
    VarianceAnalysis, 
    PMCCAnalysis, 
    ChiSquaredScipyAnalysis, 
    ChiSquaredManualAnalysis
)


def demonstrate_analysis_classes():
    """
    Demonstrate the new class-based analysis structure.
    """
    print("=" * 60)
    print("Demonstrating Class-Based Analysis Structure")
    print("=" * 60)
    
    # Example 1: Using MeanAnalysis directly
    print("\n1. MeanAnalysis Class:")
    mean_analysis = MeanAnalysis()
    print(f"Return format: {mean_analysis.return_format}")
    
    # Sample data from multiple sources
    mean_data = [
        np.array([2, 100]),  # Source 1: n=2, total=100
        np.array([3, 150]),  # Source 2: n=3, total=150
        np.array([1, 50])    # Source 3: n=1, total=50
    ]
    
    # Aggregate and analyze
    aggregated = mean_analysis.aggregate_data(mean_data)
    result = mean_analysis.analyze(aggregated)
    print(f"Mean result: {result}")
    print(f"Expected: {(100 + 150 + 50) / (2 + 3 + 1)}")
    
    # Example 2: Using VarianceAnalysis directly
    print("\n2. VarianceAnalysis Class:")
    variance_analysis = VarianceAnalysis()
    print(f"Return format: {variance_analysis.return_format}")
    
    variance_data = [
        np.array([2, 100, 100]),  # Source 1: n=2, sum_x2=100, total=100
        np.array([3, 150, 150]),  # Source 2: n=3, sum_x2=150, total=150
        np.array([1, 50, 50])     # Source 3: n=1, sum_x2=50, total=50
    ]
    
    aggregated = variance_analysis.aggregate_data(variance_data)
    result = variance_analysis.analyze(aggregated)
    print(f"Variance result: {result}")
    
    # Example 3: Using PMCCAnalysis directly
    print("\n3. PMCCAnalysis Class:")
    pmcc_analysis = PMCCAnalysis()
    print(f"Return format: {pmcc_analysis.return_format}")
    
    pmcc_data = [
        np.array([2, 10, 20, 200, 100, 400]),   # Source 1: n=2, sum_x=10, sum_y=20, sum_xy=200, sum_x2=100, sum_y2=400
        np.array([3, 15, 30, 450, 225, 900])    # Source 2: n=3, sum_x=15, sum_y=30, sum_xy=450, sum_x2=225, sum_y2=900
    ]
    
    aggregated = pmcc_analysis.aggregate_data(pmcc_data)
    result = pmcc_analysis.analyze(aggregated)
    print(f"PMCC result: {result}")
    
    # Example 4: Using ChiSquaredScipyAnalysis directly
    print("\n4. ChiSquaredScipyAnalysis Class:")
    chi_squared_analysis = ChiSquaredScipyAnalysis()
    print(f"Return format: {chi_squared_analysis.return_format}")
    
    # Sample contingency tables from multiple sources
    contingency_tables = [
        np.array([[10, 20], [15, 25]]),  # Source 1
        np.array([[5, 10], [8, 12]])     # Source 2
    ]
    
    aggregated = chi_squared_analysis.aggregate_data(contingency_tables)
    result = chi_squared_analysis.analyze(aggregated)
    print(f"Chi-squared result: {result}")
    
    # Example 5: Using ChiSquaredManualAnalysis directly
    print("\n5. ChiSquaredManualAnalysis Class:")
    chi_squared_manual = ChiSquaredManualAnalysis()
    print(f"Return format: {chi_squared_manual.return_format}")
    
    aggregated = chi_squared_manual.aggregate_data(contingency_tables)
    result = chi_squared_manual.analyze(aggregated)
    print(f"Manual chi-squared result: {result}")
    
    # Example 6: Demonstrating the abstract base class interface
    print("\n6. Abstract Base Class Interface:")
    analysis_classes = [MeanAnalysis, VarianceAnalysis, PMCCAnalysis, ChiSquaredScipyAnalysis, ChiSquaredManualAnalysis]
    
    for cls in analysis_classes:
        instance = cls()
        print(f"  {cls.__name__}:")
        print(f"    - Return format: {instance.return_format}")
        print(f"    - Has aggregate_data method: {hasattr(instance, 'aggregate_data')}")
        print(f"    - Has analyze method: {hasattr(instance, 'analyze')}")
        print(f"    - Inherits from AnalysisBase: {issubclass(cls, AnalysisBase)}")


def demonstrate_custom_analysis():
    """
    Demonstrate how to create a custom analysis class.
    """
    print("\n" + "=" * 60)
    print("Demonstrating Custom Analysis Class")
    print("=" * 60)
    
    class MedianAnalysis(AnalysisBase):
        """Custom analysis class for calculating median."""
        
        @property
        def return_format(self) -> str:
            return "sorted_values"
        
        def aggregate_data(self, input_data):
            """Aggregate data for median calculation."""
            if isinstance(input_data, list):
                # Flatten all arrays into a single array
                all_values = []
                for arr in input_data:
                    all_values.extend(arr)
                return np.array(all_values)
            return input_data
        
        def analyze(self, aggregated_data):
            """Calculate median from aggregated values."""
            return np.median(aggregated_data)
    
    # Test the custom analysis
    print("Custom MedianAnalysis Class:")
    median_analysis = MedianAnalysis()
    print(f"Return format: {median_analysis.return_format}")
    
    # Sample data
    median_data = [
        np.array([1, 3, 5]),  # Source 1
        np.array([2, 4, 6]),  # Source 2
        np.array([7, 9])      # Source 3
    ]
    
    aggregated = median_analysis.aggregate_data(median_data)
    result = median_analysis.analyze(aggregated)
    print(f"Median result: {result}")
    print(f"Expected: {np.median([1, 3, 5, 2, 4, 6, 7, 9])}")


def demonstrate_statistical_analyzer():
    """
    Demonstrate how the StatisticalAnalyzer uses the new class-based structure.
    """
    print("\n" + "=" * 60)
    print("Demonstrating StatisticalAnalyzer with Class-Based Structure")
    print("=" * 60)
    
    from statistical_analyzer import StatisticalAnalyzer
    
    analyzer = StatisticalAnalyzer()
    
    # Test supported analysis types
    supported_types = analyzer.get_supported_analysis_types()
    print(f"Supported analysis types: {supported_types}")
    
    # Test mean calculation using the analyzer
    mean_data = np.array([[2, 100], [3, 150], [1, 50]])
    mean_result = analyzer.analyze_data(mean_data, "mean")
    print(f"Mean calculation via analyzer: {mean_result}")
    
    # Test variance calculation using the analyzer
    variance_data = np.array([[2, 100, 100], [3, 150, 150], [1, 50, 50]])
    variance_result = analyzer.analyze_data(variance_data, "variance")
    print(f"Variance calculation via analyzer: {variance_result}")
    
    # Test PMCC calculation using the analyzer
    pmcc_data = np.array([[2, 10, 20, 200, 100, 400], [3, 15, 30, 450, 225, 900]])
    pmcc_result = analyzer.analyze_data(pmcc_data, "PMCC")
    print(f"PMCC calculation via analyzer: {pmcc_result}")
    
    # Test chi-squared calculation using the analyzer
    contingency_table = np.array([[10, 20], [15, 25]])
    chi_squared_result = analyzer.analyze_data(contingency_table, "chi_squared_scipy")
    print(f"Chi-squared calculation via analyzer: {chi_squared_result}")


def main():
    """Run all demonstrations."""
    try:
        demonstrate_analysis_classes()
        demonstrate_custom_analysis()
        demonstrate_statistical_analyzer()
        
        print("\n" + "=" * 60)
        print("✅ All demonstrations completed successfully!")
        print("The class-based analysis structure is working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        raise


if __name__ == "__main__":
    main() 