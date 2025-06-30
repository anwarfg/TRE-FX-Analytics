import numpy as np
from scipy import stats
from typing import Dict, Any, Union, List
from abc import ABC, abstractmethod


class AnalysisBase(ABC):
    """
    Abstract base class for statistical analyses.
    All analysis classes must inherit from this and implement required methods.
    """
    
    @property
    @abstractmethod
    def return_format(self) -> dict:
        """Return format description for the analysis."""
        pass
    
    @abstractmethod
    def aggregate_data(self, input_data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Aggregate input data for analysis.
        
        Args:
            input_data: Input data (numpy array or list of arrays)
            
        Returns:
            np.ndarray: Aggregated data ready for analysis
        """
        pass
    
    @abstractmethod
    def analyze(self, aggregated_data: np.ndarray) -> Union[float, Dict[str, Any]]:
        """
        Perform the statistical analysis.
        
        Args:
            aggregated_data: Aggregated data from aggregate_data method
            
        Returns:
            Union[float, Dict[str, Any]]: Analysis result
        """
        pass


class MeanAnalysis(AnalysisBase):
    """Analysis class for calculating mean values."""
    
    def __init__(self):
        self.aggregated_data = {}
    
    @property
    def return_format(self) -> dict:
        return {"n": None, "total": None}
    
    def aggregate_data(self, input_data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Aggregate data for mean calculation."""
        if isinstance(input_data, list):
            return np.vstack(input_data)
        return input_data
    
    def analyze(self, aggregated_data: np.ndarray) -> float:
        """Calculate mean from aggregated values."""
        n, total = np.sum(aggregated_data, axis=0)
        # Store the aggregated values
        self.aggregated_data = {"n": n, "total": total}
        return total / n


class VarianceAnalysis(AnalysisBase):
    """Analysis class for calculating variance."""
    
    def __init__(self):
        self.aggregated_data = {}
    
    @property
    def return_format(self) -> dict:
        return {"n": None, "sum_x2": None, "total": None}
    
    def aggregate_data(self, input_data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Aggregate data for variance calculation."""
        if isinstance(input_data, list):
            return np.vstack(input_data)
        return input_data
    
    def analyze(self, aggregated_data: np.ndarray) -> float:
        """Calculate variance from aggregated values."""
        n, sum_x2, total = np.sum(aggregated_data, axis=0)
        # Store the aggregated values
        self.aggregated_data = {"n": n, "sum_x2": sum_x2, "total": total}
        return (sum_x2 - (total * total) / n) / (n - 1)


class PMCCAnalysis(AnalysisBase):
    """Analysis class for calculating Pearson's correlation coefficient."""
    
    def __init__(self):
        self.aggregated_data = {}
    
    @property
    def return_format(self) -> dict:
        return {"n": None, "sum_x": None, "sum_y": None, "sum_xy": None, "sum_x2": None, "sum_y2": None}
    
    def aggregate_data(self, input_data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Aggregate data for PMCC calculation."""
        if isinstance(input_data, list):
            return np.vstack(input_data)
        return input_data
    
    def analyze(self, aggregated_data: np.ndarray) -> float:
        """Calculate Pearson's correlation coefficient from aggregated values."""
        n, sum_x, sum_y, sum_xy, sum_x2, sum_y2 = np.sum(aggregated_data, axis=0)
        
        # Store the aggregated values
        self.aggregated_data = {
            "n": n, "sum_x": sum_x, "sum_y": sum_y,
            "sum_xy": sum_xy, "sum_x2": sum_x2, "sum_y2": sum_y2
        }
        
        # Calculate means
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        # Calculate standard deviations
        std_x = np.sqrt(sum_x2 - (sum_x ** 2) / n)
        std_y = np.sqrt(sum_y2 - (sum_y ** 2) / n)
        
        # Calculate covariance
        cov = (sum_xy - (sum_x * sum_y) / n) / (n - 1)
        
        # Calculate correlation coefficient
        return cov / (std_x * std_y)


class ChiSquaredScipyAnalysis(AnalysisBase):
    """Analysis class for chi-squared test using scipy."""
    
    def __init__(self):
        self.aggregated_data = {}
    
    @property
    def return_format(self) -> dict:
        return {"contingency_table": None}
    
    def aggregate_data(self, input_data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Aggregate contingency tables."""
        if isinstance(input_data, list):
            # For contingency tables, we need to combine them
            if len(input_data) == 1:
                return input_data[0]
            else:
                # Combine multiple contingency tables
                combined = np.zeros_like(input_data[0])
                for table in input_data:
                    combined += table
                return combined
        return input_data
    
    def analyze(self, aggregated_data: np.ndarray) -> float:
        """Calculate chi-squared statistic using scipy."""
        # Store the contingency table
        self.aggregated_data = {"contingency_table": aggregated_data}
        
        # Get both corrected and uncorrected results
        chi2_corrected, p_corrected, dof, expected = stats.chi2_contingency(aggregated_data)
        chi2_uncorrected, p_uncorrected, _, _ = stats.chi2_contingency(aggregated_data, correction=False)
        
        return chi2_uncorrected


class ChiSquaredManualAnalysis(AnalysisBase):
    """Analysis class for manual chi-squared calculation."""
    
    def __init__(self):
        self.aggregated_data = {}
    
    @property
    def return_format(self) -> dict:
        return {"contingency_table": None}
    
    def aggregate_data(self, input_data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """Aggregate contingency tables."""
        if isinstance(input_data, list):
            # For contingency tables, we need to combine them
            if len(input_data) == 1:
                return input_data[0]
            else:
                # Combine multiple contingency tables
                combined = np.zeros_like(input_data[0])
                for table in input_data:
                    combined += table
                return combined
        return input_data
    
    def analyze(self, aggregated_data: np.ndarray) -> Dict[str, Any]:
        """Calculate chi-squared statistic manually."""
        # Store the contingency table
        self.aggregated_data = {"contingency_table": aggregated_data}
        
        # Manual calculation
        row_totals = np.sum(aggregated_data, axis=1)
        col_totals = np.sum(aggregated_data, axis=0)
        total = np.sum(row_totals)
        
        # Calculate expected frequencies
        expected = np.zeros_like(aggregated_data)
        for i in range(len(aggregated_data)):
            for j in range(len(aggregated_data[i])):
                expected[i][j] = row_totals[i] * col_totals[j] / total
        
        # Calculate chi-squared
        chi2 = np.sum((aggregated_data - expected) ** 2 / expected)
        dof = (len(row_totals) - 1) * (len(col_totals) - 1)
        p_value = 1 - stats.chi2.cdf(chi2, dof)
        
        return {
            'chi_squared': chi2,
            'p_value': p_value,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected
        }


class StatisticalAnalyzer:
    """
    Handles statistical calculations and analysis for federated data.
    Uses individual analysis classes that inherit from AnalysisBase.
    """
    
    def __init__(self):
        """Initialize the statistical analyzer with analysis classes."""
        self.analysis_classes = {
            "mean": MeanAnalysis(),
            "variance": VarianceAnalysis(),
            "PMCC": PMCCAnalysis(),
            "chi_squared_scipy": ChiSquaredScipyAnalysis(),
            "chi_squared_manual": ChiSquaredManualAnalysis()
        }
    
    def get_analysis_config(self, analysis_type: str) -> Dict[str, Any]:
        """
        Get configuration for a specific analysis type.
        
        Args:
            analysis_type (str): Type of analysis
            
        Returns:
            Dict[str, Any]: Analysis configuration
            
        Raises:
            ValueError: If analysis type is not supported
        """
        if analysis_type not in self.analysis_classes:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        analysis_class = self.analysis_classes[analysis_type]
        return {
            "return_format": analysis_class.return_format,
            "aggregation_function": analysis_class.aggregate_data,
            "analysis_function": analysis_class.analyze
        }
    
    def get_supported_analysis_types(self) -> List[str]:
        """
        Get list of supported analysis types.
        
        Returns:
            List[str]: List of supported analysis types
        """
        return list(self.analysis_classes.keys())
    
    def analyze_data(self, input_data: Union[np.ndarray, List[np.ndarray]], analysis_type: str) -> Union[float, Dict[str, Any]]:
        """
        Analyze data using the specified analysis type.
        
        Args:
            input_data: Input data (numpy array or list of arrays)
            analysis_type (str): Type of analysis to perform
            
        Returns:
            Union[float, Dict[str, Any]]: Analysis result
            
        Raises:
            ValueError: If analysis type is not supported
        """
        if analysis_type not in self.analysis_classes:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        analysis_class = self.analysis_classes[analysis_type]
        
        # Aggregate data
        aggregated_data = analysis_class.aggregate_data(input_data)
        
        # Perform analysis
        result = analysis_class.analyze(aggregated_data)
        return result
    
    def calculate_descriptive_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Calculate descriptive statistics for a dataset.
        
        Args:
            data (np.ndarray): Input data
            
        Returns:
            Dict[str, float]: Dictionary containing descriptive statistics
        """
        return {
            'mean': np.mean(data),
            'median': np.median(data),
            'std': np.std(data),
            'variance': np.var(data),
            'min': np.min(data),
            'max': np.max(data),
            'count': len(data)
        }
    
    def perform_hypothesis_test(self, 
                              data1: np.ndarray, 
                              data2: np.ndarray, 
                              test_type: str = "t_test") -> Dict[str, Any]:
        """
        Perform hypothesis testing between two datasets.
        
        Args:
            data1 (np.ndarray): First dataset
            data2 (np.ndarray): Second dataset
            test_type (str): Type of test ("t_test", "mann_whitney", etc.)
            
        Returns:
            Dict[str, Any]: Test results
        """
        if test_type == "t_test":
            statistic, p_value = stats.ttest_ind(data1, data2)
            return {
                'test_type': 'independent_t_test',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        elif test_type == "mann_whitney":
            statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            return {
                'test_type': 'mann_whitney_u',
                'statistic': statistic,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
    
    def calculate_confidence_interval(self, 
                                    data: np.ndarray, 
                                    confidence: float = 0.95) -> Dict[str, float]:
        """
        Calculate confidence interval for the mean.
        
        Args:
            data (np.ndarray): Input data
            confidence (float): Confidence level (default: 0.95)
            
        Returns:
            Dict[str, float]: Confidence interval bounds
        """
        mean = np.mean(data)
        std_err = stats.sem(data)
        ci = stats.t.interval(confidence, len(data) - 1, loc=mean, scale=std_err)
        
        return {
            'lower_bound': ci[0],
            'upper_bound': ci[1],
            'confidence_level': confidence,
            'mean': mean
        } 