#!/usr/bin/env python3
"""
Demo script showing the new query separation functionality.
Users only need to provide data selection queries, and the system automatically
handles the statistical analysis calculations.
"""

from query_builder import QueryBuilder
from analysis_engine import AnalysisEngine


def demo_query_builder():
    """Demonstrate the new QueryBuilder functionality."""
    print("=== QueryBuilder Demo ===\n")
    
    builder = QueryBuilder()
    
    # Example 1: Mean analysis
    print("1. Mean Analysis")
    print("-" * 50)
    
    user_query = """SELECT value_as_number FROM public.measurement 
WHERE measurement_concept_id = 3037532
AND value_as_number IS NOT NULL"""
    
    complete_query = builder.build_query("mean", user_query, column="value_as_number")
    print("User Query:")
    print(user_query)
    print("\nComplete Query:")
    print(complete_query)
    print("\n" + "="*60 + "\n")
    
    # Example 2: Variance analysis
    print("2. Variance Analysis")
    print("-" * 50)
    
    complete_query = builder.build_query("variance", user_query, column="value_as_number")
    print("User Query:")
    print(user_query)
    print("\nComplete Query:")
    print(complete_query)
    print("\n" + "="*60 + "\n")
    
    # Example 3: PMCC analysis
    print("3. PMCC Analysis")
    print("-" * 50)
    
    user_query_pmcc = """WITH x_values AS (
  SELECT person_id, measurement_date, value_as_number AS x
  FROM public.measurement
  WHERE measurement_concept_id = 3037532
    AND value_as_number IS NOT NULL
),
y_values AS (
  SELECT person_id, measurement_date, value_as_number AS y
  FROM public.measurement
  WHERE measurement_concept_id = 3037533
    AND value_as_number IS NOT NULL
)
SELECT
  x.x,
  y.y
FROM x_values x
INNER JOIN y_values y
  ON x.person_id = y.person_id
  AND x.measurement_date = y.measurement_date"""
    
    complete_query = builder.build_query("PMCC", user_query_pmcc, x_column="x", y_column="y")
    print("User Query:")
    print(user_query_pmcc)
    print("\nComplete Query:")
    print(complete_query)
    print("\n" + "="*60 + "\n")
    
    # Example 4: Chi-squared analysis
    print("4. Chi-squared Analysis")
    print("-" * 50)
    
    user_query_chi = """SELECT 
  g.concept_name AS gender_name,
  r.concept_name AS race_name
FROM person p
JOIN concept g ON p.gender_concept_id = g.concept_id
JOIN concept r ON p.race_concept_id = r.concept_id
WHERE p.race_concept_id IN (38003574, 38003584)"""
    
    complete_query = builder.build_query("chi_squared_scipy", user_query_chi, group_columns="gender_name, race_name")
    print("User Query:")
    print(user_query_chi)
    print("\nComplete Query:")
    print(complete_query)
    print("\n" + "="*60 + "\n")
    
    # Show analysis requirements
    print("5. Analysis Requirements")
    print("-" * 50)
    
    for analysis_type in ["mean", "variance", "PMCC", "chi_squared_scipy"]:
        requirements = builder.get_analysis_requirements(analysis_type)
        print(f"\n{analysis_type.upper()} Analysis:")
        print(f"  Description: {requirements['description']}")
        print(f"  User Query Requirements: {requirements['user_query_requirements']}")
        print(f"  Required Parameters: {requirements['required_parameters']}")
        print(f"  Expected Columns: {requirements['expected_columns']}")


def demo_analysis_engine():
    """Demonstrate the new AnalysisEngine functionality."""
    print("\n\n=== AnalysisEngine Demo ===\n")
    
    # Note: This demo doesn't actually run analysis (no real token)
    # It just shows the interface
    
    print("AnalysisEngine now accepts user_query instead of complete query:")
    print("-" * 60)
    
    print("""
# Old way (user had to write complete query):
engine.run_analysis(
    analysis_type="mean",
    query="WITH user_query AS (SELECT value_as_number FROM measurement WHERE concept_id = 123) SELECT COUNT(value_as_number) AS n, SUM(value_as_number) AS total FROM user_query;",
    tres=["TRE1", "TRE2"]
)

# New way (user only provides data selection):
engine.run_analysis(
    analysis_type="mean",
    user_query="SELECT value_as_number FROM measurement WHERE concept_id = 123",
    tres=["TRE1", "TRE2"],
    column="value_as_number"  # System knows what to calculate
)
""")
    
    print("Benefits:")
    print("1. Users focus only on data selection")
    print("2. System automatically handles statistical calculations")
    print("3. Reduced chance of errors in analysis formulas")
    print("4. Consistent analysis implementation across all TREs")
    print("5. Easy to add new analysis types")


def demo_example_functions():
    """Demonstrate the updated example functions."""
    print("\n\n=== Example Functions Demo ===\n")
    
    print("Updated example functions now use the new interface:")
    print("-" * 60)
    
    print("""
# Mean analysis example:
def run_mean_analysis_example(engine, concept_id, tres):
    user_query = f\"\"\"SELECT value_as_number FROM public.measurement 
WHERE measurement_concept_id = {concept_id}
AND value_as_number IS NOT NULL\"\"\"
    
    return engine.run_analysis("mean", user_query, tres, column="value_as_number")

# PMCC analysis example:
def run_pmcc_analysis_example(engine, x_concept_id, y_concept_id, tres):
    user_query = f\"\"\"WITH x_values AS (
  SELECT person_id, measurement_date, value_as_number AS x
  FROM public.measurement
  WHERE measurement_concept_id = {x_concept_id}
    AND value_as_number IS NOT NULL
),
y_values AS (
  SELECT person_id, measurement_date, value_as_number AS y
  FROM public.measurement
  WHERE measurement_concept_id = {y_concept_id}
    AND value_as_number IS NOT NULL
)
SELECT
  x.x,
  y.y
FROM x_values x
INNER JOIN y_values y
  ON x.person_id = y.person_id
  AND x.measurement_date = y.measurement_date\"\"\"
    
    return engine.run_analysis("PMCC", user_query, tres, x_column="x", y_column="y")
""")


def main():
    """Run all demos."""
    demo_query_builder()
    demo_analysis_engine()
    demo_example_functions()
    
    print("\n\n=== Summary ===")
    print("The new query separation functionality provides:")
    print("✅ Clean separation between data selection and analysis")
    print("✅ Reduced complexity for users")
    print("✅ Consistent statistical calculations")
    print("✅ Easy extensibility for new analysis types")
    print("✅ Better error handling and validation")


if __name__ == "__main__":
    main() 