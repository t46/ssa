"""WVS Wave 7 USA Data Analysis Pipeline - Dynamic version."""
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import anthropic
import os
warnings.filterwarnings('ignore')


class DynamicWVSAnalysis:
    """Dynamic analysis pipeline for WVS Wave 7 USA data."""
    
    def __init__(self):
        """Initialize analysis with data and configurations."""
        self.data_path = Path("data/processed/usa_w7.csv")
        self.research_path = Path("spec/research.yaml")
        self.codebook_path = Path("data/code-maps/codebook_map.json")
        self.output_path = Path("outputs")
        self.output_path.mkdir(exist_ok=True)
        
        # Load data and configurations
        self.df = pd.read_csv(self.data_path)
        with open(self.research_path, 'r') as f:
            self.research_config = yaml.safe_load(f)
        with open(self.codebook_path, 'r') as f:
            self.codebook = json.load(f)
            
        # Initialize Anthropic client for dynamic analysis
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    def preprocess_data(self, variables: List[str]) -> pd.DataFrame:
        """Preprocess data for analysis."""
        # Always include weight variable
        all_vars = list(set(variables + ['W_WEIGHT']))
        
        # Filter to only variables that exist in the data
        existing_vars = [var for var in all_vars if var in self.df.columns]
        missing_vars = [var for var in all_vars if var not in self.df.columns]
        
        if missing_vars:
            print(f"Warning: Variables not found in data: {missing_vars}")
        
        df_subset = self.df[existing_vars].copy()
        
        # Handle missing values based on WVS conventions
        # -1 to -5 are typically missing value codes
        for col in df_subset.columns:
            if col != 'W_WEIGHT':  # Don't modify weights
                df_subset[col] = df_subset[col].replace([-5, -4, -3, -2, -1], np.nan)
        
        # Apply population weight
        if 'W_WEIGHT' in df_subset.columns:
            df_subset['weight'] = df_subset['W_WEIGHT']
        else:
            df_subset['weight'] = 1.0
        
        return df_subset
    
    def generate_analysis_code(self, research: Dict[str, Any]) -> str:
        """Generate analysis code dynamically based on research proposal."""
        # Create a prompt for the LLM to generate analysis code
        var_info = []
        for var_type, var_list in research['variables'].items():
            if var_list:
                var_info.append(f"{var_type}: {', '.join(var_list)}")
        
        prompt = f"""Generate Python code for analyzing the following research proposal using WVS data:

Title: {research['title']}
Objective: {research['objective']}
Hypotheses: {json.dumps(research['hypotheses'], indent=2)}
Variables: {chr(10).join(var_info)}
Analytical Approach: {research['analytical_approach']}

The code should:
1. Create appropriate indices/scales from the variables
2. Implement the specified analytical approach
3. Test each hypothesis
4. Generate visualizations
5. Return results as a dictionary with keys: 'statistics', 'visualizations', 'findings'

Available data columns: df_clean with all specified variables plus 'weight'
Use weighted statistics where appropriate (using the 'weight' column).

Return only the Python code that can be executed with exec(), no explanations.
The code should define a function called analyze_data(df_clean) that returns the results dictionary.
"""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=4000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract code from response
        code = response.content[0].text
        if "```python" in code:
            code = code.split("```python")[1].split("```")[0]
        elif "```" in code:
            code = code.split("```")[1].split("```")[0]
            
        return code
    
    def execute_dynamic_analysis(self, research_id: int) -> Dict[str, Any]:
        """Execute analysis for a specific research proposal."""
        research = self.research_config['research_proposals'][research_id]
        results = {
            'research_id': research_id + 1,
            'title': research['title'],
            'hypotheses': research['hypotheses']
        }
        
        try:
            # Get all variables
            all_vars = []
            for var_type, var_list in research['variables'].items():
                if isinstance(var_list, list):
                    all_vars.extend(var_list)
            
            # Preprocess data
            df = self.preprocess_data(all_vars)
            df_clean = df.dropna()
            
            if len(df_clean) < 100:
                raise ValueError(f"Insufficient data after preprocessing: {len(df_clean)} records")
            
            # Generate and execute analysis code
            analysis_code = self.generate_analysis_code(research)
            
            # Create a safe execution environment
            exec_globals = {
                'pd': pd,
                'np': np,
                'sm': sm,
                'smf': smf,
                'stats': stats,
                'plt': plt,
                'sns': sns,
                'StandardScaler': StandardScaler,
                'SimpleImputer': SimpleImputer,
                'df_clean': df_clean,
                'output_path': self.output_path,
                'research_id': research_id
            }
            
            # Execute the generated code
            exec(analysis_code, exec_globals)
            
            # Call the analyze_data function
            if 'analyze_data' in exec_globals:
                analysis_results = exec_globals['analyze_data'](df_clean)
                results.update(analysis_results)
            else:
                raise ValueError("Generated code did not define analyze_data function")
                
        except Exception as e:
            print(f"Error in dynamic analysis for research {research_id + 1}: {e}")
            results['error'] = str(e)
            results['statistics'] = {'error': 'Analysis failed'}
            results['visualizations'] = []
            results['findings'] = f"Analysis failed due to: {str(e)}"
        
        return results
    
    def generate_summary_statistics(self, df: pd.DataFrame, research: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for the research variables."""
        stats_dict = {}
        
        # Get all variables
        all_vars = []
        for var_type, var_list in research['variables'].items():
            if isinstance(var_list, list):
                all_vars.extend(var_list)
        
        # Calculate basic statistics
        for var in all_vars:
            if var in df.columns and var != 'W_WEIGHT':
                stats_dict[var] = {
                    'mean': np.average(df[var].dropna(), weights=df.loc[df[var].notna(), 'weight']),
                    'std': np.sqrt(np.average((df[var].dropna() - np.average(df[var].dropna(), weights=df.loc[df[var].notna(), 'weight']))**2, 
                                             weights=df.loc[df[var].notna(), 'weight'])),
                    'n': df[var].notna().sum(),
                    'missing': df[var].isna().sum()
                }
        
        return stats_dict
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """Generate analysis report."""
        report_path = self.output_path / 'dynamic_analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# WVS Wave 7 USA Data Analysis Results (Dynamic)\n\n")
            f.write(f"Dataset: {self.research_config['metadata']['dataset']}\n")
            f.write(f"Country: {self.research_config['metadata']['country']}\n")
            f.write(f"Sample Weight: {self.research_config['metadata']['sample_weight']}\n\n")
            
            for result in results:
                f.write(f"## Research {result['research_id']}: {result['title']}\n\n")
                
                if 'error' in result:
                    f.write(f"### Error:\n{result['error']}\n\n")
                    continue
                
                # Write hypotheses
                f.write("### Hypotheses:\n")
                for i, hyp in enumerate(result.get('hypotheses', []), 1):
                    f.write(f"{i}. {hyp}\n")
                f.write("\n")
                
                # Write statistics
                if 'statistics' in result:
                    f.write("### Key Statistics:\n")
                    stats = result['statistics']
                    if isinstance(stats, dict):
                        for key, value in stats.items():
                            if isinstance(value, (int, float)):
                                f.write(f"- {key}: {value:.3f}\n")
                            else:
                                f.write(f"- {key}: {value}\n")
                    f.write("\n")
                
                # Write findings
                if 'findings' in result:
                    f.write("### Findings:\n")
                    findings = result['findings']
                    if isinstance(findings, dict):
                        for key, value in findings.items():
                            f.write(f"**{key}:** {value}\n\n")
                    else:
                        f.write(str(findings))
                    f.write("\n\n")
                
                # Reference visualizations
                if 'visualizations' in result and result['visualizations']:
                    f.write("### Visualizations:\n")
                    for viz in result['visualizations']:
                        f.write(f"![{viz}]({viz})\n")
                    f.write("\n")
        
        print(f"Report generated: {report_path}")
    
    def run_all_analyses(self):
        """Run all research analyses dynamically."""
        results = []
        
        for i, research in enumerate(self.research_config['research_proposals']):
            print(f"\nRunning Research {i+1}: {research['title']}...")
            try:
                result = self.execute_dynamic_analysis(i)
                results.append(result)
            except Exception as e:
                print(f"Error in Research {i+1}: {e}")
                results.append({
                    'research_id': i+1,
                    'title': research['title'],
                    'error': str(e)
                })
        
        # Generate report
        self.generate_report(results)
        
        return results


if __name__ == "__main__":
    analysis = DynamicWVSAnalysis()
    results = analysis.run_all_analyses()
    print("\nDynamic analysis complete! Check the outputs/ directory for results.")