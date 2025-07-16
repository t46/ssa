"""WVS Wave 7 USA Data Analysis Pipeline."""
import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class WVSAnalysis:
    """Analysis pipeline for WVS Wave 7 USA data."""
    
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
    
    def preprocess_data(self, variables: List[str]) -> pd.DataFrame:
        """Preprocess data for analysis."""
        # Select relevant variables including weight
        # Use W_WEIGHT as the weight variable
        all_vars = list(set(variables + ['W_WEIGHT']))  # Always include weight
        
        # Filter to only variables that exist in the data
        existing_vars = [var for var in all_vars if var in self.df.columns]
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
            # If no weight available, use equal weights
            df_subset['weight'] = 1.0
        
        return df_subset
    
    def recode_variables(self, df: pd.DataFrame, var_map: Dict[str, str]) -> pd.DataFrame:
        """Recode variables for analysis."""
        df_recoded = df.copy()
        
        # Example recoding for common variables
        # Political ideology (Q240): 1-10 scale, recode to 0-1
        if 'Q240' in df_recoded.columns:
            df_recoded['political_ideology'] = (df_recoded['Q240'] - 1) / 9
        
        # Trust variables (Q57-Q71): Usually 1-4 scale
        trust_vars = [f'Q{i}' for i in range(57, 72) if f'Q{i}' in df_recoded.columns]
        for var in trust_vars:
            # Recode to 0-1 scale (higher = more trust)
            df_recoded[f'{var}_trust'] = (df_recoded[var] - 1) / 3
        
        # Well-being variables (Q46-Q49)
        if 'Q46' in df_recoded.columns:  # Life satisfaction (1-10)
            df_recoded['life_satisfaction'] = (df_recoded['Q46'] - 1) / 9
        
        if 'Q49' in df_recoded.columns:  # Happiness (1-4)
            df_recoded['happiness'] = (df_recoded['Q49'] - 1) / 3
        
        return df_recoded
    
    def analyze_research_1(self) -> Dict[str, Any]:
        """Analyze Research 1: Political Polarization and Trust."""
        research = self.research_config['research_proposals'][0]
        results = {'research_id': 1, 'title': research['title']}
        
        # Get all variables
        all_vars = (research['variables']['dependent'] + 
                   research['variables']['independent'] + 
                   research['variables']['controls'])
        
        # Preprocess data
        df = self.preprocess_data(all_vars)
        df = self.recode_variables(df, {})
        
        # Create trust index (average of trust variables)
        trust_vars = [f'Q{i}_trust' for i in range(57, 72) if f'Q{i}_trust' in df.columns]
        df['trust_index'] = df[trust_vars].mean(axis=1)
        
        # Prepare for regression
        df_clean = df[['trust_index', 'political_ideology', 'weight']].dropna()
        
        # Weighted OLS regression
        X = sm.add_constant(df_clean['political_ideology'])
        model = sm.WLS(df_clean['trust_index'], X, weights=df_clean['weight'])
        results_ols = model.fit()
        
        results['model_summary'] = str(results_ols.summary())
        results['coefficients'] = results_ols.params.to_dict()
        results['p_values'] = results_ols.pvalues.to_dict()
        results['r_squared'] = results_ols.rsquared
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Bin political ideology for visualization
        df_clean['ideology_bin'] = pd.cut(df_clean['political_ideology'], 
                                         bins=5, labels=['Very Liberal', 'Liberal', 
                                                        'Moderate', 'Conservative', 
                                                        'Very Conservative'])
        
        # Calculate weighted means by ideology group
        trust_by_ideology = df_clean.groupby('ideology_bin').apply(
            lambda x: np.average(x['trust_index'], weights=x['weight'])
        )
        
        trust_by_ideology.plot(kind='bar', ax=ax)
        ax.set_xlabel('Political Ideology')
        ax.set_ylabel('Average Trust Index')
        ax.set_title('Trust Levels by Political Ideology')
        plt.tight_layout()
        plt.savefig(self.output_path / 'research1_trust_by_ideology.png')
        plt.close()
        
        results['visualization'] = 'research1_trust_by_ideology.png'
        
        return results
    
    def analyze_research_2(self) -> Dict[str, Any]:
        """Analyze Research 2: Economic Insecurity and Social Values."""
        research = self.research_config['research_proposals'][1]
        results = {'research_id': 2, 'title': research['title']}
        
        # Get all variables
        all_vars = (research['variables']['dependent'] + 
                   research['variables']['independent'] + 
                   research['variables']['controls'])
        
        # Preprocess data
        df = self.preprocess_data(all_vars)
        
        # Create economic insecurity index (Q50-Q55)
        econ_vars = [f'Q{i}' for i in range(50, 56) if f'Q{i}' in df.columns]
        df['econ_insecurity'] = df[econ_vars].mean(axis=1)
        
        # Create immigration attitude index (Q121-Q130)
        immig_vars = [f'Q{i}' for i in range(121, 131) if f'Q{i}' in df.columns]
        df['immigration_attitude'] = df[immig_vars].mean(axis=1)
        
        # Prepare for regression
        df_clean = df[['immigration_attitude', 'econ_insecurity', 'weight']].dropna()
        
        # Standardize variables
        scaler = StandardScaler()
        df_clean['econ_insecurity_std'] = scaler.fit_transform(df_clean[['econ_insecurity']])
        df_clean['immigration_attitude_std'] = scaler.fit_transform(df_clean[['immigration_attitude']])
        
        # Weighted OLS regression
        X = sm.add_constant(df_clean['econ_insecurity_std'])
        model = sm.WLS(df_clean['immigration_attitude_std'], X, weights=df_clean['weight'])
        results_ols = model.fit()
        
        results['model_summary'] = str(results_ols.summary())
        results['coefficients'] = results_ols.params.to_dict()
        results['p_values'] = results_ols.pvalues.to_dict()
        results['r_squared'] = results_ols.rsquared
        
        # Create visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create scatter plot with regression line
        ax.scatter(df_clean['econ_insecurity_std'], df_clean['immigration_attitude_std'], 
                  alpha=0.5, s=df_clean['weight']*10)
        
        # Add regression line
        x_pred = np.linspace(df_clean['econ_insecurity_std'].min(), 
                           df_clean['econ_insecurity_std'].max(), 100)
        y_pred = results_ols.predict(sm.add_constant(x_pred))
        ax.plot(x_pred, y_pred, 'r-', linewidth=2)
        
        ax.set_xlabel('Economic Insecurity (standardized)')
        ax.set_ylabel('Immigration Attitude (standardized)')
        ax.set_title('Economic Insecurity and Immigration Attitudes')
        plt.tight_layout()
        plt.savefig(self.output_path / 'research2_econ_immigration.png')
        plt.close()
        
        results['visualization'] = 'research2_econ_immigration.png'
        
        return results
    
    def analyze_research_3(self) -> Dict[str, Any]:
        """Analyze Research 3: Religion, Science, and Well-being."""
        research = self.research_config['research_proposals'][2]
        results = {'research_id': 3, 'title': research['title']}
        
        # Get all variables
        all_vars = (research['variables']['dependent'] + 
                   research['variables']['independent'] + 
                   research['variables'].get('mediators', []) +
                   research['variables']['controls'])
        
        # Preprocess data
        df = self.preprocess_data(all_vars)
        df = self.recode_variables(df, {})
        
        # Create religiosity index (Q164-Q173)
        relig_vars = [f'Q{i}' for i in range(164, 174) if f'Q{i}' in df.columns]
        if relig_vars:
            df['religiosity'] = df[relig_vars].mean(axis=1)
        
        # Use life satisfaction as main DV
        if 'life_satisfaction' in df.columns:
            df_clean = df[['life_satisfaction', 'religiosity', 'weight']].dropna()
            
            # Weighted OLS regression
            X = sm.add_constant(df_clean['religiosity'])
            model = sm.WLS(df_clean['life_satisfaction'], X, weights=df_clean['weight'])
            results_ols = model.fit()
            
            results['model_summary'] = str(results_ols.summary())
            results['coefficients'] = results_ols.params.to_dict()
            results['p_values'] = results_ols.pvalues.to_dict()
            results['r_squared'] = results_ols.rsquared
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Bin religiosity for visualization
            df_clean['religiosity_bin'] = pd.qcut(df_clean['religiosity'], 
                                                 q=4, labels=['Low', 'Medium-Low', 
                                                            'Medium-High', 'High'])
            
            # Calculate weighted means
            wellbeing_by_religion = df_clean.groupby('religiosity_bin').apply(
                lambda x: np.average(x['life_satisfaction'], weights=x['weight'])
            )
            
            wellbeing_by_religion.plot(kind='bar', ax=ax)
            ax.set_xlabel('Religiosity Level')
            ax.set_ylabel('Average Life Satisfaction')
            ax.set_title('Life Satisfaction by Religiosity Level')
            plt.tight_layout()
            plt.savefig(self.output_path / 'research3_religion_wellbeing.png')
            plt.close()
            
            results['visualization'] = 'research3_religion_wellbeing.png'
        
        return results
    
    def generate_report(self, results: List[Dict[str, Any]]):
        """Generate analysis report."""
        report_path = self.output_path / 'analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# WVS Wave 7 USA Data Analysis Results\n\n")
            f.write(f"Dataset: {self.research_config['metadata']['dataset']}\n")
            f.write(f"Country: {self.research_config['metadata']['country']}\n")
            f.write(f"Sample Weight: {self.research_config['metadata']['sample_weight']}\n\n")
            
            for result in results:
                title = result.get('title', 'Research Analysis')
                f.write(f"## Research {result['research_id']}: {title}\n\n")
                
                if 'error' in result:
                    f.write(f"### Error:\n{result['error']}\n\n")
                    continue
                
                if 'coefficients' in result:
                    f.write("### Key Findings:\n")
                    f.write(f"- R-squared: {result['r_squared']:.3f}\n")
                    for param, value in result['coefficients'].items():
                        p_val = result['p_values'][param]
                        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
                        f.write(f"- {param}: {value:.3f} (p={p_val:.3f}){sig}\n")
                
                f.write(f"\n### Visualization:\n")
                f.write(f"![{result['visualization']}]({result['visualization']})\n\n")
                
                f.write("### Model Summary:\n")
                f.write("```\n")
                f.write(result.get('model_summary', 'No model summary available'))
                f.write("\n```\n\n")
        
        print(f"Report generated: {report_path}")
    
    def run_all_analyses(self):
        """Run all research analyses."""
        results = []
        
        print("Running Research 1: Political Polarization and Trust...")
        try:
            results.append(self.analyze_research_1())
        except Exception as e:
            print(f"Error in Research 1: {e}")
            results.append({'research_id': 1, 'error': str(e)})
        
        print("Running Research 2: Economic Insecurity and Social Values...")
        try:
            results.append(self.analyze_research_2())
        except Exception as e:
            print(f"Error in Research 2: {e}")
            results.append({'research_id': 2, 'error': str(e)})
        
        print("Running Research 3: Religion, Science, and Well-being...")
        try:
            results.append(self.analyze_research_3())
        except Exception as e:
            print(f"Error in Research 3: {e}")
            results.append({'research_id': 3, 'error': str(e)})
        
        # Generate report
        self.generate_report(results)
        
        return results


if __name__ == "__main__":
    analysis = WVSAnalysis()
    results = analysis.run_all_analyses()
    print("\nAnalysis complete! Check the outputs/ directory for results.")