# imports
import pandas as pd
import numpy as np
import sqlite3
import re
import warnings
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import joblib
import yagmail
from fpdf import FPDF
import io
import base64
import tempfile
import os
import traceback
import streamlit as st
from streamlit_option_menu import option_menu

# Reporting imports
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics import renderPDF

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import xgboost as xgb

# Email imports
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Visualisation imports
from PIL import Image
import plotly.io as pio
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ======================================================================
# 1. ENHANCED ETL PIPELINE WITH ML PREDICTION CAPABILITIES
# ======================================================================

class EnhancedHealthDataCleaner:
    """Enhanced ETL Pipeline with ML capabilities and prediction features"""
    
    def __init__(self):
        self.df = None
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.training_features = {}  # Store features used for each model
        self.feature_medians = {}    # Store median values for features
        
    def clean_dataset(self, csv_path, output_path='cleaned_global_health_data.csv'):
        """Main cleaning pipeline"""
        print("=" * 80)
        print("ENHANCED GLOBAL HEALTH DATASET CLEANING PIPELINE")
        print("=" * 80)
        
        # Load data
        self._load_data(csv_path)
        
        # Transform data
        self._clean_country_names()
        self._clean_disease_names()
        self._clean_years()
        self._clean_numeric_columns()
        self._clean_categorical_columns()
        self._handle_missing_values()
        self._create_derived_features()
        self._remove_outliers()
        self._standardize_columns()
        self._final_cleaning()
        
        # Save cleaned data
        self._save_data(output_path)
        
        # Generate ML features
        self._prepare_ml_features()
        
        return self.df
    
    def _load_data(self, csv_path):
        """Load data with multiple encoding attempts"""
        print("\n1. LOADING DATA...")
        encodings = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
        for enc in encodings:
            try:
                self.df = pd.read_csv(
                    csv_path,
                    encoding=enc,
                    low_memory=False,
                    na_values=['', 'NaN', 'NA', 'NULL', 'None', 'nan', 'N/A', 'n/a', '~none~', '?', '-'],
                    keep_default_na=True
                )
                print(f"✓ Successfully read with encoding: {enc}")
                print(f"  Shape: {self.df.shape}")
                break
            except UnicodeDecodeError:
                continue
        
        if self.df is None:
            raise ValueError("Could not read CSV with any encoding")
    
    def load_dataframe(self, df):
        """Load an existing DataFrame"""
        self.df = df.copy()
        self._prepare_ml_features()
        return self.df
    
    def _clean_country_names(self):
        """Clean and standardize country names"""
        print("\n2. CLEANING COUNTRY NAMES...")
        
        corrections = {
            'It@lĄ': 'Italy',
            'T?u?r?k?e?y?': 'Turkey',
            'G%rmany': 'Germany',
            'Can@da': 'Canada',
            'Mex!co': 'Mexico',
            '?r?zil': 'Brazil',
            'Ind!a': 'India',
            'Ch!na': 'China',
            'J@p@n': 'Japan',
            'Kor%a': 'South Korea',
            'Russi@': 'Russia',
            'Astr@lia': 'Australia',
            'Fr@nce': 'France',
            'Sp@in': 'Spain',
            'UK': 'United Kingdom',
            'USA': 'United States'
        }
        
        def clean_name(name):
            if pd.isna(name):
                return "Unknown"
            name = str(name).strip()
            if name in corrections:
                return corrections[name]
            # Remove special characters but keep letters, spaces, hyphens, apostrophes
            cleaned = re.sub(r'[^a-zA-Z\s\-\.\']', '', name)
            cleaned = ' '.join([word.capitalize() for word in cleaned.split()])
            return cleaned if cleaned else "Unknown"
        
        if 'Country' in self.df.columns:
            self.df['Country'] = self.df['Country'].apply(clean_name)
            print(f"✓ Unique countries: {self.df['Country'].nunique()}")
    
    def _clean_disease_names(self):
        """Clean disease names"""
        print("\n3. CLEANING DISEASE NAMES...")
        
        disease_corrections = {
            'A!DS': 'AIDS',
            'Influen&za': 'Influenza',
            'Pol!o': 'Polio',
            'COVID-19': 'COVID-19',
            'COVID': 'COVID-19',
            'Mal@ria': 'Malaria',
            'Tubercul!sis': 'Tuberculosis',
            'Di@betes': 'Diabetes',
            'Cancer': 'Cancer',
            'Heart Dise@se': 'Heart Disease',
            'Stroke': 'Stroke'
        }
        
        def clean_disease(name):
            if pd.isna(name):
                return "Unknown"
            name = str(name).strip()
            if name in disease_corrections:
                return disease_corrections[name]
            # Remove special characters
            cleaned = re.sub(r'[^\w\s\-\(\)\']', '', name)
            cleaned = ' '.join(cleaned.split())
            # Preserve known acronyms
            if not any(x in cleaned.upper() for x in ['COVID', 'HIV', 'AIDS', 'SARS', 'MERS']):
                cleaned = cleaned.title()
            return cleaned
        
        if 'Disease Name' in self.df.columns:
            self.df['Disease Name'] = self.df['Disease Name'].apply(clean_disease)
            print(f"✓ Unique diseases: {self.df['Disease Name'].nunique()}")
    
    def _clean_years(self):
        """Clean year column"""
        print("\n4. CLEANING YEAR DATA...")
        if 'Year' in self.df.columns:
            # Convert to numeric, handle errors
            self.df['Year'] = pd.to_numeric(self.df['Year'], errors='coerce')
            # Fill NaN with median year
            year_median = self.df['Year'].median()
            if pd.isna(year_median):
                year_median = 2000  # Default if no valid years
            self.df['Year'] = self.df['Year'].fillna(year_median)
            self.df['Year'] = self.df['Year'].astype(int)
            
            # Filter to reasonable years
            self.df = self.df[self.df['Year'].between(1900, 2100)]
            
            print(f"✓ Years range: {self.df['Year'].min()} to {self.df['Year'].max()}")
    
    def _clean_numeric_columns(self):
        """Clean all numeric columns"""
        print("\n5. CLEANING NUMERIC COLUMNS...")
        
        numeric_columns = [
            'Country_pop', 'Incidence Rate mn (%)', 'Prevalence rate (%)',
            'Mortality Rate per 100 people (%)', 'Population affected',
            'Pop_affected(Male)', 'Pop_affected(Female)', 'Ages 0-18 (%)',
            'Ages 19-35 (%)', 'Ages 36-60 (%)', 'Ages 61+ (%)',
            'Pop_affected_U (%)', 'Pop_affected_R (%)', 'Healthcare Access (%)',
            'Doctors per 1000', 'Hospital Beds per 1000', 'Recovery Rate (%)',
            'DALYs', 'Improvement in 5 Years (%)', 'Average Annual Treatment Cost (USD)',
            'Composite Health Index (CHI)', 'Per Capita Income (USD)',
            'Education Index', 'Urbanization Rate (%)'
        ]
        
        def clean_numeric(val):
            if pd.isna(val):
                return np.nan
            val_str = str(val)
            # Remove quotes, commas, special characters
            val_str = val_str.replace("'", "").replace(",", ".")
            val_str = re.sub(r'[^\d\.\-]', '', val_str)
            if val_str == '' or val_str == '.':
                return np.nan
            try:
                return float(val_str)
            except:
                return np.nan
        
        for col in numeric_columns:
            if col in self.df.columns:
                # Store original length
                original_len = len(self.df)
                self.df[col] = self.df[col].apply(clean_numeric)
                # Count cleaned values
                cleaned_count = self.df[col].notna().sum()
                print(f"  {col}: {cleaned_count}/{original_len} values cleaned")
    
    def _clean_categorical_columns(self):
        """Clean categorical columns"""
        print("\n6. CLEANING CATEGORICAL COLUMNS...")
        
        if 'Treatment type' in self.df.columns:
            self.df['Treatment type'] = self.df['Treatment type'].fillna('Unknown')
            self.df['Treatment type'] = self.df['Treatment type'].str.capitalize()
        
        if 'Availability of Vaccines/Treatment' in self.df.columns:
            availability_map = {
                'High': 'High', 'high': 'High', 'High ': 'High',
                'Medium': 'Medium', 'medium': 'Medium',
                'Low': 'Low', 'low': 'Low',
                'None': 'None', 'none': 'None', '~none~': 'None'
            }
            self.df['Availability of Vaccines/Treatment'] = (
                self.df['Availability of Vaccines/Treatment']
                .fillna('Unknown')
                .apply(lambda x: availability_map.get(str(x).strip(), 'Medium'))
            )
    
    def _handle_missing_values(self):
        """Handle missing values intelligently"""
        print("\n7. HANDLING MISSING VALUES...")
        
        # Fill country population with country-year averages
        if 'Country_pop' in self.df.columns:
            # First try grouping by country and year
            try:
                country_year_avg = self.df.groupby(['Country', 'Year'])['Country_pop'].transform('median')
                self.df['Country_pop'] = self.df['Country_pop'].fillna(country_year_avg)
            except:
                pass
            
            # Then fill with overall median
            overall_median = self.df['Country_pop'].median()
            if not pd.isna(overall_median):
                self.df['Country_pop'] = self.df['Country_pop'].fillna(overall_median)
        
        # Fill age groups
        age_cols = ['Ages 0-18 (%)', 'Ages 19-35 (%)', 'Ages 36-60 (%)', 'Ages 61+ (%)']
        for col in age_cols:
            if col in self.df.columns:
                self.df[col] = self.df[col].fillna(25)
        
        # Fill rates with disease-country averages
        rate_cols = ['Incidence Rate mn (%)', 'Prevalence rate (%)', 'Mortality Rate per 100 people (%)']
        for col in rate_cols:
            if col in self.df.columns:
                try:
                    disease_country_avg = self.df.groupby(['Disease Name', 'Country'])[col].transform('median')
                    self.df[col] = self.df[col].fillna(disease_country_avg)
                except:
                    pass
                
                # Fill remaining with column median
                col_median = self.df[col].median()
                if not pd.isna(col_median):
                    self.df[col] = self.df[col].fillna(col_median)
    
    def _create_derived_features(self):
        """Create derived features for analysis"""
        print("\n8. CREATING DERIVED FEATURES...")
        
        # Population coverage
        if all(col in self.df.columns for col in ['Population affected', 'Country_pop']):
            # Avoid division by zero
            self.df['Country_pop'] = self.df['Country_pop'].replace(0, 1)
            self.df['Population Coverage (%)'] = (
                self.df['Population affected'] / self.df['Country_pop'] * 100
            ).round(2).clip(upper=100)
            print(f"  Created Population Coverage (%)")
        
        # Gender ratio
        if all(col in self.df.columns for col in ['Pop_affected(Male)', 'Pop_affected(Female)']):
            # Avoid division by zero
            self.df['Pop_affected(Female)'] = self.df['Pop_affected(Female)'].replace(0, 1)
            self.df['Gender Ratio (M:F)'] = (
                self.df['Pop_affected(Male)'] / self.df['Pop_affected(Female)']
            ).round(2).clip(lower=0.1, upper=10)
            print(f"  Created Gender Ratio (M:F)")
        
        # Urban/Rural ratio
        if all(col in self.df.columns for col in ['Pop_affected_U (%)', 'Pop_affected_R (%)']):
            # Avoid division by zero
            self.df['Pop_affected_R (%)'] = self.df['Pop_affected_R (%)'].replace(0, 1)
            self.df['Urban_Rural_Ratio'] = (
                self.df['Pop_affected_U (%)'] / self.df['Pop_affected_R (%)']
            ).round(2).clip(lower=0.1, upper=10)
            print(f"  Created Urban_Rural_Ratio")
        
        # Severity score
        if all(col in self.df.columns for col in ['Mortality Rate per 100 people (%)', 'DALYs']):
            # Handle NaN values
            mortality_filled = self.df['Mortality Rate per 100 people (%)'].fillna(0)
            dalys_filled = self.df['DALYs'].fillna(0)
            self.df['Severity Score'] = (
                mortality_filled * 0.7 +
                np.log1p(dalys_filled) * 0.3
            ).round(2)
            print(f"  Created Severity Score")
        
        # Risk score - check if columns exist
        if 'Incidence Rate mn (%)' in self.df.columns and 'Severity Score' in self.df.columns:
            incidence_filled = self.df['Incidence Rate mn (%)'].fillna(0)
            severity_filled = self.df['Severity Score'].fillna(0)
            self.df['Risk Score'] = (
                incidence_filled * 0.4 +
                severity_filled * 0.6
            ).round(2)
            print(f"  Created Risk Score")
    
    def _remove_outliers(self):
        """Remove outliers using IQR method"""
        print("\n9. REMOVING OUTLIERS...")
        
        outlier_cols = [
            'Average Annual Treatment Cost (USD)',
            'Per Capita Income (USD)',
            'DALYs',
            'Country_pop',
            'Mortality Rate per 100 people (%)'
        ]
        
        for col in outlier_cols:
            if col in self.df.columns:
                try:
                    # Calculate IQR
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Define bounds
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Clip values instead of filtering
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    
                    # Count outliers
                    outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                    if outliers > 0:
                        print(f"  {col}: Capped {outliers} outliers")
                        
                except Exception as e:
                    print(f"  Warning: Could not remove outliers for {col}: {str(e)}")
    
    def _standardize_columns(self):
        """Standardize column names"""
        print("\n10. STANDARDIZING COLUMN NAMES...")
        
        column_rename = {
            'Country_pop': 'Country_Population',
            'Incidence Rate mn (%)': 'Incidence_Rate',
            'Prevalence rate (%)': 'Prevalence_Rate',
            'Mortality Rate per 100 people (%)': 'Mortality_Rate',
            'Population affected': 'Population_Affected',
            'Pop_affected(Male)': 'Affected_Male',
            'Pop_affected(Female)': 'Affected_Female',
            'Ages 0-18 (%)': 'Age_0_18_Pct',
            'Ages 19-35 (%)': 'Age_19_35_Pct',
            'Ages 36-60 (%)': 'Age_36_60_Pct',
            'Ages 61+ (%)': 'Age_61_Plus_Pct',
            'Pop_affected_U (%)': 'Urban_Population_Pct',
            'Pop_affected_R (%)': 'Rural_Population_Pct',
            'Healthcare Access (%)': 'Healthcare_Access',
            'Doctors per 1000': 'Doctors_per_1000',
            'Hospital Beds per 1000': 'Hospital_Beds_per_1000',
            'Treatment type': 'Treatment_Type',
            'Recovery Rate (%)': 'Recovery_Rate',
            'DALYs': 'DALYs',
            'Improvement in 5 Years (%)': 'Improvement_5_Years',
            'Average Annual Treatment Cost (USD)': 'Avg_Treatment_Cost_USD',
            'Availability of Vaccines/Treatment': 'Vaccine_Treatment_Availability',
            'Composite Health Index (CHI)': 'Health_Index',
            'Per Capita Income (USD)': 'Per_Capita_Income_USD',
            'Education Index': 'Education_Index',
            'Urbanization Rate (%)': 'Urbanization_Rate'
        }
        
        self.df = self.df.rename(columns={k: v for k, v in column_rename.items() if k in self.df.columns})
        print(f"✓ Renamed columns")
    
    def _final_cleaning(self):
        """Final cleaning steps"""
        print("\n11. FINAL CLEANING...")
        
        # Drop duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial_rows - len(self.df)
        if removed > 0:
            print(f"✓ Removed {removed:,} duplicates")
        
        # Sort and reset index
        if 'Country' in self.df.columns and 'Year' in self.df.columns and 'Disease Name' in self.df.columns:
            self.df = self.df.sort_values(['Country', 'Year', 'Disease Name'])
        self.df = self.df.reset_index(drop=True)
        self.df['Record_ID'] = range(1, len(self.df) + 1)
        
        # Fill remaining NaN values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_median = self.df[col].median()
            if not pd.isna(col_median):
                self.df[col] = self.df[col].fillna(col_median)
                # Store median for ML predictions
                self.feature_medians[col] = col_median
        
        object_cols = self.df.select_dtypes(include=['object']).columns
        for col in object_cols:
            self.df[col] = self.df[col].fillna('Unknown')
        
        print(f"✓ Final shape: {self.df.shape}")
    
    def _save_data(self, output_path):
        """Save cleaned data"""
        print("\n12. SAVING DATA...")
        self.df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"✓ Saved to: {output_path}")
    
    def _prepare_ml_features(self):
        """Prepare features for machine learning"""
        print("\n13. PREPARING ML FEATURES...")
        
        # Only proceed if we have data
        if self.df is None or len(self.df) == 0:
            print("✗ No data available for ML preparation")
            return
        
        # Store median values for all numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Record_ID']:
                median_val = self.df[col].median()
                if not pd.isna(median_val):
                    self.feature_medians[col] = median_val
        
        # Encode categorical variables
        categorical_cols = ['Country', 'Disease Name', 'Treatment_Type', 'Vaccine_Treatment_Availability']
        for col in categorical_cols:
            if col in self.df.columns:
                try:
                    # Get unique values
                    unique_vals = self.df[col].astype(str).unique()
                    if len(unique_vals) > 1:  # Only encode if we have variation
                        le = LabelEncoder()
                        self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                        self.label_encoders[col] = le
                        print(f"  Encoded {col}: {len(unique_vals)} unique values")
                except Exception as e:
                    print(f"  Warning: Could not encode {col}: {str(e)}")
        
        # Create ML-ready dataframe with all potential features
        ml_features = [
            'Year', 'Country_Population', 'Incidence_Rate', 'Prevalence_Rate',
            'Mortality_Rate', 'Healthcare_Access', 'Doctors_per_1000',
            'Hospital_Beds_per_1000', 'Recovery_Rate', 'DALYs',
            'Avg_Treatment_Cost_USD', 'Per_Capita_Income_USD',
            'Education_Index', 'Urbanization_Rate', 'Risk Score',
            'Country_encoded', 'Disease Name_encoded', 'Treatment_Type_encoded'
        ]
        
        available_features = [col for col in ml_features if col in self.df.columns]
        if available_features:
            # Create copy to avoid modifying original
            self.ml_df = self.df[available_features].copy()
            
            # Fill any remaining NaN with medians
            for col in available_features:
                if col in self.feature_medians:
                    self.ml_df[col] = self.ml_df[col].fillna(self.feature_medians[col])
            
            # Remove any rows that still have NaN
            self.ml_df = self.ml_df.dropna()
            
            print(f"✓ ML features prepared: {len(available_features)} features, {len(self.ml_df)} samples")
        else:
            print("✗ No ML features found in data")
            self.ml_df = pd.DataFrame()
    
    def train_prediction_models(self):
        """Train various ML models for prediction"""
        print("\n14. TRAINING ML MODELS...")
        
        if self.ml_df is None or len(self.ml_df) == 0:
            print("✗ No ML features available for training")
            return
        
        # Define target variables that exist in both ml_df and df
        possible_targets = {
            'Mortality_Rate': 'Mortality_Rate',
            'Recovery_Rate': 'Recovery_Rate', 
            'Avg_Treatment_Cost_USD': 'Avg_Treatment_Cost_USD',
            'Risk Score': 'Risk Score'
        }
        
        # Filter to targets that actually exist in our data
        targets = {}
        for model_name, target_col in possible_targets.items():
            if target_col in self.df.columns and target_col in self.ml_df.columns:
                targets[model_name] = target_col
        
        if not targets:
            print("✗ No suitable target variables found for training")
            print(f"  Available columns in df: {list(self.df.columns)[:20]}...")
            print(f"  Available columns in ml_df: {list(self.ml_df.columns)}")
            return
        
        print(f"  Found {len(targets)} target variables: {list(targets.keys())}")
        
        # Train models for each target
        trained_models = 0
        for model_name, target in targets.items():
            print(f"  Training model for: {model_name}")
            
            try:
                # Prepare data - ensure target exists in ml_df
                if target not in self.ml_df.columns:
                    print(f"    ✗ Target {target} not in ml_df columns")
                    continue
                
                X = self.ml_df.drop(columns=[target], errors='ignore')
                y = self.ml_df[target]
                
                # Remove rows with NaN in target or features
                mask = y.notna() & X.notna().all(axis=1)
                
                # Apply the mask
                mask_array = mask.values if hasattr(mask, 'values') else mask
                X = X[mask_array]
                y = y[mask_array]
                
                if len(X) > 10:  # Need enough data
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Train Random Forest
                    rf_model = RandomForestRegressor(
                        n_estimators=100, 
                        random_state=42,
                        n_jobs=-1  # Use all cores
                    )
                    rf_model.fit(X_train, y_train)
                    
                    # Train XGBoost
                    xgb_model = xgb.XGBRegressor(
                        n_estimators=100, 
                        random_state=42,
                        n_jobs=-1  # Use all cores
                    )
                    xgb_model.fit(X_train, y_train)
                    
                    # Evaluate
                    rf_score = rf_model.score(X_test, y_test)
                    xgb_score = xgb_model.score(X_test, y_test)
                    
                    # Store best model and its features
                    if rf_score > xgb_score:
                        best_model = rf_model
                        print(f"    ✓ RandomForest trained. R² score: {rf_score:.3f}")
                    else:
                        best_model = xgb_model
                        print(f"    ✓ XGBoost trained. R² score: {xgb_score:.3f}")
                    
                    self.models[model_name] = best_model
                    self.training_features[model_name] = X.columns.tolist()
                    trained_models += 1
                    
                    # Print feature importance
                    if hasattr(best_model, 'feature_importances_'):
                        feature_importance = pd.DataFrame({
                            'feature': X.columns,
                            'importance': best_model.feature_importances_
                        }).sort_values('importance', ascending=False).head(5)
                        print(f"    Top 5 features: {feature_importance['feature'].tolist()}")
                else:
                    print(f"    ✗ Not enough data for {model_name} (only {len(X)} samples)")
            except Exception as e:
                print(f"    ✗ Error training {model_name}: {str(e)}")
                traceback.print_exc()
        
        if trained_models > 0:
            print(f"✓ {trained_models} models trained successfully")
            # Print available models
            print(f"  Available models: {list(self.models.keys())}")
        else:
            print("✗ No models were trained")
    
    def get_default_values_for_prediction(self):
        """Get default values for prediction from the dataset"""
        defaults = {}
        
        if self.df is not None:
            # Get median values for numeric columns
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col not in ['Record_ID', 'index']:
                    median_val = self.df[col].median()
                    if not pd.isna(median_val):
                        defaults[col] = float(median_val)
            
            # Get mode values for categorical columns
            categorical_cols = self.df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if col in ['Country', 'Disease Name', 'Treatment_Type']:
                    mode_val = self.df[col].mode()
                    if not mode_val.empty:
                        defaults[col] = str(mode_val.iloc[0])
                    else:
                        defaults[col] = 'Unknown'
        
        return defaults
    
    def predict(self, input_data):
        """Make predictions using trained models with proper feature matching"""
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                # Prepare input data
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical variables if needed
                for col in ['Country', 'Disease Name', 'Treatment_Type']:
                    if col in input_data and col in self.label_encoders:
                        try:
                            col_value = str(input_data[col])
                            input_df[f'{col}_encoded'] = self.label_encoders[col].transform([col_value])[0]
                        except:
                            # If encoding fails, use 0 (Unknown)
                            input_df[f'{col}_encoded'] = 0
                
                # Get features used during training
                if model_name in self.training_features:
                    required_features = self.training_features[model_name]
                else:
                    # Try to get features from model
                    if hasattr(model, 'feature_names_in_'):
                        required_features = model.feature_names_in_
                    else:
                        required_features = []
                
                if not required_features:
                    predictions[model_name] = "Error: No feature information available"
                    continue
                
                # Ensure all required features are present
                for feature in required_features:
                    if feature not in input_df.columns:
                        # Try to get from input_data or use median
                        if feature in input_data:
                            input_df[feature] = input_data[feature]
                        elif feature in self.feature_medians:
                            input_df[feature] = self.feature_medians[feature]
                        else:
                            input_df[feature] = 0
                
                # Reorder columns to match training order
                input_df = input_df[required_features]
                
                # Make prediction
                pred = model.predict(input_df)[0]
                predictions[model_name] = float(pred)
                
            except Exception as e:
                predictions[model_name] = f"Error: {str(e)}"
                traceback.print_exc()
        
        return predictions
    
    def cluster_countries(self, n_clusters=5):
        """Cluster countries based on health metrics"""
        print("\n15. CLUSTERING COUNTRIES...")
        
        if self.df is None or 'Country' not in self.df.columns:
            return None, None
        
        # Aggregate country data
        country_features = [
            'Healthcare_Access', 'Doctors_per_1000', 'Hospital_Beds_per_1000',
            'Mortality_Rate', 'Recovery_Rate', 'Per_Capita_Income_USD',
            'Education_Index', 'Urbanization_Rate'
        ]
        
        available_features = [col for col in country_features if col in self.df.columns]
        
        if len(available_features) >= 3:  # Need at least 3 features for meaningful clustering
            country_agg = self.df.groupby('Country')[available_features].mean()
            
            # Remove rows with NaN
            country_agg = country_agg.dropna()
            
            if len(country_agg) >= n_clusters:
                # Scale features
                scaler = StandardScaler()
                scaled_features = scaler.fit_transform(country_agg)
                
                # Apply KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                country_agg['Cluster'] = kmeans.fit_predict(scaled_features)
                
                # Analyze clusters
                cluster_summary = country_agg.groupby('Cluster').mean()
                
                return country_agg, cluster_summary
        
        return None, None
    
    def generate_insights(self):
        """Generate key insights from the data"""
        print("\n16. GENERATING INSIGHTS...")
        
        if self.df is None or len(self.df) == 0:
            print("✗ No data available for insights")
            return {}
        
        insights = {
            'top_diseases': {},
            'top_countries': {},
            'year_range': {},
            'key_metrics': {},
            'available_features': list(self.df.columns)
        }
        
        # Basic counts
        if 'Disease Name' in self.df.columns:
            disease_counts = self.df['Disease Name'].value_counts().head(10)
            insights['top_diseases'] = disease_counts.to_dict()
            print(f"  Top disease: {disease_counts.index[0] if len(disease_counts) > 0 else 'N/A'}")
        
        if 'Country' in self.df.columns:
            country_counts = self.df['Country'].value_counts().head(10)
            insights['top_countries'] = country_counts.to_dict()
            print(f"  Top country: {country_counts.index[0] if len(country_counts) > 0 else 'N/A'}")
        
        if 'Year' in self.df.columns:
            insights['year_range'] = {
                'min': int(self.df['Year'].min()),
                'max': int(self.df['Year'].max()),
                'span': int(self.df['Year'].max() - self.df['Year'].min())
            }
            print(f"  Year range: {insights['year_range']['min']} to {insights['year_range']['max']}")
        
        # Key metrics
        key_cols = ['Mortality_Rate', 'Recovery_Rate', 'Avg_Treatment_Cost_USD', 
                   'Healthcare_Access', 'Risk Score', 'Country_Population', 'DALYs']
        
        for col in key_cols:
            if col in self.df.columns:
                col_data = self.df[col]
                insights['key_metrics'][col] = {
                    'mean': float(col_data.mean()),
                    'median': float(col_data.median()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max())
                }
        
        # Trends over time
        if 'Year' in self.df.columns and 'Mortality_Rate' in self.df.columns:
            yearly_trend = self.df.groupby('Year')['Mortality_Rate'].mean()
            if len(yearly_trend) > 0:
                insights['mortality_trend'] = {
                    'best_year': int(yearly_trend.idxmin()),
                    'worst_year': int(yearly_trend.idxmax()),
                    'improvement': float(yearly_trend.diff().mean())
                }
        
        # Country performance
        if 'Country' in self.df.columns and 'Recovery_Rate' in self.df.columns:
            country_perf = self.df.groupby('Country')['Recovery_Rate'].mean()
            if len(country_perf) > 0:
                insights['best_performing_countries'] = country_perf.nlargest(5).to_dict()
                insights['worst_performing_countries'] = country_perf.nsmallest(5).to_dict()
        
        print("✓ Insights generated")
        return insights


# ======================================================================
# 2. AUTOMATED REPORTING SYSTEM
# ======================================================================

class HealthReportGenerator:
    """Generate automated reports in PDF and HTML formats"""
    
    def __init__(self, df, insights, models_info=None):
        self.df = df
        self.insights = insights
        self.models_info = models_info or {}
        self.report_date = datetime.now().strftime("%Y-%m-%d")
    
    def generate_pdf_reportlab(self, filename='health_report.pdf', 
                               compact_mode=True,
                               include_all_metrics=True):
        """Generate PDF report using ReportLab with optimized layout"""
        print(f"\nGenerating ReportLab PDF: {filename}")
        
        # Create document with optimized margins
        if compact_mode:
            doc = SimpleDocTemplate(filename, pagesize=A4, 
                                    rightMargin=54, leftMargin=54,  # Reduced margins
                                    topMargin=54, bottomMargin=54)
        else:
            doc = SimpleDocTemplate(filename, pagesize=A4, 
                                    rightMargin=72, leftMargin=72,
                                    topMargin=72, bottomMargin=72)
        
        # Get styles
        styles = getSampleStyleSheet()
        
        # Create optimized styles
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontSize=20 if compact_mode else 24,  # Smaller in compact mode
            textColor=colors.HexColor('#2E4057'),
            spaceAfter=20 if compact_mode else 30,
            alignment=TA_CENTER
        ))
        
        styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=styles['Heading2'],
            fontSize=14 if compact_mode else 16,  # Smaller
            textColor=colors.HexColor('#1A5276'),
            spaceAfter=8 if compact_mode else 12,
            spaceBefore=15 if compact_mode else 20
        ))
        
        styles.add(ParagraphStyle(
            name='SubSection',
            parent=styles['Heading3'],
            fontSize=12 if compact_mode else 14,  # Smaller
            textColor=colors.HexColor('#2874A6'),
            spaceAfter=6 if compact_mode else 8,
            spaceBefore=10 if compact_mode else 15
        ))
        
        # Helper function to truncate long text
        def truncate_text(text, max_length=25):
            if not isinstance(text, str):
                text = str(text)
            if len(text) > max_length:
                return text[:max_length-3] + "..."
            return text
        
        # Helper function to create efficient tables
        def create_efficient_table(data, col_widths, style, 
                                  max_rows=None, compact=False):
            """Create tables optimized for space"""
            if compact:
                # Reduce padding in compact mode
                style.add('TOPPADDING', (0,0), (-1,-1), 3)
                style.add('BOTTOMPADDING', (0,0), (-1,-1), 3)
                style.add('LEFTPADDING', (0,0), (-1,-1), 4)
                style.add('RIGHTPADDING', (0,0), (-1,-1), 4)
            
            # Limit rows if specified
            if max_rows and len(data) > max_rows:
                data = data[:max_rows]
            
            table = Table(data, colWidths=col_widths)
            table.setStyle(style)
            return table
        
        # Story will hold all elements
        story = []
        
        # 1. COMPACT TITLE PAGE
        story.append(Paragraph(f"GLOBAL HEALTH ANALYTICS REPORT", styles['CustomTitle']))
        story.append(Spacer(1, 10 if compact_mode else 20))
        
        # Report metadata
        meta_data = f"""
        <para>
        <font size={'9' if compact_mode else '10'}>
        <b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        <b>Data Period:</b> {int(self.df['Year'].min()) if 'Year' in self.df.columns else 'N/A'} - {int(self.df['Year'].max()) if 'Year' in self.df.columns else 'N/A'}<br/>
        <b>Total Records:</b> {len(self.df):,}<br/>
        <b>Countries:</b> {self.df['Country'].nunique() if 'Country' in self.df.columns else 0}<br/>
        <b>Diseases:</b> {self.df['Disease Name'].nunique() if 'Disease Name' in self.df.columns else 0}
        </font>
        </para>
        """
        story.append(Paragraph(meta_data, styles['Normal']))
        story.append(Spacer(1, 20 if compact_mode else 40))
        
        # Executive Summary (compact)
        story.append(Paragraph("EXECUTIVE SUMMARY", styles['SectionHeader']))
        summary_text = f"""
        <para>
        <font size={'9' if compact_mode else '10'}>
        This comprehensive report provides an analysis of global health data across multiple countries and diseases. 
        The analysis reveals key patterns in disease prevalence, mortality rates, recovery rates, and healthcare accessibility. 
        Key findings and recommendations are presented to guide policy decisions and resource allocation.
        The report covers {len(self.df):,} health records spanning {int(self.df['Year'].max()) - int(self.df['Year'].min()) if 'Year' in self.df.columns else 'multiple'} years of data collection.
        Analysis identifies {self.df['Disease Name'].nunique() if 'Disease Name' in self.df.columns else 0} distinct diseases with varying impact across {self.df['Country'].nunique() if 'Country' in self.df.columns else 0} nations.
        Critical insights include regional disparities in healthcare access, disease burden variations, and recovery rate correlations.
        This evidence-based analysis serves as a foundation for targeted interventions and strategic health policy development.
        </font>
        </para>
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 15 if compact_mode else 30))
        
        # 2. KEY METRICS TABLE (Optimized for space)
        story.append(Paragraph("KEY METRICS", styles['SectionHeader']))
        
        if self.insights and 'key_metrics' in self.insights:
            # Prepare compact table data
            table_data = [['Metric', 'Mean', 'Min', 'Max']]  # Reduced columns
            
            # Show more metrics in compact mode
            metrics_to_show = list(self.insights['key_metrics'].items())
            if not include_all_metrics:
                metrics_to_show = metrics_to_show[:15]  # Limit if requested
            
            for metric, stats in metrics_to_show:
                # Very compact metric names
                metric_name = metric.replace('_', ' ')
                if len(metric_name) > 20:
                    metric_name = metric_name[:17] + "..."
                
                row = [
                    metric_name,
                    f"{stats['mean']:.2f}",
                    f"{stats['min']:.2f}",
                    f"{stats['max']:.2f}"
                ]
                table_data.append(row)
            
            # Compact table style
            table_style = TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2E4057')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('ALIGN', (0,1), (0,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 8 if compact_mode else 9),
                ('FONTSIZE', (0,1), (-1,-1), 7 if compact_mode else 8),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),  # Thinner grid
                ('TOPPADDING', (0,0), (-1,-1), 2 if compact_mode else 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2 if compact_mode else 4),
            ])
            
            # Column widths optimized
            col_widths = [2.2*inch, 0.7*inch, 0.7*inch, 0.7*inch]
            
            # Create efficient table
            t = create_efficient_table(table_data, col_widths, table_style, compact=compact_mode)
            
            # Add alternating row colors
            for i in range(1, len(table_data)):
                if i % 2 == 0:
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,i), (-1,i), colors.HexColor('#FFFFFF')),
                    ]))
                else:
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,i), (-1,i), colors.HexColor('#F2F4F4')),
                    ]))
            
            story.append(t)
        
        story.append(Spacer(1, 15 if compact_mode else 20))
        
        # 3. TOP DISEASES (More diseases per page)
        story.append(Paragraph("TOP DISEASES", styles['SectionHeader']))
        
        if self.insights and 'top_diseases' in self.insights:
            # Prepare table data - show more diseases
            table_data = [['#', 'Disease', 'Count', '%']]  # Compact headers
            
            total_records = len(self.df)
            disease_items = list(self.insights['top_diseases'].items())
            
            # Show more diseases (up to 15)
            max_diseases = 15 if compact_mode else 10
            disease_items = disease_items[:max_diseases]
            
            for i, (disease, count) in enumerate(disease_items, 1):
                percentage = (count / total_records * 100) if total_records > 0 else 0
                
                # Compact disease names
                display_disease = truncate_text(disease, max_length=25 if compact_mode else 30)
                
                row = [
                    str(i),
                    display_disease,
                    f"{count:,}",
                    f"{percentage:.1f}%"
                ]
                table_data.append(row)
            
            # Compact table
            t = Table(table_data, colWidths=[0.3*inch, 2.5*inch, 0.8*inch, 0.6*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A5276')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('ALIGN', (1,1), (1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 8 if compact_mode else 9),
                ('FONTSIZE', (0,1), (-1,-1), 7 if compact_mode else 8),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('TOPPADDING', (0,0), (-1,-1), 2 if compact_mode else 4),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2 if compact_mode else 4),
            ]))
            
            # Add alternating row colors
            for i in range(1, len(table_data)):
                if i % 2 == 0:
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,i), (-1,i), colors.HexColor('#FFFFFF')),
                    ]))
                else:
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,i), (-1,i), colors.HexColor('#EBF5FB')),
                    ]))
            
            story.append(t)
        
        story.append(Spacer(1, 15 if compact_mode else 20))
        
        # 4. COUNTRY PERFORMANCE (Dense layout)
        story.append(Paragraph("COUNTRY PERFORMANCE", styles['SectionHeader']))
        
        if 'Country' in self.df.columns:
            # Get key metrics for countries
            key_cols = ['Mortality_Rate', 'Recovery_Rate']
            available_cols = [col for col in key_cols if col in self.df.columns]
            
            if available_cols:
                # Calculate country averages
                country_stats = self.df.groupby('Country')[available_cols].mean().round(2)
                
                # Get more countries
                top_countries = country_stats.head(12 if compact_mode else 10)
                
                # Prepare very compact table
                headers = ['Country', 'Mortality', 'Recovery']
                table_data = [headers]
                
                for country, row in top_countries.iterrows():
                    country_name = truncate_text(str(country), max_length=18 if compact_mode else 20)
                    country_row = [country_name]
                    
                    for col in available_cols:
                        value = row[col]
                        country_row.append(f"{value:.1f}%")
                    
                    table_data.append(country_row)
                
                # Very compact column widths
                col_widths = [1.5*inch, 0.8*inch, 0.8*inch]
                t = Table(table_data, colWidths=col_widths)
                
                # Ultra-compact styling
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2874A6')),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                    ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                    ('ALIGN', (0,1), (0,-1), 'LEFT'),
                    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0,0), (-1,0), 8),
                    ('FONTSIZE', (0,1), (-1,-1), 7),
                    ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                    ('TOPPADDING', (0,0), (-1,-1), 1),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 1),
                ]))
                
                # Minimal alternating colors
                for i in range(1, len(table_data)):
                    if i % 2 == 0:
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0,i), (-1,i), colors.HexColor('#FFFFFF')),
                        ]))
                    else:
                        t.setStyle(TableStyle([
                            ('BACKGROUND', (0,i), (-1,i), colors.HexColor('#F4F6F6')),
                        ]))
                
                story.append(t)
        
        story.append(Spacer(1, 15 if compact_mode else 20))
        
        # 5. DATA OVERVIEW (Single compact table)
        story.append(Paragraph("DATA OVERVIEW", styles['SectionHeader']))
        
        if self.df is not None:
            # Create compact overview
            overview_data = [
                ['Metric', 'Value'],
                ['Records', f"{len(self.df):,}"],
                ['Columns', f"{len(self.df.columns)}"],
                ['Countries', f"{self.df['Country'].nunique() if 'Country' in self.df.columns else 0}"],
                ['Diseases', f"{self.df['Disease Name'].nunique() if 'Disease Name' in self.df.columns else 0}"],
                ['Years', f"{int(self.df['Year'].min()) if 'Year' in self.df.columns else 'N/A'}-{int(self.df['Year'].max()) if 'Year' in self.df.columns else 'N/A'}"],
            ]
            
            t = Table(overview_data, colWidths=[1.5*inch, 1.5*inch])
            t.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#7D3C98')),
                ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,0), 8),
                ('FONTSIZE', (0,1), (-1,-1), 8),
                ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
                ('TOPPADDING', (0,0), (-1,-1), 2),
                ('BOTTOMPADDING', (0,0), (-1,-1), 2),
            ]))
            
            story.append(t)
        
        story.append(Spacer(1, 15 if compact_mode else 20))
        
        # 6. RECOMMENDATIONS (Compact)
        story.append(Paragraph("RECOMMENDATIONS", styles['SectionHeader']))
        
        recommendations = [
            ['High', 'Resources', 'Focus on high mortality diseases'],
            ['High', 'Access', 'Improve healthcare access'],
            ['Medium', 'Prevention', 'Targeted vaccination programs'],
            ['Medium', 'Data', 'Enhance data collection'],
            ['High', 'Infrastructure', 'Increase medical personnel'],
        ]
        
        t = Table(recommendations, colWidths=[0.6*inch, 0.8*inch, 2.6*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#27AE60')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,0), 8),
            ('FONTSIZE', (0,1), (-1,-1), 7),
            ('GRID', (0,0), (-1,-1), 0.25, colors.grey),
            ('TOPPADDING', (0,0), (-1,-1), 1),
            ('BOTTOMPADDING', (0,0), (-1,-1), 1),
        ]))
        
        story.append(t)
        
        # 7. FOOTER (Compact)
        story.append(Spacer(1, 10 if compact_mode else 20))
        footer_text = f"""
        <para>
        <font size=7>
        <i>Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</i>
        </font>
        </para>
        """
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Build the PDF
        doc.build(story)
        print(f"✓ Compact PDF generated: {filename}")
        return filename
    
    def generate_html_report(self, filename='health_report.html'):
        """Generate HTML report with interactive charts"""
        print(f"\nGenerating HTML report with charts: {filename}")
        
        # Create charts
        charts = self._create_charts()
        
        # Prepare chart HTML
        charts_html = ""
        if charts:
            for chart_name, fig in charts.items():
                # Convert plotly figure to HTML
                chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
                charts_html += f"""
                <div class="chart-section">
                    <div class="chart-container">
                        {chart_html}
                    </div>
                </div>
                <hr>
                """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Global Health Analytics Report</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <script src="https://cdn.tailwindcss.com"></script>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
                .header {{ background: rgba(255, 255, 255, 0.95); padding: 40px; text-align: center; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); }}
                .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                .card {{ background: white; border-radius: 10px; padding: 25px; margin: 20px 0; box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1); }}
                .chart-container {{ width: 100%; height: 500px; margin: 20px 0; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
                .metric-box {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
                .insight-box {{ background: #f8f9fa; border-left: 4px solid #667eea; padding: 15px; margin: 15px 0; }}
                .recommendation {{ background: #e8f4fd; border-left: 4px solid #2196F3; padding: 15px; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1 style="color: #333; font-size: 2.5rem; margin-bottom: 10px;">Global Health Analytics Report</h1>
                <p style="color: #666; font-size: 1.1rem;">Generated on: {self.report_date}</p>
            </div>
            
            <div class="container">
                <!-- Executive Summary -->
                <div class="card">
                    <h2 style="color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">Executive Summary</h2>
                    <p>This comprehensive report analyzes global health data to provide insights into disease patterns, healthcare metrics, and regional performance.</p>
                    
                    <div class="metric-grid">
                        <div class="metric-box">
                            <h3 style="margin: 0; font-size: 1.5rem;">{len(self.df):,}</h3>
                            <p style="margin: 5px 0 0 0; opacity: 0.9;">Total Records</p>
                        </div>
                        <div class="metric-box">
                            <h3 style="margin: 0; font-size: 1.5rem;">{self.df['Country'].nunique() if 'Country' in self.df.columns else 0}</h3>
                            <p style="margin: 5px 0 0 0; opacity: 0.9;">Countries</p>
                        </div>
                        <div class="metric-box">
                            <h3 style="margin: 0; font-size: 1.5rem;">{self.df['Disease Name'].nunique() if 'Disease Name' in self.df.columns else 0}</h3>
                            <p style="margin: 5px 0 0 0; opacity: 0.9;">Diseases</p>
                        </div>
                    </div>
                </div>
                
                <!-- Charts Section -->
                <div class="card">
                    <h2 style="color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">Data Visualizations</h2>
                    {charts_html if charts_html else '<p>No charts available for this dataset.</p>'}
                </div>
                
                <!-- Key Insights -->
                <div class="card">
                    <h2 style="color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">Key Insights</h2>
                    {"".join([f'''
                    <div class="insight-box">
                        <h4 style="margin: 0 0 10px 0; color: #667eea;">{metric.replace("_", " ").title()}</h4>
                        <p><strong>Average:</strong> {stats["mean"]:.2f} | <strong>Range:</strong> {stats["min"]:.2f} - {stats["max"]:.2f}</p>
                    </div>
                    ''' for metric, stats in self.insights.get('key_metrics', {}).items()])}
                </div>
                
                <!-- Recommendations -->
                <div class="card">
                    <h2 style="color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">Recommendations</h2>
                    <div class="recommendation">
                        <strong>1. Resource Allocation:</strong> Focus on diseases with highest mortality and lowest recovery rates.
                    </div>
                    <div class="recommendation">
                        <strong>2. Healthcare Infrastructure:</strong> Invest in healthcare access improvement in underperforming regions.
                    </div>
                    <div class="recommendation">
                        <strong>3. Preventive Measures:</strong> Develop targeted prevention programs for high-risk populations.
                    </div>
                    <div class="recommendation">
                        <strong>4. Data Enhancement:</strong> Improve data collection for better predictive analytics.
                    </div>
                    <div class="recommendation">
                        <strong>5. Policy Development:</strong> Create evidence-based policies using cluster analysis results.
                    </div>
                </div>
                
                <!-- Technical Details -->
                <div class="card">
                    <h2 style="color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">Technical Details</h2>
                    <p><strong>Report Generation:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>Data Source:</strong> Global Health Analytics Platform</p>
                    <p><strong>Analysis Methodology:</strong> Descriptive statistics, trend analysis, and predictive modeling</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ HTML report with interactive charts saved: {filename}")
        return filename
    def send_email_report(self, recipient, sender, password, smtp_server="smtp.gmail.com", 
                         smtp_port=587, include_data=False, data_format='csv'):
        """
        Send report via email with optional data attachment
        
        Parameters:
        - recipient: Email address to send to
        - sender: Email address sending from
        - password: Email password or app password
        - smtp_server: SMTP server address
        - smtp_port: SMTP port
        - include_data: Whether to include cleaned data file (True/False)
        - data_format: Format for data file ('csv', 'excel', 'json')
        """
        try:
            # Generate reports with unique names
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_filename = f"health_report_{timestamp}.pdf"
            html_filename = f"health_report_{timestamp}.html"
            
            # Generate the reports
            if hasattr(self, 'generate_pdf_reportlab'):
                pdf_path = self.generate_pdf_reportlab(pdf_filename)
            elif hasattr(self, 'generate_pdf_report_simple'):
                pdf_path = self.generate_pdf_report_simple(pdf_filename)
            else:
                pdf_path = self._create_simple_pdf(pdf_filename)
            
            # Generate HTML report
            html_path = self.generate_html_report(html_filename)
            
            # Generate data file if requested
            data_path = None
            if include_data and self.df is not None:
                data_filename = f"health_data_{timestamp}.{data_format}"
                data_path = self._export_data_file(data_filename, data_format)
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = sender
            msg['To'] = recipient
            msg['Subject'] = f"Global Health Analytics Report – {datetime.now().strftime('%d %B %Y')}"
            
            # Create email body
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                    <h2 style="color: #2E4057; border-bottom: 2px solid #2E4057; padding-bottom: 10px;">
                        🌍 Global Health Analytics Report
                    </h2>
                    
                    <p>Hello,</p>
                    
                    <p>Please find attached your <strong>Global Health Analytics Report</strong>.</p>
                    
                    <div style="background-color: #F8F9F9; padding: 15px; margin: 20px 0; border-left: 4px solid #2E4057;">
                        <h3 style="color: #2E4057; margin-top: 0;">📊 Report Summary</h3>
                        <ul>
                            <li><strong>Total Records:</strong> {len(self.df):,}</li>
            """
            
            # Add dynamic content based on available data
            if 'Country' in self.df.columns:
                body += f"<li><strong>Countries:</strong> {self.df['Country'].nunique()}</li>"
            
            if 'Disease Name' in self.df.columns:
                body += f"<li><strong>Diseases:</strong> {self.df['Disease Name'].nunique()}</li>"
            
            if 'Year' in self.df.columns:
                body += f"<li><strong>Time Period:</strong> {int(self.df['Year'].min())} - {int(self.df['Year'].max())}</li>"
            
            # Add key metrics if available
            if self.insights and 'key_metrics' in self.insights:
                if 'Mortality_Rate' in self.insights['key_metrics']:
                    mortality = self.insights['key_metrics']['Mortality_Rate']['mean']
                    body += f"<li><strong>Avg Mortality Rate:</strong> {mortality:.3f} per 100 people</li>"
                
                if 'Recovery_Rate' in self.insights['key_metrics']:
                    recovery = self.insights['key_metrics']['Recovery_Rate']['mean']
                    body += f"<li><strong>Avg Recovery Rate:</strong> {recovery:.1f}%</li>"
            
            body += f"""
                        </ul>
                    </div>
                    
                    <div style="background-color: #E8F6F3; padding: 15px; margin: 20px 0; border-radius: 5px;">
                        <h4 style="color: #148F77; margin-top: 0;">📎 Attachments Included:</h4>
                        <ul>
                            <li><strong>PDF Report:</strong> Comprehensive analysis with detailed tables</li>
                            <li><strong>HTML Report:</strong> Interactive web version with charts</li>
            """
            
            if include_data and data_path:
                file_size = os.path.getsize(data_path) if os.path.exists(data_path) else 0
                file_size_mb = file_size / (1024 * 1024)
                body += f"""<li><strong>Data File ({data_format.upper()}):</strong> Cleaned dataset ({file_size_mb:.2f} MB)</li>"""
            
            body += f"""
                        </ul>
                    </div>
                    
                    <div style="background-color: #FEF9E7; padding: 15px; margin: 20px 0; border-left: 4px solid #F39C12;">
                        <h4 style="color: #F39C12; margin-top: 0;">💡 Key Insights</h4>
                        <ul>
                            <li>Analysis based on {len(self.df):,} health records</li>
                            <li>Covering {self.df['Country'].nunique() if 'Country' in self.df.columns else 0} countries worldwide</li>
                            <li>Includes {self.df['Disease Name'].nunique() if 'Disease Name' in self.df.columns else 0} different diseases</li>
                        </ul>
                    </div>
                    
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <hr style="border: none; border-top: 1px solid #eee; margin: 30px 0;">
                    
                    <p style="color: #666; font-size: 12px;">
                        This email was automatically generated by the Global Health Analytics Platform.<br>
                        For questions or feedback, please contact the system administrator.
                    </p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Attach PDF file
            with open(pdf_path, "rb") as pdf_file:
                pdf_part = MIMEBase('application', 'pdf')
                pdf_part.set_payload(pdf_file.read())
                encoders.encode_base64(pdf_part)
                pdf_part.add_header('Content-Disposition', 
                                  f'attachment; filename="Health_Report_{timestamp}.pdf"')
                msg.attach(pdf_part)
            
            # Attach HTML file
            with open(html_path, "rb") as html_file:
                html_part = MIMEBase('application', 'octet-stream')
                html_part.set_payload(html_file.read())
                encoders.encode_base64(html_part)
                html_part.add_header('Content-Disposition', 
                                   f'attachment; filename="Health_Report_{timestamp}.html"')
                msg.attach(html_part)
            
            # Attach data file if requested
            if include_data and data_path and os.path.exists(data_path):
                with open(data_path, "rb") as data_file:
                    data_part = MIMEBase('application', 'octet-stream')
                    data_part.set_payload(data_file.read())
                    encoders.encode_base64(data_part)
                    data_part.add_header('Content-Disposition', 
                                       f'attachment; filename="Health_Data_{timestamp}.{data_format}"')
                    msg.attach(data_part)
            
            # Send email
            server = smtplib.SMTP(smtp_server, smtp_port)
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
            server.quit()
            
            # Clean up temporary files
            try:
                os.remove(pdf_path)
                os.remove(html_path)
                if data_path and os.path.exists(data_path):
                    os.remove(data_path)
            except:
                pass
            
            return True, f"✅ Email sent successfully to {recipient}!"
            
        except smtplib.SMTPAuthenticationError:
            return False, "❌ Authentication failed. Please check your email and password."
        except smtplib.SMTPException as e:
            return False, f"❌ SMTP error: {str(e)}"
        except Exception as e:
            return False, f"❌ Failed to send email: {str(e)}"
        
    def _export_data_file(self, filename, format='csv'):
        """Export cleaned data to specified format"""
        try:
            if self.df is None or len(self.df) == 0:
                return None
            
            if format == 'csv':
                self.df.to_csv(filename, index=False, encoding='utf-8')
            
            elif format == 'excel':
                self.df.to_excel(filename, index=False)
            
            elif format == 'json':
                self.df.to_json(filename, orient='records', indent=2)
            
            else:
                # Default to CSV
                self.df.to_csv(filename, index=False, encoding='utf-8')
            
            print(f"✓ Data exported to {filename} ({format.upper()})")
            return filename
            
        except Exception as e:
            print(f"✗ Error exporting data: {str(e)}")
            return None
        
    def _create_charts(self):
        """Create various charts for reports"""
        charts = {}
        
        try:
            # 1. Top Diseases Bar Chart
            if 'Disease Name' in self.df.columns:
                top_diseases = self.df['Disease Name'].value_counts().head(10)
                fig = px.bar(
                    x=top_diseases.values,
                    y=top_diseases.index,
                    orientation='h',
                    title='Top 10 Diseases by Frequency',
                    labels={'x': 'Count', 'y': 'Disease'},
                    color=top_diseases.values,
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                charts['top_diseases'] = fig
            
            # 2. Country Performance Scatter Plot
            if all(col in self.df.columns for col in ['Country', 'Mortality_Rate', 'Recovery_Rate']):
                country_stats = self.df.groupby('Country')[['Mortality_Rate', 'Recovery_Rate']].mean()
                fig = px.scatter(
                    country_stats,
                    x='Mortality_Rate',
                    y='Recovery_Rate',
                    title='Country Performance: Mortality vs Recovery Rates',
                    labels={'Mortality_Rate': 'Mortality Rate (per 100)', 
                           'Recovery_Rate': 'Recovery Rate (%)'},
                    hover_name=country_stats.index,
                    color='Recovery_Rate',
                    size='Recovery_Rate'
                )
                fig.update_layout(height=400)
                charts['country_performance'] = fig
            
            # 3. Mortality Trend Over Time
            if all(col in self.df.columns for col in ['Year', 'Mortality_Rate']):
                yearly_mortality = self.df.groupby('Year')['Mortality_Rate'].mean().reset_index()
                fig = px.line(
                    yearly_mortality,
                    x='Year',
                    y='Mortality_Rate',
                    title='Mortality Rate Trend Over Time',
                    labels={'Mortality_Rate': 'Mortality Rate (per 100)'},
                    markers=True
                )
                fig.update_layout(height=400)
                charts['mortality_trend'] = fig
            
            # 4. Healthcare Metrics Distribution
            healthcare_cols = ['Healthcare_Access', 'Doctors_per_1000', 'Hospital_Beds_per_1000']
            available_cols = [col for col in healthcare_cols if col in self.df.columns]
            
            if available_cols:
                fig = make_subplots(
                    rows=len(available_cols), cols=1,
                    subplot_titles=available_cols
                )
                
                for i, col in enumerate(available_cols, 1):
                    fig.add_trace(
                        go.Histogram(
                            x=self.df[col],
                            name=col,
                            nbinsx=30
                        ),
                        row=i, col=1
                    )
                
                fig.update_layout(
                    height=300 * len(available_cols),
                    title_text='Healthcare Metrics Distribution',
                    showlegend=False
                )
                charts['healthcare_distribution'] = fig
            
            # 5. Risk Score by Disease
            if all(col in self.df.columns for col in ['Disease Name', 'Risk Score']):
                disease_risk = self.df.groupby('Disease Name')['Risk Score'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(
                    x=disease_risk.values,
                    y=disease_risk.index,
                    orientation='h',
                    title='Top 10 Diseases by Risk Score',
                    labels={'x': 'Risk Score', 'y': 'Disease'},
                    color=disease_risk.values,
                    color_continuous_scale='reds'
                )
                fig.update_layout(height=400)
                charts['disease_risk'] = fig
            
            # 6. Correlation Heatmap
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = self.df[numeric_cols[:10]].corr()  # Limit to first 10 for readability
                fig = px.imshow(
                    corr_matrix,
                    text_auto='.2f',
                    aspect="auto",
                    title='Feature Correlation Heatmap',
                    color_continuous_scale='RdBu'
                )
                fig.update_layout(height=500)
                charts['correlation_heatmap'] = fig
            
        except Exception as e:
            print(f"Warning: Could not create some charts: {str(e)}")
        
        return charts
    
    def _chart_to_image_base64(self, fig):
        """Convert plotly figure to base64 encoded image"""
        try:
            # Convert to static image
            img_bytes = pio.to_image(fig, format='png', width=800, height=400)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_base64
        except:
            return None
        
    


# ======================================================================
# 3. STREAMLIT WEB APPLICATION
# ======================================================================

class HealthDashboardApp:
    """Streamlit web application for health data analysis"""
    
    def __init__(self):
        # Initialize session state
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'cleaner' not in st.session_state:
            st.session_state.cleaner = EnhancedHealthDataCleaner()
        if 'insights' not in st.session_state:
            st.session_state.insights = None
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'default_values' not in st.session_state:
            st.session_state.default_values = {}
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        
        # Page configuration
        st.set_page_config(
            page_title="Global Health Analytics Dashboard",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the Streamlit application"""
        
        # Sidebar navigation
        with st.sidebar:
            st.image("https://cdn-icons-png.flaticon.com/512/3067/3067256.png", width=100)
            st.title("🏥 Global Health Analytics")
            
            selected = option_menu(
                menu_title="Navigation",
                options=["Home", "Data Upload", "Data Analysis", 
                        "ML Predictions", "Reports", "About"],
                icons=["house", "cloud-upload", "graph-up", 
                      "robot", "file-text", "info-circle"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "5px"},
                    "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px"},
                }
            )
            
            # Show data status
            st.divider()
            if st.session_state.df is not None:
                st.success(f"✅ Data loaded: {len(st.session_state.df):,} records")
                if st.session_state.insights:
                    st.info("✅ Insights available")
                if st.session_state.models_trained:
                    st.success("✅ ML models trained")
                    # Show available models
                    models = list(st.session_state.cleaner.models.keys())
                    if models:
                        st.write("**Available models:**")
                        for model in models:
                            st.write(f"• {model}")
            else:
                st.warning("No data loaded")
        
        # Home Page
        if selected == "Home":
            self._render_home()
        
        # Data Upload Page
        elif selected == "Data Upload":
            self._render_data_upload()
        
        # Data Analysis Page
        elif selected == "Data Analysis":
            self._render_data_analysis()
        
        # ML Predictions Page
        elif selected == "ML Predictions":
            self._render_ml_predictions()
        
        # Reports Page
        elif selected == "Reports":
            self._render_reports()
        
        # About Page
        elif selected == "About":
            self._render_about()
    
    def _render_home(self):
        """Render home page"""
        st.title("🌍 Global Health Analytics Dashboard")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Countries", "150+", "Global Coverage")
        
        with col2:
            st.metric("Diseases", "100+", "Comprehensive")
        
        with col3:
            st.metric("Records", "1M+", "Detailed Analysis")
        
        st.markdown("---")
        
        st.write("""
        ### Welcome to the Global Health Analytics Platform
        
        This platform provides:
        
        📊 **Interactive Dashboards** - Visualize health trends and patterns
        🤖 **Machine Learning Predictions** - Forecast disease outcomes and costs
        📁 **Multi-format Support** - Upload CSV, Excel, SQLite files
        📈 **Advanced Analytics** - Clustering, trend analysis, and insights
        📨 **Automated Reporting** - Generate and email comprehensive reports
        
        **Get Started:**
        1. Upload your health data (CSV, Excel, or SQLite)
        2. Explore interactive visualizations
        3. Use ML models for predictions
        4. Generate automated reports
        """)
        
        if st.session_state.df is not None:
            with st.expander("Current Data Preview"):
                st.dataframe(st.session_state.df.head(), use_container_width=True)
    
    def _process_uploaded_file(self, uploaded_file, file_ext):
        """Process uploaded file based on type"""
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            if file_ext in ['.csv']:
                # Use the cleaner to process CSV
                cleaner = EnhancedHealthDataCleaner()
                df = cleaner.clean_dataset(tmp_path)
                return df, "CSV file processed successfully!"
            
            elif file_ext in ['.xlsx', '.xls']:
                # Read Excel file
                xl = pd.ExcelFile(tmp_path)
                sheet_names = xl.sheet_names
                
                # If only one sheet, use it automatically
                if len(sheet_names) == 1:
                    df = pd.read_excel(tmp_path, sheet_name=sheet_names[0])
                    return df, f"Excel file loaded from sheet: {sheet_names[0]}"
                else:
                    # Let user choose sheet
                    selected_sheet = st.selectbox("Select sheet to load", sheet_names)
                    if selected_sheet:
                        df = pd.read_excel(tmp_path, sheet_name=selected_sheet)
                        return df, f"Excel file loaded from sheet: {selected_sheet}"
            
            elif file_ext in ['.db', '.sqlite']:
                # Read SQLite database
                conn = sqlite3.connect(tmp_path)
                try:
                    # Get list of tables
                    tables = pd.read_sql_query(
                        "SELECT name FROM sqlite_master WHERE type='table';", conn
                    )
                    
                    if len(tables) == 0:
                        return None, "No tables found in the SQLite database"
                    
                    # If only one table, use it automatically
                    if len(tables) == 1:
                        table_name = tables.iloc[0]['name']
                        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
                        return df, f"SQLite table loaded: {table_name}"
                    else:
                        # Let user choose table
                        table_names = tables['name'].tolist()
                        selected_table = st.selectbox("Select table to load", table_names)
                        if selected_table:
                            df = pd.read_sql_query(f"SELECT * FROM {selected_table}", conn)
                            return df, f"SQLite table loaded: {selected_table}"
                finally:
                    conn.close()
            
            return None, f"Unsupported file type: {file_ext}"
            
        except Exception as e:
            return None, f"Error processing file: {str(e)}"
        
        finally:
            # Clean up temp file
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def _render_data_upload(self):
        """Render data upload page"""
        st.title("📁 Upload Health Data")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls', 'db', 'sqlite'],
            help="Upload CSV, Excel, or SQLite files"
        )
        
        if uploaded_file is not None:
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            
            # Process the file
            with st.spinner(f"Processing {uploaded_file.name}..."):
                df, message = self._process_uploaded_file(uploaded_file, file_ext)
                
                if df is not None:
                    # Store in session state
                    st.session_state.df = df
                    st.session_state.cleaner.load_dataframe(df)
                    st.session_state.insights = None
                    st.session_state.models_trained = False
                    st.session_state.default_values = st.session_state.cleaner.get_default_values_for_prediction()
                    st.session_state.data_loaded = True
                    
                    st.success(message)
                    
                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(df.head(100), use_container_width=True)
                    
                    # Basic statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", f"{len(df):,}")
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.1f} MB")
                    
                    # Show available columns
                    with st.expander("View Available Columns"):
                        col_list = list(df.columns)
                        cols_per_row = 3
                        rows = [col_list[i:i+cols_per_row] for i in range(0, len(col_list), cols_per_row)]
                        for row in rows:
                            cols = st.columns(len(row))
                            for idx, col_name in enumerate(row):
                                with cols[idx]:
                                    st.text(col_name)
                    
                    # Generate insights button
                    if st.button("Generate Insights", type="primary"):
                        with st.spinner("Analyzing data..."):
                            st.session_state.insights = st.session_state.cleaner.generate_insights()
                            st.success("✅ Insights generated!")
                            
                            # Display quick insights
                            st.subheader("Quick Insights")
                            if st.session_state.insights:
                                insights_col1, insights_col2 = st.columns(2)
                                
                                with insights_col1:
                                    if 'top_diseases' in st.session_state.insights:
                                        st.write("**Top 5 Diseases:**")
                                        for disease, count in list(st.session_state.insights['top_diseases'].items())[:5]:
                                            st.write(f"- {disease}: {count:,} records")
                                
                                with insights_col2:
                                    if 'year_range' in st.session_state.insights:
                                        st.write("**Time Range:**")
                                        st.write(f"- From: {st.session_state.insights['year_range']['min']}")
                                        st.write(f"- To: {st.session_state.insights['year_range']['max']}")
                                        st.write(f"- Span: {st.session_state.insights['year_range']['span']} years")
                else:
                    st.error(message)
    
    def _render_data_analysis(self):
        """Render data analysis page"""
        st.title("📊 Data Analysis Dashboard")
        
        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first from the Data Upload page")
            st.info("Go to the Data Upload page to load your dataset")
            return
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Overview", "🌍 Geographical", "🦠 Disease Analysis", "📅 Trends"
        ])
        
        with tab1:
            self._render_overview_analysis()
        
        with tab2:
            self._render_geographical_analysis()
        
        with tab3:
            self._render_disease_analysis()
        
        with tab4:
            self._render_trend_analysis()
    
    def _render_overview_analysis(self):
        """Render overview analysis"""
        col1, col2 = st.columns(2)
        
        with col1:
            # Top diseases bar chart
            if 'Disease Name' in st.session_state.df.columns:
                top_diseases = st.session_state.df['Disease Name'].value_counts().head(10)
                fig = px.bar(
                    x=top_diseases.values,
                    y=top_diseases.index,
                    orientation='h',
                    title='Top 10 Diseases by Frequency',
                    labels={'x': 'Count', 'y': 'Disease'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Countries with most records
            if 'Country' in st.session_state.df.columns:
                top_countries = st.session_state.df['Country'].value_counts().head(10)
                fig = px.pie(
                    values=top_countries.values,
                    names=top_countries.index,
                    title='Top 10 Countries by Record Count'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics distribution
        st.subheader("Key Metrics Distribution")
        
        if len(st.session_state.df.select_dtypes(include=[np.number]).columns) > 0:
            metric_options = [col for col in st.session_state.df.select_dtypes(include=[np.number]).columns 
                             if col not in ['Year', 'Record_ID', 'index']]
            
            selected_metric = st.selectbox(
                "Select metric to visualize",
                metric_options,
                index=0 if metric_options else None
            )
            
            if selected_metric:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        st.session_state.df, 
                        x=selected_metric,
                        title=f'Distribution of {selected_metric}',
                        nbins=50
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(
                        st.session_state.df,
                        y=selected_metric,
                        title=f'Box Plot of {selected_metric}'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for distribution analysis")
    
    def _render_geographical_analysis(self):
        """Render geographical analysis"""
        st.subheader("Geographical Analysis")
        
        if 'Country' in st.session_state.df.columns:
            # Get numeric columns for analysis
            numeric_cols = [col for col in st.session_state.df.select_dtypes(include=[np.number]).columns 
                           if col not in ['Year', 'Record_ID', 'index']]
            
            if numeric_cols:
                # Aggregate by country
                country_metrics = st.multiselect(
                    "Select metrics to aggregate by country",
                    numeric_cols,
                    default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols
                )
                
                if country_metrics:
                    country_agg = st.session_state.df.groupby('Country')[country_metrics].mean().reset_index()
                    
                    # Display as table
                    st.dataframe(
                        country_agg.sort_values(country_metrics[0], ascending=False).head(20),
                        use_container_width=True
                    )
                    
                    # Scatter plot for two selected metrics
                    if len(country_metrics) >= 2:
                        fig = px.scatter(
                            country_agg,
                            x=country_metrics[0],
                            y=country_metrics[1],
                            color='Country',
                            size=country_agg[country_metrics[0]] if country_metrics[0] in country_agg.columns else None,
                            hover_name='Country',
                            title=f'{country_metrics[0]} vs {country_metrics[1]} by Country'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric metrics available for geographical analysis")
        else:
            st.info("No 'Country' column found in the dataset")
    
    def _render_disease_analysis(self):
        """Render disease-specific analysis"""
        st.subheader("Disease Analysis")
        
        if 'Disease Name' in st.session_state.df.columns:
            # Select disease
            diseases = sorted(st.session_state.df['Disease Name'].unique())
            selected_disease = st.selectbox(
                "Select a disease to analyze",
                diseases
            )
            
            if selected_disease:
                disease_data = st.session_state.df[st.session_state.df['Disease Name'] == selected_disease]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Records",
                        f"{len(disease_data):,}",
                        f"{len(disease_data)/len(st.session_state.df)*100:.1f}% of total"
                    )
                
                with col2:
                    if 'Mortality_Rate' in disease_data.columns:
                        st.metric(
                            "Avg Mortality Rate",
                            f"{disease_data['Mortality_Rate'].mean():.3f}",
                            "per 100 people"
                        )
                
                with col3:
                    if 'Recovery_Rate' in disease_data.columns:
                        st.metric(
                            "Avg Recovery Rate",
                            f"{disease_data['Recovery_Rate'].mean():.1f}%",
                            "success rate"
                        )
                
                # Disease trends over time if Year column exists
                if 'Year' in disease_data.columns:
                    yearly_data = []
                    for col in ['Mortality_Rate', 'Recovery_Rate', 'Population_Affected']:
                        if col in disease_data.columns:
                            yearly_data.append(col)
                    
                    if yearly_data:
                        yearly_trend = disease_data.groupby('Year')[yearly_data].mean().reset_index()
                        
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=('Mortality Rate Trend', 'Recovery Rate Trend',
                                          'Population Affected', 'Combined View'),
                            specs=[[{}, {}],
                                  [{'colspan': 2}, None]]
                        )
                        
                        row_idx = 1
                        col_idx = 1
                        for col in yearly_data[:2]:  # First two metrics
                            if col in yearly_trend.columns:
                                fig.add_trace(
                                    go.Scatter(x=yearly_trend['Year'], y=yearly_trend[col],
                                              name=col, mode='lines+markers'),
                                    row=row_idx, col=col_idx
                                )
                                col_idx += 1
                        
                        # Population affected
                        if 'Population_Affected' in yearly_trend.columns:
                            fig.add_trace(
                                go.Bar(x=yearly_trend['Year'], y=yearly_trend['Population_Affected'],
                                      name='Population Affected'),
                                row=2, col=1
                            )
                        
                        fig.update_layout(height=800, title_text=f"{selected_disease} Trends Over Time")
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'Disease Name' column found in the dataset")
    
    def _render_trend_analysis(self):
        """Render trend analysis"""
        st.subheader("Trend Analysis Over Time")
        
        if 'Year' in st.session_state.df.columns:
            # Get numeric columns for trend analysis
            numeric_cols = [col for col in st.session_state.df.select_dtypes(include=[np.number]).columns 
                           if col not in ['Year', 'Record_ID', 'index']]
            
            if numeric_cols:
                # Select metrics for trend analysis
                trend_metrics = st.multiselect(
                    "Select metrics to analyze trends",
                    numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
                )
                
                if trend_metrics:
                    # Aggregate by year
                    yearly_trends = st.session_state.df.groupby('Year')[trend_metrics].mean().reset_index()
                    
                    # Plot trends
                    fig = px.line(
                        yearly_trends,
                        x='Year',
                        y=trend_metrics,
                        title='Trends Over Time',
                        labels={'value': 'Value', 'variable': 'Metric'}
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show correlation matrix
                    st.subheader("Correlation Analysis")
                    corr_matrix = st.session_state.df[trend_metrics].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        title="Correlation Matrix",
                        color_continuous_scale='RdBu'
                    )
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numeric metrics available for trend analysis")
        else:
            st.info("No 'Year' column found for trend analysis")
    
    def _render_ml_predictions(self):
        """Render ML predictions page"""
        st.title("🤖 Machine Learning Predictions")
        
        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first to train ML models")
            st.info("Go to the Data Upload page to load your dataset")
            return
        
        # Train models section
        st.subheader("1. Train Prediction Models")
        
        if not st.session_state.models_trained:
            if st.button("Train ML Models", type="primary"):
                with st.spinner("Training machine learning models..."):
                    st.session_state.cleaner.train_prediction_models()
                    if len(st.session_state.cleaner.models) > 0:
                        st.session_state.models_trained = True
                        st.success(f"✅ {len(st.session_state.cleaner.models)} models trained successfully!")
                        
                        # Show what models were trained
                        st.write("**Trained Models:**")
                        for model_name in st.session_state.cleaner.models.keys():
                            st.write(f"• {model_name}")
                    else:
                        st.warning("⚠️ Could not train models. Check if you have enough data and appropriate columns.")
        else:
            st.success(f"✅ {len(st.session_state.cleaner.models)} models already trained")
            st.write("**Available Models:**")
            for model_name in st.session_state.cleaner.models.keys():
                st.write(f"• {model_name}")
        
        # Prediction interface
        st.subheader("2. Make Predictions")
        
        # Get default values for prediction
        defaults = st.session_state.default_values
        
        # Get available options from data
        country_options = ["Select..."] + sorted(st.session_state.df['Country'].unique().tolist()) if 'Country' in st.session_state.df.columns else ["Select..."]
        disease_options = ["Select..."] + sorted(st.session_state.df['Disease Name'].unique().tolist()) if 'Disease Name' in st.session_state.df.columns else ["Select..."]
        treatment_options = ["Select..."] + sorted(st.session_state.df['Treatment_Type'].unique().tolist()) if 'Treatment_Type' in st.session_state.df.columns else ["Select..."]
        
        # Create input form
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                country = st.selectbox("Country", country_options, index=0)
            
            with col2:
                disease = st.selectbox("Disease", disease_options, index=0)
            
            with col3:
                year = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
            
            # Additional inputs with defaults from data
            st.subheader("Additional Parameters")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                healthcare_access = st.slider(
                    "Healthcare Access (%)", 
                    0.0, 100.0, 
                    float(defaults.get('Healthcare_Access', 75.0))
                )
                
                treatment_type = st.selectbox(
                    "Treatment Type",
                    treatment_options,
                    index=0
                )
            
            with col2:
                doctors_per_1000 = st.slider(
                    "Doctors per 1000", 
                    0.0, 50.0, 
                    float(defaults.get('Doctors_per_1000', 3.0))
                )
                
                avg_treatment_cost = st.number_input(
                    "Avg Treatment Cost (USD)",
                    min_value=0,
                    max_value=1000000,
                    value=int(defaults.get('Avg_Treatment_Cost_USD', 5000))
                )
            
            with col3:
                income = st.number_input(
                    "Per Capita Income (USD)",
                    min_value=0,
                    max_value=200000,
                    value=int(defaults.get('Per_Capita_Income_USD', 30000))
                )
                
                dalys = st.number_input(
                    "DALYs (Disability-Adjusted Life Years)",
                    min_value=0.0,
                    max_value=1000000.0,
                    value=float(defaults.get('DALYs', 1000.0))
                )
            
            # Hidden/default fields that models need
            st.subheader("Additional Model Parameters (Defaults from Data)")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                country_pop = st.number_input(
                    "Country Population",
                    min_value=0,
                    max_value=2000000000,
                    value=int(defaults.get('Country_Population', 10000000)),
                    help="Country total population"
                )
                
                incidence_rate = st.number_input(
                    "Incidence Rate",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(defaults.get('Incidence_Rate', 1.0)),
                    help="Disease incidence rate per million"
                )
            
            with col2:
                prevalence_rate = st.number_input(
                    "Prevalence Rate",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(defaults.get('Prevalence_Rate', 2.0)),
                    help="Disease prevalence rate"
                )
                
                hospital_beds = st.number_input(
                    "Hospital Beds per 1000",
                    min_value=0.0,
                    max_value=50.0,
                    value=float(defaults.get('Hospital_Beds_per_1000', 3.0)),
                    help="Hospital beds availability"
                )
            
            with col3:
                education_index = st.number_input(
                    "Education Index",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(defaults.get('Education_Index', 0.7)),
                    help="Education development index"
                )
                
                urbanization_rate = st.number_input(
                    "Urbanization Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(defaults.get('Urbanization_Rate', 50.0)),
                    help="Percentage of urban population"
                )
            
            # Additional fields for prediction
            col1, col2, col3 = st.columns(3)
            
            with col1:
                recovery_rate = st.number_input(
                    "Recovery Rate (%)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(defaults.get('Recovery_Rate', 50.0)),
                    help="Expected recovery rate"
                )
            
            with col2:
                risk_score = st.number_input(
                    "Risk Score",
                    min_value=0.0,
                    max_value=10.0,
                    value=float(defaults.get('Risk Score', 0.5)),
                    help="Disease risk score"
                )
            
            with col3:
                mortality_rate = st.number_input(
                    "Mortality Rate",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(defaults.get('Mortality_Rate', 0.1)),
                    help="Mortality rate per 100"
                )
            
            # Make prediction button
            submitted = st.form_submit_button("Make Predictions", type="primary")
        
        if submitted:
            if country == "Select..." or disease == "Select...":
                st.error("⚠️ Please select both Country and Disease")
            elif not st.session_state.models_trained:
                st.error("⚠️ Please train ML models first")
            else:
                # Prepare input data with ALL required fields
                input_data = {
                    'Country': country,
                    'Disease Name': disease,
                    'Treatment_Type': treatment_type if treatment_type != "Select..." else "Unknown",
                    'Year': year,
                    'Healthcare_Access': healthcare_access,
                    'Doctors_per_1000': doctors_per_1000,
                    'Per_Capita_Income_USD': income,
                    'Avg_Treatment_Cost_USD': avg_treatment_cost,
                    'DALYs': dalys,
                    'Country_Population': country_pop,
                    'Incidence_Rate': incidence_rate,
                    'Prevalence_Rate': prevalence_rate,
                    'Hospital_Beds_per_1000': hospital_beds,
                    'Education_Index': education_index,
                    'Urbanization_Rate': urbanization_rate,
                    'Recovery_Rate': recovery_rate,
                    'Risk Score': risk_score,
                    'Mortality_Rate': mortality_rate
                }
                
                # Make predictions
                with st.spinner("Making predictions..."):
                    predictions = st.session_state.cleaner.predict(input_data)
                    
                    # Display results
                    st.subheader("📈 Prediction Results")
                    
                    if predictions:
                        # Create columns for predictions
                        num_predictions = len(predictions)
                        cols = st.columns(num_predictions if num_predictions <= 4 else 2)
                        
                        prediction_items = list(predictions.items())
                        for idx, (metric, value) in enumerate(prediction_items):
                            col_idx = idx % len(cols)
                            with cols[col_idx]:
                                if isinstance(value, (int, float)):
                                    # Format based on metric type
                                    if 'Cost' in metric:
                                        formatted_value = f"${value:,.0f}"
                                        delta = None
                                    elif 'Rate' in metric and 'Mortality' in metric:
                                        formatted_value = f"{value:.3f}"
                                        delta = f"{value - defaults.get('Mortality_Rate', 0):.3f}" if 'Mortality_Rate' in defaults else None
                                    elif 'Rate' in metric:
                                        formatted_value = f"{value:.1f}%"
                                        delta = f"{value - defaults.get('Recovery_Rate', 0):.1f}%" if 'Recovery_Rate' in defaults else None
                                    elif 'Score' in metric:
                                        formatted_value = f"{value:.2f}"
                                        delta = f"{value - defaults.get('Risk Score', 0):.2f}" if 'Risk Score' in defaults else None
                                    else:
                                        formatted_value = f"{value:.2f}"
                                        delta = None
                                    
                                    st.metric(
                                        metric.replace('_', ' ').title(),
                                        formatted_value,
                                        delta=delta,
                                        help="Predicted value"
                                    )
                                else:
                                    st.error(f"❌ {value}")
                        
                        # Show explanation
                        with st.expander("💡 How to interpret predictions"):
                            st.write("""
                            **Interpretation Guide:**
                            
                            - **Mortality_Rate**: Predicted deaths per 100 people (lower is better)
                            - **Recovery_Rate**: Predicted recovery percentage (higher is better)
                            - **Avg_Treatment_Cost_USD**: Predicted average treatment cost in USD
                            - **Risk Score**: Composite risk score (0-10, lower is better)
                            
                            **Delta Values:**
                            The delta shows the difference from the dataset average. 
                            Positive delta for Recovery Rate is good, negative for Mortality Rate is good.
                            """)
                    
                                        # Show feature importance if available
                    if hasattr(st.session_state.cleaner, 'models') and 'Mortality_Rate' in st.session_state.cleaner.models:
                        model = st.session_state.cleaner.models['Mortality_Rate']
                        if hasattr(model, 'feature_importances_'):
                            st.subheader("🔍 Feature Importance for Mortality Prediction")
                            
                            # Get feature names
                            feature_names = None
                            
                            # Try to get feature names from model
                            if hasattr(model, 'feature_names_in_'):
                                feature_names = model.feature_names_in_
                            elif 'Mortality_Rate' in st.session_state.cleaner.training_features:
                                feature_names = st.session_state.cleaner.training_features['Mortality_Rate']
                            
                            # Check if we have feature names
                            if feature_names is not None:
                                # Convert to list for consistent handling
                                if isinstance(feature_names, pd.Index):
                                    feature_names = feature_names.tolist()
                                elif isinstance(feature_names, np.ndarray):
                                    feature_names = feature_names.tolist()
                                
                                # Now check if we have any feature names
                                if len(feature_names) > 0:
                                    importances = model.feature_importances_
                                    
                                    # Ensure lengths match
                                    if len(feature_names) == len(importances):
                                        importance_df = pd.DataFrame({
                                            'Feature': feature_names,
                                            'Importance': importances
                                        }).sort_values('Importance', ascending=False).head(10)
                                        
                                        fig = px.bar(
                                            importance_df,
                                            x='Importance',
                                            y='Feature',
                                            orientation='h',
                                            title='Top 10 Most Important Features for Mortality Prediction',
                                            labels={'Importance': 'Relative Importance', 'Feature': 'Feature Name'}
                                        )
                                        fig.update_layout(yaxis={'categoryorder':'total ascending'})
                                        st.plotly_chart(fig, use_container_width=True)
                                    else:
                                        st.info(f"Feature importance: Mismatch between features ({len(feature_names)}) and importances ({len(importances)})")
                                else:
                                    st.info("Feature importance: No feature names available")
                            else:
                                st.info("Feature importance: Feature names not available")
        
        # Clustering analysis
        st.subheader("3. Country Clustering Analysis")
        
        if st.button("🔍 Cluster Countries", type="secondary"):
            with st.spinner("Clustering countries..."):
                country_clusters, cluster_summary = st.session_state.cleaner.cluster_countries()
                
                if country_clusters is not None and not country_clusters.empty:
                    num_clusters = country_clusters['Cluster'].nunique()
                    st.success(f"✅ Countries clustered into {num_clusters} groups")
                    
                    # Display cluster results
                    if 'Healthcare_Access' in country_clusters.columns and 'Mortality_Rate' in country_clusters.columns:
                        fig = px.scatter(
                            country_clusters.reset_index(),
                            x='Healthcare_Access',
                            y='Mortality_Rate',
                            color='Cluster',
                            hover_name='Country',
                            size='Doctors_per_1000' if 'Doctors_per_1000' in country_clusters.columns else None,
                            title=f'Country Clusters by Healthcare Metrics ({num_clusters} Clusters)',
                            labels={
                                'Healthcare_Access': 'Healthcare Access (%)',
                                'Mortality_Rate': 'Mortality Rate',
                                'Cluster': 'Cluster Group'
                            }
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Cluster summary
                    st.subheader("📊 Cluster Characteristics")
                    st.dataframe(cluster_summary.style.format("{:.2f}"), use_container_width=True)
                    
                    # Interpretation
                    with st.expander("📖 Cluster Interpretation Guide"):
                        st.write("""
                        ### How to interpret clusters:
                        
                        **Cluster Analysis:**
                        - **Lower numbered clusters**: Typically better healthcare outcomes
                        - **Higher numbered clusters**: May indicate areas needing improvement
                        - **Cluster size**: Number of countries in each group
                        
                        **Key Metrics:**
                        - **Healthcare_Access**: Percentage of population with healthcare access
                        - **Mortality_Rate**: Deaths per 100 people (lower is better)
                        - **Recovery_Rate**: Recovery percentage (higher is better)
                        
                        **Recommendations:**
                        1. **High-performing clusters** (low mortality, high recovery): Study and replicate best practices
                        2. **Mid-performing clusters**: Target specific improvement areas
                        3. **Low-performing clusters**: Prioritize for international aid and intervention
                        
                        **Action Steps:**
                        - Allocate resources based on cluster needs
                        - Develop targeted interventions for each cluster
                        - Use cluster analysis for policy planning
                        """)
                else:
                    st.warning("⚠️ Could not cluster countries. Make sure you have enough country data with numeric metrics.")
    
    def _render_reports(self):
        """Render reports page"""
        st.title("📄 Reports & Export")
        
        if st.session_state.df is None:
            st.warning("⚠️ Please upload data first to generate reports")
            st.info("Go to the Data Upload page to load your dataset")
            return
        
        # Generate insights if not already done
        if st.session_state.insights is None:
            if st.button("Generate Insights First", type="primary"):
                with st.spinner("Generating insights..."):
                    st.session_state.insights = st.session_state.cleaner.generate_insights()
                    st.success("✅ Insights generated!")
        
        if st.session_state.insights:
            # ADD INSTALLATION BUTTON
            with st.expander("⚙️ Installation & Setup", expanded=False):
                st.markdown("""
                ### Chart Dependencies Setup
                
                For PDF reports with chart visualizations, you need to install the Kaleido package.
                Click the button below to install it automatically, or run manually:
                
                ```bash
                pip install kaleido
                ```
                """)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("🔧 Install Chart Dependencies", type="primary"):
                        with st.spinner("Installing Kaleido package..."):
                            try:
                                import subprocess
                                import sys
                                result = subprocess.check_call([sys.executable, "-m", "pip", "install", "kaleido"])
                                st.success("✅ Kaleido installed successfully! Please restart the app.")
                                st.balloons()
                            except Exception as e:
                                st.error(f"❌ Installation failed: {str(e)}")
                                st.info("""
                                **Manual Installation Options:**
                                1. Open terminal and run: `pip install kaleido`
                                2. Or restart the app with: `streamlit run your_app.py`
                                3. Or use: `python -m pip install kaleido`
                                """)
                
                with col2:
                    if st.button("🔄 Check Installation Status", type="secondary"):
                        try:
                            # Try multiple ways to get version
                            import kaleido
                            
                            # Method 1: Try to get version from kaleido
                            version_info = "Unknown version"
                            try:
                                # Some packages store version differently
                                if hasattr(kaleido, '__version__'):
                                    version_info = kaleido.__version__
                                else:
                                    # Try to get via pkg_resources
                                    import pkg_resources
                                    version_info = pkg_resources.get_distribution("kaleido").version
                            except:
                                # If we can't get version, just confirm it's installed
                                version_info = "Installed"
                            
                            st.success(f"✅ Kaleido is installed ({version_info})")
                            
                        except ImportError:
                            st.warning("⚠️ Kaleido is not installed. Charts won't appear in PDF reports.")
                            st.info("Install it using the button above or run: `pip install kaleido`")
                        except Exception as e:
                            st.warning(f"⚠️ Error checking Kaleido: {str(e)}")
                
                st.markdown("---")
                st.info("""
                **Note:** 
                - PDF reports will still work without Kaleido, but charts won't be included
                - HTML reports will always show interactive charts
                - Email reports will include charts if Kaleido is installed
                """)
            
            
            
                        # Report generator with multiple options
            st.subheader("📊 Generate Reports")
            
            # Create tabs for different report formats
            tab1, tab2, tab3 = st.tabs(["📄 PDF Reports", "🌐 HTML Reports", "📧 Email"])
            
            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Professional PDF (Tables)**")
                    
                    # Compact mode option
                    compact_mode = st.checkbox(
                        "Compact Mode",
                        value=True,
                        help="Reduces white space for denser layout"
                    )
                    
                    include_all = st.checkbox(
                        "Include All Metrics",
                        value=True,
                        help="Show all metrics vs. top 15"
                    )
                    
                    if st.button("📋 Generate ReportLab PDF", type="primary"):
                        with st.spinner("Generating professional PDF..."):
                            report_gen = HealthReportGenerator(st.session_state.df, st.session_state.insights)
                            pdf_file = report_gen.generate_pdf_reportlab(
                                compact_mode=compact_mode,
                                include_all_metrics=include_all
                            )
                            
                            with open(pdf_file, "rb") as f:
                                st.download_button(
                                    label="📥 Download ReportLab PDF",
                                    data=f,
                                    file_name=pdf_file,
                                    mime="application/pdf"
                                )
            
                
                with col2:
                    st.markdown("**PDF with Charts**")
                    st.markdown("*Requires Kaleido package for charts*")
                    if st.button("📊 Generate PDF with Charts", type="secondary",
                                help="PDF with embedded charts (requires Kaleido)"):
                        with st.spinner("Generating PDF with charts..."):
                            report_gen = HealthReportGenerator(st.session_state.df, st.session_state.insights)
                            pdf_file = report_gen.generate_pdf_reportlab()
                            
                            with open(pdf_file, "rb") as f:
                                st.download_button(
                                    label="📥 Download PDF with Charts",
                                    data=f,
                                    file_name=pdf_file,
                                    mime="application/pdf"
                                )
            
            with tab2:
                
                if st.button("🌐 Generate Interactive HTML Report", type="secondary",
                            help="Generate interactive HTML report with Plotly charts"):
                    with st.spinner("Generating interactive HTML report..."):
                        report_gen = HealthReportGenerator(st.session_state.df, st.session_state.insights)
                        html_file = report_gen.generate_html_report()
                        
                        with open(html_file, "r", encoding='utf-8') as f:
                            st.download_button(
                                label="📥 Download HTML Report",
                                data=f.read(),
                                file_name=html_file,
                                mime="text/html"
                            )
            
            
            
            
            with tab3:
               
            
            
                                                # Email reporting section
                st.subheader("📧 Email Report")
                
                with st.expander("Configure Email Settings", expanded=False):
                    st.info("💡 **Tip:** For Gmail, use an 'App Password' if you have 2FA enabled.")
                    
                    # Configuration form
                    with st.form("email_config_form"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            sender_email = st.text_input(
                                "Sender Email", 
                                placeholder="your.email@gmail.com",
                                help="The email address that will send the report"
                            )
                            
                            smtp_server = st.text_input(
                                "SMTP Server", 
                                value="smtp.gmail.com",
                                help="SMTP server address"
                            )
                        
                        with col2:
                            recipient_email = st.text_input(
                                "Recipient Email", 
                                placeholder="recipient@example.com",
                                help="Where to send the report"
                            )
                            
                            smtp_port = st.number_input(
                                "SMTP Port", 
                                value=587,
                                min_value=1,
                                max_value=9999,
                                help="Common ports: 587 (TLS), 465 (SSL)"
                            )
                        
                        # Password field
                        sender_password = st.text_input(
                            "Email Password / App Password", 
                            type="password",
                            placeholder="Enter your email password or app password"
                        )
                        
                        # Data attachment options
                        st.subheader("📁 Data Attachment Options")
                        
                        col3, col4 = st.columns(2)
                        
                        with col3:
                            include_data = st.checkbox(
                                "Include cleaned data file",
                                value=True,
                                help="Attach the cleaned dataset to the email"
                            )
                        
                        with col4:
                            if include_data:
                                data_format = st.selectbox(
                                    "Data Format",
                                    options=['csv', 'excel', 'json'],
                                    index=0,
                                    help="Format for the data file attachment"
                                )
                        
                        # DATA PREVIEW
                        if include_data and st.session_state.df is not None:
                            with st.expander("👁️ Data Preview (first 5 rows)", expanded=False):
                                st.dataframe(
                                    st.session_state.df.head(5),
                                    use_container_width=True,
                                    hide_index=True
                                )
                                st.caption(f"Total: {len(st.session_state.df):,} rows, {len(st.session_state.df.columns)} columns")
                        
                        # INFO BOX ABOUT DATA SIZE
                        if include_data:
                            st.info(f"""
                            **Data file will be included:**
                            - Format: {data_format.upper()}
                            - Records: {len(st.session_state.df):,}
                            - Columns: {len(st.session_state.df.columns)}
                            - Size: ~{(len(st.session_state.df) * len(st.session_state.df.columns) * 0.01):.1f} MB estimated
                            """)
                        
                        # Submit button
                        submitted = st.form_submit_button("📤 Send Report via Email", type="primary")
                    
                    # Handle form submission
                    if submitted:
                        # Validate inputs
                        if not all([sender_email, recipient_email, sender_password]):
                            st.error("⚠️ Please fill in all required fields: Sender Email, Recipient Email, and Password")
                        else:
                            # Validate email format
                            import re
                            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                            
                            if not re.match(email_pattern, sender_email):
                                st.error("❌ Invalid sender email format")
                            elif not re.match(email_pattern, recipient_email):
                                st.error("❌ Invalid recipient email format")
                            else:
                                # Send the email
                                with st.spinner("Generating reports and sending email..."):
                                    report_gen = HealthReportGenerator(st.session_state.df, st.session_state.insights)
                                    
                                    # Show what's being sent
                                    attachments = ["PDF Report", "HTML Report"]
                                    if include_data:
                                        attachments.append(f"Data File ({data_format.upper()})")
                                    
                                    st.info(f"📧 Sending email with attachments: {', '.join(attachments)}")
                                    
                                    success, message = report_gen.send_email_report(
                                        recipient=recipient_email,
                                        sender=sender_email,
                                        password=sender_password,
                                        smtp_server=smtp_server,
                                        smtp_port=smtp_port,
                                        include_data=include_data,
                                        data_format=data_format if include_data else 'csv'
                                    )
                                    
                                    if success:
                                        st.success(message)
                                        st.balloons()
                                        
                                        # Show summary
                                        with st.expander("📦 Email Summary", expanded=True):
                                            st.write(f"**To:** {recipient_email}")
                                            st.write(f"**Subject:** Global Health Analytics Report")
                                            st.write(f"**Attachments sent:**")
                                            st.write(f"1. 📄 PDF Report (professional formatting)")
                                            st.write(f"2. 🌐 HTML Report (interactive charts)")
                                            if include_data:
                                                st.write(f"3. 📁 Data File ({data_format.upper()} format)")
                                                st.write(f"   - Records: {len(st.session_state.df):,}")
                                                st.write(f"   - Columns: {len(st.session_state.df.columns)}")
                                    
                                    else:
                                        st.error(f"❌ {message}")
                                        
                                        # Show troubleshooting tips
                                        with st.expander("🔧 Troubleshooting Tips", expanded=False):
                                            st.write("""
                                            **Common solutions:**
                                            1. **For Gmail:** Use app password instead of regular password
                                            2. **Check password:** Ensure you're using the correct password
                                            3. **Verify SMTP settings:** 
                                            - Gmail: smtp.gmail.com, Port 587
                                            - Outlook: smmtp-mail.outlook.com, Port 587
                                            - Yahoo: smtp.mail.yahoo.com, Port 465
                                            4. **File size limits:** Large data files may exceed email limits
                                            5. **Network:** Check firewall/antivirus settings
                                            """)
                    
                    # Quick configuration examples
                    with st.expander("⚡ Quick Configuration Examples", expanded=False):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write("**Gmail:**")
                            st.code("""
    SMTP: smtp.gmail.com
    Port: 587
    Use App Password if 2FA enabled
                            """)
                        
                        with col2:
                            st.write("**Outlook:**")
                            st.code("""
    SMTP: smtp-mail.outlook.com
    Port: 587
    Use your Microsoft password
                            """)
                        
                        with col3:
                            st.write("**Yahoo:**")
                            st.code("""
    SMTP: smtp.mail.yahoo.com
    Port: 465
    Use your Yahoo password
                            """)
            
            # Data export
            st.subheader("💾 Export Cleaned Data")
            
            export_format = st.selectbox(
                "Select export format",
                ["CSV", "Excel", "SQLite", "JSON"]
            )
            
            if st.button(f"Export as {export_format}", type="primary"):
                with st.spinner(f"Exporting data as {export_format}..."):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    if export_format == "CSV":
                        filename = f"health_data_export_{timestamp}.csv"
                        st.session_state.df.to_csv(filename, index=False)
                        
                        with open(filename, "rb") as f:
                            st.download_button(
                                label="📥 Download CSV",
                                data=f,
                                file_name=filename,
                                mime="text/csv"
                            )
                    
                    elif export_format == "Excel":
                        filename = f"health_data_export_{timestamp}.xlsx"
                        st.session_state.df.to_excel(filename, index=False)
                        
                        with open(filename, "rb") as f:
                            st.download_button(
                                label="📥 Download Excel",
                                data=f,
                                file_name=filename,
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    elif export_format == "SQLite":
                        filename = f"health_data_export_{timestamp}.db"
                        conn = sqlite3.connect(filename)
                        st.session_state.df.to_sql('health_data', conn, if_exists='replace', index=False)
                        conn.close()
                        
                        with open(filename, "rb") as f:
                            st.download_button(
                                label="📥 Download SQLite",
                                data=f,
                                file_name=filename,
                                mime="application/x-sqlite3"
                            )
                    
                    elif export_format == "JSON":
                        filename = f"health_data_export_{timestamp}.json"
                        st.session_state.df.to_json(filename, orient='records', indent=2)
                        
                        with open(filename, "r", encoding='utf-8') as f:
                            st.download_button(
                                label="📥 Download JSON",
                                data=f.read(),
                                file_name=filename,
                                mime="application/json"
                            )
        else:
            st.warning("Please generate insights first using the button above")
    
    def _render_about(self):
        """Render about page"""
        st.title("ℹ️ About This Application")
        
        st.write("""
        ## Global Health Analytics Platform
        
        ### Overview
        This platform provides comprehensive tools for analyzing global health data,
        including data cleaning, visualization, machine learning predictions, and
        automated reporting.
        
        ### Features
        
        #### 1. **Data Management**
        - Support for multiple file formats (CSV, Excel, SQLite)
        - Automated data cleaning and preprocessing
        - Handling of missing values and outliers
        
        #### 2. **Interactive Analysis**
        - Real-time data visualization
        - Geographical analysis
        - Trend analysis over time
        - Correlation analysis
        
        #### 3. **Machine Learning**
        - Predictive modeling for mortality rates
        - Recovery rate predictions
        - Treatment cost forecasting
        - Country clustering based on health metrics
        
        #### 4. **Reporting**
        - Automated PDF report generation
        - Interactive HTML reports
        - Email distribution of reports
        - Multiple export formats
        
        ### Technical Stack
        - **Backend**: CSS, Python, Pandas, NumPy
        - **Machine Learning**: Scikit-learn, XGBoost
        - **Visualization**: Plotly, Streamlit
        - **Reporting**: ReportLab, FPDF
        - **Email**: yagmail
        
        ### Usage Instructions
        1. **Upload Data**: Navigate to the Data Upload page and upload your health data
        2. **Explore**: Use the Data Analysis page to visualize and understand your data
        3. **Predict**: Make predictions using trained ML models
        4. **Report**: Generate and export comprehensive reports
        
        ### Data Privacy
        - All data processing happens locally
        - No data is sent to external servers
        - You maintain full control of your data
        
        ### Support
        For issues, questions, or feature requests, please contact the development team.
        
        ---
        
        **Version**: 4.0.0  
        **Last Updated**: December 2025  
        **Developer**: Chiagoziem Cyriacus Ugoh
        """)


# ======================================================================
# 4. MAIN EXECUTION
# ======================================================================

def main():
    """Main execution function"""
    app = HealthDashboardApp()
    app.run()



if __name__ == "__main__":
    main()