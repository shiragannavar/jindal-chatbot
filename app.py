import streamlit as st
import pandas as pd
import numpy as np
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits import create_json_agent
from langchain.tools import tool
import os
from dotenv import load_dotenv
import datetime
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
import plotly.express as px
import plotly.graph_objects as go
import re

from typing import List, Dict, Any, Literal, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

# Pydantic Models for Orchestrator Plan
# -------------------------------------

class DataRetrieverInputs(BaseModel):
    tool_to_call: str = Field(description="The exact name of the tool to call (e.g., 'get_data_for_category', 'list_available', 'get_current_date').")
    tool_input_json: str = Field(description="The JSON string input for the specified tool.")

class FinancialAnalysisInputs(BaseModel):
    data_to_analyze: Any = Field(description="Data from a previous step (e.g., JSON string, list of dicts). Placeholder like '[data_from_step_X]' will be used by orchestrator initially.")
    analysis_task: str = Field(description="Specific question or analysis to perform on the data.")

class VisualizationInputs(BaseModel):
    data_to_visualize: Any = Field(description="Data from a previous step. Placeholder like '[data_from_step_X]' will be used by orchestrator initially.")
    chart_type: Literal["line", "bar", "pie", "scatter"] = Field(description="Type of chart to generate.")
    x_axis: str = Field(description="Key from data dictionaries to use for the x-axis.")
    y_axis: str = Field(description="Key from data dictionaries to use for the y-axis (numeric values).")
    title: str = Field(description="Title for the chart.")
    is_financial: Optional[bool] = Field(default=True, description="If true, formats y-axis and hover data with currency.")

class TOCStrategyInputs(BaseModel):
    analyzed_data: Any = Field(description="Analyzed data or insights from a previous step. Placeholder like '[analysis_from_step_X]' will be used by orchestrator initially.")
    strategic_question: str = Field(description="Specific strategic question to address based on the analyzed data.")

class FinalResponseCompilationInputs(BaseModel):
    summary_points: List[str] = Field(description="List of key information points or outputs from previous steps (can be placeholders like '[output_from_step_X]').")
    user_query: str = Field(description="The original user query for context.")

class Step(BaseModel):
    step_id: int = Field(description="An integer, starting from 1, identifying the step order.")
    persona: Literal["DataRetrieverMode", "FinancialAnalysisMode", "VisualizationMode", "TOCStrategyMode", "FinalResponseCompilationMode"] = Field(description="The specialized persona/mode required for this step.")
    goal: str = Field(description="A concise description of what this step aims to achieve.")
    # Using Dict[str, Any] for inputs for now, relying on the orchestrator prompt to guide structure per persona.
    # For stricter validation, a Union of the above input types with a discriminator would be needed,
    # which can complicate JsonOutputParser setup if not using PydanticOutputFunctionsParser.
    inputs: Dict[str, Any] = Field(description="An object containing the necessary inputs for this step, structured according to the 'persona' and its specific input model (e.g., DataRetrieverInputs, FinancialAnalysisInputs).")
    outputs_expected: str = Field(description="A brief description of what this step should produce.")
    depends_on_step: Optional[Union[int, List[int]]] = Field(default=None, description="The step_id (or list of step_ids) of previous steps whose outputs are required as input for this step, or null if no dependency.")

class ExecutionPlan(BaseModel):
    steps: List[Step] = Field(description="A list of step objects representing the execution plan.")

# -------------------------------------

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Jindal Steel TOC Advisor",
    page_icon="🏭",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.assistant {
        background-color: #475063;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "json_data" not in st.session_state:
    st.session_state.json_data = None
if "agent_memory" not in st.session_state: # Renamed for single agent
    st.session_state.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "last_chart_figure_json" not in st.session_state:
    st.session_state.last_chart_figure_json = None

def clean_and_validate_data(data: dict, llm) -> dict:
    """Clean and validate the data using LLM.
    This function is designed to be light for now to avoid token limits.
    """
    try:
        # st.write("Attempting LLM-based data validation (light check)...")
        # Minimal check to avoid large token usage for now
        if not isinstance(data, dict) or not data:
            # st.warning("Data for validation is empty or not a dictionary.")
            return {}

        # Example: Check for presence of main categories
        expected_categories = ["OVERALL", "RAIGARH", "ANGUL"]
        issues = []
        for cat in expected_categories:
            if cat not in data:
                issues.append(f"Missing expected category: {cat}")

        if issues:
            st.warning("Data validation issues found during light check:")
            for issue in issues:
                st.warning(f"- {issue}")
        # else:
            # st.write("Basic data structure validation passed.")
        return data # Return original data for now, focusing on extraction
            
    except Exception as e:
        st.error(f"Error in light data validation: {str(e)}")
        return data

def map_category(category: str) -> str:
    """Maps user-friendly category names to internal category names."""
    if not isinstance(category, str): # Handle non-string inputs gracefully
        return "OVERALL" # Or some other default/error indicator

    category_upper = category.upper()
    category_mapping = {
        "JSPL": "OVERALL",
        "JINDAL": "OVERALL",
        "JINDAL STEEL": "OVERALL",
        "JINDAL STEEL AND POWER": "OVERALL",
        "JINDAL STEEL AND POWER LIMITED": "OVERALL",
        "OVERALL": "OVERALL",
        "RAIGARH": "RAIGARH",
        "ANGUL": "ANGUL"
    }
    return category_mapping.get(category_upper, category_upper) # Default to original if not in map

def excel_to_json(uploaded_file):
    df = pd.read_excel(
        uploaded_file,
        engine='openpyxl',
        header=None, # Read all rows without assuming a header row initially
        skiprows=0, 
        dtype=object  # Use object type to prevent automatic conversions
    )
    
    try:
        st.write("Debug: Raw Excel data (first 20 rows for context):")
        st.text(df.head(20).to_string()) # Display more rows for debugging
        
        date_header_row_index = -1
        date_headers = []

        # Attempt to dynamically find the date header row
        # Look for rows with multiple potential date-like strings or keywords
        potential_header_keywords = ["date", "period", "week"] # Add more if needed
        for i, row_series in df.head(15).iterrows(): # Check first 15 rows
            row_values = [str(v).strip().lower() for v in row_series.iloc[1:] if pd.notna(v) and str(v).strip() != ""]
            # Check if a significant number of cells look like date headers or contain keywords
            date_like_cells = 0
            possible_dates_in_row = []
            for cell_val in row_series.iloc[1:]: # Check from column B onwards
                if pd.notna(cell_val) and str(cell_val).strip() != "":
                    s_cell_val = str(cell_val).strip()
                    # Heuristic: short, non-numeric, may contain '-', '/', or month names
                    if len(s_cell_val) < 25 and (any(kw in s_cell_val.lower() for kw in potential_header_keywords) or \
                       ("-" in s_cell_val or "/" in s_cell_val or any(mon.lower() in s_cell_val.lower() for mon in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]))):
                        date_like_cells += 1
                        possible_dates_in_row.append(s_cell_val)

            if date_like_cells > 2 : # If at least 3 cells in the row look like date headers
                date_header_row_index = i
                date_headers = [str(h).strip() for h in df.iloc[i, 1:] if pd.notna(h) and str(h).strip() != ""]
                # Further filter out any non-sensical headers (e.g. very long strings, or purely numeric if not expected)
                date_headers = [h for h in date_headers if len(h) < 30 and not (h.replace('.','',1).isdigit() and len(h) > 7) ]
                if len(date_headers) > 2: # Ensure we still have enough valid-looking date headers
                    st.success(f"Dynamically identified date header row at Excel row {date_header_row_index + 1}: {date_headers}")
                    break
                else: # Reset if not enough good headers found
                    date_header_row_index = -1
                    date_headers = []


        if date_header_row_index == -1:
            # Fallback to original heuristic if dynamic detection fails
            st.warning("Dynamic date header detection failed. Falling back to default row 6 (index 5).")
            date_header_row_index = 5 
            if date_header_row_index >= len(df):
                st.error(f"Default date header row index {date_header_row_index + 1} is out of bounds. Excel sheet has {len(df)} rows.")
                return None
            date_headers_raw = df.iloc[date_header_row_index, 1:]
            date_headers = [str(h).strip() for h in date_headers_raw if pd.notna(h) and str(h).strip() != ""]
            if not date_headers:
                 st.error(f"No date headers found in default Excel row {date_header_row_index + 1}. Please check Excel file.")
                 return None
            st.write(f"Using date headers from default Excel row {date_header_row_index + 1}: {date_headers}")

        
        result = {
            "OVERALL": {"dates": {}, "subcategories": []},
            "RAIGARH": {"dates": {}, "subcategories": []},
            "ANGUL": {"dates": {}, "subcategories": []}
        }
        
        current_main_category = "OVERALL"
        subcategories_map = { cat: {} for cat in result.keys() }
        
        data_start_row_index = date_header_row_index + 1
        st.write(f"Debug: Starting data processing from Excel row {data_start_row_index + 1}.")
        st.write(f"Debug: Default main category set to OVERALL until another category header is found.")

        value_tracking = pd.DataFrame(columns=["Category", "Date", "Subcategory", "Value", "Raw_Value", "Row", "Col", "Error"])
        
        for i, row_series in df.iloc[data_start_row_index:].iterrows():
            row_values = row_series.values
            
            if row_series.isnull().all():
                continue

            category_name_raw = str(row_values[0]).strip() if pd.notna(row_values[0]) else ""
            
            if not category_name_raw:
                continue

            if category_name_raw.upper() in result:
                current_main_category = category_name_raw.upper()
                st.write(f"Debug: Switched to main category: {current_main_category} at Excel row {i+1} (Value: '{category_name_raw}')")
                continue 

            subcategory_name = category_name_raw
            if not subcategory_name:
                continue
                
            subcategories_map[current_main_category][subcategory_name] = True
            
            if subcategory_name in ["Net Revenue", "Cash Score"]:
                st.write(f"Debug: Processing important subcategory '{subcategory_name}' under '{current_main_category}' from Excel row {i+1}")

            for j, date_str in enumerate(date_headers):
                col_idx = j + 1 
                
                if col_idx < len(row_values):
                    cell_value_raw = row_values[col_idx]
                    
                    if pd.isna(cell_value_raw):
                        continue 
                        
                    cell_value_str = str(cell_value_raw).strip()
                    
                    if not cell_value_str or cell_value_str == "-" or cell_value_str.lower() == "na":
                        continue
                    
                    value = cell_value_str # Default to string if conversion fails
                    conversion_error = None
                    try:
                        # Strengthened numeric value extraction
                        # Handles: ₹, $, commas, Cr, %, potential parentheses for negatives
                        cleaned_value_str = cell_value_str.replace('₹', '').replace('$', '').replace(',', '').replace('%', '').strip()
                        
                        is_negative = False
                        if cleaned_value_str.startswith('(') and cleaned_value_str.endswith(')'):
                            is_negative = True
                            cleaned_value_str = cleaned_value_str[1:-1]
                        
                        # Handle "Cr" suffix, case-insensitive
                        if cleaned_value_str.lower().endswith('cr'):
                            cleaned_value_str = cleaned_value_str[:-2].strip()
                        
                        # Remove any remaining non-numeric characters except decimal point and minus sign (if not already handled by parentheses)
                        # This is a bit aggressive, ensure it doesn't break valid numbers
                        # cleaned_value_str = re.sub(r"[^0-9.-]", "", cleaned_value_str) # Might be too broad

                        if cleaned_value_str: # Ensure not empty after cleaning
                            value = float(cleaned_value_str)
                            if is_negative:
                                value = -value
                        else: # if string becomes empty after cleaning, it wasn't a valid number
                            conversion_error = "Value became empty after cleaning"
                            value = cell_value_str # Revert to original string

                    except ValueError as ve:
                        conversion_error = f"ValueError: {str(ve)}"
                        value = cell_value_str # Keep as string if conversion fails
                    except Exception as e_conv:
                        conversion_error = f"General Error: {str(e_conv)}"
                        value = cell_value_str


                    # Add to tracking for important metrics or if there was an error
                    if subcategory_name in ["Net Revenue", "Cash Score"] or conversion_error:
                        if isinstance(value, float): # Log successful conversion for important metrics
                             st.write(f"Debug: [{current_main_category}] Excel row {i+1}, col {col_idx+1}: Converted '{cell_value_str}' to numeric: {value}")
                        
                        value_tracking = pd.concat([value_tracking, pd.DataFrame({
                            "Category": [current_main_category], "Date": [date_str],
                            "Subcategory": [subcategory_name], "Value": [value if isinstance(value, float) else None],
                            "Raw_Value": [cell_value_str], "Row": [i+1], "Col": [col_idx+1],
                            "Error": [conversion_error]
                        })], ignore_index=True)
                        if conversion_error and subcategory_name not in ["Net Revenue", "Cash Score"]: # Log errors for other metrics too
                            st.warning(f"Debug: [{current_main_category}] Excel row {i+1}, col {col_idx+1}: Failed to convert '{cell_value_str}'. Kept as string. Error: {conversion_error}")
                        
                    if date_str not in result[current_main_category]["dates"]:
                        result[current_main_category]["dates"][date_str] = {}
                    
                    result[current_main_category]["dates"][date_str][subcategory_name] = value
                    
                    if subcategory_name in ["Net Revenue", "Cash Score"] and isinstance(value, float):
                        st.write(f"Debug: Successfully stored {current_main_category} | {date_str} | {subcategory_name} = {value} (type: {type(value).__name__})")

        for category_key in result:
            result[category_key]["subcategories"] = sorted(list(subcategories_map[category_key].keys()))

        st.write("Debug: Value conversion tracking (includes errors and important metrics):")
        if not value_tracking.empty:
            st.dataframe(value_tracking)
        else:
            st.info("No values tracked for conversion (e.g. no errors for non-critical metrics, or critical metrics not found).")
        
        # Final structure validation
        st.write("Debug: Final JSON structure preview (post-processing):")
        for cat in result:
            if not result[cat]["dates"] and not result[cat]["subcategories"]:
                st.warning(f"Category '{cat}' appears empty or was not found/processed correctly.")
                continue
            st.write(f"Category: {cat}")
            st.write(f"  Subcategories ({len(result[cat]['subcategories'])}): {result[cat]['subcategories'][:5]}...")
            st.write(f"  Dates found ({len(result[cat]['dates'])}): {list(result[cat]['dates'].keys())[:3]}...")
            if result[cat]['dates']:
                first_date_key = list(result[cat]['dates'].keys())[0]
                sample_data = result[cat]['dates'][first_date_key]
                st.write(f"  Sample data for '{first_date_key}' ({len(sample_data)} items): {dict(list(sample_data.items())[:3])}...")
                
                # Extra debugging for specific key metrics
                for metric in ["Cash Score", "Net Revenue"]:
                    if metric in sample_data:
                        st.write(f"  Important metric - {cat} | {first_date_key} | {metric} = {sample_data[metric]} (type: {type(sample_data[metric]).__name__})")

        if not any(result[cat]['dates'] for cat in result if result[cat]):
            st.error("CRITICAL: No date-specific data was extracted for ANY category. This indicates a fundamental issue with Excel structure recognition.")
            return None
            
        return result
    except Exception as e:
        st.error(f"Major error in excel_to_json: {str(e)}")
        st.error(f"DataFrame info at error: Shape={df.shape if 'df' in locals() else 'N/A'}")
        import traceback
        st.error(traceback.format_exc())
        return None

@tool
def get_data_for_category(query: str = "{\"category\": \"OVERALL\"}") -> str:
    """Fetches data for a specified category, and optionally for a specific date and/or subcategory.

    Args:
        query (str): A JSON string specifying the data to fetch. 
                     Must include 'category'. 
                     Optional keys: 'date', 'subcategory'.

    Input JSON Structure Examples:
        - To get general info for a category (available dates & subcategories):
          '{{\"category\": \"OVERALL\"}}'
        - To get data for all subcategories for a category on a specific date:
          '{{\"category\": \"RAIGARH\", \"date\": \"25th-31st Jul 24\"}}'
        - To get a specific subcategory for a category on a specific date:
          '{{\"category\": \"ANGUL\", \"date\": \"last week\", \"subcategory\": \"Cash Score\"}}'

    Output JSON Structure (Success):
        - If only category is provided (general info):
          {{\"category\": \"OVERALL\", \"available_dates\": [\"date1\", \"date2\", ...], \"available_subcategories\": [\"subcat1\", \"subcat2\", ...]}}
        - If category and date are provided (all subcategories for that date):
          {{\"category\": \"RAIGARH\", \"date\": \"matched_date_string\", \"values\": {{\"subcat1\": 123.45, \"subcat2\": 67.89}}}}
          (Note: numeric values are floats, non-numeric are strings)
        - If category, date, and subcategory are provided (specific value):
          {{\"category\": \"ANGUL\", \"date\": \"matched_date_string\", \"subcategory\": \"Cash Score\", \"value\": 150.75}}
          (Note: numeric value is a float, non-numeric is a string)

    Output JSON Structure (Error):
        {{\"error\": \"Descriptive error message\"}}
    """
    try:
        if not query:
            return json.dumps({"error": "Query parameter is required. Please provide a JSON string with at least a category parameter."})
            
        params = json.loads(query)
        category = map_category(params.get("category", "")) # map_category should handle variations
        date_query = params.get("date")
        subcategory_query = params.get("subcategory")

        st.write(f"Debug get_data_for_category: Processing query: category='{category}', date='{date_query}', subcategory='{subcategory_query}'")

        if not category:
            return json.dumps({"error": "Category is required in the query."})

        data = st.session_state.json_data
        if not data or category not in data:
            return json.dumps({"error": f"Category '{category}' not found in processed data."})

        category_data = data[category]

        if date_query:
            # Date parsing/matching needs to be robust. find_closest_date should handle this.
            # It should be able to understand "last week", "last 3 months" or specific week ranges.
            closest_dates = find_closest_date(category_data.get("dates", {}), date_query)
            st.write(f"Debug get_data_for_category: Date query '{date_query}' mapped to: '{closest_dates}'")

            if not closest_dates:
                available_dates = list(category_data.get("dates", {}).keys())
                error_msg = f"No data found for date query '{date_query}' in category '{category}'."
                if available_dates:
                    error_msg += f" Available date periods include: {available_dates[:5]}"
                    if len(available_dates) > 5: error_msg += "..."
                else:
                    error_msg += " No date periods are available for this category."
                return json.dumps({"error": error_msg})
            
            # Handle single date vs multiple dates
            if isinstance(closest_dates, list):
                # Multiple dates - return data for all of them
                if subcategory_query:
                    # Return specific subcategory values for all dates
                    results = []
                    for date in closest_dates:
                        if date in category_data["dates"]:
                            date_specific_data = category_data["dates"][date]
                            if subcategory_query in date_specific_data:
                                value = date_specific_data[subcategory_query]
                                # Ensure numeric values are returned as numbers
                                if isinstance(value, str):
                                    try:
                                        clean_value = value.replace(',', '').replace('₹', '').replace('%', '').replace(' Cr', '').strip()
                                        value = float(clean_value)
                                    except ValueError:
                                        pass
                                results.append({
                                    "date": date,
                                    "value": value
                                })
                    
                    if results:
                        return json.dumps({
                            "category": category,
                            "subcategory": subcategory_query,
                            "period": date_query,
                            "data": results,
                            "summary": {
                                "count": len(results),
                                "total": sum(r["value"] for r in results if isinstance(r["value"], (int, float))),
                                "average": sum(r["value"] for r in results if isinstance(r["value"], (int, float))) / len([r for r in results if isinstance(r["value"], (int, float))]) if any(isinstance(r["value"], (int, float)) for r in results) else None
                            }
                        })
                    else:
                        return json.dumps({"error": f"Subcategory '{subcategory_query}' not found in any of the {len(closest_dates)} matching dates for '{date_query}'"})
                else:
                    # Return all data for all dates in the period
                    results = {}
                    for date in closest_dates:
                        if date in category_data["dates"]:
                            # Process all values to ensure numeric conversion
                            processed_data = {}
                            for subcat, val in category_data["dates"][date].items():
                                if isinstance(val, str):
                                    try:
                                        clean_val = val.replace(',', '').replace('₹', '').replace('%', '').replace(' Cr', '').strip()
                                        processed_data[subcat] = float(clean_val)
                                    except ValueError:
                                        processed_data[subcat] = val
                                else:
                                    processed_data[subcat] = val
                            results[date] = processed_data
                    
                    return json.dumps({
                        "category": category,
                        "period": date_query,
                        "dates_found": len(results),
                        "data": results
                    })
            else:
                # Single date - existing logic
                closest_date = closest_dates
                date_specific_data = category_data["dates"][closest_date]
                if subcategory_query:
                    if subcategory_query in date_specific_data:
                        value = date_specific_data[subcategory_query]
                        st.write(f"Debug get_data_for_category: Found value for {category} | {closest_date} | {subcategory_query} = {value} (type: {type(value).__name__})")
                        # Ensure numeric values are returned as numbers, not strings
                        if isinstance(value, str):
                            try:
                                # More robust handling of numeric string formats (commas, currency symbols, etc.)
                                clean_value = value.replace(',', '').replace('₹', '').replace('%', '').replace(' Cr', '').strip()
                                value = float(clean_value)
                                st.write(f"Debug get_data_for_category: Converted string value '{clean_value}' to number: {value}")
                            except ValueError:
                                # Keep as string if conversion fails
                                st.write(f"Debug get_data_for_category: Could not convert '{value}' to number, keeping as string")
                                pass
                        
                        return json.dumps({
                            "category": category,
                            "date": closest_date, # Return the matched date
                            "subcategory": subcategory_query,
                            "value": value
                        })
                    else:
                        available_subcats = list(date_specific_data.keys())
                        error_msg = f"Subcategory '{subcategory_query}' not found for '{category}' on '{closest_date}'."
                        if available_subcats:
                             error_msg += f" Available subcategories for this date include: {available_subcats[:5]}"
                             if len(available_subcats) > 5: error_msg += "..."
                        return json.dumps({"error": error_msg})
                else: # Return all subcategories for this date
                    # Ensure all numeric values are returned as numbers
                    processed_data = {}
                    for subcat, val in date_specific_data.items():
                        if isinstance(val, str):
                            try:
                                # More thorough cleaning of formatted strings with currency symbols, commas, etc.
                                clean_val = val.replace(',', '').replace('₹', '').replace('%', '').replace(' Cr', '').strip()
                                processed_data[subcat] = float(clean_val)
                                st.write(f"Debug get_data_for_category: Converted string '{val}' to number: {processed_data[subcat]}")
                            except ValueError:
                                processed_data[subcat] = val
                                st.write(f"Debug get_data_for_category: Could not convert '{val}' to number, keeping as string")
                        else:
                            processed_data[subcat] = val
                    
                    return json.dumps({
                        "category": category,
                        "date": closest_date,
                        "values": processed_data
                    })
        else: # No date specified, return general category info
            return json.dumps({
                "category": category,
                "available_dates": list(category_data.get("dates", {}).keys()),
                "available_subcategories": category_data.get("subcategories", [])
            })
            
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input format for query_category_data."})
    except Exception as e:
        return json.dumps({"error": f"Error in query_category_data: {str(e)}"})

@tool
def compare_category_data(query: str = "{\"category1\": \"OVERALL\", \"category2\": \"RAIGARH\"}") -> str:
    """Compares data between two specified categories, optionally for a specific date and/or subcategory.

    Args:
        query (str): A JSON string specifying the comparison parameters.
                     Must include 'category1' and 'category2'.
                     Optional keys: 'date', 'subcategory'.

    Input JSON Structure Examples:
        - To compare a specific subcategory for two categories on a specific date:
          '{{\"category1\": \"RAIGARH\", \"category2\": \"ANGUL\", \"date\": \"last week\", \"subcategory\": \"Cash Score\"}}'
        - To compare all common subcategories for two categories on a specific date:
          '{{\"category1\": \"OVERALL\", \"category2\": \"RAIGARH\", \"date\": \"25th-31st Jul 24\"}}'
        - If date is omitted, the tool will return an error recommending a date for meaningful comparison.

    Output JSON Structure (Success):
        - If comparing a specific subcategory on a specific date:
          {{\"comparison_on_date\": \"matched_date_string\", \"subcategory\": \"Cash Score\", \"<category1_name>\": 150.75, \"<category2_name>\": 160.25}}
          (e.g., "OVERALL": 150.75, "RAIGARH": 160.25)
        - If comparing all common subcategories on a specific date:
          {{\"comparison_on_date\": \"matched_date_string\", \"data\": {{\"<category1_name>\": {{\"subcatA\": 12, \"subcatB\": 20}}, \"<category2_name>\": {{\"subcatA\": 15, \"subcatB\": 18}}}}}}

    Output JSON Structure (Error):
        {{\"error\": \"Descriptive error message\"}}
    """
    try:
        if not query:
            return json.dumps({"error": "Query parameter is required. Please provide a JSON string with category1 and category2 parameters."})
            
        params = json.loads(query)
        cat1_query = params.get("category1")
        cat2_query = params.get("category2")
        date_query = params.get("date")
        subcat_query = params.get("subcategory")

        if not cat1_query or not cat2_query:
            return json.dumps({"error": "Both category1 and category2 are required."})

        cat1 = map_category(cat1_query)
        cat2 = map_category(cat2_query)
        data = st.session_state.json_data

        if cat1 not in data or cat2 not in data:
            missing = [c for c in [cat1, cat2] if c not in data]
            return json.dumps({"error": f"Categories not found: {', '.join(missing)}"})

        # Use the existing compare_categories logic by passing parsed values
        # This requires compare_categories to be adapted or for its logic to be integrated here.
        # For simplicity, let's assume compare_categories can be called with these direct args for now.
        # This part needs careful implementation matching the original compare_categories structure.
        
        # Simplified integration for now:
        val1, val2 = None, None
        matched_date1, matched_date2 = date_query, date_query # Assuming find_closest_date handles this

        if date_query:
            matched_date1 = find_closest_date(data[cat1].get("dates",{}), date_query)
            matched_date2 = find_closest_date(data[cat2].get("dates",{}), date_query)

            if not matched_date1 or not matched_date2:
                 return json.dumps({"error": f"Date '{date_query}' not found for comparison in one or both categories."})
            
            data1_on_date = data[cat1]["dates"].get(matched_date1, {})
            data2_on_date = data[cat2]["dates"].get(matched_date2, {})

            if subcat_query:
                val1 = data1_on_date.get(subcat_query)
                val2 = data2_on_date.get(subcat_query)
                if val1 is None or val2 is None:
                    return json.dumps({"error": f"Subcategory '{subcat_query}' not found for comparison on date '{date_query}'."})
                return json.dumps({
                    "comparison_on_date": date_query, # or matched_date1 if they can differ
                    "subcategory": subcat_query,
                    cat1: val1,
                    cat2: val2
                })
            else: # Compare all common subcategories for the date
                common_subcats = set(data1_on_date.keys()) & set(data2_on_date.keys())
                comparison_data = {cat1: {}, cat2: {}}
                for sc in common_subcats:
                    comparison_data[cat1][sc] = data1_on_date[sc]
                    comparison_data[cat2][sc] = data2_on_date[sc]
                return json.dumps({
                     "comparison_on_date": date_query,
                     "data": comparison_data
                })
        else: # No date, general comparison (e.g. list common dates/subcats - too broad for direct output usually)
            return json.dumps({"error": "Date parameter is recommended for meaningful comparison."})


    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input format for compare_category_data."})
    except Exception as e:
        return json.dumps({"error": f"Error in compare_category_data: {str(e)}"})


def parse_date(date_str: str) -> str:
    """Parse date string into a standardized format or identify relative periods.
    This function primarily aims to identify keywords for the agent or pass through week ranges.
    """
    if not date_str:
        return None
    
    # Check for specific week range format first (e.g., "25th-31st Jul 24")
    # This relies on parse_week_range being robust and returning the standardized range if matched.
    parsed_as_week_range = parse_week_range(date_str) 
    if parsed_as_week_range: # parse_week_range should return the "DD-Mon-YY to DD-Mon-YY" string if successful
        return parsed_as_week_range 

    date_str_lower = str(date_str).lower() # Ensure it's a string before .lower()

    # Keywords for relative dates that the agent prompt will instruct how to handle
    if "last 3 months" in date_str_lower or "last three months" in date_str_lower:
        return "last_3_months" 
    elif "last week" in date_str_lower:
        return "last_week"
    elif "last month" in date_str_lower:
        return "last_month"
    elif "this week" in date_str_lower: # Added "this week"
        return "this_week"

    # If not a known keyword or a parseable week range, return the original string.
    # The find_closest_date function will then attempt to match it against available week ranges.
    return date_str

def find_closest_date(available_dates, query_date):
    """Find the closest matching date(s) in available_dates for the given query_date.
    
    Args:
        available_dates: Dictionary with date strings as keys or list of date strings
        query_date: String date query (can be relative like "last_week" or specific date string)
    
    Returns:
        For period queries (last_month, last_week, etc.): List of ALL matching date strings
        For specific date queries: Single closest matching date string
        Returns None if no match found
    """
    if not available_dates or not query_date:
        return None
    
    # Convert to list if dictionary
    date_options = list(available_dates.keys()) if isinstance(available_dates, dict) else available_dates
    
    if not date_options:
        return None

    # Get current date info for relative date calculations
    current_date_info = json.loads(get_current_date.run({"query": "{}"}))
    
    # For period queries, we want to return ALL matching dates
    matching_dates = []
    
    # Handle relative date queries
    if query_date == "last_week":
        # Get the date range for last week
        last_week_start = datetime.datetime.strptime(current_date_info["last_week_description"].split("(")[1].split(")")[0].split(" - ")[0], "%d %b")
        last_week_end = datetime.datetime.strptime(current_date_info["last_week_description"].split("(")[1].split(")")[0].split(" - ")[1], "%d %b")
        
        # Find ALL dates that fall within last week's range
        for date_str in date_options:
            try:
                # Parse the date string (assuming format like "25th-31st Jul 24")
                date_parts = date_str.split()
                if len(date_parts) >= 3:
                    day_range = date_parts[0].split("-")
                    if len(day_range) == 2:
                        start_day = int(''.join(filter(str.isdigit, day_range[0])))
                        end_day = int(''.join(filter(str.isdigit, day_range[1])))
                        month = date_parts[1]
                        year = date_parts[2]
                        
                        # Create datetime objects for comparison
                        date_start = datetime.datetime.strptime(f"{start_day} {month} {year}", "%d %b %y")
                        date_end = datetime.datetime.strptime(f"{end_day} {month} {year}", "%d %b %y")
                        
                        # Check if this date range overlaps with last week
                        if (date_start <= last_week_end and date_end >= last_week_start):
                            matching_dates.append(date_str)
            except Exception as e:
                st.write(f"Debug find_closest_date: Error parsing date {date_str}: {str(e)}")
                continue
        
        return matching_dates if matching_dates else None
    
    elif query_date == "this_week":
        # Similar logic for this week
        this_week_start = datetime.datetime.strptime(current_date_info["this_week_description"].split("(")[1].split(")")[0].split(" - ")[0], "%d %b")
        this_week_end = datetime.datetime.strptime(current_date_info["this_week_description"].split("(")[1].split(")")[0].split(" - ")[1], "%d %b")
        
        for date_str in date_options:
            try:
                date_parts = date_str.split()
                if len(date_parts) >= 3:
                    day_range = date_parts[0].split("-")
                    if len(day_range) == 2:
                        start_day = int(''.join(filter(str.isdigit, day_range[0])))
                        end_day = int(''.join(filter(str.isdigit, day_range[1])))
                        month = date_parts[1]
                        year = date_parts[2]
                        
                        date_start = datetime.datetime.strptime(f"{start_day} {month} {year}", "%d %b %y")
                        date_end = datetime.datetime.strptime(f"{end_day} {month} {year}", "%d %b %y")
                        
                        if (date_start <= this_week_end and date_end >= this_week_start):
                            matching_dates.append(date_str)
            except Exception as e:
                st.write(f"Debug find_closest_date: Error parsing date {date_str}: {str(e)}")
                continue
        
        return matching_dates if matching_dates else None
    
    elif query_date == "last_month":
        # Get the date range for last month
        last_month_start = datetime.datetime.strptime(current_date_info["last_month_start"], "%Y-%m-%d")
        last_month_end = datetime.datetime.strptime(current_date_info["last_month_end"], "%Y-%m-%d")
        
        for date_str in date_options:
            try:
                date_parts = date_str.split()
                if len(date_parts) >= 3:
                    day_range = date_parts[0].split("-")
                    if len(day_range) == 2:
                        start_day = int(''.join(filter(str.isdigit, day_range[0])))
                        end_day = int(''.join(filter(str.isdigit, day_range[1])))
                        month = date_parts[1]
                        year = date_parts[2]
                        
                        # Handle both 2-digit and 4-digit years
                        if len(year) == 2:
                            year = "20" + year
                        
                        date_start = datetime.datetime.strptime(f"{start_day} {month} {year}", "%d %b %Y")
                        date_end = datetime.datetime.strptime(f"{end_day} {month} {year}", "%d %b %Y")
                        
                        # Check if ANY part of this date range falls within last month
                        if (date_start <= last_month_end and date_end >= last_month_start):
                            matching_dates.append(date_str)
                            st.write(f"Debug find_closest_date: Date '{date_str}' matches last month period")
            except Exception as e:
                st.write(f"Debug find_closest_date: Error parsing date {date_str}: {str(e)}")
                continue
        
        st.write(f"Debug find_closest_date: Found {len(matching_dates)} dates for last month")
        return matching_dates if matching_dates else None
    
    elif query_date == "last_3_months":
        # Get the date range for last 3 months
        last_3_months_start = datetime.datetime.strptime(current_date_info["last_3_months_start"], "%Y-%m-%d")
        last_3_months_end = datetime.datetime.strptime(current_date_info["last_3_months_end"], "%Y-%m-%d")
        
        for date_str in date_options:
            try:
                date_parts = date_str.split()
                if len(date_parts) >= 3:
                    day_range = date_parts[0].split("-")
                    if len(day_range) == 2:
                        start_day = int(''.join(filter(str.isdigit, day_range[0])))
                        end_day = int(''.join(filter(str.isdigit, day_range[1])))
                        month = date_parts[1]
                        year = date_parts[2]
                        
                        # Handle both 2-digit and 4-digit years
                        if len(year) == 2:
                            year = "20" + year
                            
                        date_start = datetime.datetime.strptime(f"{start_day} {month} {year}", "%d %b %Y")
                        date_end = datetime.datetime.strptime(f"{end_day} {month} {year}", "%d %b %Y")
                        
                        if (date_start <= last_3_months_end and date_end >= last_3_months_start):
                            matching_dates.append(date_str)
            except Exception as e:
                st.write(f"Debug find_closest_date: Error parsing date {date_str}: {str(e)}")
                continue
        
        return matching_dates if matching_dates else None
    
    # For exact date matching
    if query_date in date_options:
        return query_date
    
    # For partial date matching (e.g., "Jul" or "24")
    for date_str in date_options:
        if query_date.lower() in date_str.lower():
            return date_str
    
    # If no match found, return the most recent date as a fallback
    try:
        # Sort dates by parsing them into datetime objects
        sorted_dates = sorted(date_options, key=lambda x: datetime.datetime.strptime(x.split()[0].split("-")[0] + " " + x.split()[1] + " " + x.split()[2], "%d %b %y"))
        return sorted_dates[-1]  # Most recent date
    except Exception as e:
        st.write(f"Debug find_closest_date: Error sorting dates: {str(e)}")
        return None

def parse_week_range(date_str):
    """Parse a week range string like '25th-31st Jul 24' into a standardized format.
    Returns the standardized string or None if it doesn't match expected formats.
    This is a placeholder - implement proper parsing logic based on your exact date formats.
    """
    # This is just a placeholder implementation
    # In a real implementation, you would parse the date range properly
    # For now, we'll just return the input string if it looks like a week range
    
    if not date_str:
        return None
        
    date_str = str(date_str).strip()
    
    # Very basic check if this looks like a week range (contains a hyphen)
    if "-" in date_str and not date_str.endswith("-"):
        return date_str
    
    return None

@tool
def list_available(query: str = "{\"type\": \"categories\"}") -> str:
    """Lists available main categories, or available dates/subcategories for a specific main category.

    Args:
        query (str): A JSON string specifying what to list.
                     Must include 'type', which can be "categories", "dates", or "subcategories".
                     If 'type' is "dates" or "subcategories", then 'category' (mapped name, e.g., "OVERALL") must also be provided.

    Input JSON Structure Examples:
        - To list all main categories (OVERALL, RAIGARH, ANGUL):
          '{{\"type\": \"categories\"}}'
        - To list available dates for a specific category:
          '{{\"type\": \"dates\", \"category\": \"RAIGARH\"}}'
        - To list available subcategories for a specific category:
          '{{\"type\": \"subcategories\", \"category\": \"ANGUL\"}}'

    Output JSON Structure (Success):
        - For "categories":
          ["OVERALL", "RAIGARH", "ANGUL", ...]
        - For "dates" (for a category):
          ["date1_string", "date2_string", ...]
        - For "subcategories" (for a category):
          ["subcat1_name", "subcat2_name", ...]

    Output JSON Structure (Error):
        {{\"error\": \"Descriptive error message\"}}
        (Example: {"error": "Category 'XYZ' not found.", "available_subcategories": []} if type was subcategories)
    """
    try:
        # Ensure query parameter is provided
        if not query:
            query = "{\"type\": \"categories\"}"  # Default to listing categories if no query provided
            
        params = json.loads(query)
        list_type = params.get("type")
        
        data = st.session_state.json_data
        if not data:
            return json.dumps({"error": "Data not loaded. Please upload an Excel file."})

        if list_type == "categories":
            return json.dumps(list(data.keys()))
        
        category_query = params.get("category")
        if not category_query:
            return json.dumps({"error": "Category is required for listing dates or subcategories."})
            
        category = map_category(category_query)

        if category not in data:
            return json.dumps({"error": f"Category '{category}' (mapped from '{category_query}') not found."})

        if list_type == "dates":
            if "dates" not in data[category]:
                return json.dumps({"error": f"No 'dates' field found for category '{category}'.", "available_dates": []})
            return json.dumps(list(data[category]["dates"].keys()))
        elif list_type == "subcategories":
            if "subcategories" in data[category] and data[category]["subcategories"]:
                return json.dumps(data[category]["subcategories"])
            # Fallback to derive from first date if explicitly computed list is empty/missing but dates exist
            elif "dates" in data[category] and data[category]["dates"]:
                first_date = next(iter(data[category]["dates"]), None)
                if first_date and isinstance(data[category]["dates"][first_date], dict):
                    return json.dumps(list(data[category]["dates"][first_date].keys()))
                else:
                    return json.dumps({"error": f"Subcategories could not be derived for '{category}'. No valid date entries.", "available_subcategories": []})
            else: # No subcategories list and no dates to derive from
                return json.dumps({"error": f"No subcategories found or derivable for category '{category}'.", "available_subcategories": []})
        else:
            return json.dumps({"error": "Invalid 'type' in list_available query. Must be 'categories', 'dates', or 'subcategories'."})
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input for list_available."})
    except Exception as e:
        return json.dumps({"error": f"Error in list_available: {str(e)}"})

@tool
def create_visualization(query: str = "{\"type\": \"line\", \"data\": []}") -> str:
    """Creates and displays a Plotly visualization based on the provided configuration and data.

    Args:
        query (str): A JSON string containing the visualization configuration.

    Input JSON Structure:
        {{
            \"type\": \"line\" | \"bar\" | \"pie\" | \"scatter\",  // Required: type of chart
            \"data\": [                             // Required: list of dictionaries, each dict is a data point/row
                {{\"column1_name\": \"value1\", \"column2_name\": 10}},
                {{\"column1_name\": \"value2\", \"column2_name\": 20}}
            ],
            \"x\": \"column1_name\",                  // Required: key from data dicts to use for x-axis (e.g., date, category name)
            \"y\": \"column2_name\",                  // Required: key from data dicts to use for y-axis (numeric values)
            \"title\": \"Optional Chart Title\",      // Optional: title for the chart
            \"labels\": {{\"x_axis_label_override\": \"X-Axis Custom\", \"y_axis_label_override\": \"Y-Axis Custom\"}}, // Optional: custom labels for axes
            \"is_financial\": true | false           // Optional (defaults to true): if true, formats y-axis and hover data with ₹ Cr
        }}

    Input JSON Example:
        '{{
            \"type\": \"line\",
            \"data\": [
                {{\"Week\": \"W1\", \"Cash Score\": 70.5}},
                {{\"Week\": \"W2\", \"Cash Score\": 75.2}},
                {{\"Week\": \"W3\", \"Cash Score\": 72.1}}
            ],
            \"x\": \"Week\",
            \"y\": \"Cash Score\",
            \"title\": \"Weekly Cash Score Trend\",
            \"is_financial\": true
        }}'

    Output JSON Structure (Success):
        {{\"success\": true, \"message\": \"Line chart titled 'Weekly Cash Score Trend' has been generated and is now displayed.\", \"chart_type\": \"line\"}}
        (The chart is displayed directly in the Streamlit app and stored in session state for the current turn.)

    Output JSON Structure (Error):
        {{\"error\": \"Descriptive error message (e.g., Invalid data provided, Missing x/y axis keys, etc.)\"}}
    """
    try:
        if not query:
            return json.dumps({"error": "Query parameter is required. Please provide a JSON string with visualization configuration."})
            
        config = json.loads(query)
        chart_type = config.get('type')
        data = config.get('data') # This should be a list of dictionaries for px
        title = config.get('title', 'Data Visualization')
        labels = config.get('labels', {}) # e.g., {"x_column_name": "X-Axis Label", "y_column_name": "Y-Axis Label"}
        x_axis = config.get('x') # Explicitly define x and y columns from data keys if provided
        y_axis = config.get('y')
        is_financial = config.get('is_financial', True)  # Default to True for Jindal steel data

        # Debug input data
        st.write(f"Debug create_visualization: Input data (first 3 items):")
        if data and len(data) > 0:
            st.write(data[:min(3, len(data))])
        else:
            st.write("No data provided")

        if not data or not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            return json.dumps({"error": "Invalid or empty data provided. Data must be a list of dictionaries."})

        # IMPORTANT: Check data structure to ensure correct keys are used
        # If a generic "value" key is used, rename it to match the y-axis label for better plots
        if len(data) > 0 and "value" in data[0] and not y_axis:
            # Extract a better column name from the labels if available
            better_y_name = labels.get("y", "").split(" ")[0] if "y" in labels else "Value"
            if better_y_name and better_y_name not in ["(₹", "Cr)"]:
                # Rename the "value" key to the better name in all data items
                for item in data:
                    if "value" in item:
                        item[better_y_name] = item.pop("value")
                y_axis = better_y_name
                st.write(f"Debug create_visualization: Renamed 'value' key to '{better_y_name}' for better visualization")
                
                # Critical fix: Print the transformed data for verification
                st.write("Debug create_visualization: Transformed data (after renaming 'value'):")
                st.write(data[:min(3, len(data))])

        # Find the first key that could be an x-axis if not specified
        if not x_axis and len(data) > 0:
            for key in data[0].keys():
                if key.lower() in ['date', 'week', 'period', 'month', 'quarter', 'year', 'time']:
                    x_axis = key
                    st.write(f"Debug create_visualization: Auto-detected x-axis as '{x_axis}'")
                    break
            if not x_axis:
                x_axis = list(data[0].keys())[0]  # Take first key as default x-axis
                st.write(f"Debug create_visualization: Using first key '{x_axis}' as x-axis")

        # Create a DataFrame from the data for easier plotting with Plotly Express
        df = pd.DataFrame(data)
        
        # Show extracted DataFrame for debugging
        st.write("Debug create_visualization: DataFrame created from input data:")
        st.dataframe(df.head())

        if df.empty:
            return json.dumps({"error": "Data converted to empty DataFrame. Cannot plot."})

        # PRE-PROCESS: Convert all string values that look like numbers to actual numbers
        # This is critical - do this BEFORE attempting to identify axes
        for col in df.columns:
            if col != x_axis:  # Don't convert x-axis - keep as labels
                # First handle commas in string values - important for Indian number format
                if df[col].dtype == 'object':
                    # Try to clean and convert strings with commas, rupee symbols, etc.
                    df[col] = df[col].apply(lambda x: str(x).replace(',', '').replace('₹', '').replace(' Cr', '') 
                                            if isinstance(x, str) else x)
                
                # Then try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    st.write(f"Debug create_visualization: Converted column '{col}' to numeric. Sample: {df[col].head()}")
                except:
                    st.write(f"Debug create_visualization: Column '{col}' could not be converted to numeric. Sample: {df[col].head()}")

        # Determine y for plotting if not explicitly given
        if not y_axis and len(df.columns) >= 2:
            # Find first numeric column for y-axis if not specified
            for col in df.columns:
                if col != x_axis and pd.api.types.is_numeric_dtype(df[col]):
                    y_axis = col
                    st.write(f"Debug create_visualization: Auto-detected y-axis as numeric column '{y_axis}'")
                    break
        
        if not y_axis: # Y-axis is crucial for most plots
            return json.dumps({"error": f"Could not determine Y-axis for plotting. Available columns: {list(df.columns)}"})

        # Enhance labels for financial data
        if is_financial:
            if 'y' not in labels:
                labels['y'] = f"{y_axis} (₹ Cr)"
            
            # Special formatting for hover data - make sure data is properly formatted
            hover_template = "%{y:.2f} ₹ Cr"
        else:
            hover_template = None

        # VERY IMPORTANT: Always treat x-axis as categorical for this specific application
        if x_axis and x_axis in df.columns:
            # Explicitly convert to category type, regardless of original format
            df[x_axis] = df[x_axis].astype(str).astype('category')
            st.write(f"Debug create_visualization: Treating '{x_axis}' as categorical data for visualization")
            
            # Get unique categories in original order from the dataframe
            x_categories = df[x_axis].tolist()
            
            # Debug the categories
            st.write(f"Debug create_visualization: Categories for '{x_axis}': {x_categories}")
        
        # Debug the data being used for visualization
        if y_axis in df.columns:
            st.write(f"Debug create_visualization: Data range for {y_axis}: min={df[y_axis].min()}, max={df[y_axis].max()}, mean={df[y_axis].mean()}")
            # Display the actual values that will be plotted
            st.write(f"Debug create_visualization: Actual values being plotted for {y_axis}: {df[y_axis].tolist()}")

        fig = None
        
        # Create figure with explicit categorical x-axis
        if chart_type == "line":
            # Create basic line chart
            st.write(f"Debug create_visualization: Creating line chart with x={x_axis}, y={y_axis}")
            # Ensure data is correct before creating chart
            st.write(f"Debug create_visualization: Final DataFrame used for plotting:")
            st.dataframe(df)
            
            # Extract the exact x and y values
            x_values = df[x_axis].tolist()
            y_values = df[y_axis].tolist()
            
            # Print the actual values being plotted
            st.write(f"Debug create_visualization: X values: {x_values}")
            st.write(f"Debug create_visualization: Y values: {y_values}")
            
            # Instead of px.line, create a Figure and add a Scatter trace manually
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=y_values,
                    mode='lines+markers+text',
                    text=[f"{y:.1f}" for y in y_values],
                    textposition="top center",
                    name=y_axis
                )
            )
            
            # Add title and labels
            fig.update_layout(
                title=title,
                xaxis_title=labels.get('x', x_axis),
                yaxis_title=labels.get('y', y_axis)
            )
            
            # Ensure x-axis shows categorical values correctly
            fig.update_xaxes(type='category')
            
            if is_financial and fig is not None:
                fig.update_traces(hovertemplate=hover_template)
                
        elif chart_type == "bar":
            # Extract the exact x and y values
            x_values = df[x_axis].tolist()
            y_values = df[y_axis].tolist()
            
            # Print the actual values being plotted
            st.write(f"Debug create_visualization: X values: {x_values}")
            st.write(f"Debug create_visualization: Y values: {y_values}")
            
            # Create a Figure and add a Bar trace manually
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=x_values, 
                    y=y_values,
                    text=[f"{y:.1f}" for y in y_values],
                    textposition="outside",
                    name=y_axis
                )
            )
            
            # Add title and labels
            fig.update_layout(
                title=title,
                xaxis_title=labels.get('x', x_axis),
                yaxis_title=labels.get('y', y_axis)
            )
            
            # Ensure x-axis shows categorical values correctly
            fig.update_xaxes(type='category')
            
            if is_financial and fig is not None:
                fig.update_traces(hovertemplate=hover_template)
                
        elif chart_type == "pie":
            # For pie charts, 'names' would be the category column and 'values' the numeric column.
            if not x_axis or not y_axis:
                 return json.dumps({"error": "For pie chart, please specify 'x' for names and 'y' for values in the query."})
            fig = px.pie(df, names=x_axis, values=y_axis, title=title)
            if is_financial and fig is not None:
                fig.update_traces(texttemplate="%{value:.2f} ₹ Cr")
                
        elif chart_type == "scatter":
            # Extract the exact x and y values
            x_values = df[x_axis].tolist()
            y_values = df[y_axis].tolist()
            
            # Print the actual values being plotted
            st.write(f"Debug create_visualization: X values: {x_values}")
            st.write(f"Debug create_visualization: Y values: {y_values}")
            
            # Create a Figure and add a Scatter trace manually
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=x_values, 
                    y=y_values,
                    mode='markers+text',
                    text=[f"{y:.1f}" for y in y_values],
                    textposition="top center",
                    name=y_axis
                )
            )
            
            # Add title and labels
            fig.update_layout(
                title=title,
                xaxis_title=labels.get('x', x_axis),
                yaxis_title=labels.get('y', y_axis)
            )
            
            # Ensure x-axis shows categorical values correctly
            fig.update_xaxes(type='category')

        else:
            return json.dumps({"error": f"Unsupported visualization type: {chart_type}. Supported types: line, bar, pie, scatter."})

        if is_financial and fig is not None:
            title = f"{title} (₹ in Crores)" if "Crores" not in title and "₹" not in title else title
            fig.update_layout(title=title)
            
            # Add rupee symbol to Y-axis tick labels
            if chart_type != "pie":
                fig.update_layout(
                    yaxis=dict(
                        tickprefix="₹ ",
                        ticksuffix=" Cr"
                    )
                )
                
            # Also update the hover template for line, bar, and scatter charts
            if chart_type in ["line", "bar", "scatter"]:
                fig.update_traces(
                    hovertemplate="<b>%{x}</b><br>%{y:.2f} ₹ Cr<extra></extra>"
                )
                
            # Adjust Y-axis range for better data visibility
            if y_axis in df.columns:
                y_min = df[y_axis].min()
                y_max = df[y_axis].max()
                if not pd.isna(y_min) and not pd.isna(y_max):
                    # Add a 15% padding above the max value and below the min value for better visibility
                    y_range = y_max - y_min
                    y_range_padding = max(y_range * 0.15, 0.5)  # At least 0.5 Cr padding
                    
                    # Set Y-axis range with padding (min 0 for financial data unless negative values)
                    min_bound = min(0, y_min - y_range_padding) if y_min > 0 else y_min - y_range_padding
                    fig.update_layout(
                        yaxis_range=[min_bound, y_max + y_range_padding]
                    )
                    st.write(f"Debug create_visualization: Adjusted Y-axis range to [{min_bound:.1f}, {y_max + y_range_padding:.1f}]")

        if fig is not None:
            # Display the chart immediately in Streamlit
            st.write("### Chart Preview")
            st.plotly_chart(fig, use_container_width=True)
            
            # Before storing, verify the figure has valid data
            if hasattr(fig, 'data') and len(fig.data) > 0:
                st.write("Debug create_visualization: Figure generated successfully with data")
                # Store the figure directly in session state
                st.session_state.last_chart_figure_json = fig.to_json()
                st.session_state.last_chart_figure = fig
                return json.dumps({
                    "success": True, 
                    "message": f"{chart_type.capitalize()} chart titled '{title}' has been generated and is now displayed.", 
                    "chart_type": chart_type
                })
            else:
                st.write("Debug create_visualization: Figure generated but has no data")
                return json.dumps({"error": "Figure was generated but contains no data. Please check your data values."})
        else:
            return json.dumps({"error": "Figure could not be generated for unknown reasons."})

    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input for create_visualization."})
    except KeyError as e:
        return json.dumps({"error": f"Missing expected key in data for plotting: {str(e)}. Ensure your data list of dicts contains keys for x and y axes."})
    except Exception as e:
        import traceback
        error_traceback = traceback.format_exc()
        st.error(f"Error in create_visualization: {str(e)}")
        st.error(error_traceback)
        return json.dumps({"error": f"Error in create_visualization: {str(e)}", "traceback": error_traceback})

@tool
def analyze_trends(query: str = "{\"data\": [], \"metrics\": []}") -> str:
    """Analyzes trends in the provided time series data and provides textual insights.

    Args:
        query (str): A JSON string containing the data and parameters for trend analysis.

    Input JSON Structure:
        {{
            \"data\": [                             // Required: List of dictionaries, each representing a data point (e.g., for a time period).
                {{\"date_column_key\": \"Period1\", \"metric1_key\": 100, \"metric2_key\": 200}},
                {{\"date_column_key\": \"Period2\", \"metric1_key\": 110, \"metric2_key\": 190}}
            ],
            \"metrics\": [\"metric1_key\", \"metric2_key\"], // Required: List of metric keys (from the data dicts) to analyze.
            \"time_period_description\": \"last 3 months\", // Required: A string describing the time period for context in the output (e.g., "last 3 months", "selected weeks").
            \"is_financial\": true | false             // Optional (defaults to true): If true, formats financial values with ₹ Cr in the output.
        }}

    Input JSON Example:
        '{{
            \"data\": [
                {{\"date\": \"Week 1 (Jul)\", \"Sales\": 120.5, \"Expenses\": 80.2}},
                {{\"date\": \"Week 2 (Jul)\", \"Sales\": 125.0, \"Expenses\": 82.1}},
                {{\"date\": \"Week 3 (Jul)\", \"Sales\": 118.3, \"Expenses\": 79.5}}
            ],
            \"metrics\": [\"Sales\", \"Expenses\"],
            \"time_period_description\": \"first three weeks of July\",
            \"is_financial\": true
        }}'

    Output JSON Structure (Success):
        {{
            \"insights\": [\"Insight about metric1...\", \"Significant change in metric2...\"],
            \"trends_observed\": [
                \"For 'metric1' over [time_period_description]: - The value increased/decreased from X to Y. - Overall change: Z (P%...). - Average value: A.\",
                \"For 'metric2' over [time_period_description]: ...\"
            ],
            \"recommendations\": [], // Currently a placeholder, future enhancement
            \"currency_note\": \"All financial values are in Indian Rupees (₹) and in Crores (1 Crore = 10 million)\" // (if is_financial)
        }}

    Output JSON Structure (Error):
        {{\"error\": \"Descriptive error message\", \"traceback\": \"Optional traceback string\"}}
    """
    try:
        if not query:
            return json.dumps({"error": "Query parameter is required. Please provide a JSON string with data and metrics parameters."})
            
        config = json.loads(query)
        data_list = config.get('data', [])
        metrics_to_analyze = config.get('metrics', [])
        time_period_desc = config.get('time_period_description', 'the given period')
        # Flag to note if these are financial metrics (to format with ₹ and Crores)
        is_financial = config.get('is_financial', True)  # Default to True for Jindal steel data

        # Debug input
        st.write(f"Debug analyze_trends: Received metrics to analyze: {metrics_to_analyze}")
        st.write(f"Debug analyze_trends: Data sample (first 3 items):")
        if data_list and len(data_list) > 0:
            st.write(data_list[:min(3, len(data_list))])
        else:
            st.write("No data provided")

        if not data_list or not isinstance(data_list, list) or not all(isinstance(item, dict) for item in data_list):
            return json.dumps({"error": "Invalid or empty data provided for trend analysis. Must be a list of dictionaries."})
        
        if not metrics_to_analyze or not isinstance(metrics_to_analyze, list):
            return json.dumps({"error": "Metrics to analyze must be a list of strings (keys in your data dicts)."})

        df = pd.DataFrame(data_list)
        
        # Debug the created DataFrame
        st.write("Debug analyze_trends: DataFrame created from input data:")
        st.dataframe(df.head())
        
        if df.empty:
            return json.dumps({"error": "Data for trend analysis is empty after DataFrame conversion."})

        # Ensure all metric columns are numeric
        for metric in metrics_to_analyze:
            if metric in df.columns:
                try:
                    # Convert to numeric, coercing errors to NaN
                    df[metric] = pd.to_numeric(df[metric], errors='coerce')
                    # Show the successful conversion
                    st.write(f"Debug analyze_trends: Converted '{metric}' column to numeric values.")
                    st.write(f"Debug analyze_trends: {metric} values: {df[metric].tolist()}")
                except Exception as e:
                    st.write(f"Debug analyze_trends: Error converting '{metric}' column to numeric: {str(e)}")

        analysis_results = {
            "insights": [],
            "trends_observed": [],
            "recommendations": [], # Placeholder for now
            "currency_note": "All financial values are in Indian Rupees (₹) and in Crores (1 Crore = 10 million)" if is_financial else ""
        }

        for metric in metrics_to_analyze:
            if metric not in df.columns:
                analysis_results["insights"].append(f"Metric '{metric}' not found in the provided data.")
                continue

            series = pd.to_numeric(df[metric], errors='coerce').dropna()
            
            # Debug the numeric conversion
            st.write(f"Debug analyze_trends: {metric} values after final conversion: {series.tolist()}")
            st.write(f"Debug analyze_trends: {metric} stats: min={series.min() if not series.empty else 'N/A'}, max={series.max() if not series.empty else 'N/A'}, mean={series.mean() if not series.empty else 'N/A'}")
            
            if len(series) < 2:
                analysis_results["insights"].append(f"Not enough data points for '{metric}' to determine a trend.")
                continue

            # Simple trend: compare first and last points
            overall_change = series.iloc[-1] - series.iloc[0]
            trend_direction = "increased" if overall_change > 0 else "decreased" if overall_change < 0 else "remained stable"
            avg_value = series.mean()
            percentage_change = (overall_change / series.iloc[0]) * 100 if series.iloc[0] != 0 else 0

            value_prefix = "₹" if is_financial else ""
            value_suffix = " Cr" if is_financial else ""

            analysis_results["trends_observed"].append(f"""For '{metric}' over {time_period_desc}:
            - The value {trend_direction} from {value_prefix}{series.iloc[0]:.2f}{value_suffix} to {value_prefix}{series.iloc[-1]:.2f}{value_suffix}.
            - Overall change: {value_prefix}{overall_change:.2f}{value_suffix} (Percentage change: {percentage_change:.2f}%).
            - Average value: {value_prefix}{avg_value:.2f}{value_suffix}.""")
            
            if abs(percentage_change) > 20: # Example threshold for significant change
                 analysis_results["insights"].append(f"Significant {trend_direction} observed in '{metric}' ({percentage_change:.2f}%).")

        if not analysis_results["trends_observed"]:
             analysis_results["insights"].append("No conclusive trends could be analyzed with the provided metrics or data.")

        return json.dumps(analysis_results)

    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid JSON input for analyze_trends."})
    except Exception as e:
        import traceback
        return json.dumps({"error": f"Error in analyze_trends: {str(e)}", "traceback": traceback.format_exc()})

@tool
def get_current_date(query: str = "{}") -> str:
    """Gets the current server date and provides contextual relative date information.

    Args:
        query (str): An optional JSON string. Currently not used for any parameters, so an empty JSON string '{{}}' is typical.

    Input JSON Structure Example:
        '{{}}' 

    Output JSON Structure (Success):
        {{
            \"current_date\": \"YYYY-MM-DD\",
            \"current_week_start\": \"YYYY-MM-DD\",
            \"current_week_end\": \"YYYY-MM-DD\",
            \"current_month_start\": \"YYYY-MM-DD\",
            \"current_month_end\": \"YYYY-MM-DD\",
            \"current_quarter_start\": \"YYYY-MM-DD\",
            \"current_quarter_end\": \"YYYY-MM-DD\",
            \"last_3_months_start\": \"YYYY-MM-DD\", // Start date of the period covering the last 3 full months plus current partial month.
            \"last_3_months_end\": \"YYYY-MM-DD\",   // Typically the current date.
            \"last_month_start\": \"YYYY-MM-DD\",
            \"last_month_end\": \"YYYY-MM-DD\",
            \"last_week_description\": \"The week prior to current (approx. DD Mon - DD Mon)\",
            \"this_week_description\": \"Current week (DD Mon - DD Mon)\",
            \"last_month_description\": \"Mon YYYY\",
            \"last_3_months_description\": \"Period from DD Mon YYYY to DD Mon YYYY (approximately 12-13 weekly data points)\"
        }}

    Output JSON Structure (Error):
        {{\"error\": \"Descriptive error message\"}}
    """
    try:
        # Optional: parse query if it ever becomes useful, e.g., for specific timezone requests in future.
        # params = json.loads(query) if query else {}

        now = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        # Current week (assuming Monday is the start of the week)
        start_of_week = now - datetime.timedelta(days=now.weekday())
        end_of_week = start_of_week + datetime.timedelta(days=6)

        # Current month
        start_of_month = now.replace(day=1)
        # To get end_of_month, go to the first day of next month and subtract one day
        if now.month == 12:
            end_of_month = now.replace(year=now.year + 1, month=1, day=1) - datetime.timedelta(days=1)
        else:
            end_of_month = now.replace(month=now.month + 1, day=1) - datetime.timedelta(days=1)
        
        # Get the start of the current quarter
        current_quarter = (now.month - 1) // 3 + 1
        start_of_quarter = datetime.datetime(now.year, 3 * current_quarter - 2, 1)
        if current_quarter == 4:
            end_of_quarter = datetime.datetime(now.year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            end_of_quarter = datetime.datetime(now.year, 3 * current_quarter + 1, 1) - datetime.timedelta(days=1)
        
        # Last 3 months (more precise calculation)
        three_months_ago = now.replace(day=1)  # Start of current month
        for _ in range(3):  # Go back 3 months
            # Go to the last day of previous month
            three_months_ago = (three_months_ago - datetime.timedelta(days=1)).replace(day=1)
        
        # Last month
        if now.month == 1:
            last_month_start = now.replace(year=now.year-1, month=12, day=1)
            last_month_end = now.replace(year=now.year, month=1, day=1) - datetime.timedelta(days=1)
        else:
            last_month_start = now.replace(month=now.month-1, day=1)
            last_month_end = now.replace(day=1) - datetime.timedelta(days=1)

        date_info = {
            "current_date": today_str,
            "current_week_start": start_of_week.strftime("%Y-%m-%d"),
            "current_week_end": end_of_week.strftime("%Y-%m-%d"),
            "current_month_start": start_of_month.strftime("%Y-%m-%d"),
            "current_month_end": end_of_month.strftime("%Y-%m-%d"),
            "current_quarter_start": start_of_quarter.strftime("%Y-%m-%d"),
            "current_quarter_end": end_of_quarter.strftime("%Y-%m-%d"),
            "last_3_months_start": three_months_ago.strftime("%Y-%m-%d"),
            "last_3_months_end": today_str,
            "last_month_start": last_month_start.strftime("%Y-%m-%d"),
            "last_month_end": last_month_end.strftime("%Y-%m-%d")
        }
        
        # Add human-readable descriptions
        date_info["last_week_description"] = f"The week prior to current (approx. {(start_of_week - datetime.timedelta(days=7)).strftime('%d %b')} - {(end_of_week - datetime.timedelta(days=7)).strftime('%d %b')})"
        date_info["this_week_description"] = f"Current week ({start_of_week.strftime('%d %b')} - {end_of_week.strftime('%d %b')})"
        date_info["last_month_description"] = f"{last_month_start.strftime('%b %Y')}"
        date_info["last_3_months_description"] = f"Period from {three_months_ago.strftime('%d %b %Y')} to {now.strftime('%d %b %Y')} (approximately 12-13 weekly data points)"
        
        return json.dumps(date_info)
    except Exception as e:
        return json.dumps({"error": f"Error in get_current_date: {str(e)}"})

# Define AVAILABLE_TOOLS list
AVAILABLE_TOOLS = [
    get_data_for_category,
    compare_category_data,
    list_available,
    create_visualization,
    analyze_trends,
    get_current_date
]

# Orchestrator Agent System Prompt
ORCHESTRATOR_AGENT_PROMPT = """You are a highly intelligent Orchestrator Agent for Jindal Steel's TOC (Theory of Constraints) analysis.
Your primary role is to understand the user's request, break it down into a sequence of logical steps, and create a clear plan for execution.
You do NOT execute the tools directly. Instead, you will output a structured plan as a JSON object that the main application loop will follow.

CRITICAL: You MUST ALWAYS output a valid JSON object with a "steps" array, even for greetings or unclear queries. NEVER respond with plain text.

IMPORTANT DISTINCTION:
- Greetings are only: "hello", "hi", "hey", "good morning", etc. without any data-related content
- ANY query mentioning categories (OVERALL, RAIGARH, ANGUL, JSPL), metrics, comparisons, performance, or specific data requests is a DATA QUERY that requires proper analysis

If the user greets you (ONLY simple greetings), create a minimal plan that uses list_available to show what data is available.
If the user asks about data, comparisons, or performance, create a comprehensive plan to retrieve and analyze the requested data.

Your output MUST strictly adhere to the JSON schema provided in the formatting instructions.

Example output for a greeting:
{
  "steps": [
    {
      "step_id": 1,
      "persona": "DataRetrieverMode",
      "goal": "List available data categories to help user understand what's available",
      "inputs": {"tool_to_call": "list_available", "tool_input_json": "{\"type\": \"categories\"}"},
      "outputs_expected": "List of available categories",
      "depends_on_step": null
    },
    {
      "step_id": 2,
      "persona": "FinalResponseCompilationMode",
      "goal": "Compile a helpful greeting with information about available data",
      "inputs": {"summary_points": ["Available categories: [output_from_step_1]"], "user_query": "greeting"},
      "outputs_expected": "Friendly response with available options",
      "depends_on_step": 1
    }
  ]
}

Your Plan Output Structure:
You MUST output your plan as a JSON object string with a "steps" key. "steps" will be a list of objects, where each object represents a step.
Each step object MUST have the following keys:
- "step_id": An integer, starting from 1.
- "persona": The specialized persona/mode required for this step. Choose from: "DataRetrieverMode", "FinancialAnalysisMode", "VisualizationMode", "TOCStrategyMode", "FinalResponseCompilationMode".
- "goal": A concise description of what this step aims to achieve.
- "inputs": An object containing the necessary inputs for this step. The structure of "inputs" depends on the "persona" and should align with the Pydantic models (DataRetrieverInputs, FinancialAnalysisInputs etc.). For example:
    - For "DataRetrieverMode": {'tool_to_call': "tool_name", "tool_input_json": "{...json_string...}"}
    - For "FinancialAnalysisMode": {'data_to_analyze': "[placeholder_for_data_from_step_X]", "analysis_task": "Specific question or analysis to perform"}
- "outputs_expected": A brief description of what this step should produce.
- "depends_on_step": Either:
    - A single step_id (integer) if this step depends on one previous step
    - A list of step_ids (e.g., [1, 2]) if this step depends on multiple previous steps
    - null if this step has no dependencies

MANDATORY RULES:

1. **ALWAYS Check Current Date First for Relative Date Queries**: 
   If the user's query contains ANY relative date reference or time-related keywords, your FIRST step MUST ALWAYS be to call `get_current_date` to establish the current date context.
   
   Relative date keywords include (but are not limited to):
   - "last week", "this week", "next week", "weekly" (implies current week)
   - "last month", "this month", "next month", "monthly" (implies current month)
   - "last quarter", "this quarter", "quarterly" (implies current quarter)
   - "last year", "this year", "yearly", "annual" (implies current year)
   - "last 3 months", "last 6 months", "past X months"
   - "yesterday", "today", "tomorrow"
   - "recent", "latest", "current"
   - Any query asking about performance "vs Target" (implies current period comparison)
   
   Example first step for ANY query with these keywords:
   {
     "step_id": 1,
     "persona": "DataRetrieverMode",
     "goal": "Get current date context for relative date interpretation",
     "inputs": {"tool_to_call": "get_current_date", "tool_input_json": "{}"},
     "outputs_expected": "Current date and relative date ranges",
     "depends_on_step": null
   }
   
   IMPORTANT: After getting date information from Step 1, use it to:
   - For "last month" queries: Use "last_month" (with underscore) as the date parameter in subsequent tools
   - For "last week" queries: Use "last_week" (with underscore) as the date parameter  
   - For "this week" queries: Use "this_week" (with underscore) as the date parameter
   - For "last 3 months" queries: Use "last_3_months" (with underscore) as the date parameter
   - The find_closest_date function in tools will handle these keywords automatically
   - Do NOT pass the entire JSON output from get_current_date as the date parameter
   
   Example of correct date usage in subsequent steps:
   {
     "step_id": 3,
     "persona": "DataRetrieverMode", 
     "goal": "Get Net Revenue data for last month",
     "inputs": {"tool_to_call": "get_data_for_category", "tool_input_json": "{\"category\": \"OVERALL\", \"subcategory\": \"Net Revenue\", \"date\": \"last_month\"}"},
     "outputs_expected": "Net Revenue value for last month",
     "depends_on_step": 1
   }

2. **ALWAYS End with FinalResponseCompilationMode**: 
   EVERY plan MUST have a FinalResponseCompilationMode step as the LAST step. This step compiles all outputs from previous steps into a coherent final response. The summary_points should reference outputs from previous steps using placeholders like [output_from_step_X].

3. **Understand the Query**: Carefully analyze the user's question to determine the specific categories, subcategories, dates, and analysis type required.

4.  **State Your Plan (Before Calling Tools)**: For ANY tool call, first clearly state:
    a. Which tool you are about to call.
    b. What specific information you are trying to obtain with that tool.
    c. The *exact* JSON input you will use for the tool.
    This is critical for clarity and debugging. For example:
    "Okay, I need to find the 'Cash Score' for 'OVERALL' for 'last week'.
    I will call the `get_data_for_category` tool.
    Input: '{{\"category\": \"OVERALL\", \"subcategory\": \"Cash Score\", \"date\": \"last week\"}}' "

5.  **Verify Information with `list_available`**:
    *   If unsure about the exact spelling or availability of a category, subcategory, or date, ALWAYS use `list_available` first to confirm.
    *   Example: If asked for "Profit" but `list_available` shows "Net Profit", use "Net Profit".
    *   Example: If asked for a specific date, use `list_available` with `'{{\"type\": \"dates\", \"category\": \"<CATEGORY_NAME>\"}}'` to see available date strings and use the closest match.

6.  **Date Handling**:
    *   For queries involving relative dates (e.g., "last week", "this month", "last 3 months"), ALWAYS call `get_current_date` first to understand the current date context.
    *   Use the output of `get_current_date` to determine the correct date strings or keywords for other tools.
    *   For "last week", "this week", "last month": These keywords can often be used directly in the `date` parameter of `get_data_for_category` and `compare_category_data` tools, which will attempt to match them to available data periods.
    *   For "last 3 months" or other multi-period ranges:
        a. Call `get_current_date` to get the overall date range.
        b. Use `list_available` (e.g., `'{{\"type\": \"dates\", \"category\": \"OVERALL\"}}'`) to get all available date strings for the relevant category.
        c. Identify which of these available date strings fall within your target period.
        d. You may need to call `get_data_for_category` multiple times, once for each relevant date string, and then aggregate the results.

7.  **Tool Input Format**:
    *   ALL tools expect their input query as a single, valid JSON-formatted string.
    *   Pay close attention to the specific JSON structure required by each tool (see tool descriptions).

8.  **Data Retrieval with `get_data_for_category`**:
    *   Use this tool to fetch specific data points or sets of data.
    *   For specific dates: Input example: `'{{\"category\": \"OVERALL\", \"subcategory\": \"Cash Score\", \"date\": \"25th-31st Jul 24\"}}'`
    *   For period queries (last_month, last_week, etc.): The tool will return ALL matching dates with data and a summary
    *   Input example: `'{{\"category\": \"RAIGARH\", \"subcategory\": \"Net Revenue\", \"date\": \"last_month\"}}'`
    *   Period query response includes: individual date entries, count, total, and average

9.  **Comparing Data with `compare_category_data`**:
    *   Use for direct comparisons between two categories.
    *   Input example: `'{{\"category1\": \"RAIGARH\", \"category2\": \"ANGUL\", \"subcategory\": \"Production Volume\", \"date\": \"this_week\"}}'`

10. **Visualizations with `create_visualization`**:
    *   Before calling, ensure you have the necessary data, usually as a list of dictionaries.
    *   Specify `x` and `y` axis keys from your data.
    *   Ensure `is_financial: true` for financial data to apply ₹ and Cr formatting.
    *   The data passed to `create_visualization` should be structured correctly, often from the output of `get_data_for_category`.

11. **Trend Analysis with `analyze_trends`**:
    *   Pass data as a list of dictionaries. Specify the metrics to analyze.

12. **Error Handling & Iteration**:
    *   If a tool returns an error or unexpected data (e.g., "subcategory not found"), DO NOT give up immediately.
    *   Re-evaluate your plan:
        *   Did you use the correct spelling/name? Use `list_available` to verify.
        *   Is the data available for the requested period/category?
        *   Try a slightly different approach if appropriate (e.g., query for a broader category or a more general date).
    *   Explain the error and your recovery step.

13. **Analytical Queries (Hero/Zero Analysis, Performance Comparisons)**:
    *   For queries asking about "Hero" and "Zero" areas:
        - Hero = High-performing metrics (above average or top performers)
        - Zero = Low-performing metrics (below average or bottom performers)
    *   Steps for such analysis:
        a. Retrieve data for all relevant subcategories/metrics for the specified period
        b. Calculate averages or identify performance benchmarks
        c. Classify metrics as Hero (high) or Zero (low) based on performance
        d. For "turnarounds", compare with historical data to identify improvements or declines
    *   Example: "Show Hero and Zero areas" means identify which metrics are performing best and worst
   
   Example plan for Hero/Zero analysis:
   {
     "steps": [
       {
         "step_id": 1,
         "persona": "DataRetrieverMode",
         "goal": "Get current date context",
         "inputs": {"tool_to_call": "get_current_date", "tool_input_json": "{}"},
         "outputs_expected": "Current date information",
         "depends_on_step": null
       },
       {
         "step_id": 2,
         "persona": "DataRetrieverMode", 
         "goal": "Get all subcategory data for last week",
         "inputs": {"tool_to_call": "get_data_for_category", "tool_input_json": "{\"category\": \"OVERALL\", \"date\": \"last_week\"}"},
         "outputs_expected": "All metrics data for last week",
         "depends_on_step": 1
       },
       {
         "step_id": 3,
         "persona": "FinancialAnalysisMode",
         "goal": "Identify Hero (high performing) and Zero (low performing) metrics",
         "inputs": {"data_to_analyze": "[data_from_step_2]", "analysis_task": "Identify top 5 Hero metrics and bottom 5 Zero metrics based on values"},
         "outputs_expected": "Classified Hero and Zero metrics",
         "depends_on_step": 2
       },
       {
         "step_id": 4,
         "persona": "FinalResponseCompilationMode",
         "goal": "Compile Hero and Zero analysis results",
         "inputs": {
           "summary_points": [
             "[output_from_step_3]"
           ],
           "user_query": "Show Hero and Zero areas for last week"
         },
         "outputs_expected": "Final formatted response with Hero and Zero areas",
         "depends_on_step": 3
       }
     ]
   }

   14. **Final Answer Formulation**:
    *   Synthesize information from tool outputs into a clear, concise answer.
    *   ALWAYS present financial figures with ₹ and Cr units (e.g., "₹123.45 Cr").
    *   For period queries with multiple dates:
        - Present the summary (total/average) prominently
        - List individual date values if relevant
        - Example: "Net Revenue for last month (April 2025): Total ₹450.5 Cr across 4 weeks, averaging ₹112.6 Cr per week"
    *   If a chart was generated by `create_visualization`, mention that it has been displayed.

**PERSONA INPUT STRUCTURES:**

1. **DataRetrieverMode**:
   ```json
   {
     "tool_to_call": "get_data_for_category",
     "tool_input_json": "{\"category\": \"OVERALL\", \"date\": \"last_week\", \"subcategory\": \"Cash Score\"}"
   }
   ```

2. **VisualizationMode** (ALL fields are REQUIRED):
   ```json
   {
     "data_to_visualize": "[data_from_step_X]",
     "chart_type": "line",  // Required: "line", "bar", "pie", or "scatter"
     "x_axis": "date",       // Required: key from data for x-axis
     "y_axis": "value",      // Required: key from data for y-axis
     "title": "Cash Score Trend - Last 3 Months",  // Required: descriptive title
     "is_financial": true    // Optional, defaults to true
   }
   ```
   
   Example for Cash Score visualization:
   ```json
   {
     "data_to_visualize": "[data_from_step_3]",
     "chart_type": "line",
     "x_axis": "date",
     "y_axis": "Cash Score",
     "title": "JSPL Cash Score Performance - Last 3 Months",
     "is_financial": true
   }
   ```

3. **FinancialAnalysisMode**:
   ```json
   {
     "data_to_analyze": "[data_from_step_2]",
     "analysis_task": "Analyze the trend and performance of Cash Score over the last 3 months"
   }
   ```

**COMPLETE EXAMPLE PLAN - Cash Score Analysis with Visualization:**

For query: "How is the cash score of JSPL performing from last 3 months? Please comment. Also show graphically."

```json
{
  "steps": [
    {
      "step_id": 1,
      "persona": "DataRetrieverMode",
      "goal": "Get current date context",
      "inputs": {"tool_to_call": "get_current_date", "tool_input_json": "{}"},
      "outputs_expected": "Current date and date ranges",
      "depends_on_step": null
    },
    {
      "step_id": 2,
      "persona": "DataRetrieverMode",
      "goal": "Get Cash Score data for JSPL (OVERALL) for last 3 months",
      "inputs": {"tool_to_call": "get_data_for_category", "tool_input_json": "{\"category\": \"OVERALL\", \"subcategory\": \"Cash Score\", \"date\": \"last_3_months\"}"},
      "outputs_expected": "Cash Score values for multiple dates in last 3 months",
      "depends_on_step": 1
    },
    {
      "step_id": 3,
      "persona": "FinancialAnalysisMode",
      "goal": "Analyze Cash Score performance and trends",
      "inputs": {"data_to_analyze": "[data_from_step_2]", "analysis_task": "Analyze the performance trend of Cash Score over the last 3 months, identify if it's improving or declining, calculate average, and note any significant changes"},
      "outputs_expected": "Analysis insights about Cash Score performance",
      "depends_on_step": 2
    },
    {
      "step_id": 4,
      "persona": "VisualizationMode",
      "goal": "Create line chart showing Cash Score trend",
      "inputs": {
        "data_to_visualize": "[data_from_step_2]",
        "chart_type": "line",
        "x_axis": "date",
        "y_axis": "value",
        "title": "JSPL Cash Score Performance - Last 3 Months",
        "is_financial": true
      },
      "outputs_expected": "Line chart visualization",
      "depends_on_step": 2
    },
    {
      "step_id": 5,
      "persona": "FinalResponseCompilationMode",
      "goal": "Compile analysis and confirm visualization",
      "inputs": {
        "summary_points": [
          "Cash Score data for last 3 months: [output_from_step_2]",
          "Performance analysis: [output_from_step_3]",
          "Visualization status: [output_from_step_4]"
        ],
        "user_query": "How is the cash score of JSPL performing from last 3 months? Please comment. Also show graphically."
      },
      "outputs_expected": "Complete response with analysis and chart confirmation",
      "depends_on_step": [3, 4]
    }
  ]
}
```

Be methodical, show your reasoning, and ensure all steps are transparent. Your goal is to be a helpful and accurate analyst.


**EXAMPLE PLAN - Cash Score vs Target Analysis:**

For query: "What is our weekly Cash Score vs Target? Who contributed the most and least?"

```json
{
  "steps": [
    {
      "step_id": 1,
      "persona": "DataRetrieverMode",
      "goal": "Get current date context (required for 'weekly' keyword)",
      "inputs": {"tool_to_call": "get_current_date", "tool_input_json": "{}"},
      "outputs_expected": "Current date and week information",
      "depends_on_step": null
    },
    {
      "step_id": 2,
      "persona": "DataRetrieverMode",
      "goal": "Get Cash Score for this week",
      "inputs": {"tool_to_call": "get_data_for_category", "tool_input_json": "{\"category\": \"OVERALL\", \"subcategory\": \"Cash Score\", \"date\": \"this_week\"}"},
      "outputs_expected": "Cash Score value for current week",
      "depends_on_step": 1
    },
    {
      "step_id": 3,
      "persona": "DataRetrieverMode",
      "goal": "Get Cash Score Target for this week",
      "inputs": {"tool_to_call": "get_data_for_category", "tool_input_json": "{\"category\": \"OVERALL\", \"subcategory\": \"Cash Score Target\", \"date\": \"this_week\"}"},
      "outputs_expected": "Cash Score Target value for current week",
      "depends_on_step": 1
    },
    {
      "step_id": 4,
      "persona": "DataRetrieverMode",
      "goal": "Get all subcategory data to identify contributors",
      "inputs": {"tool_to_call": "get_data_for_category", "tool_input_json": "{\"category\": \"OVERALL\", \"date\": \"this_week\"}"},
      "outputs_expected": "All metrics for contribution analysis",
      "depends_on_step": 1
    },
    {
      "step_id": 5,
      "persona": "FinancialAnalysisMode",
      "goal": "Analyze Cash Score vs Target and identify top/bottom contributors",
      "inputs": {"data_to_analyze": "[data_from_step_2], [data_from_step_3], [data_from_step_4]", "analysis_task": "Compare Cash Score vs Target and identify which metrics contributed most and least to overall performance"},
      "outputs_expected": "Comparison analysis and contributor identification",
      "depends_on_step": [2, 3, 4]
    },
    {
      "step_id": 6,
      "persona": "FinalResponseCompilationMode",
      "goal": "Compile comprehensive response with analysis",
      "inputs": {
        "summary_points": [
          "Cash Score vs Target comparison: [output_from_step_5]",
          "Top and bottom contributors: [output_from_step_5]"
        ],
        "user_query": "What is our weekly Cash Score vs Target? Who contributed the most and least?"
      },
      "outputs_expected": "Final response with expert commentary",
      "depends_on_step": 5
    }
  ]
}
```


Example minimal plan structure:
1. DataRetrieverMode steps to get data
2. Analysis/Visualization steps as needed
3. FinalResponseCompilationMode to compile and generate expert commentary

The only exception is for simple listing queries (like "list available categories") where no analysis is performed.

**EXAMPLE PLAN - Category Comparison:**

For query: "Compare RAIGARH and ANGUL performance"

```json
{
  "steps": [
    {
      "step_id": 1,
      "persona": "DataRetrieverMode",
      "goal": "Get current date context",
      "inputs": {"tool_to_call": "get_current_date", "tool_input_json": "{}"},
      "outputs_expected": "Current date information",
      "depends_on_step": null
    },
    {
      "step_id": 2,
      "persona": "DataRetrieverMode",
      "goal": "Get performance data for RAIGARH",
      "inputs": {"tool_to_call": "get_data_for_category", "tool_input_json": "{\"category\": \"RAIGARH\", \"date\": \"last_month\"}"},
      "outputs_expected": "RAIGARH performance metrics",
      "depends_on_step": 1
    },
    {
      "step_id": 3,
      "persona": "DataRetrieverMode",
      "goal": "Get performance data for ANGUL",
      "inputs": {"tool_to_call": "get_data_for_category", "tool_input_json": "{\"category\": \"ANGUL\", \"date\": \"last_month\"}"},
      "outputs_expected": "ANGUL performance metrics",
      "depends_on_step": 1
    },
    {
      "step_id": 4,
      "persona": "FinancialAnalysisMode",
      "goal": "Compare performance metrics between RAIGARH and ANGUL",
      "inputs": {"data_to_analyze": "[data_from_step_2], [data_from_step_3]", "analysis_task": "Compare key performance metrics between RAIGARH and ANGUL, identify which is performing better and in which areas"},
      "outputs_expected": "Comparison analysis",
      "depends_on_step": [2, 3]
    },
    {
      "step_id": 5,
      "persona": "FinalResponseCompilationMode",
      "goal": "Compile comparison results",
      "inputs": {
        "summary_points": [
          "RAIGARH data: [output_from_step_2]",
          "ANGUL data: [output_from_step_3]", 
          "Comparison analysis: [output_from_step_4]"
        ],
        "user_query": "Compare RAIGARH and ANGUL performance"
      },
      "outputs_expected": "Complete comparison report",
      "depends_on_step": 4
    }
  ]
}
```


**EXAMPLE PLAN - Cash Score vs Target Analysis:**
"""

# Function to initialize the Orchestrator Chain
def initialize_orchestrator_chain():
    """Initializes and returns the main Orchestrator LCEL chain."""
    try:
        llm = ChatOpenAI(
            model_name="gpt-4-turbo", 
            temperature=0.1,
            max_tokens=4096,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_kwargs={
                "frequency_penalty": 0.1,
                "presence_penalty": 0.1,
                "top_p": 0.95
            }
        )
        
        output_parser = JsonOutputParser(pydantic_object=ExecutionPlan)
        
        # Bypass ChatPromptTemplate entirely to avoid its aggressive variable parsing
        # Create a custom chain that formats messages manually
        def format_messages(inputs):
            """Format the input query into messages for the LLM."""
            user_query = inputs.get("query", "")
            messages = [
                {"role": "system", "content": ORCHESTRATOR_AGENT_PROMPT},
                {"role": "user", "content": user_query}
            ]
            return messages
        
        # Create a simple chain: format messages -> LLM -> parse output
        # Using LCEL (LangChain Expression Language) composition
        from langchain_core.runnables import RunnableLambda
        
        orchestrator_chain = (
            RunnableLambda(format_messages) 
            | llm 
            | output_parser
        )
        
        st.success("Orchestrator Chain initialized successfully (using direct message formatting)!")
        return orchestrator_chain
    except Exception as e:
        st.error(f"Error initializing Orchestrator Chain: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None

# Sidebar for file upload and preview
with st.sidebar:
    st.title("📁 TOC Analysis Data")
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            st.write("Processing uploaded Excel file...")
            # Clear any existing data and agent to ensure fresh processing
            keys_to_delete = [
                "json_data", 
                "toc_agent",                # Old agent key
                "agent_executor",           # Another possible old agent key
                "orchestrator_chain",       # New chain, clear to re-initialize
                "agent_memory",             # Old memory object
                "messages",                 # Chat messages
                "last_chart_figure_json",
                "last_chart_figure"
            ]
            for key in keys_to_delete:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Re-initialize critical session components after clearing
            st.session_state.messages = []
            st.session_state.agent_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True) # If orchestrator uses memory
            
            json_data = excel_to_json(uploaded_file)
            if json_data:
                st.session_state.json_data = json_data
                st.success("Excel file processed and data extracted!")
                
                st.subheader("📊 Data Structure Preview")
                for main_category_key in ["OVERALL", "RAIGARH", "ANGUL"]:
                    if main_category_key in json_data and json_data[main_category_key].get("dates"): # Check if dates exist
                        st.markdown(f"**{main_category_key}**:")
                        subcategories = json_data[main_category_key].get("subcategories", [])
                        st.write(f"- Subcategories ({len(subcategories)}): `{', '.join(subcategories[:3])}...`")
                        
                        dates_available = list(json_data[main_category_key]["dates"].keys())
                        st.write(f"- Date ranges ({len(dates_available)}): `{', '.join(dates_available[:2])}...`")
                        if dates_available:
                            first_date_found = dates_available[0]
                            sample_data_items = list(json_data[main_category_key]["dates"][first_date_found].items())
                            st.write(f"  Sample for `{first_date_found}` ({len(sample_data_items)} items): `{dict(sample_data_items[:2])}...`")
                        st.markdown("---")
                    elif main_category_key in json_data:
                         st.warning(f"Category '{main_category_key}' processed but no date-specific data found.")
            else:
                st.error("Failed to process Excel file into usable JSON data. Please check the debug messages above and the Excel sheet format.")
                st.session_state.json_data = None # Clear previous data if processing fails

        except Exception as e:
            st.error(f"Error during Excel file processing in sidebar: {str(e)}")
            st.session_state.json_data = None
            import traceback
            st.error(traceback.format_exc())
    elif st.session_state.get("json_data") is None: # Only show if no data is loaded yet
        st.info("👈 Please upload your Excel file to begin analysis.")

# Main chat area
st.title("🏭 Jindal Steel TOC Advisor")

if st.session_state.get("json_data") is not None:
    if "orchestrator_chain" not in st.session_state or st.session_state.orchestrator_chain is None:
        # Initialize the orchestrator chain if it's not already in session state
        st.session_state.orchestrator_chain = initialize_orchestrator_chain()
    
    orchestrator_chain = st.session_state.orchestrator_chain

    if orchestrator_chain:
        # Display existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Handle user input
        if user_prompt := st.chat_input("Ask a question about your Jindal Steel TOC data..."):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            # Initialize for this interaction
            step_outputs = {} 
            final_agent_response_content = "Processing your request..." # Default
            
            with st.chat_message("assistant"):
                # ---- START: New Orchestration Logic ----
                try:
                    with st.spinner("Orchestrator is planning your request..."):
                        # 1. Invoke Orchestrator Chain to get the plan
                        plan_dict = orchestrator_chain.invoke({"query": user_prompt})
                        # Validate plan with Pydantic model (will raise error if not compliant)
                        execution_plan = ExecutionPlan(**plan_dict) 
                        
                        st.markdown("#### Orchestrator's Plan:")
                        for s_idx, s_detail in enumerate(execution_plan.steps):
                            with st.expander(f"Step {s_detail.step_id}: {s_detail.persona} - {s_detail.goal}", expanded=(s_idx==0)):
                                st.json({
                                    "persona": s_detail.persona,
                                    "goal": s_detail.goal,
                                    "inputs": s_detail.inputs,
                                    "outputs_expected": s_detail.outputs_expected,
                                    "depends_on_step": s_detail.depends_on_step
                                })
                        execution_steps = execution_plan.steps

                except Exception as e_plan:
                    st.error(f"Error during plan generation by Orchestrator: {e_plan}")
                    import traceback
                    st.error(f"Traceback: {traceback.format_exc()}")
                    execution_steps = []
                    final_agent_response_content = "I encountered an error while planning how to respond to your request."

                # 2. Execute the plan step-by-step if plan generation was successful
                if execution_steps:
                    for step_details in sorted(execution_steps, key=lambda x: x.step_id):
                        step_id = step_details.step_id
                        persona = step_details.persona
                        goal = step_details.goal
                        inputs_template = step_details.inputs # This is a dict from Pydantic model
                        depends_on_step_id = step_details.depends_on_step

                        st.markdown(f"--- \n**Executing Step {step_id}: {persona}**\n*Goal: {goal}*")
                        current_step_output = None
                        
                        try:
                            # Substitute inputs from previous steps
                            processed_inputs = inputs_template.copy()
                            
                            # Handle dependencies - could be single int or list of ints
                            if depends_on_step_id:
                                # Normalize to list for uniform handling
                                if isinstance(depends_on_step_id, int):
                                    dependency_list = [depends_on_step_id]
                                elif isinstance(depends_on_step_id, list):
                                    dependency_list = depends_on_step_id
                                else:
                                    dependency_list = []
                                
                                # Process each dependency for non-list values first
                                for dep_id in dependency_list:
                                    if dep_id in step_outputs:
                                        previous_output = step_outputs[dep_id]
                                        for key, value_template in list(processed_inputs.items()):  # Use list() to avoid dict size change during iteration
                                            if isinstance(value_template, str) and key != "summary_points":
                                                # General placeholder for direct data substitution
                                                placeholder = f"[data_from_step_{dep_id}]"
                                                if placeholder in value_template:
                                                    processed_inputs[key] = value_template.replace(placeholder, str(previous_output))
                                                    value_template = processed_inputs[key]  # Update for next placeholder
                                                
                                                # Placeholder for analysis output
                                                analysis_placeholder = f"[analysis_from_step_{dep_id}]"
                                                if analysis_placeholder in value_template:
                                                    processed_inputs[key] = value_template.replace(analysis_placeholder, str(previous_output))
                                                    value_template = processed_inputs[key]  # Update for next placeholder
                                                
                                                # Generic output placeholder
                                                output_placeholder = f"[output_from_step_{dep_id}]"
                                                if output_placeholder in value_template:
                                                    processed_inputs[key] = value_template.replace(output_placeholder, str(previous_output))

                                # Now handle summary_points separately, processing all dependencies
                                if "summary_points" in processed_inputs and isinstance(processed_inputs["summary_points"], list):
                                    new_summary_points = []
                                    for point_template in processed_inputs["summary_points"]:
                                        if isinstance(point_template, str):
                                            new_point = point_template
                                            # Process all dependencies for this point
                                            for dep_id in dependency_list:
                                                if dep_id in step_outputs:
                                                    previous_output = step_outputs[dep_id]
                                                    # Check for any placeholder pattern with this dependency
                                                    for placeholder_pattern in [f"[output_from_step_{dep_id}]", f"[data_from_step_{dep_id}]", f"[analysis_from_step_{dep_id}]"]:
                                                        if placeholder_pattern in new_point:
                                                            # If the point is ONLY the placeholder, replace entirely
                                                            if new_point.strip() == placeholder_pattern:
                                                                new_point = str(previous_output)
                                                                st.write(f"Debug: Replaced entire placeholder '{placeholder_pattern}' with output")
                                                            # If it ends with the placeholder after a colon
                                                            elif new_point.strip().endswith(": " + placeholder_pattern):
                                                                prefix = new_point.replace(placeholder_pattern, "").strip().rstrip(":")
                                                                new_point = f"{prefix}: {previous_output}"
                                                                st.write(f"Debug: Replaced placeholder after colon in '{point_template[:50]}...'")
                                                            # Otherwise replace inline
                                                            else:
                                                                new_point = new_point.replace(placeholder_pattern, str(previous_output))
                                                                st.write(f"Debug: Replaced inline placeholder '{placeholder_pattern}'")
                                            new_summary_points.append(new_point)
                                        else:
                                            new_summary_points.append(point_template)
                                    processed_inputs["summary_points"] = new_summary_points
                            
                            st.write("Inputs for this step (after substitution):")
                            st.json(processed_inputs)

                            # Persona-based execution
                            if persona == "DataRetrieverMode":
                                dr_inputs = DataRetrieverInputs(**processed_inputs)
                                tool_name = dr_inputs.tool_to_call
                                tool_input_json_str = dr_inputs.tool_input_json
                                target_tool = next((t for t in AVAILABLE_TOOLS if t.name == tool_name), None)
                                if target_tool:
                                    with st.spinner(f"Calling tool: {tool_name}..."):
                                        current_step_output = target_tool.run(tool_input_json_str)
                                else:
                                    current_step_output = json.dumps({"error": f"Tool '{tool_name}' not found."})
                                st.markdown(f"**Output from {tool_name}:**")
                                try: st.json(json.loads(current_step_output))
                                except: st.text(str(current_step_output))

                            elif persona == "VisualizationMode":
                                viz_inputs = VisualizationInputs(**processed_inputs)
                                data_for_viz = viz_inputs.data_to_visualize
                                if isinstance(data_for_viz, str):
                                    try: 
                                        data_for_viz = json.loads(data_for_viz)
                                        # If the data has a "data" field (from get_data_for_category multi-date response), extract it
                                        if isinstance(data_for_viz, dict) and "data" in data_for_viz:
                                            st.write("Debug: Extracting 'data' array from get_data_for_category response")
                                            data_for_viz = data_for_viz["data"]
                                    except json.JSONDecodeError: 
                                        raise ValueError("Visualization data is a string but not valid JSON.")
                                
                                # Ensure data_for_viz is a list of dictionaries
                                if not isinstance(data_for_viz, list):
                                    st.error(f"Visualization data must be a list of dictionaries, got {type(data_for_viz)}")
                                    raise ValueError(f"Invalid data format for visualization: expected list, got {type(data_for_viz)}")
                                
                                # Create tool input with proper field mapping
                                viz_tool_input = {
                                    "type": viz_inputs.chart_type,  # Map chart_type to type
                                    "data": data_for_viz,
                                    "x": viz_inputs.x_axis,
                                    "y": viz_inputs.y_axis,
                                    "title": viz_inputs.title,
                                    "is_financial": viz_inputs.is_financial
                                }
                                
                                # Validate that x_axis and y_axis keys exist in the data
                                if data_for_viz and len(data_for_viz) > 0:
                                    first_item = data_for_viz[0]
                                    if viz_inputs.x_axis not in first_item:
                                        st.warning(f"x_axis '{viz_inputs.x_axis}' not found in data. Available keys: {list(first_item.keys())}")
                                    if viz_inputs.y_axis not in first_item:
                                        st.warning(f"y_axis '{viz_inputs.y_axis}' not found in data. Available keys: {list(first_item.keys())}")
                                
                                with st.spinner(f"Generating {viz_inputs.chart_type} chart..."):
                                     current_step_output = create_visualization.run(json.dumps(viz_tool_input))
                                st.markdown("**Visualization Tool Output:**"); st.json(json.loads(current_step_output))

                            # --- Placeholder Personas --- 
                            elif persona == "FinancialAnalysisMode":
                                fa_inputs = FinancialAnalysisInputs(**processed_inputs)
                                st.markdown(f"**Financial Analysis for:** {fa_inputs.analysis_task}")
                                
                                # Parse the data to analyze
                                data_to_analyze = fa_inputs.data_to_analyze
                                if isinstance(data_to_analyze, str):
                                    try:
                                        data_to_analyze = json.loads(data_to_analyze)
                                    except:
                                        pass
                                
                                # Perform the analysis based on the task
                                analysis_result = {}
                                
                                if "Hero" in fa_inputs.analysis_task and "Zero" in fa_inputs.analysis_task:
                                    # Hero/Zero analysis
                                    st.write("Performing Hero/Zero analysis...")
                                    
                                    if isinstance(data_to_analyze, dict):
                                        # Extract metrics and values
                                        metrics_data = []
                                        
                                        # Handle different data structures
                                        if "values" in data_to_analyze:
                                            # Single date data
                                            for metric, value in data_to_analyze["values"].items():
                                                if isinstance(value, (int, float)):
                                                    metrics_data.append({"metric": metric, "value": value})
                                        elif "data" in data_to_analyze:
                                            # Multiple dates - calculate averages
                                            if isinstance(data_to_analyze["data"], dict):
                                                # Aggregate metrics across dates
                                                metric_totals = {}
                                                metric_counts = {}
                                                
                                                for date, date_data in data_to_analyze["data"].items():
                                                    for metric, value in date_data.items():
                                                        if isinstance(value, (int, float)):
                                                            if metric not in metric_totals:
                                                                metric_totals[metric] = 0
                                                                metric_counts[metric] = 0
                                                            metric_totals[metric] += value
                                                            metric_counts[metric] += 1
                                                
                                                # Calculate averages
                                                for metric in metric_totals:
                                                    avg_value = metric_totals[metric] / metric_counts[metric]
                                                    metrics_data.append({"metric": metric, "value": avg_value})
                                        
                                        # Sort metrics by value
                                        metrics_data.sort(key=lambda x: x["value"], reverse=True)
                                        
                                        # Identify Heroes and Zeros
                                        hero_count = min(5, len(metrics_data) // 2)
                                        zero_count = min(5, len(metrics_data) // 2)
                                        
                                        heroes = metrics_data[:hero_count]
                                        zeros = metrics_data[-zero_count:] if zero_count > 0 else []
                                        
                                        analysis_result = {
                                            "hero_areas": [
                                                {
                                                    "metric": h["metric"],
                                                    "value": h["value"],
                                                    "performance": "High Performer"
                                                } for h in heroes
                                            ],
                                            "zero_areas": [
                                                {
                                                    "metric": z["metric"],
                                                    "value": z["value"],
                                                    "performance": "Low Performer"
                                                } for z in zeros
                                            ],
                                            "total_metrics_analyzed": len(metrics_data)
                                        }
                                        
                                        # Display the results
                                        st.write("**Hero Areas (Top Performers):**")
                                        for hero in analysis_result["hero_areas"]:
                                            st.write(f"- {hero['metric']}: ₹{hero['value']:.2f} Cr")
                                        
                                        st.write("**Zero Areas (Low Performers):**")
                                        for zero in analysis_result["zero_areas"]:
                                            st.write(f"- {zero['metric']}: ₹{zero['value']:.2f} Cr")

                                        current_step_output = json.dumps(analysis_result)
                                        st.json(analysis_result)
                                
                                elif "compare" in fa_inputs.analysis_task.lower() or "comparison" in fa_inputs.analysis_task.lower():
                                    # Comparison analysis between categories
                                    st.write("Performing comparison analysis...")
                                    
                                    # Parse multiple data inputs (expecting list or comma-separated)
                                    data_parts = []
                                    if isinstance(data_to_analyze, str):
                                        # Try to parse as JSON array first
                                        try:
                                            parsed_data = json.loads(data_to_analyze)
                                            if isinstance(parsed_data, list):
                                                data_parts = parsed_data
                                            else:
                                                data_parts = [parsed_data]
                                        except:
                                            # Split by comma if not valid JSON
                                            parts = data_to_analyze.split("], ")
                                            for part in parts:
                                                try:
                                                    clean_part = part.strip()
                                                    if not clean_part.endswith("]"):
                                                        clean_part += "]"
                                                    data_parts.append(json.loads(clean_part))
                                                except:
                                                    pass
                                    elif isinstance(data_to_analyze, list):
                                        data_parts = data_to_analyze
                                    
                                    if len(data_parts) >= 2:
                                        # Extract category names and data
                                        comparisons = {}
                                        for part in data_parts:
                                            if isinstance(part, dict):
                                                category = part.get("category", "Unknown")
                                                if "values" in part:
                                                    comparisons[category] = part["values"]
                                                elif "data" in part:
                                                    # Aggregate if multiple dates
                                                    aggregated = {}
                                                    for date_data in part["data"].values():
                                                        for metric, value in date_data.items():
                                                            if isinstance(value, (int, float)):
                                                                if metric not in aggregated:
                                                                    aggregated[metric] = []
                                                                aggregated[metric].append(value)
                                                    # Calculate averages
                                                    comparisons[category] = {
                                                        metric: sum(values)/len(values) 
                                                        for metric, values in aggregated.items()
                                                    }
                                        
                                        # Perform comparison
                                        if len(comparisons) >= 2:
                                            categories = list(comparisons.keys())
                                            cat1, cat2 = categories[0], categories[1]
                                            
                                            # Find common metrics
                                            common_metrics = set(comparisons[cat1].keys()) & set(comparisons[cat2].keys())
                                            
                                            analysis_result = {
                                                "comparison": f"{cat1} vs {cat2}",
                                                "metrics_compared": len(common_metrics),
                                                "details": []
                                            }
                                            
                                            for metric in common_metrics:
                                                val1 = comparisons[cat1][metric]
                                                val2 = comparisons[cat2][metric]
                                                diff = val2 - val1
                                                pct_diff = (diff / val1 * 100) if val1 != 0 else 0
                                                
                                                analysis_result["details"].append({
                                                    "metric": metric,
                                                    cat1: val1,
                                                    cat2: val2,
                                                    "difference": diff,
                                                    "percentage_difference": pct_diff,
                                                    "better_performer": cat1 if val1 > val2 else cat2
                                                })
                                            
                                            # Sort by absolute percentage difference
                                            analysis_result["details"].sort(
                                                key=lambda x: abs(x["percentage_difference"]), 
                                                reverse=True
                                            )
                                            
                                            # Display results
                                            st.write(f"**Comparison: {cat1} vs {cat2}**")
                                            st.write(f"Metrics compared: {len(common_metrics)}")
                                            
                                            for detail in analysis_result["details"][:5]:  # Top 5 differences
                                                metric = detail["metric"]
                                                better = detail["better_performer"]
                                                pct = abs(detail["percentage_difference"])
                                                st.write(f"- **{metric}**: {better} performs {pct:.1f}% better")
                                                st.write(f"  - {cat1}: ₹{detail[cat1]:.2f} Cr")
                                                st.write(f"  - {cat2}: ₹{detail[cat2]:.2f} Cr")
                                            
                                            current_step_output = json.dumps(analysis_result)
                                        else:
                                            current_step_output = json.dumps({"error": "Not enough data for comparison"})
                                    else:
                                        current_step_output = json.dumps({"error": "Need at least 2 datasets for comparison"})
                                
                                elif "trend" in fa_inputs.analysis_task.lower() or "performance" in fa_inputs.analysis_task.lower():
                                    # Trend analysis
                                    st.write("Performing trend analysis...")
                                    
                                    # Handle data structure with period data
                                    if isinstance(data_to_analyze, dict) and "data" in data_to_analyze:
                                        # Extract the time series data
                                        time_series_data = data_to_analyze["data"]
                                        metric_name = data_to_analyze.get("subcategory", "Metric")
                                        summary = data_to_analyze.get("summary", {})
                                        
                                        # Sort data by date for proper trend analysis
                                        sorted_dates = sorted(time_series_data, key=lambda x: x.get("date", ""))
                                        
                                        # Extract values for analysis
                                        values = [item["value"] for item in sorted_dates if isinstance(item.get("value"), (int, float))]
                                        dates = [item["date"] for item in sorted_dates]
                                        
                                        if len(values) >= 2:
                                            # Calculate trend metrics
                                            first_value = values[0]
                                            last_value = values[-1]
                                            overall_change = last_value - first_value
                                            percentage_change = (overall_change / first_value * 100) if first_value != 0 else 0
                                            
                                            # Determine trend direction
                                            if percentage_change > 5:
                                                trend_direction = "improving"
                                            elif percentage_change < -5:
                                                trend_direction = "declining"
                                            else:
                                                trend_direction = "stable"
                                            
                                            # Find high and low points
                                            max_value = max(values)
                                            min_value = min(values)
                                            max_index = values.index(max_value)
                                            min_index = values.index(min_value)
                                            
                                            # Check for volatility
                                            avg_value = sum(values) / len(values)
                                            deviations = [abs(v - avg_value) for v in values]
                                            avg_deviation = sum(deviations) / len(deviations)
                                            volatility_pct = (avg_deviation / avg_value * 100) if avg_value != 0 else 0
                                            
                                            # Create analysis result
                                            analysis_result = {
                                                "metric": metric_name,
                                                "period_analyzed": f"{dates[0]} to {dates[-1]}",
                                                "data_points": len(values),
                                                "trend_direction": trend_direction,
                                                "overall_change": {
                                                    "absolute": overall_change,
                                                    "percentage": percentage_change
                                                },
                                                "starting_value": first_value,
                                                "ending_value": last_value,
                                                "average_value": avg_value,
                                                "peak": {
                                                    "value": max_value,
                                                    "date": dates[max_index]
                                                },
                                                "trough": {
                                                    "value": min_value,
                                                    "date": dates[min_index]
                                                },
                                                "volatility_percentage": volatility_pct,
                                                "insights": []
                                            }
                                            
                                            # Generate insights
                                            insights = []
                                            
                                            # Trend insight
                                            if trend_direction == "improving":
                                                insights.append(f"Cash Score shows an improving trend with {percentage_change:.1f}% growth over the period")
                                            elif trend_direction == "declining":
                                                insights.append(f"Cash Score shows a declining trend with {abs(percentage_change):.1f}% decrease over the period")
                                            else:
                                                insights.append(f"Cash Score remains relatively stable with only {abs(percentage_change):.1f}% change")
                                            
                                            # Volatility insight
                                            if volatility_pct > 30:
                                                insights.append(f"High volatility observed ({volatility_pct:.1f}% average deviation) indicating unstable performance")
                                            elif volatility_pct < 10:
                                                insights.append(f"Low volatility ({volatility_pct:.1f}% average deviation) indicates consistent performance")
                                            
                                            # Performance levels
                                            if avg_value < 50:
                                                insights.append(f"Average Cash Score of ₹{avg_value:.1f} Cr is below optimal levels (target: ₹100+ Cr)")
                                            elif avg_value > 100:
                                                insights.append(f"Strong average Cash Score of ₹{avg_value:.1f} Cr exceeds target levels")
                                            
                                            # Recent performance
                                            if len(values) > 4:
                                                recent_avg = sum(values[-4:]) / 4
                                                older_avg = sum(values[:-4]) / (len(values) - 4)
                                                if recent_avg > older_avg * 1.1:
                                                    insights.append("Recent performance (last 4 weeks) shows improvement compared to earlier periods")
                                                elif recent_avg < older_avg * 0.9:
                                                    insights.append("Recent performance (last 4 weeks) shows decline compared to earlier periods")
                                            
                                            analysis_result["insights"] = insights
                                            
                                            # Display results
                                            st.write(f"**Trend Analysis for {metric_name}**")
                                            st.write(f"- Period: {analysis_result['period_analyzed']} ({len(values)} data points)")
                                            st.write(f"- Trend: **{trend_direction.upper()}** ({percentage_change:+.1f}%)")
                                            st.write(f"- Range: ₹{first_value:.1f} Cr → ₹{last_value:.1f} Cr")
                                            st.write(f"- Average: ₹{avg_value:.1f} Cr")
                                            st.write(f"- Peak: ₹{max_value:.1f} Cr on {dates[max_index]}")
                                            st.write(f"- Trough: ₹{min_value:.1f} Cr on {dates[min_index]}")
                                            
                                            st.write("\n**Key Insights:**")
                                            for insight in insights:
                                                st.write(f"- {insight}")
                                            
                                            current_step_output = json.dumps(analysis_result)
                                        else:
                                            current_step_output = json.dumps({"error": "Not enough data points for trend analysis"})
                                    else:
                                        current_step_output = json.dumps({"error": "Invalid data format for trend analysis"})
                                
                                else:
                                    # Generic financial analysis
                                    current_step_output = f"Financial analysis completed for: {fa_inputs.analysis_task}"
                                    st.info(current_step_output)

                            elif persona == "TOCStrategyMode":
                                ts_inputs = TOCStrategyInputs(**processed_inputs)
                                st.markdown(f"**Conceptual TOC Strategy for:** {ts_inputs.strategic_question}")
                                # Placeholder: Invoke LLM with TOC strategy persona & ts_inputs.analyzed_data, ts_inputs.strategic_question
                                current_step_output = f"TOC strategic advice for '{ts_inputs.strategic_question}' based on analyzed data would appear here."
                                st.info(current_step_output)
                            
                            elif persona == "FinalResponseCompilationMode":
                                fr_inputs = FinalResponseCompilationInputs(**processed_inputs)
                                st.markdown("**Compiling Final Response...**")
                                
                                # Debug: Show what we received
                                st.write("Debug: Summary points received:")
                                for i, sp in enumerate(fr_inputs.summary_points):
                                    sp_str = str(sp)
                                    st.write(f"Point {i+1} (type: {type(sp).__name__}, length: {len(sp_str)})")
                                    if len(sp_str) > 200:
                                        st.write(f"  Content: {sp_str[:200]}...")
                                    else:
                                        st.write(f"  Content: {sp_str}")
                                    
                                    # Check if it looks like JSON
                                    if sp_str.strip().startswith("{") and sp_str.strip().endswith("}"):
                                        st.write("  ^ Looks like JSON data")
                                    elif sp_str.strip().startswith("[") and not sp_str.strip().startswith("[output"):
                                        st.write("  ^ Looks like a JSON array")
                                    elif "[output_from_step" in sp_str or "[data_from_step" in sp_str:
                                        st.write("  ^ Contains unreplaced placeholder!")
                                
                                # Enhanced compilation logic for different types of analysis
                                response_parts = []
                                
                                # First, let's parse all JSON data from summary points
                                parsed_data = {}
                                for point in fr_inputs.summary_points:
                                    point_str = str(point)
                                    # Try to parse as JSON
                                    try:
                                        if point_str.startswith("{") and point_str.endswith("}"):
                                            data = json.loads(point_str)
                                            # Store parsed data by type
                                            if "hero_areas" in data or "zero_areas" in data:
                                                parsed_data["hero_zero"] = data
                                            elif "trend_direction" in data:
                                                parsed_data["trend"] = data
                                            elif "success" in data and "chart" in str(data).lower():
                                                parsed_data["visualization"] = data
                                    except:
                                        pass
                                
                                # Check for trend analysis
                                if parsed_data.get("trend") or any("trend" in str(sp).lower() for sp in fr_inputs.summary_points):
                                    # Handle trend analysis results
                                    if parsed_data.get("trend"):
                                        analysis_data = parsed_data["trend"]
                                        # Format trend analysis results
                                        if "metric" in analysis_data:
                                            response_parts.append(f"## {analysis_data['metric']} Analysis\n")
                                            response_parts.append(f"**Period Analyzed**: {analysis_data.get('period_analyzed', 'N/A')}")
                                            response_parts.append(f"**Data Points**: {analysis_data.get('data_points', 'N/A')}\n")
                                            
                                            # Trend summary
                                            trend = analysis_data.get('trend_direction', 'unknown').upper()
                                            change = analysis_data.get('overall_change', {})
                                            response_parts.append(f"### Trend: {trend}")
                                            if isinstance(change, dict):
                                                response_parts.append(f"- Overall Change: ₹{change.get('absolute', 0):.1f} Cr ({change.get('percentage', 0):+.1f}%)")
                                            
                                            # Performance metrics
                                            response_parts.append(f"\n### Performance Metrics")
                                            response_parts.append(f"- Starting Value: ₹{analysis_data.get('starting_value', 0):.1f} Cr")
                                            response_parts.append(f"- Ending Value: ₹{analysis_data.get('ending_value', 0):.1f} Cr")
                                            response_parts.append(f"- Average: ₹{analysis_data.get('average_value', 0):.1f} Cr")
                                            
                                            peak = analysis_data.get('peak', {})
                                            trough = analysis_data.get('trough', {})
                                            if peak and trough:
                                                response_parts.append(f"- Peak: ₹{peak.get('value', 0):.1f} Cr on {peak.get('date', 'N/A')}")
                                                response_parts.append(f"- Trough: ₹{trough.get('value', 0):.1f} Cr on {trough.get('date', 'N/A')}")
                                            
                                            # Key insights
                                            insights = analysis_data.get('insights', [])
                                            if insights:
                                                response_parts.append(f"\n### Key Insights")
                                                for insight in insights:
                                                    response_parts.append(f"- {insight}")
                                    
                                    # Add visualization status if present
                                    if parsed_data.get("visualization", {}).get("success"):
                                        response_parts.append(f"\n✅ {parsed_data['visualization'].get('message', 'Chart generated successfully.')}")
                                
                                # Check for Hero/Zero analysis
                                elif parsed_data.get("hero_zero") or any("hero" in str(sp).lower() or "zero" in str(sp).lower() for sp in fr_inputs.summary_points):
                                    # Format Hero/Zero results
                                    if parsed_data.get("hero_zero"):
                                        data = parsed_data["hero_zero"]
                                        response_parts.append("## Hero and Zero Areas Analysis\n")
                                        
                                        if "hero_areas" in data and data["hero_areas"]:
                                            response_parts.append("### 🌟 Hero Areas (Top Performers):")
                                            for hero in data["hero_areas"]:
                                                response_parts.append(f"- **{hero['metric']}**: ₹{hero['value']:.2f} Cr")
                                        
                                        if "zero_areas" in data and data["zero_areas"]:
                                            response_parts.append("\n### ⚠️ Zero Areas (Low Performers):")
                                            for zero in data["zero_areas"]:
                                                response_parts.append(f"- **{zero['metric']}**: ₹{zero['value']:.2f} Cr")
                                        
                                        # Add summary
                                        total_analyzed = data.get("total_metrics_analyzed", 0)
                                        if total_analyzed:
                                            response_parts.append(f"\n*Total metrics analyzed: {total_analyzed}*")
                                    else:
                                        # If no parsed data but keywords present, show a message
                                        response_parts.append("## Hero and Zero Areas Analysis")
                                        response_parts.append("Analysis data is being processed. Please check the results above.")
                                
                                # Check for comparison analysis
                                elif any("compar" in str(sp).lower() for sp in fr_inputs.summary_points):
                                    # Look for comparison data in summary points
                                    for point in fr_inputs.summary_points:
                                        point_str = str(point)
                                        if "comparison" in point_str or "metrics_compared" in point_str:
                                            try:
                                                if point_str.startswith("{"):
                                                    comp_data = json.loads(point_str)
                                                    response_parts.append(f"## {comp_data.get('comparison', 'Comparison Analysis')}")
                                                    response_parts.append(f"*Metrics compared: {comp_data.get('metrics_compared', 0)}*\n")
                                                    
                                                    # Show top differences
                                                    details = comp_data.get('details', [])
                                                    if details:
                                                        response_parts.append("### Key Differences:")
                                                        for detail in details[:5]:
                                                            metric = detail.get('metric', 'Unknown')
                                                            better = detail.get('better_performer', 'N/A')
                                                            pct = abs(detail.get('percentage_difference', 0))
                                                            response_parts.append(f"- **{metric}**: {better} performs {pct:.1f}% better")
                                            except:
                                                pass
                                
                                else:
                                    # Default compilation - show any remaining content
                                    has_content = False
                                    for point in fr_inputs.summary_points:
                                        point_str = str(point)
                                        # Skip placeholders and redundant text
                                        if (not point_str.startswith("[output_from") and 
                                            not point_str.startswith("[data_from") and
                                            not point_str.startswith("Hero areas from") and
                                            not point_str.startswith("Zero areas from") and
                                            not point_str.startswith("Turnarounds in") and
                                            point_str.strip()):
                                            
                                            # Try to parse and format any JSON data
                                            if point_str.startswith("{") and point_str.endswith("}"):
                                                try:
                                                    data = json.loads(point_str)
                                                    response_parts.append("### Analysis Results:")
                                                    response_parts.append(f"```json\n{json.dumps(data, indent=2)}\n```")
                                                    has_content = True
                                                except:
                                                    response_parts.append(point_str)
                                                    has_content = True
                                            else:
                                                response_parts.append(point_str)
                                                has_content = True
                                    
                                    if not has_content:
                                        response_parts.append("Analysis completed. Please check the detailed results above.")
                                
                                # Join all parts with proper formatting
                                if response_parts:
                                    final_agent_response_content = "\n".join(response_parts)
                                else:
                                    final_agent_response_content = "Analysis completed. Please see the results above."
                                
                                current_step_output = final_agent_response_content
                                st.info("Response compiled successfully.")
                            else:
                                current_step_output = json.dumps({"error": f"Unknown persona: {persona}"})
                                st.error(current_step_output)

                            step_outputs[step_id] = current_step_output

                        except Exception as step_exec_e:
                            st.error(f"Error executing step {step_id} ({persona}): {step_exec_e}")
                            import traceback; st.error(f"Traceback: {traceback.format_exc()}")
                            step_outputs[step_id] = json.dumps({"error": f"Execution error in step {step_id} ({persona}): {str(step_exec_e)}"})
                            final_agent_response_content = f"I encountered an error trying to complete step {step_id}."
                            break # Stop further execution
                    
                    # Fallback if FinalResponseCompilationMode didn't explicitly set the final content
                    if final_agent_response_content == "Processing your request..." and step_outputs:
                        last_successful_output = step_outputs.get(max(step_outputs.keys()) if step_outputs else None)
                        if last_successful_output: final_agent_response_content = str(last_successful_output)
                        else: final_agent_response_content = "Processing completed, but no final response was generated."

                elif not final_agent_response_content or final_agent_response_content == "Processing your request...": # No steps from orchestrator, or plan error set specific message
                    if not final_agent_response_content or final_agent_response_content == "Processing your request...":
                        final_agent_response_content = "I was unable to create a processing plan for your request."
                    st.warning(final_agent_response_content)
                
                # ---- END: New Orchestration Logic ----

                # Display chart if one was generated
                if "last_chart_figure" in st.session_state and st.session_state.last_chart_figure is not None:
                    st.markdown("### Generated Chart")
                    chart_fig = st.session_state.last_chart_figure
                    if hasattr(chart_fig, 'data') and len(chart_fig.data) > 0:
                        st.plotly_chart(chart_fig, use_container_width=True)
                    else: st.error("Chart object exists but has no data.")
                    st.session_state.last_chart_figure = None # Clear after displaying
                
                # Display the final response
                st.markdown("---")
                st.markdown(final_agent_response_content)

            # Store the response in message history
            st.session_state.messages.append({"role": "assistant", "content": final_agent_response_content})

    else: # Orchestrator chain failed to initialize
        st.error("Orchestrator Chain failed to initialize. Please check configuration or API keys.")
elif not uploaded_file: # Check if uploaded_file is None from the sidebar context
    st.info("👈 Please upload your Excel file in the sidebar to begin TOC analysis.")