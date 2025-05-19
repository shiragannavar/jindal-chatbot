import streamlit as st
import pandas as pd
import numpy as np
import json
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
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
        
        # Heuristic: Date headers are expected in row 6 (index 5)
        date_header_row_index = 5 
        
        if date_header_row_index >= len(df):
            st.error(f"Date header row index {date_header_row_index + 1} is out of bounds. Excel sheet has {len(df)} rows.")
            return None

        date_headers_raw = df.iloc[date_header_row_index, 1:] # Dates start from column B (index 1)
        # Filter out NaN and empty strings more robustly
        date_headers = [str(h).strip() for h in date_headers_raw if pd.notna(h) and str(h).strip() != ""]
        
        st.write(f"Debug: Extracted Date headers from Excel row {date_header_row_index + 1}: {date_headers}")
        
        if not date_headers:
            st.error(f"No date headers found in Excel row {date_header_row_index + 1} (starting from column B).")
            # Fallback search for rows that look like date headers
            st.write("Attempting fallback to find date headers in nearby rows...")
            found_fallback_dates = False
            for i in range(max(0, date_header_row_index - 3), min(len(df), date_header_row_index + 4)):
                if i == date_header_row_index: continue # Already checked
                temp_headers_raw = df.iloc[i, 1:]
                temp_headers = [str(h).strip() for h in temp_headers_raw if pd.notna(h) and str(h).strip() != ""]
                # A simple check: if a row has several short, non-numeric strings, it might be dates
                if len(temp_headers) > 3 and all(len(th) < 20 and not th.replace('.','',1).isdigit() for th in temp_headers[:3]): # Check first 3
                    st.warning(f"Potential fallback date headers found in Excel row {i+1}: {temp_headers}. Using these.")
                    date_headers = temp_headers
                    date_header_row_index = i
                    found_fallback_dates = True
                    break
            if not found_fallback_dates:
                 st.error("Fallback failed. No valid date headers found. Please ensure dates are in a single row, typically row 6.")
                 return None
        
        result = {
            "OVERALL": {"dates": {}, "subcategories": []},
            "RAIGARH": {"dates": {}, "subcategories": []},
            "ANGUL": {"dates": {}, "subcategories": []}
        }
        
        current_main_category = "OVERALL"  # IMPORTANT CHANGE: Start with OVERALL as the default category
        subcategories_map = { cat: {} for cat in result.keys() }
        
        data_start_row_index = date_header_row_index + 1
        st.write(f"Debug: Starting data processing from Excel row {data_start_row_index + 1}.")
        st.write(f"Debug: Default main category set to OVERALL until another category header is found.")

        # Create DataFrame to track extracted numeric values for validation
        value_tracking = pd.DataFrame(columns=["Category", "Date", "Subcategory", "Value", "Row", "Col"])
        
        for i, row_series in df.iloc[data_start_row_index:].iterrows():
            row_values = row_series.values # Get row as numpy array
            
            # Skip completely empty rows
            if row_series.isnull().all():
                continue

            category_name_raw = str(row_values[0]).strip() if pd.notna(row_values[0]) else ""
            
            if not category_name_raw: # If first cell is empty, skip (might be formatting row)
                continue

            # Check for main category markers
            if category_name_raw.upper() in result: # Check if it's one of our main categories
                current_main_category = category_name_raw.upper()
                st.write(f"Debug: Switched to main category: {current_main_category} at Excel row {i+1} (Value: '{category_name_raw}')")
                continue 

            # If we are here, it's a subcategory row under the current_main_category
            subcategory_name = category_name_raw
            if not subcategory_name: # Should not happen if we check above, but as safeguard
                st.write(f"Debug: Skipping row {i+1} due to empty subcategory name under {current_main_category}.")
                continue
                
            subcategories_map[current_main_category][subcategory_name] = True
            
            # Add more verbose debugging for key subcategories
            if subcategory_name in ["Net Revenue", "Cash Score"]:
                st.write(f"Debug: Processing important subcategory '{subcategory_name}' under '{current_main_category}' from Excel row {i+1}")

            for j, date_str in enumerate(date_headers):
                col_idx = j + 1  # Data for this date_str is in column (j+1) of the original DataFrame
                
                if col_idx < len(row_values): # Ensure we don't go out of bounds for the row
                    cell_value_raw = row_values[col_idx]
                    
                    # Extract cell value very carefully
                    if pd.isna(cell_value_raw):
                        continue  # Skip NaN values
                        
                    # Convert to string and strip whitespace
                    cell_value_str = str(cell_value_raw).strip()
                    
                    # Skip empty strings
                    if not cell_value_str or cell_value_str == "-" or cell_value_str.lower() == "na":
                        continue
                        
                    try:
                        # Try to extract numeric value cleanly
                        # Remove commas, %, $, ₹ symbols that might be in financial data
                        cleaned_value_str = cell_value_str.replace(',', '').replace('%', '').replace('$', '').replace('₹', '')
                        value = float(cleaned_value_str)
                        
                        # Add to tracking for important metrics
                        if subcategory_name in ["Net Revenue", "Cash Score"]:
                            st.write(f"Debug: [{current_main_category}] Excel row {i+1}, col {col_idx+1}: Converting '{cell_value_str}' to numeric: {value}")
                            value_tracking = pd.concat([value_tracking, pd.DataFrame({
                                "Category": [current_main_category],
                                "Date": [date_str],
                                "Subcategory": [subcategory_name],
                                "Value": [value],
                                "Row": [i+1],
                                "Col": [col_idx+1]
                            })])
                    except (ValueError, TypeError):
                        # If conversion fails, keep as string
                        value = cell_value_str
                        if subcategory_name in ["Net Revenue", "Cash Score"]:
                            st.write(f"Debug: [{current_main_category}] Excel row {i+1}, col {col_idx+1}: Keeping as string: '{cell_value_str}'")
                        
                    # Store the value in our result structure
                    if date_str not in result[current_main_category]["dates"]:
                        result[current_main_category]["dates"][date_str] = {}
                    
                    result[current_main_category]["dates"][date_str][subcategory_name] = value
                    
                    # Extra debug for important metrics
                    if subcategory_name in ["Net Revenue", "Cash Score"]:
                        st.write(f"Debug: Successfully stored {current_main_category} | {date_str} | {subcategory_name} = {value} (type: {type(value).__name__})")


        # Generate subcategory lists
        for category_key in result:
            result[category_key]["subcategories"] = sorted(list(subcategories_map[category_key].keys()))

        # Value validation and debugging
        st.write("Debug: Value tracking for important metrics:")
        if not value_tracking.empty:
            st.dataframe(value_tracking)
        else:
            st.error("No numeric values were successfully extracted for important metrics!")
        
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
def get_data_for_category(query: str = "{\"category\": \"OVERALL\"}") -> str: # Modified to accept JSON string
    """Fetches data for a specified category, date, and subcategory.
    Input should be a JSON string with 'category' (required), and optional 'date', 'subcategory'.
    Example: '{"category": "OVERALL", "date": "last week", "subcategory": "Cash Score"}'
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
            closest_date = find_closest_date(category_data.get("dates", {}), date_query)
            st.write(f"Debug get_data_for_category: Date query '{date_query}' mapped to closest date: '{closest_date}'")

            if not closest_date:
                available_dates = list(category_data.get("dates", {}).keys())
                error_msg = f"No data found for date query '{date_query}' in category '{category}'."
                if available_dates:
                    error_msg += f" Available date periods include: {available_dates[:5]}"
                    if len(available_dates) > 5: error_msg += "..."
                else:
                    error_msg += " No date periods are available for this category."
                return json.dumps({"error": error_msg})
            
            date_specific_data = category_data["dates"][closest_date]
            if subcategory_query:
                if subcategory_query in date_specific_data:
                    value = date_specific_data[subcategory_query]
                    st.write(f"Debug get_data_for_category: Found value for {category} | {closest_date} | {subcategory_query} = {value} (type: {type(value).__name__})")
                    # Ensure numeric values are returned as numbers, not strings
                    if isinstance(value, str):
                        try:
                            value = float(value.replace(',', '').replace('%', ''))
                            st.write(f"Debug get_data_for_category: Converted string value to number: {value}")
                        except ValueError:
                            # Keep as string if conversion fails
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
                            processed_data[subcat] = float(val.replace(',', '').replace('%', ''))
                        except ValueError:
                            processed_data[subcat] = val
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
def compare_category_data(query: str = "{\"category1\": \"OVERALL\", \"category2\": \"RAIGARH\"}") -> str: # Modified to accept JSON string
    """Compares data between two categories.
    Input: JSON string with 'category1', 'category2'. Optional 'date', 'subcategory'.
    Example: '{"category1": "RAIGARH", "category2": "ANGUL", "date": "last week", "subcategory": "Cash Score"}'
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
    """Find the closest matching date in available_dates for the given query_date.
    
    Args:
        available_dates: Dictionary with date strings as keys or list of date strings
        query_date: String date query (can be relative like "last_week" or specific date string)
    
    Returns:
        Closest matching date string from available_dates, or None if no match found
    """
    if not available_dates or not query_date:
        return None
    
    # Convert to list if dictionary
    date_options = list(available_dates.keys()) if isinstance(available_dates, dict) else available_dates
    
    if not date_options:
        return None
    
    # Handle relative date queries
    if query_date == "last_week":
        # Simply return the most recent date in the available dates
        # This is a simplification - in production you might want more sophisticated matching
        # For now, we'll just sort the dates as strings and get the last one
        # This works for formats like "25th-31st Jul 24" because they sort chronologically
        sorted_dates = sorted(date_options)
        if sorted_dates:
            return sorted_dates[-1]  # Most recent date
    
    elif query_date == "this_week":
        # Similar logic but perhaps we want the most recent date
        sorted_dates = sorted(date_options)
        if sorted_dates:
            return sorted_dates[-1]  # Most recent date
    
    elif query_date == "last_month":
        # Find a date from the previous month
        # This is simplified - in production you'd parse the actual date strings
        sorted_dates = sorted(date_options)
        if len(sorted_dates) > 1:
            return sorted_dates[-2]  # Second most recent date as a simple approximation
    
    elif query_date == "last_3_months" or query_date == "last_three_months":
        # For "last 3 months" we want to return the most recent date
        # The agent's process is to:
        # 1. Get the list of available dates
        # 2. Take the most recent ~12 weeks (approximately 3 months)
        # 3. Make separate queries for each week
        
        # Here we'll just return the most recent date to get started
        # The agent should then use list_available to get all dates and process them
        sorted_dates = sorted(date_options)
        if sorted_dates:
            # We should return ALL dates from last 3 months, but this function
            # can only return one date. We'll return the most recent so the
            # agent can list_available and then query each date separately.
            return sorted_dates[-1]  # Most recent date
    
    # For exact date matching or more complex queries
    # For now just do a simple string comparison to find the closest match
    # In production, you'd parse these into actual dates for better comparison
    if query_date in date_options:
        return query_date  # Exact match
    
    # Simple fuzzy matching - find the date string that contains the query
    for date_str in date_options:
        if query_date.lower() in date_str.lower():
            return date_str
    
    # If no match found, return the most recent date as a fallback
    sorted_dates = sorted(date_options)
    if sorted_dates:
        return sorted_dates[-1]
    
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
    """Lists available main categories, or dates/subcategories for a specific main category.
    Input should be a JSON string.
    To list all main categories: '{"type": "categories"}'
    To list available dates for a category: '{"type": "dates", "category": "RAIGARH"}'
    To list available subcategories for a category: '{"type": "subcategories", "category": "ANGUL"}'
    Returns a JSON string with the list or an error message."""
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
    """Creates visualizations based on the provided configuration.
    Input should be a JSON string containing visualization configuration with:
        - type: type of visualization ('line', 'bar', 'pie', 'scatter')
        - data: data to visualize (e.g., list of dicts like [{'date': d, 'value': v}])
        - title: optional title for the visualization
        - labels: optional dict for customizing labels (e.g., {'x': 'Date', 'y': 'Cash Score'})
        - is_financial: boolean indicating if this is financial data (default: True)
    The tool will attempt to generate the chart and store it for display. 
    It returns a JSON string confirming success or detailing an error.
    NOTE: All financial data is in Indian Rupees (₹) and in Crores (1 Cr = 10M).
    Example: '{"type": "line", "data": [{"date": "W1", "Cash Score": 70}, {"date": "W2", "Cash Score": 75}], "title": "Cash Score Trend", "labels": {"x": "Week", "y": "Score"}}'
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

        # Create a DataFrame from the data for easier plotting with Plotly Express
        df = pd.DataFrame(data)
        
        # Show extracted DataFrame for debugging
        st.write("Debug create_visualization: DataFrame created from input data:")
        st.dataframe(df.head())

        if df.empty:
            return json.dumps({"error": "Data converted to empty DataFrame. Cannot plot."})

        # Ensure numeric values are actually numeric
        for col in df.columns:
            if col != x_axis:  # Don't convert x-axis if it's dates or categories
                try:
                    # Try to convert to numeric, coercing errors to NaN
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    st.write(f"Debug create_visualization: Converted column '{col}' to numeric. Sample: {df[col].head()}")
                except:
                    st.write(f"Debug create_visualization: Column '{col}' could not be converted to numeric. Sample: {df[col].head()}")

        # Determine x and y for plotting if not explicitly given but data structure is simple (e.g. date/value)
        if not x_axis and len(df.columns) >= 1:
            x_axis = df.columns[0]
        if not y_axis and len(df.columns) >= 2:
            y_axis = df.columns[1]
        elif not y_axis and len(df.columns) >=1 and chart_type in ["bar", "line"]: # Allow single column for y if x is index
             y_axis = df.columns[0]
             if x_axis == y_axis: x_axis = None # Use index for x if x and y are the same single column

        if not y_axis: # Y-axis is crucial for most plots
            return json.dumps({"error": f"Could not determine Y-axis for plotting. Available columns: {list(df.columns)}"})

        # Enhance labels for financial data
        if is_financial:
            if 'y' not in labels:
                labels['y'] = f"{y_axis} (₹ Cr)"
            elif 'Cr' not in labels['y'] and '₹' not in labels['y']:
                labels['y'] = f"{labels['y']} (₹ Cr)"
            
            # Add ₹ symbol to hover template for financial values
            hover_template = "%{y:.2f} ₹ Cr"
        else:
            hover_template = None

        fig = None
        
        # Debug the data being used for visualization
        st.write(f"Debug create_visualization: Creating {chart_type} chart with x={x_axis}, y={y_axis}")
        if y_axis in df.columns:
            st.write(f"Debug create_visualization: Data range for {y_axis}: min={df[y_axis].min()}, max={df[y_axis].max()}, mean={df[y_axis].mean()}")

        # Create figure based on chart type
        if chart_type == "line":
            fig = px.line(df, x=x_axis, y=y_axis, title=title, labels=labels, markers=True)
            if is_financial and fig is not None:
                fig.update_traces(hovertemplate=hover_template)
        elif chart_type == "bar":
            fig = px.bar(df, x=x_axis, y=y_axis, title=title, labels=labels)
            if is_financial and fig is not None:
                fig.update_traces(hovertemplate=hover_template)
        elif chart_type == "pie":
            # For pie charts, 'names' would be the category column and 'values' the numeric column.
            # The agent needs to specify this in the 'x' (names) and 'y' (values) params in the query.
            if not x_axis or not y_axis:
                 return json.dumps({"error": "For pie chart, please specify 'x' for names and 'y' for values in the query."})
            fig = px.pie(df, names=x_axis, values=y_axis, title=title)
            if is_financial and fig is not None:
                fig.update_traces(texttemplate="%{value:.2f} ₹ Cr")
        elif chart_type == "scatter":
            fig = px.scatter(df, x=x_axis, y=y_axis, title=title, labels=labels)
            if is_financial and fig is not None:
                fig.update_traces(hovertemplate=hover_template)
        else:
            return json.dumps({"error": f"Unsupported visualization type: {chart_type}. Supported types: line, bar, pie, scatter."})

        if is_financial and fig is not None:
            title = f"{title} (₹ in Crores)" if "Crores" not in title and "₹" not in title else title
            fig.update_layout(title=title)

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
    """Analyzes trends in the provided data (e.g., time series data) and provides insights.
    Input should be a JSON string containing:
        - data: list of dicts (e.g., [{'date': d1, 'metric': v1}, {'date': d2, 'metric': v2}])
        - metrics: list of metric keys from the data dicts to analyze (e.g., ["metric"])
        - time_period_description: string describing the time period (e.g., "last 3 months")
    Returns a JSON string containing trend analysis, insights, and recommendations.
    NOTE: All financial data is in Indian Rupees (₹) and in Crores (1 Crore = 10 million).
    Example: '{"data": [{"date": "W1", "Sales": 100}, {"date": "W2", "Sales": 120}], "metrics": ["Sales"], "time_period_description": "Last 2 Weeks"}'
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
    """Gets the current date and contextual date information like start/end of current week and month.
    Accepts an optional JSON string query, but typically doesn't require input to fetch current date details.
    Returns a JSON string containing current date, current week (start/end), and current month (start/end).
    Example output: '{"current_date": "YYYY-MM-DD", "current_week_start": "YYYY-MM-DD", "current_week_end": "YYYY-MM-DD", ...}'
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

# TOC Analysis Agent System Prompt - Single, comprehensive prompt
TOC_ANALYSIS_AGENT_PROMPT = """You are an expert TOC (Theory of Constraints) analyst for Jindal Steel.
Analyze Excel data to answer questions accurately and provide actionable insights. Base answers ONLY on uploaded data.

Data Context:
- Financial/operational metrics for: OVERALL, RAIGARH, ANGUL
- IMPORTANT: All financial values in Indian Rupees (₹) and Crores (1 Cr = 10M)
- Data in weekly periods (e.g., '25th-31st Jul 24')
- For relative periods:
  1. ALWAYS call get_current_date first
  2. For "last week", "this week", "last month": Use these keywords directly in query_category_data
  3. For "last 3 months": 
     a. Call get_current_date for date range
     b. Get dates with list_available, choose dates in range
     c. Make multiple query_category_data calls for each date
     d. Aggregate results

Data Retrieval Strategy:
- If data not found: Check category names (JSPL → OVERALL), use list_available to verify subcategory names and dates
- Double-check exact case/spelling of subcategories/dates using list_available

Key Analysis Types:
1. Performance Over Time:
   - Call get_current_date first, get relevant dates, collect data points
   - Visualize with create_visualization including Indian currency
   - Comment on trends with proper financial units

2. Hero/Zero Areas:
   - Identify highest/lowest performing metrics from latest available date
   - Compare with historical data for context

3. Constraint Identification:
   - Use list_available to confirm constraint metrics exist
   - Check key metrics across all sites to identify bottlenecks

4. Actual vs. Target:
   - Query actual + target values
   - Compare RAIGARH vs ANGUL contribution

5. Delta Analysis:
   - Calculate differences between key metrics
   - Identify potential "leaks"

Tool Usage:
- Always use JSON-formatted strings as inputs
- query_category_data: '{{"category": "OVERALL", "subcategory": "Cash Score", "date": "last week"}}'
- list_available: '{{"type": "dates", "category": "RAIGARH"}}'
- create_visualization: Ensure financial data properly labeled with ₹ and Cr
- get_current_date: Always call first for date-based queries

Be methodical, clear, and show all financial values with ₹ and Cr units.
"""

def initialize_agent_executor(): # Renamed and simplified
    try:
        llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0.1, openai_api_key=os.getenv("OPENAI_API_KEY")) # Updated model
        
        tools = [
            get_data_for_category, # Corrected name
            compare_category_data,
            list_available,        # Ensure this is now correctly defined
            create_visualization,
            analyze_trends, 
            get_current_date
        ]
        
        # Custom Prompt Template - Using a simpler template to avoid unwanted variables
        prompt = ChatPromptTemplate.from_messages([
            ("system", TOC_ANALYSIS_AGENT_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        agent = create_openai_functions_agent(llm, tools, prompt)
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True, 
            memory=st.session_state.agent_memory,
            handle_parsing_errors="Check your output and make sure it conforms!", # More direct parsing error handling
            max_iterations=15,  # Increase from default of 10
            max_execution_time=120,  # Add a timeout of 120 seconds
            early_stopping_method="force",  # Changed from "generate" to "force"
        )
        
        st.success("TOC Analysis Agent initialized successfully with custom prompt!")
        return agent_executor
    except Exception as e:
        st.error(f"Error initializing TOC Analysis Agent: {str(e)}")
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
            if "json_data" in st.session_state:
                del st.session_state.json_data
            if "toc_agent" in st.session_state:
                del st.session_state.toc_agent
            if "last_chart_figure_json" in st.session_state:
                del st.session_state.last_chart_figure_json
                
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
    if "toc_agent" not in st.session_state or st.session_state.toc_agent is None:
        st.session_state.toc_agent = initialize_agent_executor() # Initialize single agent
    
    toc_agent = st.session_state.toc_agent

    if toc_agent:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if user_prompt := st.chat_input("Ask a question about your Jindal Steel TOC data..."):
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("The TOC Analyst is thinking..."):
                    try:
                        # Invoke the single agent
                        response = toc_agent.invoke({
                            "input": user_prompt,
                            # chat_history is handled by memory object
                        })
                        agent_response_content = response.get("output", "Sorry, I encountered an issue generating a response.")
                        
                        st.markdown(agent_response_content)
                        
                        # Improved visualization handling:
                        # First check if a chart figure is directly stored
                        if "last_chart_figure" in st.session_state and st.session_state.last_chart_figure is not None:
                            st.write("### Generated Chart")
                            st.plotly_chart(st.session_state.last_chart_figure, use_container_width=True)
                            st.session_state.last_chart_figure = None  # Clear after displaying
                        # Fallback to json representation if needed
                        elif "last_chart_figure_json" in st.session_state and st.session_state.last_chart_figure_json:
                            try:
                                st.write("### Generated Chart")
                                fig = go.Figure(json.loads(st.session_state.last_chart_figure_json))
                                st.plotly_chart(fig, use_container_width=True)
                            except Exception as chart_e:
                                st.error(f"Error displaying chart: {chart_e}")
                            finally:
                                st.session_state.last_chart_figure_json = None  # Clear after displaying
                        
                        st.session_state.messages.append({"role": "assistant", "content": agent_response_content})
                        
                    except TimeoutError:
                        error_message = "I apologize, but your request timed out. For complex analyses like spanning multiple months of data, please try breaking your question into smaller, more specific queries. For example, instead of asking about 3 months at once, you could ask about one month at a time."
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                    except Exception as e:
                        error_message = f"An error occurred with the TOC Analyst: {str(e)}"
                        
                        # Check for iteration limit errors
                        if "iteration limit" in str(e) or "Agent stopped due to iteration limit" in str(e):
                            error_message = "I apologize, but I reached my iteration limit while analyzing your request. This usually happens with very complex queries. Please try breaking your question into smaller parts, or be more specific about exactly what information you're looking for."
                            
                        # Check for token limit errors
                        elif "maximum context length" in str(e) or "context_length_exceeded" in str(e):
                            error_message = "I apologize, but your conversation has reached the token limit. Please start a new conversation by refreshing the page or try a more focused query."
                        
                        st.error(error_message)
                        import traceback
                        st.error(f"Traceback: {traceback.format_exc()}")
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
    else:
        st.error("TOC Analysis Agent failed to initialize. Please check configuration, API keys, or upload a file again.")
elif not uploaded_file: # Check if uploaded_file is None from the sidebar context
    st.info("👈 Please upload your Excel file in the sidebar to begin TOC analysis.") 