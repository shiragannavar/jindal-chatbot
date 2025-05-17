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

def excel_to_json(uploaded_file):
    # Read Excel with pandas
    df = pd.read_excel(
        uploaded_file,
        engine='openpyxl',
        header=None,
        skiprows=0
    )
    
    try:
        # Extract date headers (row 5, index 5 is 6th row in Excel)
        date_headers = [str(h).strip() for h in df.iloc[5, 1:].values if not pd.isna(h)]
        
        # Create JSON structure
        result = {
            "OVERALL": {"dates": {}, "subcategories": []},
            "RAIGARH": {"dates": {}, "subcategories": []},
            "ANGUL": {"dates": {}, "subcategories": []}
        }
        
        # Process the data row by row
        current_main_category = "OVERALL"  # Start with OVERALL as default
        encountered_raigarh = False
        encountered_angul = False
        
        # Initialize subcategories dict for all categories
        subcategories = {
            "OVERALL": {},
            "RAIGARH": {},
            "ANGUL": {}
        }
        
        # First pass: Process the data row by row
        for i, row in df.iloc[6:].iterrows():  # Skip header rows
            if i < len(df) and 0 < len(row):  # Ensure row exists and has at least one column
                category_value = row[0]
                category_name = str(category_value).strip() if not pd.isna(category_value) else ""
                
                # Check for main category transitions
                if category_name.upper() == "RAIGARH":
                    current_main_category = "RAIGARH"
                    encountered_raigarh = True
                    continue  # Skip to next row
                elif category_name.upper() == "ANGUL":
                    current_main_category = "ANGUL"
                    encountered_angul = True
                    continue  # Skip to next row
                elif category_name.upper() == "OVERALL":
                    current_main_category = "OVERALL"
                    continue  # Skip to next row
                
                # If we reach here, this is a subcategory under the current main category
                if category_name:  # Only process non-empty subcategories
                    subcategory = category_name
                    
                    # Add subcategory to our tracking dict
                    subcategories[current_main_category][subcategory] = True
                    
                    # Process data for this subcategory row
                    for j, date in enumerate(date_headers):
                        col_idx = j + 1  # +1 because column 0 is category
                        if col_idx < len(row) and not pd.isna(row[col_idx]):
                            value = row[col_idx]
                            if isinstance(value, (np.integer, np.floating)):
                                value = float(value)
                            
                            # Ensure the date key exists
                            if date not in result[current_main_category]["dates"]:
                                result[current_main_category]["dates"][date] = {}
                            
                            # Store value for this subcategory and date
                            result[current_main_category]["dates"][date][subcategory] = value
        
        # Store the subcategory lists in the result
        for category in ["OVERALL", "RAIGARH", "ANGUL"]:
            if category in subcategories:
                result[category]["subcategories"] = list(subcategories[category].keys())
        
        # Diagnostics
        st.write(f"OVERALL subcategories: {len(result['OVERALL']['subcategories'])}")
        st.write(f"RAIGARH subcategories: {len(result['RAIGARH']['subcategories'])}")
        st.write(f"ANGUL subcategories: {len(result['ANGUL']['subcategories'])}")
        
        return result
    except Exception as e:
        st.error(f"Error in excel_to_json: {str(e)}")
        # Debugging info
        st.error(f"Excel shape: {df.shape}")
        st.error(f"First few rows: {df.iloc[:10, :5]}")
        raise e

@tool
def get_data_for_category(category, date=None, subcategory=None):
    """
    Get data for a specific category and optionally a specific date and subcategory.
    
    Args:
        category (str): The main category to get data for (e.g., "RAIGARH", "ANGUL", "OVERALL")
        date (str, optional): The specific date to look up (e.g., "23rd-29th June")
        subcategory (str, optional): The specific subcategory to look up (e.g., "Cash Score", "Net Revenue")
        
    Returns:
        dict: The data for the specified category, date, and/or subcategory
    """
    data = st.session_state.json_data
    category = category.upper()
    
    if category not in data:
        return {"error": f"Category {category} not found"}
        
    if date:
        # Find the closest matching date
        closest_date = find_closest_date(data[category]["dates"], date)
        if not closest_date:
            return {"error": f"No data found for date {date} in category {category}"}
            
        if subcategory:
            # Look for specific subcategory
            if subcategory in data[category]["dates"][closest_date]:
                return {
                    "category": category,
                    "date": closest_date,
                    "subcategory": subcategory,
                    "value": data[category]["dates"][closest_date][subcategory]
                }
            else:
                return {"error": f"Subcategory {subcategory} not found for {category} on {closest_date}"}
        else:
            # Return all subcategories for this date
            return {
                "category": category,
                "date": closest_date,
                "values": data[category]["dates"][closest_date]
            }
    else:
        # Return all dates for this category
        return {
            "category": category,
            "dates": list(data[category]["dates"].keys()),
            "subcategories": data[category].get("subcategories", ["Total"])
        }

@tool
def compare_categories(category1, category2, date=None, subcategory=None):
    """
    Compare data between two categories for an optional specific date and subcategory.
    
    Args:
        category1 (str): First category (e.g., "RAIGARH")
        category2 (str): Second category (e.g., "ANGUL")
        date (str, optional): Specific date to compare (e.g., "23rd-29th June")
        subcategory (str, optional): Specific subcategory to compare (e.g., "Cash Score")
        
    Returns:
        dict: Comparison data between the two categories
    """
    data = st.session_state.json_data
    category1 = category1.upper()
    category2 = category2.upper()
    
    if category1 not in data:
        return {"error": f"Category {category1} not found"}
    if category2 not in data:
        return {"error": f"Category {category2} not found"}
    
    if date:
        # Find the closest matching dates
        date1 = find_closest_date(data[category1]["dates"], date)
        date2 = find_closest_date(data[category2]["dates"], date)
        
        if not date1 or not date2:
            return {"error": f"Date {date} not found for comparison"}
            
        if subcategory:
            # Compare specific subcategory
            if subcategory in data[category1]["dates"][date1] and subcategory in data[category2]["dates"][date2]:
                return {
                    "date": date,
                    category1: data[category1]["dates"][date1][subcategory],
                    category2: data[category2]["dates"][date2][subcategory],
                    "subcategory": subcategory
                }
            else:
                return {"error": f"Subcategory {subcategory} not found for comparison"}
        else:
            # Compare common subcategories
            common_subcats = set(data[category1]["dates"][date1].keys()) & set(data[category2]["dates"][date2].keys())
            result = {
                "date": date,
                category1: {},
                category2: {}
            }
            for subcat in common_subcats:
                result[category1][subcat] = data[category1]["dates"][date1][subcat]
                result[category2][subcat] = data[category2]["dates"][date2][subcat]
            return result
    else:
        # Find common dates
        common_dates = set(data[category1]["dates"].keys()) & set(data[category2]["dates"].keys())
        result = {
            category1: {},
            category2: {}
        }
        for date in common_dates:
            # Only compare "Total" or another common metric to avoid too much data
            if "Total" in data[category1]["dates"][date] and "Total" in data[category2]["dates"][date]:
                result[category1][date] = data[category1]["dates"][date]["Total"]
                result[category2][date] = data[category2]["dates"][date]["Total"]
        return result

@tool
def get_available_categories():
    """
    Get a list of all available categories in the data.
    
    Returns:
        list: All available main categories
    """
    data = st.session_state.json_data
    return list(data.keys())

@tool
def get_available_dates(category="OVERALL"):
    """
    Get a list of all available date ranges in the data.
    
    Args:
        category (str, optional): The category to get dates for. Defaults to "OVERALL".
        
    Returns:
        list: All available date ranges
    """
    data = st.session_state.json_data
    category = category.upper()
    
    if category not in data or "dates" not in data[category]:
        return []
    
    return list(data[category]["dates"].keys())

@tool
def get_subcategories(category):
    """
    Get a list of all subcategories for a specific main category.
    
    Args:
        category (str): The main category (e.g., "RAIGARH", "ANGUL")
        
    Returns:
        list: All subcategories for the specified category
    """
    data = st.session_state.json_data
    category = category.upper()
    
    if category not in data:
        return {"error": f"Category {category} not found"}
    
    if "subcategories" in data[category]:
        return data[category]["subcategories"]
    elif "dates" in data[category]:
        # Get subcategories from the first date entry
        first_date = next(iter(data[category]["dates"]))
        return list(data[category]["dates"][first_date].keys())
    else:
        return []

def find_closest_date(dates_dict, target_date):
    """Helper function to find the closest matching date in the dates dictionary"""
    target_date = target_date.lower()
    for date in dates_dict.keys():
        if target_date in date.lower():
            return date
    return None

# Sidebar for file upload and preview
with st.sidebar:
    st.title("📁 TOC Analysis Data")
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        try:
            # Convert Excel to JSON
            json_data = excel_to_json(uploaded_file)
            st.session_state.json_data = json_data
            
            # Show preview
            st.subheader("📊 Data Structure")
            st.write("Data is organized by main categories with subcategories and dates:")

            # Extract fields for reference
            if json_data:
                # Show main categories structure
                for main_category in ["OVERALL", "RAIGARH", "ANGUL"]:
                    if main_category in json_data:
                        st.write(f"**{main_category}**:")
                        
                        # Show subcategories (up to 10)
                        if "subcategories" in json_data[main_category] and json_data[main_category]["subcategories"]:
                            subcategories = json_data[main_category]["subcategories"]
                            st.write(f"- Subcategories: {len(subcategories)} items")
                            for subcat in subcategories[:5]:  # Show first 5 subcategories
                                st.write(f"  • {subcat}")
                            if len(subcategories) > 5:
                                st.write(f"  • ... and {len(subcategories) - 5} more")
                        
                        # Show dates (up to 5)
                        if "dates" in json_data[main_category]:
                            dates = list(json_data[main_category]["dates"].keys())
                            st.write(f"- Date ranges: {len(dates)} periods")
                            for date in dates[:3]:  # Show first 3 date ranges
                                st.write(f"  • {date}")
                            if len(dates) > 3:
                                st.write(f"  • ... and {len(dates) - 3} more")
                        
                        st.write("")  # Add spacing
                    
                # Show data info
                st.subheader("ℹ️ Data Info")
                total_data_points = 0
                for category in json_data:
                    if "dates" in json_data[category]:
                        for date in json_data[category]["dates"]:
                            total_data_points += len(json_data[category]["dates"][date])
                st.write(f"Total data points: {total_data_points}")
                st.write("All values are in crores")
            
        except Exception as e:
            st.error(f"Error processing the Excel file: {str(e)}")
            st.error("Please make sure your Excel file is properly formatted.")
    else:
        st.info("👈 Please upload your Excel file in the sidebar to begin your TOC analysis.")

# Main chat area
st.title("🏭 Jindal Steel TOC Advisor")

if st.session_state.json_data is not None:
    # Initialize the OpenAI chat model
    try:
        llm = ChatOpenAI(
            model="gpt-4o"
        )
        
        # Create custom tools list
        tools = [
            get_data_for_category,
            compare_categories,
            get_available_categories,
            get_available_dates,
            get_subcategories
        ]
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are Ravi Gilani, a renowned expert in Theory of Constraints (TOC) and management consultant working with Jindal Steel. 
            
            Your communication style:
            - Direct and data-focused: You cut through noise to address the real constraints
            - Practical and solution-oriented: You emphasize identifying bottlenecks and actionable improvements
            - Cash flow focused: You prioritize cash generation and throughput over traditional accounting metrics
            - Analytical with clear priorities: You focus on the vital few factors rather than trivial many
            - Experience-backed: You reference your 25+ years working with manufacturing giants in India
            
            When analyzing Jindal Steel data:
            1. Immediately identify the constraint or bottleneck in the system based on the data
            2. Focus on throughput (cash generated through sales), inventory, and operating expenses
            3. Emphasize delivery reliability metrics when relevant
            4. Compare plant performance between RAIGARH and ANGUL where appropriate
            5. Always mention that monetary values are in crores
            
            Respond as though you are personally advising executives at Jindal Steel.
            
            Use the available tools to extract and analyze the data. If asked about values without specific date ranges, ASK the user to specify a date range before providing an answer.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create agent - use only factory method with no additional parameters
        agent = create_openai_functions_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=True,
            return_intermediate_steps=False
        )
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your data"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Get AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = agent_executor.invoke({
                            "input": prompt,
                            "chat_history": st.session_state.messages
                        })
                        output = response["output"]
                        st.markdown(output)
                        st.session_state.messages.append({"role": "assistant", "content": output})
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                        # Print extended error info
                        import traceback
                        st.error(f"Error details: {traceback.format_exc()}")
                        st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {str(e)}"})
    except Exception as e:
        st.error(f"Error initializing the model: {str(e)}")
        st.error("Please ensure your OpenAI API key is correct and has access to the gpt-4o model.")
else:
    st.info("👈 Please upload your Excel file in the sidebar to begin your TOC analysis.") 