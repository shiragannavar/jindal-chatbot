# Jindal Steel Chatbot

An AI-powered chatbot built to analyze and visualize data from Excel files for Jindal Steel. This project provides a user-friendly interface to query financial and operational data across different categories and time periods.

## Features

- **Excel Data Processing**: Upload Excel files with financial/operational data
- **Multi-category Analysis**: Analyze data across OVERALL, RAIGARH, and ANGUL categories
- **Date Range Support**: Query data for specific time periods
- **AI-Powered Insights**: Natural language queries to get insights from your data
- **Data Visualization**: Clean presentation of financial information

## Technologies Used

- **Streamlit**: For the web interface
- **LangChain**: For conversational AI and agent functionality
- **OpenAI**: Powering the intelligent chatbot with GPT-4o
- **Pandas**: For data processing
- **Python**: Programming language

## Usage

1. Upload your Excel file with financial data
2. Ask questions about your data in natural language
3. Get AI-powered insights and analysis
4. Compare data across different categories and time periods

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your OpenAI API key in a `.env` file: `OPENAI_API_KEY=your_key_here`
4. Run the app: `streamlit run app.py`

## Example Queries

- "What was the EBITDA for RAIGARH in June 2022?"
- "Compare Net Revenue between RAIGARH and ANGUL for Q1 2023"
- "Show me the Cash Score trend for OVERALL across all dates"
- "What were the Operational Expenses for ANGUL in July 2022?" 