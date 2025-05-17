# Jindal Steel TOC Advisor

An AI-powered Theory of Constraints (TOC) advisor built to analyze and provide strategic insights from Excel data for Jindal Steel. This application helps identify bottlenecks, improve throughput, and optimize cash flow based on financial and operational data.

## Features

- **Theory of Constraints Analysis**: Identify system constraints and bottlenecks
- **Cash Flow Optimization**: Focus on throughput and operational expenses
- **Multi-plant Comparison**: Analyze data across OVERALL, RAIGARH, and ANGUL plants
- **Date Range Analysis**: Query data for specific time periods
- **Strategic Advisor**: Get expert TOC-based recommendations and insights
- **Data Visualization**: Clean presentation of financial metrics in crores

## Technologies Used

- **Streamlit**: For the web interface
- **LangChain**: For conversational AI and agent functionality
- **OpenAI**: Powering the intelligent advisor with GPT-4o
- **Pandas**: For data processing
- **Python**: Programming language

## Usage

1. Upload your Excel file with financial and operational data
2. Ask questions about constraints, throughput, and operational metrics
3. Get TOC-focused analysis and strategic recommendations
4. Compare performance across different plants and time periods

## Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your OpenAI API key in a `.env` file: `OPENAI_API_KEY=your_key_here`
4. Run the app: `streamlit run app.py`

## Example Queries

- "What are the main constraints limiting throughput at RAIGARH in June 2022?"
- "Compare cash generation between RAIGARH and ANGUL for Q1 2023"
- "Show me the delivery reliability metrics for OVERALL across all dates"
- "What's the key bottleneck affecting performance at ANGUL in the latest period?"
- "How can we improve cash flow based on the current operational data?" 