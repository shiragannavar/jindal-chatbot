# Jindal Steel TOC Advisor - Technical Architecture

## System Architecture Deep Dive

### Overview

The Jindal Steel TOC Advisor uses a sophisticated multi-agent orchestration pattern to handle complex analytical queries. Unlike traditional single-agent systems, our architecture decomposes complex tasks into specialized sub-tasks handled by expert agents.

## Core Components

### 1. Orchestrator Agent

The orchestrator is the brain of the system, powered by GPT-4 Turbo with a carefully crafted system prompt.

```python
# Key responsibilities:
- Query understanding and decomposition
- Execution plan generation
- Dependency management
- Result aggregation
```

**Execution Plan Structure:**
```json
{
  "steps": [
    {
      "step_id": 1,
      "persona": "DataRetrieverMode",
      "goal": "Get current date context",
      "inputs": {...},
      "outputs_expected": "Current date information",
      "depends_on_step": null
    }
  ]
}
```

### 2. Agent Personas

Each persona is a specialized mode of operation with specific capabilities:

#### DataRetrieverMode
- **Purpose**: Interface with data access tools
- **Input Model**: `DataRetrieverInputs`
- **Tools Access**: All data retrieval tools
- **Example Flow**:
  ```
  User Query → Tool Selection → Parameter Formatting → Tool Execution → Result Return
  ```

#### FinancialAnalysisMode
- **Purpose**: Complex financial calculations and insights
- **Input Model**: `FinancialAnalysisInputs`
- **Capabilities**:
  - Trend analysis with statistical metrics
  - Performance comparison
  - Hero/Zero identification
  - Volatility analysis

#### VisualizationMode
- **Purpose**: Create interactive charts
- **Input Model**: `VisualizationInputs`
- **Integration**: Direct Plotly integration
- **Chart Types**: Line, Bar, Pie, Scatter

#### TOCStrategyMode
- **Purpose**: Theory of Constraints strategic insights
- **Input Model**: `TOCStrategyInputs`
- **Focus**: Bottleneck identification and optimization

#### FinalResponseCompilationMode
- **Purpose**: Format and present results
- **Input Model**: `FinalResponseCompilationInputs`
- **Features**: Markdown formatting, JSON parsing, insight summarization

### 3. Tool System

Tools are decorated functions that agents can invoke:

```python
@tool
def get_data_for_category(query: str) -> str:
    """Fetches data for a specified category"""
    # Implementation
```

**Available Tools:**
1. `get_data_for_category` - Fetch metric data
2. `compare_category_data` - Compare between categories
3. `list_available` - List available options
4. `create_visualization` - Generate charts
5. `analyze_trends` - Trend analysis
6. `get_current_date` - Date context

### 4. Data Models (Pydantic)

```python
class Step(BaseModel):
    step_id: int
    persona: Literal[...]
    goal: str
    inputs: Dict[str, Any]
    outputs_expected: str
    depends_on_step: Optional[Union[int, List[int]]]

class ExecutionPlan(BaseModel):
    steps: List[Step]
```

## Execution Flow

### 1. Query Processing
```
User Query → Orchestrator → Execution Plan
```

### 2. Step Execution
```python
for step in execution_plan.steps:
    # Dependency resolution
    if step.depends_on_step:
        inputs = substitute_dependencies(step.inputs, previous_outputs)
    
    # Persona-based execution
    output = execute_persona(step.persona, inputs)
    
    # Store output
    step_outputs[step.step_id] = output
```

### 3. Dependency Resolution
- Steps can depend on single or multiple previous steps
- Placeholders like `[data_from_step_1]` are replaced with actual outputs
- Supports complex data flow between steps

## Advanced Features

### 1. Relative Date Handling
The system intelligently handles relative date queries:
- "last week" → Maps to actual date ranges
- "last 3 months" → Aggregates multiple periods
- Uses `get_current_date` tool for context

### 2. Multi-Period Aggregation
For queries spanning multiple dates:
```python
# Returns all matching dates
matching_dates = find_closest_date(available_dates, "last_month")

# Aggregates data
summary = {
    "count": len(results),
    "total": sum(values),
    "average": mean(values)
}
```

### 3. Financial Data Processing
- Handles Indian currency format (₹, Cr)
- Automatic numeric conversion
- Preserves precision for calculations

### 4. Error Handling
- Graceful fallbacks at each step
- Detailed error messages
- Continues execution when possible

## Performance Optimizations

1. **Lazy Loading**: Data loaded only when needed
2. **Caching**: Session state caches processed data
3. **Streaming**: Results displayed as available
4. **Memory Management**: Efficient DataFrame operations

## Security Considerations

1. **API Key Management**: Environment variables
2. **Input Validation**: Pydantic models
3. **Tool Access Control**: Limited tool exposure
4. **Data Isolation**: Session-based data storage

## Extensibility

### Adding New Personas
1. Define Pydantic input model
2. Add persona to Step literal type
3. Implement execution logic
4. Update orchestrator prompt

### Adding New Tools
1. Create tool function with @tool decorator
2. Add to AVAILABLE_TOOLS list
3. Document in orchestrator prompt
4. Test with sample queries

## Debugging Features

1. **Step-by-Step Visualization**: Shows execution progress
2. **Input/Output Logging**: Displays data at each step
3. **Debug Messages**: Extensive logging throughout
4. **Error Tracebacks**: Full error context

## Future Enhancements

1. **Parallel Step Execution**: For independent steps
2. **Dynamic Tool Loading**: Plugin system for tools
3. **Advanced Caching**: Results caching across queries
4. **Multi-Model Support**: Different LLMs for different personas
5. **Real-time Collaboration**: Multiple users on same data

## Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_key_here
```

### Model Configuration
```python
llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0.1,
    max_tokens=4096
)
```

### Prompt Engineering
The orchestrator prompt is carefully engineered with:
- Clear role definition
- Structured output requirements
- Examples for each scenario
- Error handling instructions

## Monitoring and Logging

1. **Execution Metrics**: Time per step
2. **Token Usage**: Track API costs
3. **Error Rates**: Monitor failures
4. **User Satisfaction**: Query success rates 