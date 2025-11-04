# INTELLISEARCH Workflow Diagram (Updated November 2025)

## LangGraph Workflow Architecture

```mermaid
graph TD
    START([START]) --> create_queries[Create Queries<br/>LLM-Based Query Generation]
    
    create_queries --> user_approval{User Approval<br/>Interactive/Automated}
    
    user_approval -->|Approved| evaluate_search_results[Evaluate Search Results<br/>Web Search + AI Evaluation]
    user_approval -->|Refine| create_queries
    user_approval -->|Max Iterations| choose_report_type[Choose Report Type<br/>Concise/Detailed]
    
    evaluate_search_results --> extract_content[Extract Content<br/>Multi-Strategy Web Scraping]
    
    extract_content --> embed_index_and_extract[Embed Index & Extract<br/>Hybrid Retrieval System]
    
    embed_index_and_extract --> AI_evaluate{AI Evaluation<br/>Information Sufficiency}
    
    AI_evaluate -->|Sufficient| choose_report_type
    AI_evaluate -->|Need More| evaluate_search_results
    AI_evaluate -->|Max Iterations| choose_report_type
    
    choose_report_type --> write_report[Write Report<br/>AI-Generated Research Report]
    
    write_report --> END([END])
    
    %% Styling
    classDef startEnd fill:#e1f5fe,stroke:#01579b,stroke-width:3px
    classDef process fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef decision fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef hybrid fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
    
    class START,END startEnd
    class create_queries,evaluate_search_results,extract_content,write_report process
    class user_approval,AI_evaluate decision
    class embed_index_and_extract hybrid
```

## Key Workflow Components

### 🧠 **LLM-Based Query Generation**
- **Node**: `create_queries`
- **Enhancement**: Replaced rule-based question_analyzer with AI-powered query generation
- **Features**: Context-aware, domain-specific, multi-language support

### 🔍 **Hybrid Retrieval System** 
- **Node**: `embed_index_and_extract`
- **Enhancement**: Custom EnsembleRetriever with BM25 + Vector Search
- **Features**: Reciprocal Rank Fusion, weighted scoring, deduplication

### 🌐 **Multi-Strategy Web Scraping**
- **Node**: `extract_content`
- **Features**: requests-html, aiohttp, fallback methods
- **Error Handling**: Robust retry mechanisms

### 🤖 **AI Evaluation Loop**
- **Nodes**: `AI_evaluate`, `user_approval`
- **Features**: Information sufficiency assessment, iterative refinement
- **Limits**: Max 5 search iterations, 3 approval cycles

## Workflow Decision Points

### User Approval Routing
```
if approval_iteration_count >= 3:
    → choose_report_type (skip to report)
elif proceed == True:
    → evaluate_search_results (continue search)
else:
    → create_queries (refine queries)
```

### AI Evaluation Routing
```
if search_iteration_count >= 5:
    → choose_report_type (force completion)
elif proceed == True:
    → choose_report_type (sufficient info)
else:
    → evaluate_search_results (gather more info)
```

## Integration Points

### 📊 **Data Flow**
- **AgentState**: Unified state management across nodes
- **Document Processing**: Consistent document handling with metadata
- **Progress Tracking**: Iteration counts and decision history

### 🔧 **Configuration**
- **API Keys**: Google Gemini, Serper integration
- **Hybrid Retrieval**: Configurable weights and fusion methods
- **Report Types**: Concise (~500 words) or Detailed (~1000 words)

---

**Updated**: November 4, 2025  
**Status**: Post-Cleanup Optimization  
**Architecture**: LangGraph + Google Gemini + Hybrid Retrieval