# Reasoning vs Research Mode Enhancement

## Overview
Enhanced the distinction between Reasoning Mode and Research Mode to make the difference crystal clear:

- **Reasoning Mode**: AI provides opinions, analysis, and expert conclusions
- **Research Mode**: No opinions - just facts and data only

## Changes Made

### 1. Backend Prompt Instructions (`src/nodes.py`)

#### Reasoning Mode Instructions (NEW)
```
Generate an analytical report with your expert opinions and interpretations about the research question.
Provide your analysis, insights, and reasoned conclusions based on the data.
Make connections between ideas, offer interpretations of trends and patterns.
Present your professional judgment on implications, significance, and potential outcomes.
Use phrases like 'This suggests...', 'The evidence indicates...', 'It appears that...', 'This could mean...'.
Give your expert opinion on what the data reveals and what it might mean for the future.
Be analytical and interpretive while supporting your views with citations.
```

#### Research Mode Instructions (NEW)
```
Generate a strictly factual report presenting only the relevant data found in the sources.
Do NOT provide opinions, interpretations, analysis, or conclusions.
Present only verified facts, statistics, quotes, and documented information.
Use neutral language and avoid any subjective commentary or speculation.
Structure the information clearly but let the data speak for itself.
Use phrases like 'According to...', 'The data shows...', 'Sources indicate...', 'Reports state...'.
Focus exclusively on documenting what is known without adding interpretation.
Be comprehensive in presenting all relevant factual information found.
```

### 2. Frontend UI (`web-app/frontend/src/components/ResearchForm.tsx`)

#### Reasoning Mode Card
- **Title**: "Reasoning Mode"
- **Description**:
  - • AI provides opinions & analysis
  - • Interpretive insights
  - • Expert conclusions
  - • "What this means" perspective

#### Research Mode Card
- **Title**: "Research Mode"
- **Description**:
  - • No opinions or analysis
  - • Pure facts and data only
  - • Objective information
  - • "Just the facts" approach

### 3. CLI Interface (`app.py`)

#### Interactive Mode
```
Select reasoning mode:
1: Reasoning (AI provides opinions, analysis, and expert conclusions)
2: Research (No opinions - just facts and data only)
```

#### Command Line Help
```
--reasoning-mode: reasoning: AI gives opinions/analysis, research: facts only (default: reasoning)
```

## Expected Output Differences

### Example Topic: "Tesla Q3 Performance"

#### Reasoning Mode Output:
```
Based on the data, this suggests Tesla is performing strongly. The 15% stock 
increase indicates investor confidence, and the $23.3 billion revenue appears 
robust. This could mean the company is well-positioned for future growth.
```

#### Research Mode Output:
```
According to reports, Tesla's stock price increased 15% last quarter. The data 
shows revenue of $23.3 billion. Sources indicate that CEO Elon Musk stated 
production targets were met.
```

## Key Language Patterns

### Reasoning Mode Uses:
- "This suggests..."
- "The evidence indicates..."
- "It appears that..."
- "This could mean..."
- "Based on the analysis..."
- "The implications are..."

### Research Mode Uses:
- "According to..."
- "The data shows..."
- "Sources indicate..."
- "Reports state..."
- "Documentation reveals..."
- "Statistics demonstrate..."

## User Benefits

1. **Clear Choice**: Users now understand exactly what they'll get with each mode
2. **Appropriate Output**: Opinion-seekers get analysis, fact-seekers get data
3. **Professional Use**: Research mode perfect for academic/legal contexts where opinions aren't wanted
4. **Business Analysis**: Reasoning mode ideal for strategic decision-making where insights matter

## Technical Implementation

The system uses different prompt instructions based on the `reasoning_mode` boolean:
- `True` → Uses reasoning_instruction (with opinions)
- `False` → Uses researcher_instruction (facts only)

This ensures the AI assistant adapts its writing style and approach completely based on user preference.