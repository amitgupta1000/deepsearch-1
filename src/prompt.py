from datetime import datetime
from typing import Dict, Any

# Utility function to safely format prompts with content that may contain curly braces
def safe_format(template: str, **kwargs: Any) -> str:
    """
    Safely format a template string, escaping any curly braces in the values.
    This prevents ValueError when content contains unexpected curly braces.
    """
    # Escape any curly braces in the values
    safe_kwargs = {k: v.replace('{', '{{').replace('}', '}}') if isinstance(v, str) else v
                  for k, v in kwargs.items()}
    return template.format(**safe_kwargs)

# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")



#======================================
web_search_validation_instructions = """Evaluate search results in relation to a query".

Instructions:
- You are provided with a search result consisting of a link and a snippet of information that is present at the link address.
- Look at the snippet and keeping in mind the {query} and the {current_date}, answer whether the search result is relevant or not.
- If relevant, answer with a simple message "yes", else "no'.
- Do not add any further text to your response.

QUERY:
{query}
"""

#=======================================
reflection_instructions = """You are an expert research assistant analyzing answers about "{research_topic}".

Instructions:
- Identify knowledge gaps or areas that need deeper exploration and generate a follow-up query. (1 or multiple).
- If provided answers are sufficient to answer the user's question, don't generate a follow-up query.
- If there is a knowledge gap, generate a follow-up query that would help expand your understanding.
- Focus on technical details, implementation specifics, or emerging trends that weren't fully covered.

Requirements:
- Ensure the follow-up query is self-contained and includes necessary context for web search.

Output Format:
- Format your response as a JSON object with these exact keys:
   - "is_sufficient": true or false
   - "knowledge_gap": Describe what information is missing or needs clarification
   - "follow_up_queries": Write a specific question to address this gap

Example:
```json
{{
    "is_sufficient": true, // or false
    "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
    "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"] // [] if is_sufficient is true
}}
```

Reflect carefully on the Summaries to identify knowledge gaps and produce a follow-up query. Then, produce your output following this JSON format:

Answers:
{summaries}
"""

#================================================
reflection_instructions_modified = """You are an expert research assistant analyzing extracted information about "{research_topic}".

    Instructions:
    - Evaluate the provided extracted information to determine if it is sufficient to answer the original user question.
    - If the extracted information is sufficient, set "is_sufficient" to true and leave "knowledge_gap" and "follow_up_queries" as empty/null.
    - If the extracted information is NOT sufficient, identify knowledge gaps or areas that need deeper exploration based on the original user question and the provided extracted information.
    - For each identified knowledge gap, generate 1 or multiple specific follow-up queries that would help expand your understanding to fully answer the original user question.
    - Focus on technical details, implementation specifics, or emerging trends that weren't fully covered in the extracted information.

    Requirements:
    - Ensure any follow-up query is self-contained and includes necessary context for web search.

    Output Format:
    - Format your response as a JSON object with these exact keys:
       - "is_sufficient": true or false
       - "knowledge_gap": Describe what information is missing or needs clarification (empty string if is_sufficient is true)
       - "follow_up_queries": Write a list of specific questions to address this gap (empty array if is_sufficient is true)

    Example:
    ```json
    {{
        "is_sufficient": true, // or false
        "knowledge_gap": "The summary lacks information about performance metrics and benchmarks", // "" if is_sufficient is true
        "follow_up_queries": ["What are typical performance benchmarks and metrics used to evaluate [specific technology]?"]
    }}
    ```

    Reflect carefully on the Extracted Information to identify knowledge gaps and produce a follow-up query if necessary. Then, produce your output following this JSON format:

    Extracted Information:
    {extracted_info_json}
    """

#=====================================
## Query Writer Instructions
#=====================================
query_writer_instructions_legal = """
You are an expert research assistant generating search queries for legal and financial issues related to: {topic}

**CURRENT DATE: {current_date}. Prioritize recent developments from 2024-2025.**

Generate {number_queries} targeted search queries covering:

- Regulatory actions (SEBI, MCA, NCLT, SAT)
- Litigation and legal disputes  
- Financial irregularities and audit issues
- Corporate governance concerns
- Director-related issues and compliance

Use domain-specific terms and site filters where helpful.

Return JSON format:
{{"query": ["query1", "query2", ...]}}

Research Topic: {topic}

Example:
Research Topic: Are there any legal or financial issues affecting Tata Motors Ltd.?

Output:
```json
{{
  "query": [
    "Tata Motors Ltd. legal cases site:https://indiankanoon.org/",
    "Tata Motors Ltd. legal cases site:https://www.casemine.com",
    "Tata Motors Ltd. legal cases site:https://airrlaw.com/",
    "Tata Motors Ltd. NCLT cases site:https://nclt.gov.in/",
    "Tata Motors Ltd. SAT appeals site:https://sat.gov.in/",
    "Tata Motors Ltd. SEBI regulatory actions site:https://sebi.gov.in/",
    "Tata Motors Ltd. audit qualifications site:nseindia.com OR site:bseindia.com",
    "Tata Motors Ltd. corporate governance issues site:bseindia.com",
    "Tata Motors Ltd. MCA filings site:mca.gov.in",
    "Tata Motors Ltd. ratings news",
    "Tata Motors Ltd. Director resignations news",
    "Tata Motors Ltd. Director cases news",
    "Tata Motors Ltd. Complaints against Directors",
    "Tata Motors Ltd. SAT appeals site:sat.gov.in",
    "Tata Motors Ltd. forensic audit news",
  ]
}}```

Research Topic: {topic}
"""
#======================================
query_writer_instructions_macro = """You are a global macro research assistant analyzing a specific commodity mentioned in the research topic: {topic}. Your goal is to uncover insights into market dynamics, pricing trends, recent events, and fundamental global factors influencing the commodity.

**CURRENT DATE CONTEXT: Today is {current_date}. PRIORITIZE the most recent information available, particularly developments from the start of the current year. Always search for the latest updates and current developments.**

Generate {number_queries} commodity research queries using:
    - Focus each query on a specific dimension of the commodity's macro landscape.
    - Use authoritative filters or context signals (e.g., site:eia.gov, site:bloomberg.com).
    - Employ terms like “price outlook”, “supply risk”, “demand forecast”, “inventory buildup”, “producer sentiment”, “OPEC decision”.
    - Aim for queries that reveal near-term and medium-term implications.

Format:
- Format your response as a JSON object with this key:
   - "query": A list of search queries.

Example:
Research Topic: What are the near-term risks to crude oil prices globally?

Output:
```json
{{
  "query": [
    "Crude oil price outlook July 2025 site:bloomberg.com OR site:reuters.com",
    "Global oil inventory levels site:eia.gov",
    "OPEC supply adjustment July 2025 site:opec.org",
    "Oil demand forecast medium term site:imf.org OR site:worldbank.org",
    "Impact of recent shipping disruptions on crude oil prices site:ft.com",
    "Effect of U.S. interest rates on commodity prices site:bloomberg.com",
    "Strategic petroleum reserve releases impact site:eia.gov",
    "Brent vs WTI price divergence July 2025 site:oilprice.com"
  ]
}}```

Research Topic: {topic}
"""

#======================================
query_writer_instructions_general = """You are a research assistant exploring: {topic}

**CURRENT DATE: {current_date}. Prioritize developments from the start of the current year.**

Generate {number_queries} search queries covering key aspects:
- Historical context and background
- Recent developments and news
- Expert perspectives and analysis
- Controversies and debates
- Policy and regulatory aspects

Use site-specific constraints where helpful (e.g., site:nature.com, site:nytimes.com).
    - One aspect per query (no multitopic prompts).
    - Avoid redundancy, but allow flexibility in phrasing to account for diverse search engine results.
    - You may include site-specific constraints where helpful (e.g., site:nature.com, site:nytimes.com).
    - Prefer phrasing that mirrors how users naturally ask questions.

Format:
- Format your response as a JSON object with this key:
   - "query": A list of search queries.

Example:
Research Topic: What are the scientific and policy issues around climate geoengineering?

Output:
```json
{{
  "query": [
    "History of geoengineering techniques site:wikipedia.org",
    "Solar radiation management risks site:nature.com",
    "Geoengineering policy positions US EU China site:ipcc.ch",
    "Recent controversies in climate geoengineering site:nytimes.com",
    "Long-term effectiveness of climate engineering site:sciencedirect.com",
    "Climate scientist perspectives on geoengineering site:noaa.gov"
  ]
}}```

Research Topic: {topic}
"""


#======================================
query_writer_instructions_deepsearch = """
You are a deep search assistant tasked with uncovering the most recent, relevant and information about the topic: {topic}. Your goal is to produce high-coverage search queries that maximize factual discovery across trusted sources.

**CURRENT DATE CONTEXT: Today is {current_date}. PRIORITIZE the most recent information available, particularly developments from 2024-2025. Always search for the latest updates and current developments.**

---

## OBJECTIVE

- Focus on **information retrieval**, not interpretation.
- Prioritize **recency**, **coverage**, and **source diversity**.
- Avoid speculative or opinion-based phrasing.

---

## INSTRUCTIONS

1. Break down the topic into granular subtopics (e.g., recent developments, key players, controversies, data points).
2. Generate queries that target **latest news**, **official updates**, **expert commentary**, and **primary sources**.
3. Use filters like `site:reuters.com`, `site:gov.in`, `site:bloomberg.com`, `site:business-standard.com`, etc.
4. Include date signals (e.g., “August 2025”, “last 6 months”) where helpful.
5. Avoid multitopic prompts — each query should target one aspect.
6. Ensure queries are phrased naturally for search engines.

---

## FORMAT

Return a JSON object with:
- "query": A list of search queries.

Example:
Research Topic: Recent developments in Adani Group’s financial disclosures

Output:
```json
{{
  "query": [
    "Adani Group financial disclosures August 2025 site:business-standard.com",
    "Adani Group audit notes site:nseindia.com OR site:bseindia.com",
    "Adani Group credit rating changes 2025 site:crisil.com",
    "Adani Group debt restructuring news site:reuters.com",
    "Adani Group SEBI updates site:sebi.gov.in",
    "Adani Group financial filings site:mca.gov.in"
  ]
}}```

Research Topic: {topic}
"""


#======================================
query_writer_instructions_person_search = """
You are a research assistant generating ethical search queries for person research: {topic}

**CURRENT DATE: {current_date}. Prioritize recent developments.**

Generate {number_queries} platform-specific search queries covering:
- Professional background (LinkedIn, Naukri.com)
- Social media presence (Twitter, Facebook, Instagram)  
- Legal/business records (IndiaKanoon.org, CaseMine.com)
- Educational background and achievements
- Public statements and professional activities

Focus only on publicly available information and ethical research practices.

---

## TASK INSTRUCTIONS

Generate at least **{number_queries}** targeted search queries:
- Platform-specific queries using site: filters
- Professional background and career queries
- Social media presence queries
- Legal and regulatory involvement queries
- Educational and achievement-based queries
- Cross-platform verification queries

---

## Response Format

Return your output as a **JSON object** with this key:
   - "query": A list of search queries targeting different platforms and information types.

Example:
Research Topic: Create a profile of Sundar Pichai

Output:
```json
{{
  "query": [
    "Sundar Pichai LinkedIn profile site:linkedin.com",
    "Sundar Pichai career background site:naukri.com",
    "Sundar Pichai Google CEO site:facebook.com",
    "Sundar Pichai tweets technology leadership site:twitter.com",
    "Sundar Pichai Instagram professional site:instagram.com",
    "Sundar Pichai legal cases site:indiankanoon.org",
    "Sundar Pichai court cases site:casemine.com",
    "Sundar Pichai regulatory matters site:airrlaw.com",
    "Sundar Pichai IIT Stanford education background",
    "Sundar Pichai awards achievements recognition",
    "Sundar Pichai interviews statements public speaking",
    "Sundar Pichai business ventures investments"
  ]
}}```

Research Topic: {topic}
"""

#======================================
query_writer_instructions_investment = """
You are an investment research assistant generating search queries for: {topic}

**CURRENT DATE: {current_date}. Prioritize 2024-2025 developments.**

Generate {number_queries} investment research queries covering:
- Financial performance and metrics
- Business fundamentals and competitive position
- Growth prospects and market opportunities
- Valuation metrics and peer comparisons
- Risk factors and regulatory compliance
- Management quality and strategic direction

Target financial databases, exchanges, analyst reports, and credible business sources.

---

## TASK INSTRUCTIONS

Generate at least **{number_queries}** targeted investment research queries:
- Financial performance and metrics queries
- Business fundamentals and competitive position queries
- Management and governance analysis queries
- Market opportunity and growth prospect queries
- Risk assessment and regulatory compliance queries
- Peer comparison and sector analysis queries

---

## Response Format

Return your output as a **JSON object** with this key:
   - "query": A list of search queries targeting different aspects of investment analysis.

Example:
Research Topic: Investment analysis of Reliance Industries Ltd

Output:
```json
{{
  "query": [
    "Reliance Industries quarterly results Q2 2025",
    "Reliance Industries annual report 2024-25 financial performance site:ril.com",
    "Reliance Industries debt equity ratio cash flow analysis site:screener.in",
    "Reliance Industries Jio ARPU subscriber growth metrics site:moneycontrol.com",
    "Reliance Industries retail expansion strategy growth plans site:economictimes.indiatimes.com",
    "Reliance Industries green energy investment capex plans site:livemint.com",
    "Reliance Industries valuation PE ratio analyst target price site:investing.com",
    "Reliance Industries vs Asian Paints vs HDFC Bank peer comparison site:valueresearchonline.com",
    "Reliance Industries management commentary investor call transcript",
    "Reliance Industries ESG rating sustainability initiatives site:sustainalytics.com",
    "Reliance Industries regulatory compliance SEBI filings site:sebi.gov.in",
    "Reliance Industries oil refining margins GRM analysis site:bloomberg.com"
  ]
}}```

Research Topic: {topic}
"""


#=================================================
## Report Writer Instructions
#=================================================

report_writer_instructions_legal = """
# Legal & Financial Risk Report: {research_topic}

## Date: {current_date}

Write a comprehensive report analyzing legal and financial developments affecting the company. Focus on actionable insights for decision-makers.

**Structure:**
- **Company Profile**: Business model, sector, recent performance
- **Legal Landscape**: Regulatory actions, litigation, governance issues
- **Financial Impact**: Revenue effects, compliance costs, investor perception
- **Strategic Response**: Company actions and mitigation strategies  
- **Risk Outlook**: Short-term (0-6 months) and medium-term (6-18 months) threats
- **Benchmarks**: Peer comparisons and industry trends

**Requirements:**
- Use markdown formatting with clear headings and structure
- Cite all claims from summaries using [1], [2] notation
- Include reference list at end
- Start directly with content, no introductory sections
- Use full token capacity for thorough analysis

**Data Source:** {summaries}
"""

#======================================

report_writer_instructions_general = """
# Research Report: {research_topic}

## Date
{current_date}

## Objective
Streamlined structure for research reports.

## Contextual Overview
- Current relevant aspects & recent developments
- Key players, policy updates & technological shifts
- Public sentiment & supporting examples

## Thematic Analysis
- 3-5 major themes with:
  - Historical context & current implications
  - Future possibilities & concerns
  - Supporting evidence & citations

## Data Insights
- Quantifiable information (statistics, trends, projections)
- Visual structure for enhanced readability

## Risks & Uncertainties
- Unknowns, open questions & risk factors
- Data gaps & controversial viewpoints

Cite all claims [1], [2], etc. Use markdown formatting.

**Research Topic:** {research_topic}
**Data:** {summaries}
"""

#======================================
report_writer_instructions_macro = """
# Commodity Macro Report: {research_topic}
Date: {current_date}

## Recent Developments
- Price trends & geopolitical shifts
- Regulatory updates & policy announcements
- Global events & institutional positions

## Market Dynamics
- Supply & Demand Analysis (seasonal effects, trade flows, inventories)
- Macroeconomic Influences (interest rates, inflation, currency)
- Short-Term Outlook (0-3 months)
- Medium-Term Outlook (3-12 months)

## Risks & Uncertainties
- Policy shifts & weather anomalies
- Supply chain disruptions & geopolitical tensions
- Market volatility factors

## Theoretical Context
- Historical trends & economic models
- Regional case studies & comparative analysis

Cite all data [1], [2], etc. Use markdown formatting.

**Research Topic:** {research_topic}
**Data:** {summaries}
"""
#================================================================================

report_writer_instructions_deepsearch = """
# Factual Summary Report: {research_topic}
Date: {current_date}

## Key Findings
- All facts, events & updates discovered
- Dates, names & source references
- Recent information prioritized

## Source Highlights
- Leading informative sources
- Coverage gaps or discrepancies  
- Multiple source confirmations

Cite all facts [1], [2], etc. Maintain neutral, factual tone.

**Research Topic:** {research_topic}
**Data:** {summaries}
"""

#======================================

report_writer_instructions_person_search = """
# Digital Profile Report: {research_topic}
Date: {current_date}

## Personal & Professional Overview
- Current position & professional affiliations
- Educational background & qualifications
- Career timeline & major milestones

## Professional Network Analysis
- LinkedIn presence & career progression
- Professional achievements & recognition
- Industry expertise & specializations

## Social Media & Public Presence
- Twitter/X thought leadership & engagement
- Facebook/Instagram professional content
- Media mentions & interviews

## Legal & Regulatory Involvement
- Court cases & business litigation
- Professional legal opinions
- Regulatory matters

## Achievements & Recognition
- Awards, honors & industry recognition
- Speaking engagements & published works
- Digital reputation analysis

## Verification & Analysis
- Cross-platform information consistency
- Network size & influence metrics
- Information reliability assessment

Cite all information [1], [2], etc. Maintain professional, respectful tone.

**Person Profile:** {research_topic}
**Data:** {summaries}
"""

#======================================

report_writer_instructions_investment = """
# Investment Research Report: {research_topic}
Date: {current_date}

## Executive Summary
- Investment Recommendation (Buy/Hold/Sell)
- Target Price & Key Highlights  
- Risk Rating & Investment Horizon

## Financial Performance Analysis
- Revenue & Profitability Trends
- Balance Sheet & Cash Flow Strength
- Valuation Multiples & Peer Comparison

## Business Fundamentals
- Business Model & Competitive Advantages
- Management Quality & Strategy
- Market Position & Competitive Landscape

## Growth Prospects & Risks
- Growth Drivers & Market Opportunities
- Business & Financial Risk Assessment
- ESG Considerations

## Investment Recommendation
- Bull Case, Bear Case & Base Case Scenarios
- Valuation Analysis & Price Target
- Timeline & Key Catalysts

Cite all data [1], [2], etc. Use markdown formatting.

**Analysis Subject:** {research_topic}
**Data:** {summaries}
"""

#================================================================================
