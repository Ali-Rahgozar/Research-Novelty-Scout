# Agent Definition
This project is an AI agent specialized in **machine learning research** that investigates whether a proposed research idea has already been explored in the existing literature. It searches and analyzes relevant ML papers to identify closely related work, similarities, differences, and potential research gaps.

```mermaid
flowchart LR
    U[Researcher] --> A[Agent]
    A <--> DB[(ML Literature Sources)]
    A --> O[Research Report]
    O --> U
```


# Agent Architecture


```mermaid
flowchart TD
    I[Research Idea] --> D[Decompose Idea]
    D --> Q[Generate Search Queries]
    Q --> S[Search Literature]
    S --> P[Analyze Candidate Papers]
    P --> C[Compare With Research Idea]
    C --> E{Enough Evidence?}
    E -->|No| Q
    E -->|Yes| R[Generate Research Report]

```