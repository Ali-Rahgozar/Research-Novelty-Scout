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
    I[Research Idea] --> LLM[LLM]

    LLM -->|Tool call| T[search_papers]
    T --> R[Tool Result]
    R --> LLM

    LLM -->|No tool call| O[Final Answer]

    LLM -.->|Repeated iterations| LLM
```