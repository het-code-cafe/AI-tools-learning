```mermaid
graph TD
    A[Initial question/observation]:::start --> B[Background research]:::process
    B --> C[Form a hypothesis]:::decision
    C --> D[Empiric testing]:::process
    D --> E{Analyze results and evaluate hypothesis}:::decision
    E --> F[Communicate results]:::endnode
    E -- Hypothesis false? --> C

    classDef start fill:#FFF066,stroke:#113428,stroke-width:2px,color:black;
    classDef process fill:#ACD1AF,stroke:#113428,stroke-width:2px,color:black;
    classDef decision fill:#ACD1AF,stroke:#113428,stroke-width:2px,color:black;
    classDef endnode fill:#FFF066,stroke:#113428,stroke-width:2px,color:black;

    linkStyle 5 stroke:#459578,stroke-width:2px,color:113428,fill:none,color:black;
```
