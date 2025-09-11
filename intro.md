# Enriching Digital Heritage with LLMs and Linked Open Data

This cookbook explores the transformative potential of combining Large Language Models (LLMs) and Linked Open Data (LOD) to enrich cultural heritage metadata and foster FAIR (Findable, Accessible, Interoperable, Reusable) usage.

# FAQs
## What's a recipe?
You can think of 'recipes' as a set of instructions for creating a specific thing with a set of ingredients. Recipes can be bundled together into 'cookbooks'.

In this case, we've created recipes for people who work with cultural heritage collections to show something of what's possible with AI and linked open data.

## Research Challenge
Cultural Heritage (CH) collections represent one of Europe's greatest assets, but their metadata faces significant challenges:

### Current Problems
1. **Limited Coverage**: Digital catalogue metadata exists for only a small proportion of large national collections
2. **Poor Quality**: Where metadata exists, it is often sparse, unstructured, and contains varying forms of bias
3. **Lack of Standardization**: Structured metadata is often unstandardized and unaligned with Persistent Identifiers (PIDs) from external authorities

These issues limit user queries, hinder discovery, and make integration between institutions difficult, preventing the FAIR use of heritage objects.

### Our Approach
This workshop explores synergies between two different approaches:
- **Linked Open Data (LOD)**: Ontologies that provide structured, standardized metadata but can be laborious to produce
- **Large Language Models (LLMs)**: Vector embeddings that excel at interpreting natural language but produce probabilistic, variable outputs

By carefully combining both approaches, we can harness the benefits and mitigate the weaknesses of each to radically improve FAIRness and engagement with CH collections.
## Workshop Objectives

This workshop explores the transformative potential of combining LLMs and LOD to enrich cultural heritage metadata. We focus on three core processes for working with Named Entities:

### Key Focus Areas

#### 1. Named Entity Recognition (NER)
- **Process**: Identifying and extracting named entities from unstructured text
- **Output**: Character strings representing proper nouns and their location within source text
- **Example**: Identifying "Pieter Bruegel" as a person in metadata
- **Focus**: How LLMs compare to and integrate with traditional NER techniques

#### 2. Named Entity Disambiguation (NED)
- **Process**: Associating textual references to entries in authority files
- **Goal**: Distinguish between similar entities (e.g., Bruegel the Elder vs. Bruegel the Younger)
- **Resources**: Authority files such as ULAN, VIAF, NACO, and Geonames
- **Focus**: How LLMs can link recognized entities to authority files

#### 3. Named Entity Relations

- **Process**: Identifying relationships between named entities and described objects
- **Example**: Determining whether "Pieter" is the producer or subject of a painting
- **Importance**: Crucial for meaningful indexing and querying
- **Focus**: Using LLMs to automate relation identification through contextual analysis
- 
### Geographic Focus

We use **geographic Named Entities** as our principal case study, drawing on the experience of the **Pelagios Network** - a community dedicated to developing efficient LOD practices for cultural heritage with emphasis on geographic aspects.



## Workshop Organizers

- **Arno Bosse** - WP lead for Infrastructure, Globalise project, Digital Infrastructure Dept., KNAW HuC, NL
- **Rossana Damiano** - Assoc. Prof. of Computer Science, Dept. of Informatics, U. of Turin, IT
- **Leif Isaksen** - Prof. of Digital Humanities, Digital Humanities Lab, University of Exeter, UK
- **Gethin Rees** - Lead Curator, Digital Mapping, British Library, UK - Main Contact
- **Tariq Yousef** - Asst. Prof. of Data Science, Dept. of Maths & Computer Sci., U. of Southern Denmark, DK
