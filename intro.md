<h1 style="text-align: center">Enriching Digital Heritage with LLMs and Linked Open Data</h1>

<div style="margin: 20px auto; text-align: center; max-width: 900px;">
  <!-- First row: 5 people -->
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 20px;">
    <div><strong>Khalid Al Khatib</strong><br><em>Rijksuniversiteit Groningen</em></div>
    <div><strong>Anne Chen</strong><br><em>Bard College</em></div>
    <div><strong>Enrico Daga</strong><br><em>The Open University</em></div>
    <div><strong>Stephen Gadd</strong><br><em>Docuracy</em></div>
    <div><strong>William Mattingly</strong><br><em>Yale</em></div>
  </div>
  <!-- Second row: 4 people -->
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;">
    <div><strong>Diana Maynard</strong><br><em>University of Sheffield</em></div>
    <div><strong>Chiara Palladino</strong><br><em>Durham University</em></div>
    <div><strong>Sebastian Peeters</strong><br><em>University of Twente</em></div>
    <div><strong>Nina Claudia Rastinger</strong><br><em>Austrian Academy of Sciences</em></div>
  </div>
  <!-- Third row: 5 people -->
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 20px;">
    <div><strong>Mia Ridge</strong><br><em>British Library</em></div>
    <div><strong>Matteo Romanello</strong><br><em>Odoma</em></div>
    <div><strong>Robert Sanderson</strong><br><em>Yale</em></div>
    <div><strong>Marco Antonio Stranisci</strong><br><em>University of Turin</em></div>
    <div><strong>William Thorne</strong><br><em>University of Sheffield</em></div>
  </div>
  <!-- Fourth row: 4 people -->
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
    <div><strong>Erik Tjong Kim Sang</strong><br><em>Netherlands eScience Center</em></div>
    <div><strong>Leon van Wissen</strong><br><em>University of Amsterdam</em></div>
    <div><strong>Mónica Marrero</strong><br><em>Europeana</em></div>
    <div><strong>Margherita Fantoli</strong><br><em>Catholic University of Leuven</em></div>
  </div>
</div>

The workshop explored the transformative potential of combining Large Language Models (LLMs) and Linked Open Data (LOD) to enrich cultural heritage metadata in ways that foster FAIR usage. The workshop also provided a forum for knowledge exchange amongst participants with experience in Cultural Heritage standards, LOD, LLMs and Named Entity Recognition, and explicitly engaged with the challenges presented by validation, bias, and the ethical processing of heritage data. The workshop made the problem space of working with Named Entities more tractable by dividing it into three core processes: Recognition, Disambiguation, and Relations, each explored at a specific stage of the workshop.

# FAQs
## What's a recipe?
You can think of 'recipes' as a set of instructions for creating a specific thing with a set of ingredients. Recipes can be bundled together into 'cookbooks'.

We're imagining users who are interested in experimenting with recipes that demonstrate what is possible and potentially valuable for Culturual Heritage contexts when combinging LOD and LLMs. Less technically-oriented users may consider browsing the recipes to provide adaptable inspiration to share with funding bodies, technical and collections colleagues. Developers may be interested in mining the cookbook for ideas and adaptable code elements.

In this case, we've created recipes for people who work with cultural heritage collections to show something of what's possible with AI and linked open data.

[//]: # (## Research Challenge)

[//]: # (Cultural Heritage &#40;CH&#41; collections represent one of Europe's greatest assets, but their metadata faces significant challenges:)

[//]: # ()
[//]: # (### Current Problems)

[//]: # (1. **Limited Coverage**: Digital catalogue metadata exists for only a small proportion of large national collections)

[//]: # (2. **Poor Quality**: Where metadata exists, it is often sparse, unstructured, and contains varying forms of bias)

[//]: # (3. **Lack of Standardization**: Structured metadata is often unstandardized and unaligned with Persistent Identifiers &#40;PIDs&#41; from external authorities)

[//]: # ()
[//]: # (These issues limit user queries, hinder discovery, and make integration between institutions difficult, preventing the FAIR use of heritage objects.)

[//]: # ()
[//]: # (### Our Approach)

[//]: # (This workshop explores synergies between two different approaches:)

[//]: # (- **Linked Open Data &#40;LOD&#41;**: Ontologies that provide structured, standardized metadata but can be laborious to produce)

[//]: # (- **Large Language Models &#40;LLMs&#41;**: Vector embeddings that excel at interpreting natural language but produce probabilistic, variable outputs)

[//]: # ()
[//]: # (By carefully combining both approaches, we can harness the benefits and mitigate the weaknesses of each to radically improve FAIRness and engagement with CH collections.)

[//]: # (## Workshop Objectives)

[//]: # ()
[//]: # (This workshop explores the transformative potential of combining LLMs and LOD to enrich cultural heritage metadata. We focus on three core processes for working with Named Entities:)

[//]: # ()
[//]: # (### Key Focus Areas)

[//]: # ()
[//]: # (#### 1. Named Entity Recognition &#40;NER&#41;)

[//]: # (- **Process**: Identifying and extracting named entities from unstructured text)

[//]: # (- **Output**: Character strings representing proper nouns and their location within source text)

[//]: # (- **Example**: Identifying "Pieter Bruegel" as a person in metadata)

[//]: # (- **Focus**: How LLMs compare to and integrate with traditional NER techniques)

[//]: # ()
[//]: # (#### 2. Named Entity Disambiguation &#40;NED&#41;)

[//]: # (- **Process**: Associating textual references to entries in authority files)

[//]: # (- **Goal**: Distinguish between similar entities &#40;e.g., Bruegel the Elder vs. Bruegel the Younger&#41;)

[//]: # (- **Resources**: Authority files such as ULAN, VIAF, NACO, and Geonames)

[//]: # (- **Focus**: How LLMs can link recognized entities to authority files)

[//]: # ()
[//]: # (#### 3. Named Entity Relations)

[//]: # ()
[//]: # (- **Process**: Identifying relationships between named entities and described objects)

[//]: # (- **Example**: Determining whether "Pieter" is the producer or subject of a painting)

[//]: # (- **Importance**: Crucial for meaningful indexing and querying)

[//]: # (- **Focus**: Using LLMs to automate relation identification through contextual analysis)

[//]: # (- )

[//]: # (### Geographic Focus)

[//]: # ()
[//]: # (We use **geographic Named Entities** as our principal case study, drawing on the experience of the **Pelagios Network** - a community dedicated to developing efficient LOD practices for cultural heritage with emphasis on geographic aspects.)
