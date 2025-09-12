# <p style="text-align: center">The Cultural Heritage Connectivity Cookbook (2025 Edition)</h2>

<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px auto; text-align: center; max-width: 800px;">
  <div><strong>Gethin Rees</strong><br><em>King's College London</em></div>
  <div><strong>Arno Bosse</strong><br><em>KNAW HuC</em></div>
  <div><strong>Rossana Damiano</strong><br><em>University of Turin</em></div>
  <div><strong>Leif Isaksen</strong><br><em>University of Exeter</em></div>
  <div><strong>Tariq Yousef</strong><br><em>University of Southern Denmark</em></div>
  <div><strong>Elton Barker</strong><br><em>The Open University</em></div>
</div>

<div style="margin: 20px auto; text-align: center; max-width: 900px;">
  <!-- First row: 5 people -->
  <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px; margin-bottom: 20px;">
    <div><strong>Khalid Al Khatib</strong><br><em>Rijksuniversiteit Groningen</em></div>
    <div><strong>Anne Chen</strong><br><em>Bard College</em></div>
    <div><strong>Enrico Daga</strong><br><em>The Open University</em></div>
    <div><strong>Stephen Gadd</strong><br><em>University of Pittsburgh</em></div>
    <div><strong>William Mattingly</strong><br><em>Yale</em></div>
  </div>
  <!-- Second row: 4 people -->
  <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px;">
    <div><strong>Diana Maynard</strong><br><em>University of Sheffield</em></div>
    <div><strong>Chiara Palladino</strong><br><em>Durham University</em></div>
    <div><strong>Sebastiaan Peeters</strong><br><em>University of Twente</em></div>
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

<h2>Introduction</h2>
The depth and diversity of Cultural Heritage collections are recognised as invaluable for enriching lives, fostering social and cultural cohesion, and acting as a valuable economic resource. Yet making full use of those collections and the individual records within them remains hampered by a series of interrelated problems:
1.	digital catalogue metadata tend to exist for only a small proportion of CH collections;
2.	where it exists, it is often sparse, unstructured and contains varying forms of bias;
3.	where structured, it is often not aligned with external authorities.

This means that it is currently difficult to discover individual items and almost impossible to link them to other records within the same collection, let alone between different resources.

To address these issues, guidelines have been produced to improve the Findability, Accessibility, Interoperability and Reusability of digital assets through machine-actionable methods. Based on FAIR principles, Linked Open Data (LOD) has proven an effective mechanism for identifying, disambiguating and linking key entities, such as place, people, objects and events, but implementing LOD tends to require massive investment in time, resource and expertise. More recently, transformer-based AI Large Language Models (LLMs) have demonstrated a remarkable capacity to interpret and contextualise natural language. However, while LLMs are far more intuitive to use, their probabilistic and variable outputs make data enrichment unstable and unpredictable: they can return simply too many errors to make their use worthwhile for data curation. The particular scenario set out here uses a combination of LOD and LLM technologies to enable digital assets to be enriched through the processes of Named Entity Recognition, Named Entity Disambiguation, and Relationship Extraction. 

The following cookbook provides different recipes, derived from LOD and LLM technologies, for enabling CH institutions to enrich their metadata at scale. We envisage two user profiles of the cookbook. One user will be a collections manager who is interested in making use of digital technologies for enriching their objects, but won't necessarily have the technical expertise to do this for themselves. The second user, who has more technical proficiency, will be able to use our recipes as an inspiration or basis for their own work.

The cookbook has the following structure. It has notebooks for:
1. Data preparation and processes — in which we set out: (i) how to get the data in a format that can be used in these processes; and (ii) the different ways of identifying named entities and then disambiguating them.
2. Evaluation - in which we set out how to assess the results of the data processing according to standard metrics.
3. Applications - in which we set out examples use cases for what you'll be able to do with the processed data.
There is also a Glossary of concepts and an About page.

A final note: this work is very much of the moment: September 2025. Given the rapid pace of technological change, particularly in LLMs, we anticipate that the specific tools and methods that we outline here will not be so cutting edge in a year. In other words, the recipes should *not* be considered a maintained service or best practice that is future-proofed, nor, indeed, a ready to go implementation. That said, we believe that these simple-to-follow recipes can be easily adapted to different scenarios, updated by new technologies, and extended for greater coverage. If you have any comments or suggestions, please do raise a GitHub ticket on this repo or email officers@pelagios.org.

