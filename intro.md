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
The depth and diversity of Cultural Heritage collections are recognised as invaluable for enriching lives, fostering social and cultural cohesion, and acting as a valuable economic resource. Yet making full use of those collections and the individual records within them is still hampered by several interrelated problems:
1.	digital catalogue metadata exists for only a small proportion of CH collections;
2.	where it exists, it is often sparse, unstructured and contains varying forms of bias;
3.	where structured, it is often not aligned with external authorities.

This means that the richness of such collections is obscured, hindering both discovery of individual objects and linking between different resources.

To address these issues, CH institutions have embraced the FAIR principles of Findability, Accessibility, Interoperability and Reusability. Linked Open Data (LOD) technologies like the CIDOC-CRM have proven an effective mechanism for identifying, disambiguating and linking key entities, such as place, people, objects and events, but require large investment in time, resources and expertise. More recently, transformer-based AI Large Language Models (LLMs) have demonstrated a remarkable capacity to interpret and contextualise natural language, thereby making them easy to use, but their probabilistic and variable outputs make integration with other CH infrastructure unstable and unpredictable.

The following cookbook sets out different recipes, derived from LOD and LLM technologies, for enabling CH institutions to enrich their metadata at scale. We envisage two user profiles of the cookbook. One user will be a collections manager who is interested in making use of digital technologies for enriching their objects, but won't necessarily have the technical expertise to do this for themselves. The second user will have more technical proficiency, who will be able to use our recipes as an inspiration or basis for their own work.

The cookbook has the following structure.

A final note: this work is very much of the moment, which is September 2025. Given the rapid pace of technological change, particularly in LLM, we anticipate that the specific tools and methods that we outline here will not be cutting edge in a year. In other words, the recipes should *not* be considered a ready to go implementation, best practice forever, or to be maintained beyond this moment. That said, we believe that these simple-to-follow recipes can be easily adapted to different scenarios, updated by new technologies, and extended for greater coverage.

