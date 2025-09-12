# Awesome Lorentz list of resources

A curated list of resources mentioned during the Lorentz workshop. Resources are organised by type (datasets, models, tools) and possibly also by the processing task they help to perform. (If you are not familiar with the format, do check out existing [awesome lists on GitHub](https://github.com/awesomelistsio)).

Contents

[Resources for evaluation](#resources-for-evaluation)

[Models](#models)

[Tools](#tools)

[Applications](#applications)

[Authority Lists](#authority-lists-\(for-reconciliation\))

## Resources for evaluation {#resources-for-evaluation}

This includes datasets (e.g. for benchmarking), annotation guidelines, shared tasks, etc.

- …

## Models {#models}

### Document processing (OCR, page layout analysis, etc.)

- [dots.ocr](https://github.com/rednote-hilab/dots.ocr) – Multilingual Document Layout Parsing in a Single Vision-Language Model  
  - Online demo (for quick testing): [https://github.com/rednote-hilab/dots.ocr](https://github.com/rednote-hilab/dots.ocr) 

Model overviews

- [European Open Source AI Index](https://osai-index.eu/the-index) \- index on openness of AI models, rated on various criteria

## Tools  {#tools}

### Annotation

- [INCEpTION](https://github.com/inception-project/inception) – A semantic annotation platform suitable for various types of textual annotations (NER, EL, etc.).   
- [Recogito Studio](https://recogitostudio.org) \- An Extensible Platform for Collaborative, Standards-Based Annotation of TEI Text, IIIF Images, and PDFs, including geotagging and reconciliation with different gazetteers (WHG, Pleiades, Wikidata, etc.).  
- [Immarkus](https://immarkus.xmarkus.org/) \- open-source tool for semantic image annotation  
- [Image Positions](https://wd-image-positions.toolforge.org/) – Image Annotation platform inside the Wikidata environment  
- [FairCopy](https://faircopyeditor.com) \- tool for reading, transcribing, and encoding text with custom annotations  
- [CATma](https://catma.de) \- mark-up and analysis tool   
- [Prodi.gy](http://Prodi.gy) \- annotation tool for SpaCy (not open-source)  
- [Liiive](https://liiive.now/) \- Real-time collaborative viewing & annotation for IIIF image collections

See also:  
[ATRIUM T4.5.2 Annotation tools overview.xlsx](https://docs.google.com/spreadsheets/d/1GZwJ2sBeIB8IbUu56x6WymYuqcNKxotc/edit?usp=sharing&ouid=100418980193070696410&rtpof=true&sd=true)

### Named Entity Recognition

- [GATE geotagger](https://cloud.gate.ac.uk/shopfront/displayItem/geographical-ner) — This service identifies geographical named entities and disambiguates them against [GeoNames](https://www.geonames.org/). The service currently makes use of the [Mordecai3 geoparser](https://github.com/ahalterman/mordecai3); more details on Mordecai3 can be found in [this paper](https://arxiv.org/abs/2303.13675).  
- [GATE Pleiades NER](https://cloud.gate.ac.uk/shopfront/displayItem/pleiades-ner) — This service identifies geographical named entities and disambiguates them against the [Pleiades](https://pleiades.stoa.org/) dataset. The approach taken is to use all the names from each entry in Pleiades (that contains a representative point) to build a simple gazetteer. Locations which are ambiguous (i.e. those where multiple lookups overlap) are disambiguated using a geometrical approach. We assume that, in a similar way to word sense disambiguation, a document is likely to be discussing a single area, and so we choose the set of locations which minimise the area covered by the set of selected points; this is currently done by calculating axis aligned bounding boxes for efficiency purposes.

### 

### Entity Linking & Reconciliation

- [Spacyfishing](https://github.com/Lucaterre/spacyfishing) – A spaCy Python wrapper for the [entity-fishing](https://github.com/kermitt2/entity-fishing) tool for entity linking against Wikidata.   
- [OpenRefine](https://openrefine.org) – open source tool to manipulate datasets, including semi-automatic entity linking and variant clustering.   
- [TagMe](https://sobigdata.d4science.org/web/tagme/demo) – a tool to identify short phrases or entities and match them against Wikipedia pages.    
- [Ariadne Services](https://portal.ariadne-infrastructure.eu/services) for Entity Linking and Disambiguation – …

## Applications {#applications}

- [https://github.com/britishlibrary/peripleo](https://github.com/britishlibrary/peripleo) \- a browser-based tool for the mapping of things related to place.  
- [Vistorian](https://vistorian.github.io/vistorian/) \- online environment to visualize spatial and networked data.

## Formats

- [https://github.com/LinkedPasts/linked-places-format](https://github.com/LinkedPasts/linked-places-format) \- Linked Places format is used to describe attestations of places in a standard way, primarily for linking gazetteer datasets.  
- [https://github.com/LinkedPasts/linked-traces-format](https://github.com/LinkedPasts/linked-traces-format) \- Patterns based on the W3C Web Annotation Model, primarily for use in linking resources describing historical phenomena with the places relevant to them. 

## 

## Authority Lists (for Reconciliation) {#authority-lists-(for-reconciliation)}

[https://docs.google.com/spreadsheets/d/1AoGTRArfx7EodTrU8GPdCWJDqqt0fNMg3gcSui2LCxk/edit?gid=0\#gid=0](https://docs.google.com/spreadsheets/d/1AoGTRArfx7EodTrU8GPdCWJDqqt0fNMg3gcSui2LCxk/edit?gid=0#gid=0) 