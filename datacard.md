# Dataset Card for Dataset Name

<!-- Provide a quick summary of the dataset. -->
Language models (LMs) commonly report perplexity on monolithic data held out from the training distribution. 
Implicitly or explicitly, this data is composed of domains—variations in the distribution of language. Does rising performance lift all data? Or do some domains capture most improvement in LM fit? 
And rigorous experimental controls for confounders of perplexity, such as benchmark contamination, are so far neglected. 
To answer, we present  Perplexity Analysis for Language Model Assessment (Paloma), a benchmark curated from 18 sources comprising 585 text domains. We also introduce a set of standards for subsampling, decontamination, training data order, tokenization, and evaluation format detailed in our paper. 

## Dataset Details

### Dataset Description



<!-- Provide a longer summary of what this dataset is. -->
Paloma is for examining relative differences in LM fit on domains. We take these relative differences as a proxy of model fit to the shared knowledge, values, and social context that position the humans producing language in a domain. While we expect contemporary LMs to have a limited fit to the most complex of these latent factors of domains, improving fit to all factors is necessary both to improve perplexity and for any actual use of the LM. For example, better perplexity on a particular dialect of English suggests that that model will make a better chatbot for people that speak that dialect.

Our selection of datasets covers both performance within the distribution of common pretraining corpora as well as examining how well models are fit to distributions of language that are not specifically included in pretraining data. Sources were selected based on the following desiderata: 1) including known resources, 2) including fine-grained domains, 3) including domains repre- senting specific communities of interest.

It is beyond the scope of any one paper to prescribe an exhaustive set of domains that should be examined for a LM. Rather Paloma brings together a substantial selection of domains that are identifiable from already available metadata to demonstrate the kinds of analyses possible with hundreds of domains and rigorous experimental controls.
Different research goals will motivate different definitions and selections of domains, but other researchers can apply the guidelines we detail in our paper to novel fine-grained domains suitable for their research questions. One of the key advantages of evaluating a model by its fit to a collection of text representing a domain is that such domains can be identified not just by researchers who study LMs. We hope future work will identify many more domains that no one discipline would think to look at.


- **Curated by:** Ian Magnusson, Akshita Bhagia, Valentin Hofmann, Luca Soldaini, Ananya Harsh Jha, Oyvind Tafjord, Dustin Schwenk, Evan Pete Walsh, Kyle Lo, Dirk Groeneveld, Iz Beltagy, Hannaneh Hajishirzi, Noah A. Smith, Kyle Richardson, and Jesse Dodge
- **Language(s) (NLP):** English and other languages incidentally occuring in source corpora
- **License:** All data subsets in this dataset are licensed under the LR Agreement, except for those as listed in the "License" section of the Dataset Card.



### Dataset Sources

<!-- Provide the basic links for the dataset. -->

- [Paper]() -- (TODO update when paper is preprinted)
- [Website](paloma.allen.ai)



## Uses

<!-- Address questions around how the dataset is intended to be used. -->

This benchmark is intended for use to evaluate language model fit to fine-grained domains.

### Direct Use

<!-- This section describes suitable use cases for the dataset. -->

This data is intended for evaluating the likilihood of text from a given domain by a language model.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the dataset will not work well for. -->

Note that the sources contained in this benchmark include varying liscences with differing restrictions

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

The sources in this dataset are each organized into their own subcorpus. This consists of a `val` and `test` split. Data within this is organized as lines of JSON data where each line represents a document and its associated metadata. The type of metadata available varies from source to source, but each line contains at least a field `'text'` which contains the text of the document.

## Dataset Creation

### Curation Rationale

<!-- Motivation for the creation of this dataset. -->

[More Information Needed]

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Standard language modeling benchmarks 
Though it is common practice to evaluate on held out data from the pretraining corpus of a given model, we evaluate *across* several major pretraining corpora and standard language modeling benchmarks. We also break down performance per domain within the datasets that have multiple domains. Note that although the Paloma benchmark analysis in our paper describes results on the Pile, we are not able to re-host this data.

| Source            | Citation                                      | Description                                                                                                                                                               |
|-------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| c4-en             | Raffel et al (2019) via Dodge et al (2021)    | Standard contemporary LM pretraining corpus automatically filtered from the April 2019 Common Crawl scrape                                                                |
| mc4-en            | Xue et al (2021)                              | The English language portion of a pretraining corpus automatically filtered from 71 Common Crawl  scrapes                                                                 |
| Pile              | Gao et al (2020)                              | Standard contemporary LM benchmark from curated multi-source data including large scale non-webscraped sources                                                            |
| Wikitext-103      | Merity et al (2016)                           | A standard collection of verified “Good” and “Featured” articles on Wikipedia                                                                                             |
| Penn Tree Bank    | Marcus et al (1999) via Nunes, Davide. (2020) | Classic Wall Street Journal benchmark with linguistic structure annotations omitted                                                                                       |
| RedPajama         | Together Computer (2023)                      | A publicly available reproduction of the LLaMA (Touvron et al., 2023) pretraining source mixture, combining large amounts of webscraped text with smaller curated sources |
| Falcon-RefinedWeb | Penedo et al. (2023)                          | A corpus of English sampled from all Common Crawl scrapes until June 2023, more aggressively filtered and deduplicated than c4 and mc4-en                                 |
| Dolma v1.5        | Soldaini et al. (2023)                        | A three trillion token corpus that samples sources commonly used to train LMs in order to enable open research on pretraining data                                       |

#### Fine-grained domain benchmarks

Where typical pretraining corpora offer at most tens of labeled domains usually based on where the data is sourced, we examine datasets with up to an order of magnitude more domains.  Existing datasets (M2D2 and c4 100 Domains) and datasets we curate from Dolma v1.5 use metadata to define hundreds of domains over Wikipedia, Semantic Scholar, Common Crawl, Reddit, and Github data. These include diverse domains from *Culture and the arts: Performing arts*, a topic on Wikipedia, to *r/depression*, a forum on Reddit for mental health support.

| Source                          | Citation                                         | Description                                                                       |
|---------------------------------|--------------------------------------------------|-----------------------------------------------------------------------------------|
| M2D2 S2ORC                      | Reid et al (2022)                                | Papers from Semantic Scholar grouped by hierarchical academic field categories    |
| M2D2 Wiki                       | Reid et al (2022)                                | Wikipedia articles grouped by hierarchical categories in the Wikipedia ontology   |
| c4 100 Domains                  | Chronopoulou et al (2021)                        | Balanced samples of the top 100 URL domains in C4                                 |
| Dolma 100 Subreddits            | Soldaini et al. (2023)                           | Balanced samples of the top 100 Subreddits from the Dolma Reddit subset           |
| Dolma 100 Programming Languages | Kocetkov et al. (2022)via Soldaini et al. (2023) | Balanced samples of the top 100 programming languages from the Dolma Stack subset |

#### Disparities between speech communities

Some communities are known to be underserved by existing models. Following HELM, We measure disparities in performance on corpora of African American English and White aligned English from TwitterAAE, as well as nine corpora of English from different countries with the ICE dataset. Note that although the Paloma benchmark analysis in our paper describes results on ICE, we are not able to re-host this data.

| Source     | Citation                                           | Description                                                                                                                                                           |
|------------|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ICE        | Greenbaum and Nelson (1996) via Liang et al (2022) | English from around the world curated by local experts, with subsets for Canada, East Africa, Hong Kong, India, Ireland, Jamaica, Philippines, Singapore, and the USA |
| TwitterAAE | Blodgett et al. (2016) via Liang et al (2022)      | Balanced sets of tweets classified as African American or White aligned English                                                                                          |

#### Fringe sources previously studied for problematic discourse

Text from some fringe online communities has been shown to contain larger proportions of hate speech and toxicity than more mainstream sources. [Longpre et al. (2023)](https://arxiv.org/abs/2305.13169) has shown that varying amount of toxic content in pretraining data exhibits a tradeoff between non-toxic generation and ability to classify toxicity, indicating that model fit to discourse with toxicity is worth measuring. Measuring perplexity on Manosphere, Gab, and 4chan characterises model familiarity with distinct social contexts in which toxic language arises.

| Source            | Citation               | Description                                                                                                                                 |
|-------------------|------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| Manosphere Corpus | Ribeiro et al (2020)   | 9 forums where a set of related masculinist ideologies developed over the 2000s and 2010s                                                       |
| Gab Corpus        | Zannettou et al (2018) | Data from 2016-18 from an alt-right, free-speech-oriented social media platform shown to contain more hate speech than mainstream platforms |
| 4chan Corpus      | Papasavva et al (2020) | Data from 2016-19 from a politics subforum of an anonymity-focused forum found to contain among the highest rates of toxic content          |

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

[More Information Needed]

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

[More Information Needed]

### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

[More Information Needed]

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

[More Information Needed]

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

In this paper, we distinguish sources from domains, although not all cases permit such easy distinction. We use *source* to refer to a selection of data that is characterized by the decisions of the people who curated that data, whether that curation is automatic as in scraping c4 or manual as in selecting the subcorpora of the The Pile. By contrast we use *domain* to refer to a set of documents that belong together because they are originally produced by a group of humans that share a distinct social context. Considered as such, domains may overlap; a document's author may belong to the set of English speakers in Jamaica and the set of AI researchers. Further note, that domains are often latent categorizations which we only approximate because complete metadata does not exist.

Also, some domains in Paloma appear in multiple sources, such as academic papers. Though The Pile and RedPajama process academic papers differently, the subcorpora on academic papers in each source represent different approximations of the same or very similar domains. However for the sake of simplicity, we make the reductive assumption of counting all 585 domains in Paloma as fully distinct.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be made aware of the risks, biases and limitations of the dataset. More information needed for further recommendations.

## Citation [optional]

<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Dataset Card Authors [optional]

[More Information Needed]

## Dataset Card Contact

[More Information Needed]