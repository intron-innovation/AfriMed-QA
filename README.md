# AfriMed-QA: A pan-African Medical QA Dataset

Supported by:

[Intron Health](https://www.intron.io),
[SisonkeBiotik](https://www.sisonkebiotik.africa/),
[BioRAMP](https://www.bioramp.org),
[Google Research](https://www.research.google.com),
[Bill & Melinda Gates Foundation](https://www.gatesfoundation.org/),
[PATH](https://www.path.org),
[MasakhaneNLP](https://www.masakhane.io/)


#### Summary

AfriMed-QA creates a novel large foundational open-source dataset of 20,000 pan-African clinically diverse questions and
 answers to rigorously evaluate LLMs for accuracy, factuality, hallucination, demographic bias, potential for harm, comprehension, and recall.
 The dataset’s geographical and clinical diversity facilitates robust contextual evaluation of LLMs in African healthcare and provides a sufficiently large corpus to finetune LLMs to mitigate biases discovered.

#### Background

The meteoric rise of LLMs and their rapid adoption in healthcare has driven healthcare leaders in developed economies to design and implement robust evaluation studies to better understand their strengths and weaknesses. Despite their stellar performance across multiple task domains, LLMs are known to hallucinate, propagate biases in their training data, spill potentially harmful information, and are prone to misuse. Since a large proportion of LLM training data are sourced from predominantly western web text, the resulting LLMs have limited exposure to LMIC-specific knowledge. Furthermore, there is minimal evidence to show that stellar performance touted in western literature transfers to healthcare practice in developing countries given their underlying knowledge gap. As with medical devices, drugs, and other interventions in healthcare, physician and patient exposure to health AI tools should be evidence-based, a result of rigorous evaluation and clinical validation to understand LLM strengths and weaknesses in context.

This proposal creates a novel foundational open-source dataset of 20,000 pan-African clinically diverse crowdsourced questions with clinician answers leveraging an existing pan-African validated medical crowdsourcing platform. The dataset’s geographical and clinical diversity facilitates robust contextual evaluation of LLMs in African healthcare and provides a sufficiently large corpus to finetune LLMs to mitigate biases discovered.

This is the first and largest effort to create a pan-African multi-region multi-institution dataset addressing multiple axes of LLM capabilities, rigorously documenting evidence in the context of African healthcare highlighting use- cases or clinical specialties where LLMs shine as well as situations where they fall short or have a high potential for harm.

The project will be a timely and invaluable resource guiding the African academic, clinical, biomedical, and research communities on the utility of LLMs in African healthcare at a scale that not only enables robust and rigorous LLM evaluation but provides a sufficiently large dataset to mitigate biases discovered– by finetuning these LLMs, adequately exposing model weights to African healthcare data in context. Such a rigorous evaluation could also uncover desirable and highly valuable but unexplored applications of LLMs in African healthcare, enabling African healthcare professionals to use LLMs in novel and relevant ways that improve patient outcomes.


### Dataset Stats [Phase 1 release]

- Number of Questions: 10,000
- Number of Human Answers with explanations: 4,961
- Total Number of Unique Contributors: 727
- Countries: 15 ['KE', 'NG', 'TZ', 'UG', 'GH', 'BW', 'ZA', 'PH', 'MZ', 'LS', 'AU',
       'US', 'SZ', 'ZW', 'ZM']
- Medical Specialties: 32
- Medical Schools: 50+
- Gender: 54.45/44.36/1.19

#### Question Type

|  | num questions  | 
| -------- | ------- | 
| consumer queries  |   5500  | 
| mcq          |         3000 | 
| saq        |           1500 | 