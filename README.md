# AfriMed-QA: A pan-African Medical QA Dataset

[![DOI](https://zenodo.org/badge/792690879.svg)](https://zenodo.org/doi/10.5281/zenodo.11630146)

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg


Collaborating Organizations:

[Intron Health](https://www.intron.io),
[SisonkeBiotik](https://www.sisonkebiotik.africa/),
[BioRAMP](https://www.bioramp.org),
[Georgia Institute of Technology](https://www.gatech.edu/),
[MasakhaneNLP](https://www.masakhane.io/),
[Google Research](https://www.research.google.com)

Funded by:
[Google Research](https://www.research.google.com),
[Bill & Melinda Gates Foundation](https://www.gatesfoundation.org/),
[PATH](https://www.path.org),



#### Summary

AfriMed-QA creates a novel large foundational open-source dataset of 24,000 pan-African clinically diverse questions and
 answers to rigorously evaluate LLMs for accuracy, factuality, hallucination, demographic bias, potential for harm, comprehension, and recall.
 The dataset’s geographical and clinical diversity facilitates robust contextual evaluation of LLMs in African healthcare and provides a sufficiently large corpus to finetune LLMs to mitigate biases discovered.

#### Background

The meteoric rise of LLMs and their rapid adoption in healthcare has driven healthcare leaders in developed economies to design and implement robust evaluation studies to better understand their strengths and weaknesses. Despite their stellar performance across multiple task domains, LLMs are known to hallucinate, propagate biases in their training data, spill potentially harmful information, and are prone to misuse. Since a large proportion of LLM training data are sourced from predominantly western web text, the resulting LLMs have limited exposure to LMIC-specific knowledge. Furthermore, there is minimal evidence to show that stellar performance touted in western literature transfers to healthcare practice in developing countries given their underlying knowledge gap. As with medical devices, drugs, and other interventions in healthcare, physician and patient exposure to health AI tools should be evidence-based, a result of rigorous evaluation and clinical validation to understand LLM strengths and weaknesses in context.

This is the first and largest effort to create a pan-African multi-region multi-institution dataset addressing
 multiple axes of LLM capabilities, rigorously documenting evidence in the context of African healthcare highlighting use cases or clinical specialties where LLMs shine as well as situations where they fall short or have a high potential for harm.

The project will be a timely and invaluable resource guiding the African academic, clinical, biomedical, and research communities on the utility of LLMs in African healthcare at a scale that not only enables robust and rigorous LLM evaluation but provides a sufficiently large dataset to mitigate biases discovered– by finetuning these LLMs, adequately exposing model weights to African healthcare data in context. Such a rigorous evaluation could also uncover desirable and highly valuable but unexplored applications of LLMs in African healthcare, enabling African healthcare professionals to use LLMs in novel and relevant ways that improve patient outcomes.


### Dataset Stats [Phase 2 release]

- Number of Questions: 24,348
- Number of Human Answers with explanations: 14,517
- Total Number of Unique Contributors: 621
- Countries: 18 ['NG', 'TZ', 'KE', 'GH', 'UG', 'BW', 'PH', 'ZA', 'ZW', 'LS', 'ZM',
       'MZ', 'AU', 'SZ', 'US', 'FR', 'MW', 'ET']
- Medical Specialties: 32
- Medical Schools: 60+
- Gender: Female 48.35% / Male 51.35% / Other 0.3%

#### Question Type

|  | tier  |  num questions  | 
| -------- | -------- | ------- | 
| AfriMed-QA-Consumer-Queries  | Crowdsourced |  10,000  | 
| AfriMed-QA-MCQ          |  Crowdsourced  |     6,066 | 
 AfriMed-QA-SAQ        |   Crowdsourced  |      4,013 | 
| AfriMed-QA-Expert-MCQ          |  Experts  |     3,910 | 
| AfriMed-QA-Expert-SAQ        |   Experts  |      359 | 

- Crowdsourced = 20,079
- Expert = 4,269

### Repo Structure

- data - the dataset csv stays here
- src - code (e.g. python file) goes here
- results - csvs with LLM outputs go here
- notebooks - jupyter notebooks go here

### How to Use the Data

AfriMed-QA-MCQ \[Multiple Choice Questions\]: These are multiple choice questions where 2 - 5 answer options are
 provided with at least one correct
 answer. Each question includes the correct answers(s) along with an explanation or rationale. Data columns Required
 : `question`, `answer_options`, `correct_answer`, `answer_rationale`. Please note that there are 3 types of
  MCQ questions. 1) True/False where only 2 answer options are provided, 2) Single correct answer, and 3) Multiple
   correct answers

AfriMed-QA-SAQ \[Short Answer Questions\]: These are open-ended questions that require a short essay, usually one to
 three paragraphs. Answers must include context, rationale, or explanations. Data columns Required
 : `question`, `answer_rationale`. Evaluate model performance based on overlap with human answer/rationale

AfriMed-QA-Consumer-Queries: These represent questions provided by contributors in response to a prompt or clinical
 scenario. For example, the prompt could be, "Your friend felt feverish and lightheaded and feels she has Malaria. What
  questions should she ask her doctor?". The contributor could then respond by asking, "How many days should I wait
  to be sure of my symptoms before seeing the doctor". A clinician contributor could then respond with an answer
   along with the rationale. Data columns Required
 : `prompt`, `question`, `answer_rationale`. You will need to concatenate the prompt with the question for full context.


#### Data Description

The dataset (csv) contains a few important columns:
- sample id: unique record identifier
- question_type: Multiple Choice (MCQ), Short Answer (SAQ), or Consumer Queries
- prompt: the clinical scenario on which the contributor will base their question. This field is valid only for
 consumer queries
- question: the human-generated question
- question_clean: post-processed question text to fix issues with new lines and spacing around numbers, units, and
 punctuations
- answer_options: 2 to 5 possible answers for MCQs. Only valid for MCQs
- correct_answer: the correct answer(s)
- answer_rationale: explanation or rationale for selected correct answer
- specialty: indicates question (sub)specialty
- tier: crowdsourced or expert-generated

Other fields provide more context on the contributor's (self-reported) background:
- gender: Female/Male/Other
- country: 2-letter contributor country code
- discipline: healthcare related or not, e.g. Nursing, Pharmacist, etc.
- clinical_experience: for contributors with healthcare backgrounds, this indicates if they are a student/trainee
, resident, attending, etc. 

Other fields report reviewer ratings of contributor questions/answers:
- quality: boolean thumbs up or down indicating reviwer opinion of question or answer quality or formatting issues
- neg_percent: a measure of how much we can rely on responses provided by the contributor. It is the percentage of
 thumbs down ratings the contributor has received out of all responses reviewed for the contributor on the project.

The following are 5-point scale ratings by reviewers on the following criteria:
- rated_african: Requires African local expertise
- rated_correct: Correct and consistent with scientific consensus
- rated_omission: Omission of relevant info
- rated_hallucination: Includes irrelevant, wrong, or extraneous information
- rated_reasonable: Evidence of correct reasoning or logic, even if the answer is wrong
- rated_bias: Indication of demographic bias
- rated_harmful: Possibility of harm 

Columns for Bias Detection or Counterfactual Analysis, Boolean
- region_specific: contributor indicates if question requires local African medical expertise
- mentions_Africa: question mentions an African country
- mentions_age: question refers to patient age
- mentions_age: question mentions patient gender

The `split` column indicates samples assigned to train/test split. 
LLM responses to questions in the test split will be sent for human evaluation.

#### Data Quality Issues
This dataset was crowdsourced from clinician experts and non-clinicians across Africa.
Although the dataset has gone through rigorous review to weed our low-quality responses before release, it is
 possible that some issues may have been missed by our review team. Please report any lingering issues found by
 raising an issue, or send an email to tobi@intron.io.

Due to a bug in the web application for collecting questions, when saving the entry, new lines were ignored for a 
subset of questions. This predominantly impacts questions that list lab results. This was eventually found and fixed.
Text for questions impacted were fixed by adding a space to separate concatenated text. 

For example, in the following text where new lines are missing between each lab result:
```
Laboratory studies show:Hematocrit 42%Leukocyte count 16,000/mm3Segmented neutrophils 85%Lymphocytes 15% Platelet count 200,000/mm3Arterial blood gas analysis on an FIO2 of 1.0 shows:pH 7.35PCO2 42 mm HgPO2 55 mm HgChest x-ray shows diffuse alveolar infiltrates bilaterally and no cardiomegaly.
```
was transformed to
```
Laboratory studies show: Hematocrit 42% Leukocyte count 16,000/mm3 Segmented neutrophils 85% Lymphocytes 15% Platelet
 count 200,000/mm3 Arterial blood gas analysis on an FIO2 of 1.0 shows: pH 7.35 PCO2 42 mmHg PO2 55 mmHg Chest x-ray
 shows diffuse alveolar infiltrates bilaterally and no cardiomegaly.
```

## Model Performance Metrics

## No Explanation (MCQ - Accuracy, SAQ BertScore)

| Model Name            | Owner/Contributor   | Afrimed-QA v1 | USMLE | AfriMed-QA Experts | AFR-MCQ | AFR-SAQ BertScore | 
|-----------------------|---------------------|--------------|------------------|------------------|------------------|------------------|
| Phi-3-mini-128k-instruct | Abraham          |0.6813        |0.5750            | 0.5903  |0.5688         |0.8804         | 
| Phi-3-mini-4k-instruct   | Amina            |0.6803        |0.5766            |0.6036   |0.6298|0.8740|
| Phi-3-medium-128k-instruct| Abraham          |0.7520        |0.6842          |0.6708   |0.7405         |0.8661         | 
| Meta-Llama-3-8B          | Abraham            |0.6003        |0.4973            |0.7889    |0.6194         |0.8661         | 
| Meta-Llama-3.1-405B          | Abraham            |0.8210        |0.8068           |0.7627    |0.7958         |0.7964         | 
| JSL-MedLlama-3-8B-v2.0 | Amina          |0.6723        |0.6072            |0.5726    | 0.6713|0.8901|
| Meta-Llama-3.1-8B-Instruct | Amina          |0.6933        |0.6269            |0.6189   |0.7509|0.8677|
| Mistral-7B-Instruct-v0.2 | Amina          |0.5837        |0.5003            |0.4847   |0.6159|0.8709|
| Mistral-7B-Instruct-v0.3 | Amina          |0.6100        |0.5130            |0.5084   |0.6263|0.8744| 
| Claude 3.5 sonnet | Mardhiyah          |0.8423       |0.8327       |0.7770 |  0.7820       |0.8574         | 
| Claude 3 sonnet | Mardhiyah          | 0.7330      |0.6489       |0.6504 | 0.6817        |0.8719         | 
| Claude 3 Opus | Mardhiyah          |0.8110       | 0.7800      |0.7455 | 0.7820        |  0.8696       | 
| Claude 3 Haiku | Mardhiyah          |0.7433       |0.6709       | 0.6639| 0.7439        |  0.8656       | 
| Gpt 4 | Mardhiyah          |0.8247       |0.7989       |0.7568 | 0.7993        |0.8727         | 
| Gpt 4o | Mardhiyah          |0.8500       |0.8814       |0.7928 | 0.7924        | 0.8825        | 
| Gpt 4o mini | Mardhiyah          |0.7880       |0.7400       |0.7176 | 0.7405        |0.8808         | 
| Gpt 3.5 Turbo 1106| Mardhiyah          |0.6890       |0.5750       |0.5629 |   0.6332      | 0.8813        | 
| PMC-Llama-7B | Charles          |0.5433       |0.5090       |0.4629 |  0.3599       |    0.8650     | 
| Meditron-7B | Charles          |0.5807       |0.5334       |0.5102 |   0.4360      |  0.8547       | 
| Meta-Llama-3-70B | Charles          |0.8043       |0.7808       |0.7379 | 0.7716        |    0.7945     | 
| OpenBioLLM-70B | Charles          |0.6863       |0.5862       |0.6661 |    0.6955     |  0.8292       | 
| OpenBioLLM-8B | Charles          |0.5327       |0.4674       |0.4499 | 0.5363        | 0.8629        | 
| BioMistral-7B | Charles          |0.5353       |0.4564       |0.4402 | 0.3806        | 0.7938        | 
| Mixtral-8x7B-Instruct-v0.1 | Charles          |0.7023       |0.6002       |0.6033 |  0.6678       |    0.8455     | 
| Gemini Pro  | Mercy         |         |0.5962         |0.7455        |0.6678         |         |
| Gemini Ultra  | Mercy        |         |0.7879         |0.7390         |0.7578         |         |         
|MedLM medium        | Mercy         |         |0.5962         |0.6036         |0.6678         |         |         


## With Explanations


| Model Name            | Owner/Contributor   | MCQ Accuracy | MCQ BertScore F1 | MCQ Avg Rouge | SAQ BertScore F1 | SAQ Avg Rouge | Consumer Queries BertScore F1 | Consumer Queries Avg Rouge |
|-----------------------|---------------------|--------------|------------------|---------------|------------------|---------------|-------------------------------|-----------------------------|
| GPT-4 turbo          | Mardhiyah            |0.8243        |0.8559            |0.1999         |0.8654            |0.2055         |   0.8252                      | 0.0625            |
| GPT-4o                | Tobi                |0.8276        |0.8614            |0.2293         |0.8776            |0.2426         | 0.8254                        | 0.0674                      |
| GPT-4                 | Tobi                |0.8253        |0.8617            |0.2225         |0.8732            |0.2199         | 0.8385                        | 0.0808                      |
| GPT-4o mini         |                 |        |           |         |            |         |                        |                       |
| GPT 3.5 turbo         | Mardhiyah           |0.683        | 0.8667           |0.2536        |0.8765            |0.2542         | 0.8313        |0.0761                              |
| Gemma-7B-Instruct     |                     |              |                  |               |                  |               |                               |                     |
| Phi 3                 |                     |              |                  |               |                  |               |                               |                             |
| Claude 3.5 sonnet     |            |       |            |        |            |         |                         |                       |
| Claude 3 sonnet       | Mardhiyah           |0.6893        |0.8564            |0.2178         |0.8681            |0.2234         |0.8141                          |0.0540                       |
| Claude 3 Opus       | Mardhiyah             |0.7907        |0.8590            |0.2232         |0.8661            |0.2138         |0.8172                           |0.0544                   |
| Claude 3 Haiku       | Timothy             |0.712        |0.8558            |0.2276         |0.8620            |0.2140         |0.8163                           |0.0505                    |
| Meta-LLAMA-3.1-8B       |   Amina         |               |                  |               |                  |               |                               |                 |
| Meta-LLAMA-3.1-70B       |                     |               |                  |               |                  |               |                               |              |
| Meta-LLAMA-3.1-405B       |                     |              |                  |               |                  |               |                               |              |
| Cohere CommandR       | Henok               |              |                  |               |                  |               |                               |                             |
| OpenBioLLM-8B        | Charles           | 0.5193             |                  |               |                  |               |                               |                             |
| OpenBioLLM-70B        | Charles          |  0.7280            |                  |               |                  |               |                               |                             |
|Phi-3-mini-128k-instruct        |Abraham     |0.6676        |0.866        |0.2308        |0.8718        |0.227        |0.8266        |0.0672        |
|Llama3 8B              |Abraham              |0.635        |0.8592        |0.2286        |0.8624        |0.2094        |0.8344        |0.0909        |
|Phi-3-mini-4k-instruct |Amina                |0.6606        |0.866        |0.2432        |0.8681        |0.2214        |0.8186        |0.0595        |
|Mistral-7B-Instruct-v0.2    |Amina           |0.551        |0.8613        |0.2296        |0.8505        |0.1763        |0.8259        |0.0706        |
|Mistral-7B-Instruct-v0.3    |Amina           |       |        |        |        |        |       |       |
|JSL-MedLlama-3-8B-v2.0 |Amina                |0.6606        |0.8577        |0.2248        |0.8721        |0.2273        |0.8303        |0.0793        |
|Gemini Ultra           |Mercy                |0.8003        |0.8716        |0.2642        |0.8754        |0.2356        |0.8362        |0.079        |
|MedPalm 2              |Mercy                |0.7456        |0.8735        |0.2451        |0.8716        |0.2234        |0.8379        |0.0774        |
|Gemini pro             |Mercy                |0.631         |0.8677        |0.2413        |0.8601        |0.2025        |0.8213        |0.0579        |
|MedLM                  |Mercy                |0.7043        |0.8666        |0.247         |0.8633        |0.2083        |0.8303        |0.0785        |
| Meditron              | Charles             |0.5653              |                  |               |                  |               |                               |                             |
| BioMistral-7B | Charles              |              |                  |               |                  |               |                               |                             |
| Orpo-Med-v0           |                     |              |                  |               |                  |               |                               |                             |
| Mixtral-8x7B-Instruct-v0.1        | Charles                     | 0.7203             |                  |               |                  |               |                               |                             |
| PMC-LLama-7B             | Charles             | 0.5197             |                  |               |                  |               |                               |                             |
| Open-bio-med-merge    |                     |              |                  |               |                  |               |                               |                             |
| Med42                 |                     |              |                  |               |                  |               |                               |                             |
| Meta-LLAMA-3-8B       |   Charles                  | 0.6676              |                  |               |                  |               |                               |                   |
| Meta-LLAMA-3-70B       |   Charles                  |  0.8036             |                  |               |                  |               |                               |                  |
|Mistral-7B-Instruct-v0.2| Charles           |0.6757        |                  |               |                  |               |                               |     
| MedAlpaca             | Charles             |              |                  |               |                  |               |                               |                             |





### Base vs Instruct Prompts

##### Zero-Shot Evals with Base Prompt

| Model Name            | Owner/Contributor   | MCQ Accuracy | MCQ BertScore F1 | MCQ Avg Rouge | SAQ BertScore F1 | SAQ Avg Rouge | Consumer Queries BertScore F1 | Consumer Queries Avg Rouge |
|-----------------------|---------------------|--------------|------------------|---------------|------------------|---------------|-------------------------------|-----------------------------|
| OpenBioLLM-8B-Instruct| Charles           |0.5193        |0.8503           |0.1414        |N/A          |N/A      |N/A                           |N/A                          |
| OpenBioLLM-70B-Instruct |    Charles       | 0.7211   | 0.8594             |0.1829               | 0.8533              | 0.1668                 |    N/A           |   N/A                            |                             |
| Meta-Llama-3-8B-Instruct        | Charles           |0.6676        |0.8625            |0.2400         | N/A            |N/A       |   N/A                    | N/A            |
| Meta-Llama-3-70B-Instruct      | Charles              |0.7963        |0.8683            |0.2513         | N/A           | N/A        | N/A                      | N/A                      |
| Mixtral-8x7B-Instruct-v0.1            | Charles      | 0.7203        | 0.8644           | 0.2492         | 0.8691           | 0.2309        | N/A                       | N/A               |
| Gemma-7B-Instruct    |         Charles               |              |                  |               |                  |               |                               |                             |
| BioMistral-7B        |        Charles             |              |                  |               |                  |               |                               |                             |


##### Zero-Shot Evals with Instruct Prompt

| Model Name            | Owner/Contributor   | MCQ Accuracy | MCQ BertScore F1 | MCQ Avg Rouge | SAQ BertScore F1 | SAQ Avg Rouge | Consumer Queries BertScore F1 | Consumer Queries Avg Rouge |
|-----------------------|---------------------|--------------|------------------|---------------|------------------|---------------|-------------------------------|-----------------------------|
| OpenBioLLM-8B-Instruct| Charles           |0.5100       |0.8454         |0.1148         |N/A        |N/A        |N/A                           |N/A                          |
| OpenBioLLM-70B-Instruct |    Charles       |  0.7280     |  0.8542            |  0.1460                 |    0.8548           |      0.1695            | N/A               |    N/A    |                             |
| Meta-Llama-3-8B-Instruct  | Charles           |0.6583        |0.8533            |0.2105         | N/A            | N/A         |   n/A                     | N/A          | N/A
| Meta-Llama-3-70B-Instruct    | Charles              |0.8036       |0.8644           |0.2376        | N/A           |N/A        | N/A                        |N/A                     |
| Mixtral-8x7B-Instruct-v0.1            | Charles      | 0.7046        | 0.8626          | 0.2365       | 0.8639            | 0.2090        | N/A                       | N/A                     |
| Gemma-7B-Instruct    |         Charles               |              |                  |               |                  |               |                               |                             |
| BioMistral-7B        |        Charles             |              |                  |               |                  |               |                               |                             |
| GPT-4o        |        Mardhiyah             | 0.848             |  0.8528                |  0.2019             |                  |               |                               |                             |
| Claude 3 Opus        |        Mardhiyah             | 0.677             |  0.8561                |  0.2138             |                  |               |                               |                             |
|Gemini Ultra        |Mercy        |0.7973        |0.875        |0.2662        |         |         |         |         |
|Gemini pro        |Mercy        |0.684        |0.869        |0.2547        |         |         |         |         |
|MedLM        |Mercy        |0.702        |0.8678        |0.2495        |         |         |         |         |
|MedPalm 2        |Mercy        |0.742        |0.873        |0.2428        |         |         |         |         |



##### 3-Shot (5-Shot) Evals with Instruct Prompt
| Model Name            | Owner/Contributor   | MCQ Accuracy | MCQ BertScore F1 | MCQ Avg Rouge | SAQ BertScore F1 | SAQ Avg Rouge | Consumer Queries BertScore F1 | Consumer Queries Avg Rouge |
|-----------------------|---------------------|--------------|------------------|---------------|------------------|---------------|-------------------------------|-----------------------------|
| GPT-4o       |        Mardhiyah             | 0.7427             |  0.8610                |  0.2256             |                  |               |                               |                             |
| Claude 3 Opus       |        Mardhiyah             | 0.763             |  0.8572                |  0.2181             |                  |               |                               |                             |
|Gemini Ultra        |Mercy        |0.808 (0.8063)        |0.875 (0.8752)        |0.2519 (0.2535)        |         |         |         |         |
|Gemini pro        |Mercy        |. (0.7013)        |. (0.8725)        |. (0.2548)        |         |         |         |         |
|MedLM        |Mercy        |0.7013 (0.6993)        |0.8725 (0.8732)        |0.2548 (0.2585)        |         |         |         |         |
|MedPalm 2        |Mercy        |. (0.751)        | . (0.8708)        |. (0.2474)        |         |         |         |         |
|Meta-Llama-3-70B-Instruct       |Charles        |0.78.12        |       |      |         |         |         |         |
|OpenBioLLM-70B       |Charles        |0.78.91        |       |      |         |         |         |         |

### MedQA (USMLE) Prompts

##### Zero-Shot Evals with Base Prompt
| Model Name            | Owner/Contributor   | MCQ Accuracy | MCQ BertScore F1 | MCQ Avg Rouge | SAQ BertScore F1 | SAQ Avg Rouge | Consumer Queries BertScore F1 | Consumer Queries Avg Rouge |
|-----------------------|---------------------|--------------|------------------|---------------|------------------|---------------|-------------------------------|-----------------------------|
| GPT-4o                |        Mardhiyah             | 0.8539             |  0.8329                |  0.1975             |                  |               |                               |                             |
| Claude 3 Opus         |        Mardhiyah             | 0.7321             |  0.8288               |  0.1934             |                  |               |                               |                             |
|MedLM        |Mercy        |0.60015        |0.8318        |0.1901        |         |         |         |         |
|MedPalm2        |Mercy        |0.7038        |0.8265        |0.1451        |         |         |         |         |
|Gemini Ultra        |Mercy        |0.794        |0.8316        |0.1691        |         |         |         |         |
|Gemini pro        |Mercy        |0.5915        |0.83614        |0.2074        |         |         |         |         |
|Meta-Llama-3-70B-Instruct       |Charles        |0.7669        |       |      |         |         |         |         |
|OpenBioLLM-70B       |Charles        |0.7714        |       |      |         |         |         |         |


### Notes:
- **MCQ**: Multiple-Choice Questions.
- **SAQ**: Short Answer Questions.
- **Consumer Queries**: Queries posed by end-users, likely in a real-world application.
- **Metrics**:
  - **Accuracy**: Correct answers as a percentage of total answers.
  - **BertScore F1**: F1 score as computed by BertScore.
  - **Avg Rouge**: Avg rouge is arithmetic mean of ROUGE-1, ROUGE-2 and ROUGE-L


#### Prompting Strategies
Prompting Strategies include: 
- Zero shot
- Few shot (3 shot)
- Instruct: provide text instruction to the guide the model

#### LLM Output format

For MCQ questions, present the question and options to the model and report it's correct
 answer selection along with its explanation or rationale. Please provide responses in the following fields:
 - prompt [string]: the input string to the LLM. It should contain the question and options 
 - correct_answer [string]: model's selected answer from options provided 
 - answer_rationale [string]: LLMs explanation
 - quality [boolean]: 1 if model answer matches correct answer in dataset, 0 otherwise.

For SAQ questions, present the question to the model and report its answer and explanation or rationale. Please
 provide responses in the following fields:
 - prompt [string]: the input string to the LLM. It should contain the question 
 - answer_rationale [string]: LLMs short essay, answer, or explanation
 
For Consumer queries, present the question and options to the model and report it's correct
 answer selection along with its explanation or rationale. Please provide responses in the following fields:
 - prompt [string]: the input string to the LLM. It should contain the prompt (check data description above) and the
  contributor's question as well as any instruction provided.
 - answer_rationale [string]: LLMs answer with explanation
 
 If human answers (explanations) are provided for the question, please include:
 - ROUGE-1 [float]: ngram overlap of human with LLM explanation
 - ROUGE-2 [float]: ngram overlap of human with LLM explanation
 - ROUGE-L [float]: ngram overlap of human with LLM explanation
 - BertScore [float]: semantic similarity of human with LLM explanation

#### Result format

Please send back all responses aggregated into a single csv. If you prompt multiple LLMs, please combine all results
into one csv.
Your csv should contain the `sample_id` from the AfriMed-QA dataset along with the LLM outputs using the fields
 described above.
 Please include a column in your csv called `source` indicating the specific LLM. For clarity, transparency, and
  reproducibility, please note that your LLM name should include the size of the model. For example, LLAMA should be
   written as LLAMA-8B or LLAMA-40B or whichever size was prompted.

#### Human Evaluations
All LLM responses for question in the test partition of the dataset will be sent for human evaluation on the Intron
 Health crowdsourcing platform. Human Eval will be two-fold:
 - Non-clinicians: will rate the LLM responses for relevance, helpfulness, and bias.
 - Clinicians: will rate LLM responses using the 7 criteria in the data description section above with a 5-point scale. 

Raters will be blinded to response source. Raters will randomly evaluate model and human responses. 
Some questions will receive single or multiple-reader ratings. Inter-rater reliability or agreement will be
 computed for answers with multiple ratings

Human evaluation files were too large to push to github. They were stored on google drive.
Access them via this link: https://drive.google.com/drive/folders/1feRZ524FajVGsBvt13u-QYk_OvKgea8X?usp=sharing

# Working with the Codebase:

### Installation
Clone the repository, create a new environment and install the required dependencies:

Check pytorch installation for your machine at https://pytorch.org/get-started/locally/

```bash
git clone https://github.com/intron-innovation/AfriMed-QA
cd AfriMed-QA
conda create -n afrimed python=3.10
conda activate afrimed

conda install pytorch torchvision -c pytorch 

pip install -r requirements.txt
```

### Setting Up Your Model
1. **Subclass the Model Class**:
   - Navigate to `src/models/`.
   - Define your own custom model class by subclassing the provided `Model` class in `models.py`.
   - Ensure your class initializes the model and includes a `predict` method that returns a prediction as a string.

### Running the Code
To run the model and generate predictions, use the provided bash script in the `scripts` folder. 
Simply change the pretrained_model_path argument to run this script with your predefined model class.
If your model requires more arguments, recreate your own bash script using the same naming template. The script
 requires specific arguments to function correctly.

#### Required Arguments:
- `pretrained_model_path`: The path to the pretrained model.
- `data_path`: The path to the data file.
- `prompt_file_path`: The path to the prompt file.
- `q_type`: The type of questions to process (e.g., `mcq`).
- `num_few_shot`: The number of few shot examples to use (e.g., 0). default is 0

#### Running the Script
Navigate to the `scripts` directory and execute the bash script with the required arguments. Example usage:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/your/project/"
bash scripts/run.sh 
```
- An example bash script for Phi has been provided.

#### Few-shot prompting
- In your script, set `num_few_shot_values` = the number of shots your choose

#### Instruction tuning
- In your script, set prompt_type to either base or instruct.
   - `base`: prompt the model without any instruction tuning
   - `instruct`: prompt with model with a instruction e.g as an African Doctor.





### Output
- The predictions will be evaluated and saved to the `results` folder.


#### License

&copy; 2024. This work is licensed under a CC BY-NC-SA 4.0 license.
