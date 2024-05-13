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

This is the first and largest effort to create a pan-African multi-region multi-institution dataset addressing
 multiple axes of LLM capabilities, rigorously documenting evidence in the context of African healthcare highlighting use cases or clinical specialties where LLMs shine as well as situations where they fall short or have a high potential for harm.

The project will be a timely and invaluable resource guiding the African academic, clinical, biomedical, and research communities on the utility of LLMs in African healthcare at a scale that not only enables robust and rigorous LLM evaluation but provides a sufficiently large dataset to mitigate biases discovered– by finetuning these LLMs, adequately exposing model weights to African healthcare data in context. Such a rigorous evaluation could also uncover desirable and highly valuable but unexplored applications of LLMs in African healthcare, enabling African healthcare professionals to use LLMs in novel and relevant ways that improve patient outcomes.


### Dataset Stats [Phase 1 release]

- Number of Questions: 10,000
- Number of Human Answers with explanations: 4,601
- Total Number of Unique Contributors: 746
- Countries: 15 ['NG', 'TZ', 'KE', 'GH', 'UG', 'BW', 'PH', 'ZA', 'ZW', 'LS', 'ZM',
       'MZ', 'AU', 'SZ', 'US']
- Medical Specialties: 32
- Medical Schools: 60+
- Gender: Female 49.49% / Male 50.21% / Other 0.3%

#### Question Type

|  | num questions  | 
| -------- | ------- | 
| AfriMed-QA-Consumer-Queries  |   5500  | 
| AfriMed-QA-MCQ          |         3000 | 
| AfriMed-QA-SAQ        |           1500 | 

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
- question_type: Multiple Choice (MCQ), Short Answer (SAQ), or Consumer Queries
- prompt: the clinical scenario on which the contributor will base their question. This field is valid only for
 consumer queries
- question: the human-generated question
- answer_options: 2 to 5 possible answers for MCQs. Only valid for MCQs
- correct_answer: the correct answer(s)
- answer_rationale: explanation or rationale for selected correct answer
- question_source: indicates the medical school or online question bank from which the question was selected

Other fields provide more context on the contributor's (self-reported) background:
- age_group: indicates age bracket, e.g. 18-25
- gender: Female/Male/Other
- country: 2-letter country code
- discipline: healthcare related or not, e.g. Nursing, Pharmacist, etc.
- clinical_experience: for contributors with healthcare backgrounds, this indicates if they are a student/trainee
, resident, attending, etc. 

Other fields report reviewer ratings of contributor questions/answers:
- quality: boolean thumbs up or down
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

The `split` column indicates samples assigned to train/test split. 
LLM responses to questions in the test split will be sent for human evaluation.

#### Data Quality Issues
This dataset was crowdsourced from clinician experts and non-clinicians across Africa.
Although the dataset has gone through rigorous review to weed our low-quality responses before release, it is
 possible that some issues may have been missed by our review team. Please report any lingering issues found by
 raising an issue, posting on BioRAMP slack, or send an email to tobi@intron.io.

### LLMs
The following is a non-exhaustive list of open/closed or general/biomedical LLMs to be evaluated as part of this
 project along with contributor names
- MedPalm-2 [Mercy, Fola]
- GPT-4 turbo [Foutse, Fola]
- Med-Gemini [Mercy]
- Gemini Pro [Mercy]
- Gemini Ultra [Mercy, Fola]
- Gemma 1.2
- Phi 3
- Claude 3 sonnet [Mercy]
- Cohere CommandR [Henok]
- GPT 3.5 turbo [Henok]
- OpenBioLLM-70B [Ify, Fola]
- JSL-MedLlama-3-8B
- Meditron [Charles]
- Meta-LLaMa-3 [Charles]
- BioMistral [Foutse]
- Orpo-Med-v0
- Mixtral 8x22B
- PMC-LLama [Charles]
- Open-bio-med-merge
- Med42
- Meta-LLAMA-3-8B
- MedAlpaca [Charles]
- etc

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


# Working with the Codebase:

## Requirements
- Python 3.9 or higher
- PyTorch
- Any other dependencies listed in `requirements.txt`

### Installation
Clone the repository and install the required Python packages:
```bash
git clone <repository-url>
cd AfriMed-QA
pip install -r requirements.txt
```

### Setting Up Your Model
1. **Subclass the Model Class**:
   - Navigate to `src/models/`.
   - Define your own custom model class by subclassing the provided `Model` class in `models.py`.
   - Ensure your class initializes the model and includes a `predict` method that returns a prediction as a string.

### Running the Code
To run the model and generate predictions, use the provided bash script in the `scripts` folder. Recreate your own bash script using the same naming template. The script requires specific arguments to function correctly.

#### Required Arguments:
- `pretrained_model_path`: The path to the pretrained model.
- `data_path`: The path to the data file.
- `prompt_file_path`: The path to the prompt file.
- `q_type`: The type of questions to process (e.g., `mcq`).

#### Running the Script
Navigate to the `scripts` directory and execute the bash script with the required arguments. Example usage:
```bash
bash scripts/run_prediction.sh 
```

### Output
- The predictions will be evaluated and saved to the `results` folder.


#### License

&copy; 2024. This work is licensed under a CC BY-NC-SA 4.0 license.
