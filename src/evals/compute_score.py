import pandas as pd 

from bert_score import score as bert_score
from rouge import Rouge


def compute_score(q_type, data):
    valid_data = data[data["rationale"].str.len() > 4].copy()
    if len(valid_data) < 2:
        return (data, 0,0,0,0,0,0,0)

    scores = bert_score(valid_data["outputs"].tolist(), valid_data["rationale"].tolist(), lang="en", verbose=True)
    p, r, f1 = scores[0], scores[1], scores[2]
    BERTScore_Precision, BERTScore_Recall, BERTScore_F1 = p.numpy().mean(), r.numpy().mean(), f1.numpy().mean()
    valid_data["BERTScore_Precision"] = p.numpy()
    valid_data["BERTScore_Recall"] = r.numpy()
    valid_data["BERTScore_F1"] = f1.numpy()
    

    rouge = Rouge()
    rouge_scores = valid_data.apply(
        lambda row: rouge.get_scores(row["outputs"], row["rationale"]), axis=1
    )
    valid_data["ROUGE-1"] = [score[0]["rouge-1"]["f"] for score in rouge_scores]
    valid_data["ROUGE-2"] = [score[0]["rouge-2"]["f"] for score in rouge_scores]
    valid_data["ROUGE-L"] = [score[0]["rouge-l"]["f"] for score in rouge_scores]

    rg1, rg2, rl = valid_data["ROUGE-1"].mean(), valid_data["ROUGE-2"].mean(), valid_data["ROUGE-L"].mean()
    average_rouge = (rg1 + rg2 + rl) / 3    

    if q_type == "mcq":
        data["correct"] = data["answer"] == data["preds"]
        accuracy = data["correct"].mean()
        print("Accuracy:", accuracy)
    else:
        accuracy = ""

    print("BERTScore Precision (mean):", BERTScore_Precision)
    print("BERTScore Recall (mean):", BERTScore_Recall)
    print("BERTScore F1 (mean):", BERTScore_F1)
    print("Average ROUGE-1 Score:", rg1)
    print("Average ROUGE-2 Score:", rg2)
    print("Average ROUGE-L Score:", rl)
    print("Average ROUGE:", average_rouge)



    data = data.merge(valid_data[["sample_id", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore_Precision", "BERTScore_Recall", "BERTScore_F1"]], on="sample_id", how="left")


    return data, BERTScore_Precision, BERTScore_Recall, BERTScore_F1, rg1, rg2, rl, accuracy


def write_scores():
    pass
    



def process_mcq(text):
    
    text = str(text).replace("*", "")
    text = text.replace("#", "")
    text = text.replace("<b>", "")
    text = text.replace("Option:", "")
    text = text.replace("Option", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("Instruction:", "")
    text = text.replace("The correct answer is ", "")
    option_t = text.replace("Answer", "")
    option_t = option_t.replace(":", "").strip(" ")
    
    

    option = option_t.strip()[0] if option_t.strip()[0].upper() in "ABCDE" else option_t.strip(" ")[1]
    if "Answer:" in text:
        option_t = text.split("Answer:\n")[-1].strip().strip()[0]
        print(option_t)
    if len(option_t) >1:
        option_t = option_t[0]
    rationale = text.strip().replace("\n\nRationale:\n\n", " ")
    return option_t.upper(), rationale

def process_others(text):
    text = str(text)
    text = text.replace("*", "")
    text = text.replace("#", "")
    text = text.replace("<b>", "")
    text = text.replace("Option:", "")
    text = text.replace("Option", "")
    text = text.replace("Answer:", "")
    text = text.replace("Rationale", "")
    text = text.replace("Rationale:", "")
    text = text.split("Question:")[-1]


    rationale = text.strip().replace("\n\nRationale:\n\n", "\n")
    return  rationale



mcq= pd.read_csv("/data3/abraham/AfriMed-QA/results/afrimed-qa_mcq - few shot + instuction tuned.csv")


base = ['sample_id', 'question', 'rationale', 'options_len',
       'answer', 'A', 'B', 'C', 'D',  'model_prompt']
model = "MedLM  (3shot)"

models = [
    'Gemini Ultra (3 shot)', 'Gemini_ultra (5 shot)',
       'MedPalm 2 (5 shot)', 'MedPalm 2 (3 shot)', 'Gemini Pro (5 shot)',
       'Gemini pro (3 shot)', 'MedLM (5 shot)', 'MedLM (3shot)'
]
model_dirs = [
    "gemini_ultra",
    "gemini_ultra",
    "medpalm2",
    "medpalm2",
    "gemini_pro",
    "gemini_pro",
    "medlm",
    "medlm"
]

shot_counts = [
    "3",
    "5",
    "5",
    "3",
    "5",
    "3",
    "5",
    "3"
]



for index in range(len(models)): 
    print(f"------------------------{[models[index]]}")
    if mcq[models[index]].notna().any():
        gemini_ultra = mcq[base+[models[index]]].copy()
        gemini_ultra.rename(columns={models[index]:"model_output"}, inplace=True)

        gemini_ultra['preds'] = gemini_ultra['model_output'].apply(lambda x: process_mcq(x)[0])
        gemini_ultra['outputs'] = gemini_ultra['model_output'].apply(lambda x: process_mcq(x)[1])
        final = compute_score("mcq", gemini_ultra)
        print(f"../results/{model_dirs[index]}/afrimed-qa_mcq_instruct_prompt_{shot_counts[index]}shot.csv")
        final[0].to_csv(f"../results/{model_dirs[index]}/afrimed-qa_mcq_instruct_prompt_{shot_counts[index]}shot.csv", index=False)
        print("\n\n\n\n")
