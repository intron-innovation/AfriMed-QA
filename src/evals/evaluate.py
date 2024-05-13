from bert_score import score as bert_score
from rouge import Rouge

def compute_score(args, data):
    p, r, f1 = bert_score(data["outputs"].tolist(), data["rationale"].tolist(), lang="en", verbose=True)
    data["BERTScore_Precision"] = p.numpy()
    data["BERTScore_Recall"] = r.numpy()
    data["BERTScore_F1"] = f1.numpy()
    BERTScore_Precision, BERTScore_Recall, BERTScore_F1 = p.numpy().mean(), r.numpy(), f1.numpy()
    rouge = Rouge()
    rouge_scores = data.apply(
        lambda row: rouge.get_scores(row["outputs"], row["rationale"]), axis=1
    )
    data["ROUGE-1"] = [score[0]["rouge-1"]["f"] for score in rouge_scores]
    data["ROUGE-2"] = [score[0]["rouge-2"]["f"] for score in rouge_scores]
    data["ROUGE-L"] = [score[0]["rouge-l"]["f"] for score in rouge_scores]

    rg1, rg2, rl = data["ROUGE-1"].mean(), data["ROUGE-2"].mean(), data["ROUGE-L"].mean()
    if args.q_type == "mcq":
        data["correct"] = data["answer"] == data["preds"]
        accuracy = data["correct"].mean()
    else:
        accuracy = ""
    
    return (data, BERTScore_Precision, BERTScore_Recall, BERTScore_F1, rg1, rg2, rl,  accuracy)

def write_scores():
    pass
    