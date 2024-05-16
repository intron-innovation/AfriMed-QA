from bert_score import score as bert_score
from rouge import Rouge

def compute_score(q_type, data):
    
    if q_type=="consumer_queries":
        return (data, 0,0,0,0,0,0,0)
    scores = bert_score(data["outputs"].tolist(), data["rationale"].tolist(), lang="en", verbose=True)
    p, r, f1 = scores[0], scores[1], scores[2]
    data["BERTScore_Precision"] = p.numpy()
    data["BERTScore_Recall"] = r.numpy()
    data["BERTScore_F1"] = f1.numpy()
    BERTScore_Precision, BERTScore_Recall, BERTScore_F1 = p.numpy().mean(), r.numpy().mean(), f1.numpy().mean()
    rouge = Rouge()
    rouge_scores = data.apply(
        lambda row: rouge.get_scores(row["outputs"], row["rationale"]), axis=1
    )
    data["ROUGE-1"] = [score[0]["rouge-1"]["f"] for score in rouge_scores]
    data["ROUGE-2"] = [score[0]["rouge-2"]["f"] for score in rouge_scores]
    data["ROUGE-L"] = [score[0]["rouge-l"]["f"] for score in rouge_scores]

    rg1, rg2, rl = data["ROUGE-1"].mean(), data["ROUGE-2"].mean(), data["ROUGE-L"].mean()
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

    return (data, BERTScore_Precision, BERTScore_Recall, BERTScore_F1, rg1, rg2, rl,  accuracy)

def write_scores():
    pass
    