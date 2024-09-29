from bert_score import score as bert_score
from rouge import Rouge


def compute_score(q_type, explanation, data):

    valid_data = data[data["rationale"].str.len() > 4].copy()
    if len(valid_data) < 2:
        return (data, 0, 0, 0, 0, 0, 0, 0)

    if q_type == "mcq":
        data["correct"] = data["answer"] == data["preds"]
        accuracy = data["correct"].mean()
        print("Accuracy:", accuracy)
        if not explanation:
            return (data, 0, 0, 0, 0, 0, 0, accuracy)
    else:
        accuracy = ""

    scores = bert_score(
        valid_data["outputs"].tolist(),
        valid_data["rationale"].tolist(),
        lang="en",
        verbose=True,
    )
    p, r, f1 = scores[0], scores[1], scores[2]
    BERTScore_Precision, BERTScore_Recall, BERTScore_F1 = (
        p.numpy().mean(),
        r.numpy().mean(),
        f1.numpy().mean(),
    )
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

    rg1, rg2, rl = (
        valid_data["ROUGE-1"].mean(),
        valid_data["ROUGE-2"].mean(),
        valid_data["ROUGE-L"].mean(),
    )
    average_rouge = (rg1 + rg2 + rl) / 3

    print("BERTScore Precision (mean):", BERTScore_Precision)
    print("BERTScore Recall (mean):", BERTScore_Recall)
    print("BERTScore F1 (mean):", BERTScore_F1)
    print("Average ROUGE-1 Score:", rg1)
    print("Average ROUGE-2 Score:", rg2)
    print("Average ROUGE-L Score:", rl)
    print("Average ROUGE:", average_rouge)

    data = data.merge(
        valid_data[
            [
                "sample_id",
                "ROUGE-1",
                "ROUGE-2",
                "ROUGE-L",
                "BERTScore_Precision",
                "BERTScore_Recall",
                "BERTScore_F1",
            ]
        ],
        on="sample_id",
        how="left",
    )

    return (
        data,
        BERTScore_Precision,
        BERTScore_Recall,
        BERTScore_F1,
        rg1,
        rg2,
        rl,
        accuracy,
    )


def write_scores():
    pass
