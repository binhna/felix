from nltk.translate.bleu_score import sentence_bleu
from sari import SARIsent
from infer import rewrite
from tqdm import tqdm
import constants

import json, os
from tagging_model import FelixTagger
from transformers import AutoTokenizer, AutoConfig
import torch
import sys


def bleu_score(file, model, tokenizer, device):
    with open(file) as f:
        data = f.read().split("\n")
    data = [d.split("\t") for d in data if d]
    score_bleu = []
    score_sari = []
    for (src, tgt) in tqdm(data):
        src = " ".join(src.split()[1:-1])
        tgt = " ".join(tgt.split()[1:-1])
        infer_s = rewrite(src, model, tokenizer, device)
        score_bleu.append(sentence_bleu([tgt.split()], infer_s.split()))
        score_sari.append(SARIsent(src, infer_s, [tgt]))
    print("BLEU", sum(score_bleu) / len(score_bleu))
    print("SARI", sum(score_sari) / len(score_sari))
    return sum(score_bleu) / len(score_bleu), sum(score_sari) / len(score_sari)


if __name__ == "__main__":
    # "ngày tháng năm nào?"
    replacement = {
        "lúc nào?": ["năm nào?", "ngày nào?", "tháng nào?"],
        "?": [
            "tỉnh nào?",
            "thành phố nào?",
            "quận nào?",
            "huyện nào?",
            "xã nào?",
            "nơi nào?",
            "chỗ nào?",
            "khu nào?",
            "thị trấn nào?",
            "phường nào?",
        ],
    }

    replacement = {
        key: sorted(v, key=lambda x: len(x), reverse=True)
        for key, v in replacement.items()
    }

    model_path = "./models" if len(sys.argv) < 2 else sys.argv[1]
    pretrained_path = "../shared_data/BDIRoberta_4L"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    with open(os.path.join(model_path, "args.json")) as f:
        args = json.load(f)

    config = AutoConfig.from_pretrained(pretrained_path)
    model = FelixTagger(
        model_name=pretrained_path,
        device=device,
        max_sub_word_length=config.max_position_embeddings - 2,
        num_classes=len(constants.ID2TAGS),
        is_training=False,
        query_dim=args["query_dim"],
    )
    model.load_pretrained(model_path)
    # model.load_state_dict(
    #     torch.load(
    #         os.path.join(model_path, "best_model_correct_tagging.pt"),
    #         map_location=torch.device(device),
    #     )
    # )
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
    print(f"Model {model_path} loading is done!")

    bleu_score("./data/sentence_pair.txt", model, tokenizer, device)
