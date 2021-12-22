from nltk.translate.bleu_score import sentence_bleu
from sari import SARIsent
from infer import rewrite
from tqdm import tqdm

# candidate_corpus = ['My', 'full', 'pytorch', 'test']
# references_corpus = [['My', 'full', 'pytorch']]

# print(sentence_bleu(references_corpus, candidate_corpus, weights=(1/3, 1/3, 1/3)))

# import random
# import pandas as pd
# import underthesea

# with open("./data/not_revise_marks.txt") as f1, open("./data/revise_marks.txt") as f2:
#     df = f1.read().split("\n") + f2.read().split("\n")
#     df = sorted(list(set(df)), key=lambda x: len(x))
#     random.shuffle(df)
#     df = [line.split("\t") for line in df]
#     df = pd.DataFrame(df, columns=["question", "short answer", "rewrite1"])

# text_samples = []
# data = []
# fail = []
# src_data = []
# tgt_data = []
# for i, row in tqdm(df.iterrows(), total=df.shape[0]):
#     if row["rewrite1"] and row["question"] and row["short answer"]:
#         q = row.question.lower().strip()
#         q += "?" if not q.endswith("?") else ""
#         q = q.replace(":?", "?")
#         a = str(row['short answer']).lower().strip()
#         rewrite_s = row.rewrite1.lower().strip()
#         src = "<s> " + " ".join(underthesea.word_tokenize(q)) + " " + " ".join(underthesea.word_tokenize(a)) + " </s>"
#         tgt = "<s> " + " ".join(underthesea.word_tokenize(rewrite_s)) + " </s>"
#         src_data.append(src)
#         tgt_data.append(tgt)


# with open("./data/original_sp.txt", "w") as f:
#     for s, t in zip(src_data, tgt_data):
#         f.write(f"{s}\t{t}\n")

# pair = []
# with open("./data/valid.conll") as f:
#     valid_data = f.read().split("\n\n")
# for line in tqdm(valid_data):
#     if not line:
#         continue
#     line = line.split("\n")
#     # print(line)
#     line = [w.split()[0] for w in line]
#     line = " ".join(line)
#     try:
#         index = src_data.index(line)
#         target = tgt_data[index]
#         pair.append((line, target))   
#     except:
#         print(line)

# print(len(valid_data))
# print(len(pair))

# with open("./data/sentence_pair.txt", "w") as f:
#     for sp in pair:
#         f.write(f"{sp[0]}\t{sp[1]}\n")


def bleu_score(file):
    with open(file) as f:
        data = f.read().split("\n")
    data = [d.split("\t") for d in data if d]
    score_bleu = []
    score_sari = []
    for (src, tgt) in tqdm(data):
        src = " ".join(src.split()[1:-1])
        tgt = " ".join(tgt.split()[1:-1])
        infer_s = rewrite(src)
        score_bleu.append(sentence_bleu([tgt.split()], infer_s.split()))
        score_sari.append(SARIsent(src, infer_s, [tgt]))
    print(sum(score_bleu)/len(score_bleu))
    print(sum(score_sari)/len(score_sari))


bleu_score("./data/sentence_pair.txt")