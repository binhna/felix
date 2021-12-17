import pandas as pd
from tqdm import tqdm
import re
# from underthesea import word_tokenize

fromthis2that = [("năm nào?", "ngày nào?"), ("ngày nào?", "lúc nào?"), ("ngày nào?", "năm nào?")]

def augment_question(question):
    question = re.sub(r"năm nào\?", "ngày nào")


def contains_mark(text):
    if re.search(r"[\'\!\"\#\$\%\&\\\'\(\)\*\+\,\-\.\/\:\;\<\=\>]", text):
        return True
    return False

df = pd.read_excel("./data/RewriteAnnotation.xlsx", sheet_name="Rewrite", engine='openpyxl')
df.fillna(False, inplace=True)

data = set()

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    if row["rewrite1"] and row["question"] and row["short answer"]:
        q = row.question.lower().strip()
        q += "?" if not q.endswith("?") else ""
        a = str(row['short answer']).lower().strip()
        rewrite = row.rewrite1.lower().strip()
        data.add((q, a, rewrite))

print(len(data))
contain_mark_text = []
not_contain_mark_text = []
for q, a, r in data:
    if contains_mark(q+a):
        contain_mark_text.append(f"{q}\t{a}\t{r}")
    else:
        not_contain_mark_text.append(f"{q}\t{a}\t{r}")

contain_mark_text = sorted(contain_mark_text, key=lambda x: len(x))
not_contain_mark_text = sorted(not_contain_mark_text, key=lambda x: len(x))

with open("revise_marks.txt", "w") as f:
    f.write("\n".join(contain_mark_text))
with open("not_revise_marks.txt", "w") as f:
    f.write("\n".join(not_contain_mark_text))

print(len(contain_mark_text))
