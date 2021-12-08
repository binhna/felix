import torch
import constants
from transformers import RobertaTokenizer
from tagging_model import FelixTagger
import underthesea
import time
from tqdm import tqdm
import re
import numpy as np
import itertools


from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
import torch
import time
import re


# device = "cuda"
# tokenizer = RobertaTokenizer.from_pretrained(model_path)
# model = AutoModelWithLMHead.from_pretrained(model_path).to(device)

def insertion(model, tokenizer, sequence, device):
    # sequence = input("enter: ")#f"Vua Lê Thánh Tông qua đời {tokenizer.mask_token} năm 1497 {tokenizer.mask_token} bị bệnh nặng"
    # sequence = sequence.replace("_", tokenizer.mask_token)
    # print(sequence)

    start = time.time()
    input_ids = tokenizer.encode(sequence, return_tensors="pt").to(device)
    mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

    token_logits = model(input_ids)[0]
    
    mask_token_logits = token_logits[0, mask_token_index, :]
    mask_token_logits = torch.softmax(mask_token_logits, dim=1)
    ids = torch.argmax(mask_token_logits, dim=1)

    ids = ids.cpu().tolist()
    sequence = sequence.split()
    for id in ids:
        idx = sequence.index(tokenizer.mask_token)
        sequence[idx] = tokenizer.decode([id])

    print(" ".join(sequence))
    print(round(time.time() - start, 5)*1000, "ms")
    return " ".join(sequence)



def bpe_tokenizer(words, tokenizer, max_word_length, max_subword_length):
    token_tmp = [tokenizer.bos_token] + words + [tokenizer.eos_token]
    
    attention_mask_words = np.ones(len(token_tmp))
    attention_mask_words = attention_mask_words[:max_word_length]
    attention_mask_words = np.hstack([attention_mask_words, np.zeros(max_word_length - len(attention_mask_words))])
    
    sub_words = [
        tokenizer.encode(token, add_special_tokens=False)
        for token in token_tmp
    ]
    sub_words = sub_words[: max_subword_length]

    word_matrix = np.zeros((max_word_length, max_subword_length))

    j = 0
    for i, tks in enumerate(sub_words):
        if tks[0] == tokenizer.pad_token_id:
            break
        for _ in tks:
            word_matrix[i, j] = 1
            j += 1
    sub_word_ids = list(itertools.chain.from_iterable(sub_words))
    sub_word_ids.extend(
        [tokenizer.pad_token_id] * (max_subword_length - len(sub_word_ids))
    )  # <pad> index
    attention_mask = np.ones(len(sub_word_ids))
    attention_mask[np.array(sub_word_ids) == tokenizer.pad_token_id] = 0
    return sub_word_ids, attention_mask, attention_mask_words, word_matrix


def ner_extract(text, model, tokenizer, devide='cuda'):
    model.eval()
    ners = {
        "address": []
    }
    
    words = " ".join(underthesea.word_tokenize(text)).split()
    words = [tokenizer.bos_token] + words + [tokenizer.eos_token]
    print(words)
    len_seq = len(words)
    # words_ = words.copy()

    with torch.no_grad():
        sub_word_ids, attention_mask, attention_mask_words, word_matrix = bpe_tokenizer(words, tokenizer, 128, 192)
        sub_word_ids = torch.tensor(sub_word_ids).unsqueeze(dim=0).to(devide)
        attention_mask = torch.tensor(attention_mask).unsqueeze(dim=0).to(devide)
        attention_mask_words = torch.tensor(attention_mask_words).unsqueeze(dim=0).to(devide)
        word_matrix = torch.tensor(word_matrix, dtype=torch.float32).unsqueeze(dim=0).to(devide)
        inputs = (sub_word_ids, word_matrix, attention_mask, attention_mask_words)

        # start = time.time()
        tag_logits, point_logits = model(inputs)
        # print("output model", time.time() - start)
    tag_logits = torch.argmax(tag_logits, dim=-1)[0]
    tag_logits = tag_logits.detach().cpu().numpy()
    print(tag_logits[:len_seq])
    tag_outputs = [constants.ID2TAGS[i] for i in tag_logits]

    # print(point_logits.size())
    point_outputs = torch.argmax(point_logits, dim=-1)[0]
    point_outputs = point_outputs.detach().cpu().numpy()
    for i, point in enumerate(point_outputs):
        if i == point:
            # print(point_logits[0, :, i])
            print(torch.argmax(point_logits[0, i, :]))
            print(point, torch.topk(point_logits[0, i, :], len_seq))
            point_outputs[i] = torch.topk(point_logits[0, i, :], 2).indices[-1].detach().cpu().item()
    
    print(point_outputs[:len_seq], tag_outputs[:len_seq])

    
# ai là chủ tịch nước? bác hồ

    new_tokens = []
    tmp_w, tmp_label = '', None
    for i, (w, lb) in enumerate(zip(words, tag_outputs[1:])):
        if lb in constants.ID2TAGS:
            if not lb.startswith("KEEP|"):
                new_tokens.append(w)
            else:
                num_mask = int(lb.split("|")[-1])
                postfix = " ".join([tokenizer.mask_token]*num_mask)
                new_tokens.append(f"{w} {postfix}")
        else:
            new_tokens.append("")
    print(new_tokens)
    final_str = []
    n = 0
    i = 0
    while (n<len_seq):
        position = point_outputs[i]
        if position == 0:
            break
        final_str.append(new_tokens[position])
        i = position
        n += 1
    # for word, position in zip(new_tokens, point_outputs):
    #     if position != 0 and position < len(new_tokens):
    #         final_str.append(new_tokens[position])
    
    print(" ".join(final_str))
    return " ".join(final_str)


if __name__ == '__main__':
    #'selected_model/best_model.pth'
    #'models/best_model_correct.pth'
    model_path = './models/best_model_correct_tagging.pt'
    pretrained_path = "../shared_data/BDIRoBerta"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model = FelixTagger(model_name=pretrained_path, device=device, num_classes=len(constants.ID2TAGS), is_training=False)
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
    print(f"Model {model_path} loading is done!")

    while True:
        text = input("Enter text: ").strip().lower()#"chỉ đường tao đến số 8 ngõ 114 vũ trọng phụng giúp với"
        if not text:
            exit()
        start_time = time.time()
        ner_extract(text, model, tokenizer, device)
        # print(ners)
        print(round((time.time() - start_time)*1000, 2), "ms")



    # eval_navigation(model, tokenizer, device)