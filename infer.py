import torch
import constants
from transformers import RobertaTokenizer, AutoModelWithLMHead
from tagging_model import FelixTagger
import underthesea
import time
import numpy as np
import itertools
import json
import os
import re


def preprocess_input(sentence):
    sentence = re.sub(r"năm nào\?", "")


def insertion(model, tokenizer, sequence, device):

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
    for i, (w, lb) in enumerate(zip(words, tag_outputs)):
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
    position_added = []
    while (n<len_seq):
        position = point_outputs[i]
        if position == 0 or position in position_added:
            break
        position_added.append(position)
        token = new_tokens[position] if position < len(new_tokens) else ""
        final_str.append(token if token != tokenizer.eos_token else "")
        i = position
        n += 1

    
    print(" ".join(final_str))
    return " ".join(final_str)


if __name__ == '__main__':

    model_path = './models'
    pretrained_path = "../shared_data/BDIRoBerta"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    with open(os.path.join(model_path, "args.json")) as f:
        args = json.load(f)

    model_insertion = AutoModelWithLMHead.from_pretrained(pretrained_path).to(device)
    model = FelixTagger(
        model_name=pretrained_path, 
        device=device, 
        num_classes=len(constants.ID2TAGS), 
        is_training=False,
        position_embedding_dim=args["position_embedding_dim"],
        query_dim=args["query_dim"])
    model.load_state_dict(torch.load(os.path.join(model_path, "best_model_correct_tagging.pt"), map_location=torch.device(device)))
    model.to(device)
    model.eval()
    model_insertion.to(device)
    model_insertion.eval()
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_path)
    print(f"Model {model_path} loading is done!")

    while True:
        text = input("Enter text: ").strip().lower()
        if not text:
            exit()
        start_time = time.time()
        seq_out = ner_extract(text, model, tokenizer, device)
        # print(ners)
        print(round((time.time() - start_time)*1000, 2), "ms")
        if tokenizer.mask_token in seq_out:
            insertion(model_insertion, tokenizer, seq_out, device)
