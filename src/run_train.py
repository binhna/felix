import gc
import argparse
import torch
from torch.utils.data import DataLoader, SequentialSampler
from transformers import (
    get_linear_schedule_with_warmup,
    AdamW,
    AutoTokenizer,
    AutoConfig,
)
import torch.nn as nn
import json
import os

import constants
from tagging_model import FelixTagger
from rewrite_dataset import RewriteDataset
from engine import train_fn, validation_fn
from eval import bleu_score


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        default="vinai/phobert-base",
        type=str,
        help="name of pretrained language models",
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="num examples per batch",
    )

    parser.add_argument(
        "--weighted_loss",
        action="store_true",
        help="whether to use weighted to compute loss",
    )

    parser.add_argument(
        "--lr",
        default=3e-5,
        type=float,
        help="learning rate",
    )

    parser.add_argument(
        "--n_epochs",
        default=30,
        type=int,
        help="num epochs required for training",
    )

    parser.add_argument(
        "--seed",
        default=2048,
        type=int,
        help="seed for reproceduce",
    )

    parser.add_argument(
        "--accu_step",
        default=1,
        type=int,
        help="accu_grad_step",
    )

    parser.add_argument(
        "--position_embedding_dim",
        default=64,
        type=int,
        help="position_embedding_dim",
    )

    parser.add_argument(
        "--query_dim",
        default=6,
        type=int,
        help="query_dim",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="continue finetune",
    )

    args = parser.parse_args()
    argparse_dict = vars(args)

    # seed_all(seed_value=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    training_samples, valid_samples = [], []
    with open(constants.TRAINING_FILE, "r") as f_r:
        sentences = f_r.read().split("\n\n")
        for sent in sentences:
            tokens = sent.strip().split("\n")
            training_samples.append(
                [token.split() for token in tokens if len(token.split()) == 3]
            )

    if args.weighted_loss:
        # label_weight_ = Counter(sum([[token[1] for token in sample] for sample in training_samples], []))
        # max_freq = label_weight_.most_common()[0][1]
        # print(label_weight_)
        # label_weight_ = {constants.TAGS2ID[tag]: freq for tag, freq in label_weight_.items()}
        # label_weight = [max_freq]*len(constants.ID2TAGS)
        # for id, weight in label_weight_.items():
        #     label_weight[id] = weight
        # # print(label_weight)
        # label_weight = np.array(label_weight)
        # label_weight = sum(label_weight)/label_weight
        # label_weight = label_weight/sum(label_weight)
        label_weight = [1, 1, 2, 4]
        argparse_dict["label_weight"] = label_weight
        label_weight = torch.FloatTensor(label_weight).to(device)
        # print(label_weight)
        # exit()

    with open("./models/args.json", "w") as f:
        json.dump(argparse_dict, f, ensure_ascii=False)

    with open(constants.VALID_FILE, "r") as f_r:
        sentences = f_r.read().split("\n\n")
        for sent in sentences:
            tokens = sent.strip().split("\n")
            valid_samples.append(
                [token.split() for token in tokens if len(token.split()) == 3]
            )
    # print(Counter(sum([[token[1] for token in sample] for sample in valid_samples], [])))
    # exit()

    print("Number of training samples: ", len(training_samples))
    print("Number of validation samples: ", len(valid_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)

    train_dataset = RewriteDataset(
        training_samples,
        tokenizer=tokenizer,
        max_subword_length=config.max_position_embeddings - 2,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1
    )

    valid_dataset = RewriteDataset(
        valid_samples,
        tokenizer=tokenizer,
        max_subword_length=config.max_position_embeddings - 2,
    )
    valid_sampler = SequentialSampler(valid_dataset)
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=valid_sampler,
        num_workers=1,
    )

    model = FelixTagger(
        model_name=args.model_name,
        device=device,
        max_sub_word_length=config.max_position_embeddings - 2,
        num_classes=len(constants.ID2TAGS),
        query_dim=args.query_dim,
    )
    model.load_pretrained()
    if args.checkpoint:
        model.load_state_dict(
            torch.load(
                os.path.join(args.checkpoint, "best_model_correct_tagging.pt"),
                map_location=torch.device(device),
            )
        )
    # print('The number of parameters of the model: ', count_parameters(model))
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if (not any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if (any(nd in n for nd in no_decay))
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)

    total_steps = len(train_loader) * args.n_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=200, num_training_steps=total_steps
    )

    if args.weighted_loss:
        tag_criterion = nn.CrossEntropyLoss(
            weight=label_weight, ignore_index=constants.TAGS2ID["PAD"]
        )
    else:
        tag_criterion = nn.CrossEntropyLoss()  # ignore_index=constants.TAGS2ID["PAD"])

    pointer_criterion = nn.CrossEntropyLoss()

    max_score = -1
    best_loss = float("inf")
    best_correct = 0
    for epoch in range(args.n_epochs):
        gc.collect()
        print("Training on epoch", epoch + 1)

        total_loss = train_fn(
            dataloader=train_loader,
            model=model,
            tag_criterion=tag_criterion,
            pointer_criterion=pointer_criterion,
            optimizer=optimizer,
            device=device,
            scheduler=scheduler,
            accu_step=args.accu_step,
        )
        print("Training loss: ", total_loss)

        # total_loss, avg_f1 = validation_fn(
        #     valid_loader, model, tag_criterion, pointer_criterion, device
        # )
        model._is_training = False
        bleu, sari = bleu_score("./data/sentence_pair.txt", model, tokenizer, device)
        correct = (bleu + sari) / 2
        model._is_training = True
        # correct += eval_navigation(model, tokenizer, device)

        print("Validation loss", total_loss)
        if total_loss < best_loss:
            print(f"New best model {total_loss}")
            best_loss = total_loss
            torch.save(model.state_dict(), f"models/best_model_loss_tagging.pt")
        if correct > best_correct:
            print(f"********* New best correct {correct} *********")
            best_correct = correct
            torch.save(model.state_dict(), f"models/best_model_correct_tagging.pt")
        torch.save(model.state_dict(), f"models/last_model_tagging.pt")
        print("*" * 100)

    print(f"Best avg f1 is {best_correct}")
