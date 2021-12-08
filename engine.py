import torch
from tqdm.auto import tqdm
from seqeval.metrics import f1_score
import constants


def tagging_evaluate(y_true_tag, y_pred_tag, y_true_point, y_pred_point):
    pres, trues = [], []
    for sent_true, sent_out in zip(y_true_tag, y_pred_tag):
        tmp = [constants.ID2TAGS[i] for i in sent_true if i != -100]
        trues.append(tmp)
        pres.append([constants.ID2TAGS[i] for i in sent_out[:len(tmp)]])
    tag_f1 = f1_score(trues, pres)
    print("F1 TAGGING:", tag_f1)

    pres, trues = [], []
    for sent_true, sent_out in zip(y_true_point, y_pred_point):
        tmp = [str(i) for i in sent_true if i != -100]
        trues.append(tmp)
        pres.append([str(i) for i in sent_out[:len(tmp)]])
    point_f1 = f1_score(trues, pres)
    print("F1 POINTER:", point_f1)

    return (tag_f1+point_f1)/2


def train_fn(
    dataloader, model, tag_criterion, pointer_criterion, optimizer, scheduler, device="cuda", accu_step=1
):
    model.train()
    total_loss = 0

    pbar = tqdm(dataloader, total=len(dataloader), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}')
    for i, (batch) in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        input_mask = batch["input_mask"].to(device)
        input_mask_words = batch["input_mask_words"].to(device)
        word_matrix = batch["word_matrix"].to(device)
        tag_labels = batch["tag_labels"].to(device)
        point_labels = batch["point_labels"].to(device)
        
        inputs = (input_ids, tag_labels, point_labels, word_matrix, input_mask, input_mask_words)

        tag_logits, point_logits = model(inputs)

        # Loss calculate

        tag_logits = torch.transpose(tag_logits, 2, 1)  # loss
        tag_loss = tag_criterion(tag_logits, tag_labels)

        point_logits = torch.transpose(point_logits, 2, 1)  # loss
        point_loss = pointer_criterion(point_logits, point_labels)

        loss = tag_loss + point_loss

        # Loss backward
        loss.backward()
        if (i + 1) % accu_step == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()

        total_loss += loss.item()

    total_loss /= len(dataloader)

    return total_loss



def validation_fn(dataloader, model, tag_criterion, pointer_criterion, device="cuda"):
    model.eval()
    total_loss = 0
    tag_pres, tag_golds, point_pres, point_golds = [], [], [], []

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), bar_format='{desc:<5.5}{percentage:3.0f}%|{bar:30}{r_bar}')
        for i, (batch) in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            input_mask_words = batch["input_mask_words"].to(device)
            word_matrix = batch["word_matrix"].to(device)
            tag_labels = batch["tag_labels"].to(device)
            point_labels = batch["point_labels"].to(device)
            inputs = (input_ids, tag_labels, point_labels, word_matrix, input_mask, input_mask_words)

            tag_logits, point_logits = model(inputs)

            # Loss calculate

            tag_logits = torch.transpose(tag_logits, 2, 1)  # loss
            tag_loss = tag_criterion(tag_logits, tag_labels)

            point_logits = torch.transpose(point_logits, 2, 1)  # loss
            point_loss = pointer_criterion(point_logits, point_labels)

            loss = tag_loss + point_loss

            total_loss += loss.item()

            # Evaluate
            tag_logits = torch.transpose(tag_logits, 2, 1)
            tag_outputs = torch.argmax(tag_logits, dim=-1)
            tag_outputs = tag_outputs.detach().cpu().numpy()
            
            tag_labels = tag_labels.detach().cpu().numpy()
            tag_pres.extend(tag_outputs)
            tag_golds.extend(tag_labels)

            point_logits = torch.transpose(point_logits, 2, 1)
            point_outputs = torch.argmax(point_logits, dim=-1)
            point_outputs = point_outputs.detach().cpu().numpy()
            
            point_labels = point_labels.detach().cpu().numpy()
            point_pres.extend(point_outputs)
            point_golds.extend(point_labels)

        entity_f1 = tagging_evaluate(tag_golds, tag_pres, point_golds, point_pres)

        print("F1 score: ", entity_f1)

        total_loss /= len(dataloader)

        return total_loss, entity_f1
