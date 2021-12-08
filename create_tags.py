import collections
import frozendict
from constants import TAGS2ID, ID2TAGS
import underthesea
import pandas as pd
from tqdm import tqdm

class Point(object):
    """Point that corresponds to a token edit operation.

    Attributes:
      point_index: The index of the next token in the sequence.
      added_phrase: A phrase that's inserted before the next token (can be empty).
    """

    def __init__(self, point_index, added_phrase=''):
        """Constructs a Point object .

        Args:
          point_index: The index the of the next token in the sequence.
          added_phrase: A phrase that's inserted before the next token.

        Raises:
          ValueError: If point_index is not an Integer.
        """

        self.added_phrase = added_phrase

        try:
            self.point_index = int(point_index)
        except ValueError:
            raise ValueError(
                'point_index should be an Integer, not {}'.format(point_index))

    def __str__(self):
        return '{}|{}'.format(self.point_index, self.added_phrase)


def _compute_points(source_tokens, target_tokens):
    """Computes points needed for converting the source into the target.

    Args:
      source_tokens: List of source tokens.
      target_tokens: List of target tokens.

    Returns:
      List of pointing.Pointing objects. If the source couldn't be converted
      into the target via pointing, returns an empty list.
    """
    source_tokens_indexes = collections.defaultdict(set)
    for i, source_token in enumerate(source_tokens):
        source_tokens_indexes[source_token].add(i)

    target_points = {}
    last = 0
    token_buffer = ""

    def find_nearest(indexes, index):
        # In the case that two indexes are equally far apart
        # the lowest index is returned.
        return min(indexes, key=lambda x: abs(x - index))

    for target_token in target_tokens[1:]:
        # Is the target token in the source tokens and is buffer in the vocabulary
        # " ##" converts word pieces into words
        if (source_tokens_indexes[target_token] and
            (not token_buffer or not _phrase_vocabulary or
             token_buffer in _phrase_vocabulary)):
            # Maximum length expected of source_tokens_indexes[target_token] is 512,
            # median length is 1.
            src_indx = find_nearest(source_tokens_indexes[target_token], last)
            # We can only point to a token once.
            source_tokens_indexes[target_token].remove(src_indx)
            target_points[last] = Point(src_indx, token_buffer)
            last = src_indx
            token_buffer = ""

        else:
            token_buffer = (token_buffer + " " + target_token).strip()

    # Buffer needs to be empty at the end.
    if token_buffer.strip():
        return []

    points = []
    for i in range(len(source_tokens)):
        # If a source token is not pointed to,
        # then it should point to the start of the sequence.
        if i not in target_points:
            points.append(Point(0))
        else:
            points.append(target_points[i])

    # print([str(p) for p in points])
    return points


if __name__ == "__main__":
    _use_open_vocab = True
    _phrase_vocabulary = set()
    label_map = TAGS2ID

    _inverse_label_map = ID2TAGS
    
    df = pd.read_excel("./data/RewriteAnnotation.xlsx", sheet_name="Rewrite", engine='openpyxl')
    df.fillna(False, inplace=True)
    
    text_samples = []
    data = []
    fail = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if row["rewrite1"] and row["question"] and row["short answer"]:
            q = row.question.lower().strip()
            q += "?" if not q.endswith("?") else ""
            a = str(row['short answer']).lower().strip()
            rewrite = row.rewrite1.lower().strip()
            src = "<s> " + " ".join(underthesea.word_tokenize(q)) + " " + " ".join(underthesea.word_tokenize(a)) + " </s>"
            tgt = "<s> " + " ".join(underthesea.word_tokenize(rewrite)) + " </s>"
    #         text_samples.append((src, tgt))

    # for src, tgt in tqdm(text_samples):
            points = _compute_points(src.split(),
                                    tgt.split())
            # points = _compute_points(['[CLS]', 'the', 'big', 'very', 'loud', 'cat', '[SEP]'], ['[CLS]', 'the', 'very', 'old', 'big', 'cat', '[SEP]'])
            labels = [t.added_phrase for t in points]
            point_indexes = [t.point_index for t in points]
            point_indexes_set = set(point_indexes)

            try:
                new_labels = []
                for i, added_phrase in enumerate(labels):
                    if i not in point_indexes_set:
                        new_labels.append(label_map['DELETE'])
                    elif not added_phrase:
                        new_labels.append(label_map['KEEP'])
                    else:
                        if _use_open_vocab:
                            new_labels.append(label_map['KEEP|' +
                                                            str(len(added_phrase.split()))])
                        else:
                            new_labels.append(
                                label_map['KEEP|' + str(added_phrase)])
                    labels = new_labels
                    # print(labels)
            except Exception as e:
                # added_phrase is not in label_map.
                # print(e)
                continue

            if not labels:
                continue
                # print("failed creating labels")

            label_tokens = [
                _inverse_label_map[label_id] if label_id < len(_inverse_label_map) else "PAD"
                for label_id in labels
            ]
            if len(label_tokens) == len(points) and len(label_tokens) == len(src.split()):
                data.append((src.split(), label_tokens, [p.point_index for p in points]))
            else:
                fail.append([src.split(), label_tokens, len(src.split()), len(label_tokens)])
    print(len(data))
    print(len(fail))
    num_train = int(len(data)*0.9)
    train = data[:num_train]
    valid = data[num_train:]
    
    train_count = 0
    valid_count = 0
    with open("./data/train.conll", "w") as f:
        for words, tags, points in train:
            if set(tags).intersection(set(["KEEP|2", "KEEP|3", "KEEP|4", "KEEP|5", "KEEP|6", "KEEP|7", "KEEP|8", "KEEP|9", "KEEP|10"])):
                continue
            train_count += 1
            for w, t, p in zip(words, tags, points):
                f.write(f"{w} {t} {p}\n")
            f.write("\n")
    
    with open("./data/valid.conll", "w") as f:
        for words, tags, points in valid:
            if set(tags).intersection(set(["KEEP|2", "KEEP|3", "KEEP|4", "KEEP|5", "KEEP|6", "KEEP|7", "KEEP|8", "KEEP|9", "KEEP|10"])):
                continue
            valid_count += 1
            for w, t, p in zip(words, tags, points):
                f.write(f"{w} {t} {p}\n")
            f.write("\n")
            # print(src)
            # print(tgt)
            # print([p.point_index for p in points])
            # print(label_tokens)
            # exit()
    print(f"training sample: {train_count}\nvalid sample: {valid_count}")
