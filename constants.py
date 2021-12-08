ID2TAGS = ["PAD",
           "KEEP",
           "DELETE",
           "KEEP|1",
        #    "KEEP|2",
        #    "KEEP|3",
        #    "KEEP|4",
        #    "KEEP|5",
        #    "KEEP|6",
        #    "KEEP|7",
        #    "KEEP|8",
        #    "KEEP|9",
        #    "KEEP|10"
           ]

TAGS2ID = {
    "PAD": 0,
    "KEEP": 1,
    "DELETE": 2,
    "KEEP|1": 3,
    # "KEEP|2": 4,
    # "KEEP|3": 5,
    # "KEEP|4": 6,
    # "KEEP|5": 7,
    # "KEEP|6": 8,
    # "KEEP|7": 9,
    # "KEEP|8": 10,
    # "KEEP|9": 11,
    # "KEEP|10": 12
}

TRAINING_FILE = "./data/train.conll"
VALID_FILE = "./data/valid.conll"