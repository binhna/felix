# Felix

This is a re-implementation of the felix repo from google in pytorch, I use the model for rewrite the answer from a question and a short answer.
For example:

```
question: Who is the president of the USA?

answer: Joe Biden

expected output: Joe Biden is the president of the USA
```

This project is still in process.

## Usage Instructions
1. Install requirements dependencies
```
pip install -r requirements
```
2. Create CONLL training data from excel file by running `create_tags.py`, the expected data format from excel file is at [1]
3. Run the training code `python run_train.py --model_name ../shared_resource/roberta_example --batch_size 64 --n_epochs 20 ...`




[1]: data format from excel file contains `question`, `short answer`, `rewrite1`. Where `question` is the user question to the QA system, `short answer` is the output answer from the QA system, it could be from KBQA or MRC or what ever. `rewrite1` is the expected completed answer that we wish the TTS to speak out to the user.
