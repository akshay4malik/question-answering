

from bentoml import api, artifacts, env, ver, BentoService
from bentoml.adapters import StringInput, DefaultOutput
from bentoml.saved_bundle.config import SavedBundleConfig
from bentoml.frameworks.transformers import TransformersModelArtifact
from bentoml.frameworks.onnx import OnnxModelArtifact

from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from transformers import ElectraTokenizerFast
from transformers.convert_graph_to_onnx import convert
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.utils import shuffle
# from simpletransformers.question_answering import QuestionAnsweringModel, QuestionAnsweringArgs
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
import pyarrow as pa
from datasets.dataset_dict import Dataset, DatasetDict

from collections import defaultdict
import pathlib
import tempfile
import numpy as np
import pandas as pd
import re
from typing import List, Union, Dict
import os
import uuid
import warnings
import shutil
from tqdm import tqdm

os.environ["WANDB_DISABLED"] = "true"


# @ver(major=1, minor=0)  # major and minor version of this service
# @env(infer_pip_packages=True)    # let Bento infer required pip packages, this needs to be off is re-saving the Bento from inside the class itself!
@artifacts([TransformersModelArtifact("qa_model"), OnnxModelArtifact('onnx_model',backend='onnxruntime')])  # this is the artifact of our classification pipeline
class QAModel(BentoService):
    def __init__(self):
        super().__init__()
        self.tokenizer_fast = AutoTokenizer.from_pretrained(
            os.path.join(pathlib.Path(__file__).parent.resolve(), "artifacts", "qa_model"), use_fast=True)

    def get_answer_context(self,
                           text,
                           question,
                           ans_start,
                           ans_end,
                           buffer_before: int = 200,
                           buffer_after: int = 200):
        """
        This function returns a context window for the answer in a document text.

        Context window's larger than the maximum size supported by the model will be truncated.

        Args:
            text: text contained in the document
            question: question for extracting answer
            ans_start: answer start index
            ans_end: answer end index 
            buffer_before (int): How many characters before the start of the answer to prepend
            buffer_after (int): How many charactres after the end of the answer to prepend
        Returns:
            txt: answer context
            start: answer start indices
            end: answer end indices
        """
        tokenizer = self.artifacts.qa_model.get("tokenizer")
        question = question
        start, end = ans_start, ans_end

        count = 0
        start, end = max(0, start - buffer_before), min(len(text), end + buffer_after)
        txt = text[start:end]
        while len(tokenizer(txt).input_ids) + len(tokenizer(question + ' [SEP] ').input_ids) > 510:
            # subsequently increase the number of words and take care not to cross the zero or max len limit
            start, end = max(0, start + 20), min(len(text), end - 20)
            txt = text[start:end]
            count += 1
            if count > 100:
                break
        return txt, start, end

    def prepare_dataset_features(self, examples):
        """
        This function will tokenize the prepared dataset
        Args:
            example: a dict of data containing the following columns:
                    1. question
                    2. answers
                    3. context
                    4. is_impossible
                    5. id
                    6. title : This has been set to None
        Returns:
            return tokenized examples

        """
        # Define tokenizer
        tokenizer = ElectraTokenizerFast.from_pretrained(os.path.join(os.path.dirname(__file__), "artifacts/qa_model"))
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=512,
            stride=128,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    #@api(api_name="evaluate")
    def evaluate(self,
                 evaluation_data: List[Dict],
                 buffer_before: int = 200,
                 buffer_after: int = 200
                 ):
        """
        This function evaluate the question answering model based two primary metrics: exact match between the predicted and ground-truth answers,
        and the intersection-over-union of the predicted and ground-truth character spans.

        Args:
            evaluation_data (List[Dict]): List of dictionary containing data for evaluation
        Returns:
            evaluation metrics: the evaluation scores
            predictions: dictionary containing the actual answers, predicted answers, and their start and end indices
        """
        ious = []
        exacts = []
        tn = []
        n_pos = 0
        n_neg = 0
        metrics = {}
        answer_container = defaultdict(list)
        for data in tqdm(evaluation_data):
            #target_elements = [i for i in ex.elements if i.element_type in target_element_types]
            for ex in data['qas']:
                with warnings.catch_warnings():  # suppress annoying warning message
                    warnings.filterwarnings("ignore",
                                            message="The `max_len` attribute has been deprecated and will be removed in a future version, use `model_max_length` instead.")
                    context, ctx_start, ctx_end = self.get_answer_context(data['text'],
                                                                          ex['question'],
                                                                          ex['ans_start'],
                                                                          ex['ans_end'],
                                                                          buffer_before=buffer_before,
                                                                          buffer_after=buffer_after)
                    pred = self.single_prediction(ex['question'], context)[0]
                    pred_start = ctx_start + pred['start']
                    pred_end = pred_start + len(pred['answer'])
                if ex['ans_end'] != 0:  # indicates that there is an answer to the question
                    n_pos += 1
                    iou = self.text_iou(
                        ex['ans_start'],
                        ex['ans_end'],
                        pred_start,
                        pred_end
                    )
                    ious.append(iou)
                    if pred['answer'] == ex['ans_text']:
                        exacts.append(1)
                    else:
                        exacts.append(0)
                    answer_dict = {
                        "actual_answer": ex['ans_text'],
                        "actual_answer_start": ex['ans_start'],
                        "actual_answer_end": ex['ans_end'],
                        "predicted_answer": pred['answer'],
                        "predicted_answer_start": pred_start,
                        "predicted_answer_end": pred_end,
                        "intersection_over_union": iou
                    }
                    answer_container["answers"].append(answer_dict)
                else:
                    n_neg += 1
                    if pred['answer'] == "":
                        tn.append(1)
        try:
            val_true_neg_rate = str(np.round(len(tn) / n_neg, 2))
        except:
            val_true_neg_rate = 0

        metrics = {
            "validation_exact_match": str(np.round(np.mean(exacts), 2)),
            "validation_IoU": str(np.round(np.mean(ious), 2)),
            "validation_false_negative_rate": str(1 - np.round(len(ious) / n_pos, 2)),
            "validation_true_negative_rate": val_true_neg_rate,

        }
        return metrics, answer_container

  #  @api(api_name='string', input=StringInput())
    def single_prediction(
            self,
            question: str,
            context: str,
            max_answer_len: int = 16,
            topk: int = 1,
            handle_impossible_answer: bool = False
    ):
        """
        Use the model to perform a single prediction on the provided question and context pair.  Uses an ONNX version of the QA
        model, with custom pre/post processing of the input and output. Reproduces (within rounding error) the output of the
        Transformers question answering pipeline implementation (https://huggingface.co/transformers/task_summary.html#question-answering).

        Args:
            question (str): The question text
            context (str): The context text
            max_answer_len (int): The maximum length (in wordpiece tokens) that a predicted answer can be
            topk (int): How many predictions to return, in order of descending score
            handle_impossible_answer (bool): Whether the model should always return a blank answer, if the score is sufficiently high

        Returns:
            dict: A dictionary with the QA model's predictions
        """

        # Tokenize inputs
        inputs_tok = self.tokenizer_fast(question, context, add_special_tokens=True, truncation=True, max_length=512,
                                         return_tensors="pt")  # need to use "fast" rust-based tokenizer

        # Get logits from input text
        pred_onnx = self.artifacts.onnx_model.run(None, {
            "input_ids": inputs_tok['input_ids'].tolist(),
            "attention_mask": inputs_tok['attention_mask'].tolist(),
            "token_type_ids": inputs_tok['token_type_ids'].tolist()
        })
        ans_start_logits, ans_end_logits = pred_onnx[0], pred_onnx[1]

        # Make sure non-context indexes in the tensor cannot contribute to the softmax
        mask = np.zeros(ans_start_logits.shape).flatten()
        mask[1:np.argmax(inputs_tok[
                             'input_ids'].flatten() == 102) + 1] = 1  # find the location of the [SEP] token to mask question, but don't mask [CLS] token at index 0 in the sequence
        start_ = np.where(mask, -10000.0, ans_start_logits)
        end_ = np.where(mask, -10000.0, ans_end_logits)
        # Calculate the undesired tokesn
        undesired_tokens = np.abs(mask - 1)
        undesired_tokens_mask = undesired_tokens == 0.0
        # Normalize logits and spans with softmax to retrieve the answer
        start_ = np.exp(start_ - np.log(np.sum(np.exp(start_), axis=-1, keepdims=True)))
        end_ = np.exp(end_ - np.log(np.sum(np.exp(end_), axis=-1, keepdims=True)))

        #undesired_tokens = 0.0

        # Check for blank predictions (e.g., unanswerable)
        min_null_score = 1000000  # large and positive
        if handle_impossible_answer:
            min_null_score = min(min_null_score, start_[0][0] * end_[0][
                0])  # by convention QA models provide unanswerable predictions with the [CLS] token at index 0
        start_[0][0] = end_[0][0] = 0.0  # Mask CLS token (after checking for unanswerable prediction)

        # Get final spans and scores
        nlp = pipeline("question-answering", model=self.artifacts.qa_model.get("model"),
                       tokenizer=self.artifacts.qa_model.get("tokenizer"))
        starts, ends, scores = nlp.decode(start_, end_, max_answer_len=max_answer_len, topk=topk,
                                          undesired_tokens=undesired_tokens)

        # Post-process final predictions
        answers = []
        answers += [
            {
                "score": score.item(),
                "start": inputs_tok.token_to_chars(s)[0],
                "end": inputs_tok.token_to_chars(e)[1],
                "answer": context[inputs_tok.token_to_chars(s)[0]:inputs_tok.token_to_chars(e)[1]],
            }
            for s, e, score in zip(starts, ends, scores)
        ]

        if handle_impossible_answer:
            answers.append({"score": min_null_score, "start": 0, "end": 0, "answer": ""})

        answers = sorted(answers, key=lambda x: x["score"], reverse=True)[:topk]
        return answers

    #@api(api_name='string', input=StringInput())
    def single_prediction_transformers_pipeline(
            self,
            question: str,
            context: str,
            handle_impossible_answer: bool = False
    ):
        """
        Use the model to perform a single prediction on the provided question and context pair.  Uses the HuggingFace question-answering
        pipeline implementation (https://huggingface.co/transformers/task_summary.html#question-answering).

        Args:
            question (str): The question text
            context (str): The context text
            handle_impossible_answer (bool): Whether the model should always return an answer or not, regardless of confidence

        Returns:
            dict: A dictionary with the QA model's predictions
        """
        model = self.artifacts.qa_model.get("model")
        tokenizer = self.artifacts.qa_model.get("tokenizer")
        nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)
        with warnings.catch_warnings():  # suppress annoying warning message
            warnings.filterwarnings("ignore",
                                    message="The `max_len` attribute has been deprecated and will be removed in a future version, use `model_max_length` instead.")
            pred = nlp(context=context, question=question, handle_impossible_answer=handle_impossible_answer)
            return pred


   # @api(api_name="train_model")
    def train(
            self,
            dataset: List[Dict],
            output_path: str,
            context_size: int = 200,
            generate_negative_examples: bool = True,
            negative_example_method: str = "random",
            validation_split: Union[float, int] = 0.1,
            random_state: int = 42,
            num_train_epochs:int = 2,
            output_version: str = "finetuned_model",
            use_cuda: bool = False,
            **kwargs
    ) -> bool:
        """
        Fine-tunes a question-answering model based on Google's Electra transformer architecture
        (https://huggingface.co/mrm8488/electra-small-finetuned-squadv2) that has been trained on
        Standfords SQuADv2 dataset using the question-answer pairs present in the provided Dataset.

        The fine-tuned model will be saved as a new Bento at the path specified.

        Args:
            dataset (List[Dict]): The input Dataset contain the list of Dictionaries with appropriate training data
            output_path (str): Where to save the fine-tuned model as a new Bento package. If set to be the same directory
                               as the starting Bento, after training only the artifact files and `bentoml.yml` metadata
                               files will be updated.
            context_size (int): The approximate length of the total context surrounding each answer to be
                                used for training. The answer will always be centered in this context. (default: 200)
            generate_negative_examples (bool): Whether to generate negative examples where the correct prediction
                                               is an empty string. (default: true)
            negative_example_method (str): How to generate the negative examples (default: random)
                                               random: The negative contexts will be chosen at random from
                                                   spans of text that do not overlap with the answer of the
                                                   associated question
                                               question_word_overlap: The negative contexts will be chosen
                                                   from spans of text that contain the most overlap with
                                                   non-stopwords (english) parts of the question. This produces
                                                   more challenging negative examples for the model to learn,
                                                   and can improve performanc in cases where the model needs to
                                                   differentiate between semantically similar, but unrelated, contexts.
                                                false_positive_mining: The negative contexts will be chosen from false-positive
                                                   predictions in randomly selected spans that don't overlap with the answer
                                                   in the labeled element
            validation_split (Union[float,list]):  a fraction (0-1) for random splitting train and validation split
                                                  
                                                  
            random_state (int): The random seed used during data preparation. Ensures some reproducability between
                          different runs. (default: None)
            output_version (str): The version written in the metadata for the newly created Bento. (default: "finetuned_model")
                                The user is encouraged to follow semantic versioning principles.
            use_cuda (str): Whether to use a CUDA-enabled GPU for model-training (default: False)
            kwargs: Any other keyword arguments to pass to the QuestionAnsweringArgs object from the
                    simpletransformers library (https://simpletransformers.ai/docs/qa-model/#configuring-a-questionansweringmodel,
                    https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model), which
                    is used to train the model.

        Returns:
            bool: True after training the model
        """

        ## Set random state
        self.random_state = random_state
        np.random.seed(random_state)

        ## Prepare data and generate features
        # Get positive examples
        train_positive_examples = []
        val_positive_examples = []
        for data in tqdm(dataset, desc="Collecting positive examples"):
            var = np.random.random()

            for ex in data['qas']:
                example = self.make_squad_example(data['text'], ex['ans_start'],
                                ex['ans_end'],ex['question'], 
                                context_size // 2, context_size // 2)
                if type(validation_split) in [int, float] and var >= validation_split:
                    train_positive_examples.append(example)
                else:
                    val_positive_examples.append(example)

        # Get negative examples
        train_negative_examples = []
        val_negative_examples = []
        for data in tqdm(dataset, desc="Collecting negative examples"):
            var = np.random.random()
            for ex in data['qas']:
                example = self.make_squad_negative_example(data['text'], ex['ans_start'], 
                                                            ex['ans_end'],
                                                            ex['question'], context_size,
                                                            negative_example_method)
                if type(validation_split) in [int, float] and var >= validation_split:
                    train_negative_examples.append(example)
                else:
                    val_negative_examples.append(example)

        # Define train/validation splits
        train_data = train_positive_examples + train_negative_examples
        val_data = val_positive_examples + val_negative_examples
        train_data = shuffle(train_data, random_state=random_state)

        # Creating the DataFrame from the dataset lists
        train_data_df = pd.DataFrame(train_data)
        val_data_df = pd.DataFrame(val_data)

        # Creating a pyarrow table for Creating the Dataset instance of Transformers library
        train_arrow_table = pa.Table.from_pandas(train_data_df)
        val_arrow_table = pa.Table.from_pandas(val_data_df)

        # Creating the dataset from pyarrow tables
        train_dataset = Dataset(train_arrow_table)
        val_dataset = Dataset(val_arrow_table)

        # Dataset features
        train_dataset_dict = DatasetDict({"train": train_dataset, 'val': val_dataset})
        tokenized_datasets = train_dataset_dict.map(self.prepare_dataset_features, batched=True)

        ## Prepare for training via simpletransformers
        # Save the current model to temporary storage for simpletransformers to load
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.artifacts.qa_model.get("model").save_pretrained(os.path.join(tmp_dir, "current_model"))
            self.artifacts.qa_model.get("tokenizer").save_pretrained(os.path.join(tmp_dir, "current_model"))

            # Configure the model with sensible defaults for important parameters

            # Allow user to set/override any other arguments
            #             for key in kwargs.keys():
            #                 if key in model_args_dict.keys():
            #                     model_args.__setattr__(key, kwargs[key])

            model_args = TrainingArguments(
                output_dir=os.path.join(tmp_dir, "new_model"),
                overwrite_output_dir=True,
                per_device_train_batch_size=8,
                per_device_eval_batch_size=8,
                num_train_epochs=num_train_epochs,
                save_steps=500,
                seed=random_state,
                resume_from_checkpoint=False,
                no_cuda=True
            )

            # Transformers model
            model = AutoModelForQuestionAnswering.from_pretrained(os.path.join(tmp_dir, "current_model"))
            tokenizer = self.artifacts.qa_model.get("tokenizer")
            trainer = Trainer(
                model=model,
                args=model_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['val'],
                tokenizer=tokenizer,
            )

            trainer.train()

            trainer.save_model(os.path.join(tmp_dir, "final_model"))

            # Evaluate the model
            print("Running model Evaluation...")
            ious = []
            exacts = []
            tn = []
            n_pos = 0
            n_neg = 0
            metrics = {}

            if len(val_data) > 0:
                tokenizer = self.artifacts.qa_model.get("tokenizer")
                model = AutoModelForQuestionAnswering.from_pretrained(os.path.join(tmp_dir, "final_model"))
                nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

                for ex in tqdm(val_data):
                    with warnings.catch_warnings():  # suppress annoying warning message
                        warnings.filterwarnings("ignore",
                                                message="The `max_len` attribute has been deprecated and will be removed in a future version, use `model_max_length` instead.")
                        pred = nlp(
                            context=ex["context"],
                            question=ex['question'],
                            handle_impossible_answer=True if generate_negative_examples else False
                        )

                    if ex['answers']:
                        if len(ex['answers']['answer_start']) > 0:
                            n_pos += 1
                            print (ex['answers'])
                            iou = self.text_iou(
                                ex['answers']['answer_start'][0],
                                ex['answers']['answer_start'][0] + len(ex['answers']['text'][0]),
                                pred['start'],
                                pred['end']
                            )
                            ious.append(iou)
                            # print(ex['qas'][0]['question'], "###", ex['qas'][0]['answers'][0]['text'], "###", ex['context'], pred, iou, "\n#############\n")
                            if pred['answer'] == ex['answers']['text'][0]:
                                exacts.append(1)
                            else:
                                exacts.append(0)
                    else:
                        n_neg += 1
                        if pred['answer'] == "":
                            tn.append(1)

                metrics = {
                    "validation_exact_match": str(np.round(np.mean(exacts), 2)),
                    "validation_IoU": str(np.round(np.mean(ious), 2)),
                    "validation_false_negative_rate": str(1 - np.round(len(ious) / n_pos, 2)),
                    #"validation_true_negative_rate": str(np.round(len(tn) / n_neg, 2)),
                    #                     "training_loss": results[1].get("train_loss", None),
                    #                     "validation_loss": results[1].get("eval_loss", None)
                }
                print(metrics)

            ## Update Bento artifacts and model metadata

            # Export ONNX version of model, and pack new Bento
            with tempfile.TemporaryDirectory() as tmp_dir_onnx:
                # update the artifacts in the Bento
                if len(val_data) > 0:
                    model = AutoModelForQuestionAnswering.from_pretrained(os.path.join(tmp_dir, "final_model"))
                    convert(framework="pt", model=os.path.join(tmp_dir, "final_model"),
                            output=pathlib.Path(os.path.join(tmp_dir_onnx, "onnx_model.onnx")), opset=11,
                            pipeline_name="question-answering")
                else:
                    model = AutoModelForQuestionAnswering.from_pretrained(
                        os.path.join(tmp_dir, "final_model"))  # best model is only saved when validation data is used
                    convert(framework="pt", model=os.path.join(tmp_dir, "final_model"),
                            output=pathlib.Path(os.path.join(tmp_dir_onnx, "onnx_model.onnx")), opset=11,
                            pipeline_name="question-answering")

                tokenizer = self.artifacts.qa_model.get("tokenizer")
                artifact = {"model": model, "tokenizer": tokenizer}
                self.pack("onnx_model", os.path.join(tmp_dir_onnx, "onnx_model.onnx"))
                self.pack("qa_model", artifact, metadata=metrics)

                # increment versioning
                self.set_version(output_version)

                module_base_path = os.path.join(output_path, self.name)
                if os.path.exists(module_base_path):
                    # Save artifacts
                    shutil.rmtree(os.path.join(module_base_path, "artifacts"), ignore_errors=True)
                    self.artifacts.save(module_base_path)

                    # Update metadata files
                    config = SavedBundleConfig(self)
                    config.write_to_path(output_path)
                    config.write_to_path(module_base_path)

                else:
                    # save the model (to disk, for now, in a new directory)
                    self.save_to_dir(output_path)

        return True

    def text_iou(self, true_start, true_end, pred_start, pred_end):
        """Get the index range Intersection-over-Union (IOU) of the true and predicted answer"""
        if true_start == true_end == pred_start == pred_end: return 1
        return max(0, min(pred_end, true_end) - max(pred_start, true_start)) / \
               (max(pred_end, true_end) - min(pred_start, true_start))

    def make_squad_negative_example(self, text, ans_start,ans_end, question, context_length=750, method="random"):
        """
        Make a negative SQuAD format QA example via several different methods:

        random: Random choose a context of the specified length from the document that does not
        overlap with the answer in the labeled element

        question_word_overlap: use simple white-space tokenization to select contexts with the highest non-stopword
        overlap with the question

        false_positive_mining: search for false positive predictions in randomly selected spans that don't overlap
        with the answer in the labeled element
        """

        if method == "false_positive_mining":
            # Initialize model to test for false-positives
            model = self.artifacts.qa_model.get("model")
            tokenizer = self.artifacts.qa_model.get("tokenizer")
            nlp = pipeline("question-answering", model=model, tokenizer=tokenizer)

            # Get random examples
            start = np.random.randint(0, len(text) - context_length) if len(text) - context_length > 0 else 0
            end = start + context_length
            cnt = 0
            n_fp_candidates = 0
            n_attempts = 0
            flag = True
            fps = []
            while n_fp_candidates < 20:  # arbitrary amount
                start = np.random.randint(0, len(text) - context_length) if len(
                    text) - context_length > 0 else 0
                end = start + context_length
                if self.text_iou(ans_start, ans_end, start,
                                 end) != 0:
                    start = np.random.randint(0, len(text) - context_length) if len(
                        text) - context_length > 0 else 0
                    end = start + context_length
                    cnt += 1
                    if cnt > 100:
                        print(f"Warning! Could not find a negative example for the element {el}")
                        break
                    continue

                pred = self.single_prediction(context=text[start:end], question=question,
                                              handle_impossible_answer=True)[0]
                if pred['answer'] != "":
                    fps.append((start, end, pred['score']))
                    n_fp_candidates += 1
                else:
                    n_attempts += 1

                if n_attempts > 20:
                    break

            if not fps:
                print(
                    f"Warning! Couldn't find a false-positive example for element {question}! Returning random example.")
            else:
                fps = sorted(fps, key=lambda x: x[-1], reverse=True)
                start, end = fps[0][0], fps[0][1]

        if method == "random":
            start = np.random.randint(0, len(text) - context_length) if len(text) - context_length > 0 else 0
            end = start + context_length
            cnt = 0
            while self.text_iou(ans_start, ans_end, start,
                                end) != 0:
                start = np.random.randint(0, len(text) - context_length) if len(
                    text) - context_length > 0 else 0
                end = start + context_length
                cnt += 1
                if cnt > 100:
                    print(f"Warning! Could not find a negative example for the element {question}")
                    start = 0
                    end = ans_start

        if method == "question_word_overlap":
            q_tokens = re.sub(r'[^\w\s]', '', question).lower().split()
            q_tokens = set(q_tokens) - ENGLISH_STOP_WORDS
            window_ndcs = [(i, i + context_length) for i in range(0, len(text), context_length // 2)]
            windows_tokens = [re.sub(r'[^\w\s]', '', text[i:i + context_length]).lower().split() \
                              for i in range(0, len(text), context_length // 2)]
            windows_tokens = [set(i) - ENGLISH_STOP_WORDS for i in windows_tokens]
            intersections = sorted(
                [(len(set(q_tokens).intersection(set(i))), j) for i, j in zip(windows_tokens, window_ndcs)],
                key=lambda x: x[0], reverse=True)

            ndx = 0
            while self.text_iou(ans_start, ans_end,
                                intersections[ndx][1][0], intersections[ndx][1][1]) != 0:
                ndx += 1

            start = intersections[ndx][1][0]
            end = intersections[ndx][1][1]

        ctx = text[start:end]
        id_ = uuid.uuid4().hex
        is_impossible = True
        question = question

        squad_json = {
            "answers": {
                "answer_start": [],
                "text": []
            },
            "context": ctx,
            "id": id_,
            "question": question,
            "title": "None",
            "is_impossible": is_impossible
        }
        return squad_json

    def make_squad_example(self, text, ans_start, ans_end,
                     question, buffer_before=100, buffer_after=100):
        ctx = text[max(0, ans_start - buffer_before):min(ans_end + buffer_after,len(text))]
        id_ = uuid.uuid4().hex
        is_impossible = False
        answer_text = text[ans_start:ans_end]
        answer_start = buffer_before if ans_start - buffer_before > 0 else ans_start

        squad_json = {
            "answers": {
                "answer_start": [answer_start],
                "text": [answer_text]
            },
            "context": ctx,
            "id": id_,
            "question": question,
            "title": "None",
            "is_impossible": is_impossible
        }
        return squad_json
