# Imports
import sys
import bentoml
from ignite_lume.lume import Lume, LumeDataset, LumeElement
import os

# Load Bento
bento_service = bentoml.load("./")

# Define tests
def test_evaluate():
    lume_dir = os.path.join(os.path.dirname(__file__), "resources")
    lume_dataset = LumeDataset.load_from_directory(lume_dir)
    element_type_for_val = ["__elem__Effective Date__annotation","__elem__Parties__annotation","__elem__Governing Law__annotation",
           "__elem__Renewal Term__annotation","__elem__Notice Period To Terminate Renewal__annotation"]
    eval_metric, ans = bento_service.evaluate(lume_dataset.lumes, element_type_for_val)
    assert isinstance(eval_metric, dict)
    assert set(eval_metric.keys()) == set(['validation_exact_match', 'validation_IoU', 'validation_false_negative_rate', 'validation_true_negative_rate'])

def test_single_prediction_transformers_pipeline():
    context = "The quick brown fox jumps over the lazy dog."
    question = "What color was the fox?"
    answers = bento_service.single_prediction_transformers_pipeline(question, context)
    
    assert isinstance(answers, dict)
    assert set(answers.keys()) == set(['score', 'start', 'end', 'answer'])
    
def test_possible_answer():
    element = LumeElement("__elem__candidate_span", attributes = {
        "__attr__start_index": 0,
        "__attr__end_index": 1000
    })
    lume = Lume(name='test_lume', data = "The quick brown fox jumps over the lazy dog.", elements = [element])

    pred_lume = bento_service.predict(
        lume,
        element_type="__elem__candidate_span",
        output_element_type="__elem__qa_prediction",
        questions=["What color was the fox?"],
        handle_impossible_answer=False
    )

    assert isinstance(pred_lume, Lume)
    assert set(["__attr__start_index", "__attr__end_index", "__attr__answer", "__attr__score"]) == set(pred_lume.elements[-1].attributes)
    assert pred_lume.elements[-1].element_type == "__elem__qa_prediction"
    assert pred_lume.elements[-1].attributes["__attr__answer"] == "brown"

def test_impossible_answer():
    element = LumeElement("__elem__candidate_span", attributes = {
        "__attr__start_index": 0,
        "__attr__end_index": 1000
    })
    lume = Lume(name='test_lume', data = "The quick brown fox jumps over the lazy dog.", elements = [element])

    pred_lume = bento_service.predict(
        lume,
        element_type="__elem__candidate_span",
        output_element_type="__elem__qa_prediction",
        questions=["How big is the pacific ocean?"],
        handle_impossible_answer=True
    )

    assert isinstance(pred_lume, Lume)
    assert set(["__attr__start_index", "__attr__end_index", "__attr__answer", "__attr__score"]) == set(pred_lume.elements[-1].attributes)
    assert pred_lume.elements[-1].element_type == "__elem__qa_prediction"
    assert pred_lume.elements[-1].attributes["__attr__answer"] == ""

def test_multiple_questions():
    element = LumeElement("__elem__candidate_span", attributes = {
        "__attr__start_index": 0,
        "__attr__end_index": 1000
    })
    lume = Lume(name='test_lume', data = "The quick brown fox jumps over the lazy dog.", elements = [element])

    pred_lume = bento_service.predict(
        lume,
        element_type="__elem__candidate_span",
        output_element_type="__elem__qa_prediction",
        questions=["What color was the fox?", "How big is the pacific ocean?"],
        handle_impossible_answer=False
    )

    assert isinstance(pred_lume, Lume)
    assert set(["__attr__start_index", "__attr__end_index", "__attr__answer", "__attr__score"]) == set(pred_lume.elements[-1].attributes)
    assert pred_lume.elements[-1].element_type == "__elem__qa_prediction"
    assert pred_lume.elements[-1].attributes["__attr__answer"] == "brown"