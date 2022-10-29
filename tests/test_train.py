# Imports
import sys
import os
import tempfile
import bentoml
from ignite_lume.lume import Lume, LumeElement, LumeDataset

# Load Bento
bento_service = bentoml.load(os.path.abspath("./"))

# Define tests
def test_random_negative_example_generation():
    bento_service.random_state = 1

    qa_pair_element = LumeElement("__elem__qa_pair", attributes = {
        "__attr__start_index": 20,
        "__attr__end_index": 25,
        "__attr__question": "What is the fox's action?"
    })
    text = """The quick brown fox jumps over the lazy dog. This is another sentence in the Lume, which has some semantic overlap with the first due to the words jump, lazy, quick, and fox."""
    lume = Lume(name='train_lume', data = text, elements = [qa_pair_element])
    for _ in range(10):
        neg_example = bento_service.make_squad_negative_example(lume, qa_pair_element, "__attr__question", context_length=50, method="random")
        start, end = lume.data.find(neg_example['context']), lume.data.find(neg_example['context']) + 50
        assert bento_service.text_iou(qa_pair_element.attributes["__attr__start_index"], qa_pair_element.attributes["__attr__end_index"], start, end) == 0

def test_overlap_negative_example_generation():
    qa_pair_element = LumeElement("__elem__qa_pair", attributes = {
        "__attr__start_index": 20,
        "__attr__end_index": 25,
        "__attr__question": "What is the action of the fox?"
    })
    text = """The quick brown fox jumps over the lazy dog. This is another sentence in the Lume, which has some semantic overlap with the question due to the words fox, action, and what."""
    lume = Lume(name='train_lume', data = text, elements = [qa_pair_element])
    neg_example = bento_service.make_squad_negative_example(lume, qa_pair_element, "__attr__question", context_length=50, method="question_word_overlap")
    start, end = lume.data.find(neg_example['context']), lume.data.find(neg_example['context']) + 50
    assert bento_service.text_iou(qa_pair_element.attributes["__attr__start_index"], qa_pair_element.attributes["__attr__end_index"], start, end) == 0

def test_false_positive_negative_example_generation():
    qa_pair_element = LumeElement("__elem__qa_pair", attributes = {
        "__attr__start_index": 20,
        "__attr__end_index": 25,
        "__attr__question": "What is the action of the fox?"
    })
    text = """The quick brown fox jumps over the lazy dog. This is another sentence in the Lume, which has some semantic overlap with the question due to the words fox, action, and what."""
    lume = Lume(name='train_lume', data = text, elements = [qa_pair_element])
    neg_example = bento_service.make_squad_negative_example(lume, qa_pair_element, "__attr__question", context_length=50, method="false_positive_mining")
    start, end = lume.data.find(neg_example['context']), lume.data.find(neg_example['context']) + 50
    assert bento_service.text_iou(qa_pair_element.attributes["__attr__start_index"], qa_pair_element.attributes["__attr__end_index"], start, end) == 0

def test_train_model_no_negatives():
    element = LumeElement("__elem__candidate_span", attributes = {
        "__attr__start_index": 0,
        "__attr__end_index": 1000
    })
    qa_pair_element = LumeElement("__elem__qa_pair", attributes = {
        "__attr__start_index": 20,
        "__attr__end_index": 25,
        "__attr__question": "What is the fox's action?"
    })
    train_lume = Lume(name='train_lume', data = "The quick brown fox jumps over the lazy dog.", elements = [element, qa_pair_element])
    test_lume = Lume(name='test_lume', data = "The quick brown fox jumps over the lazy dog.", elements = [element, qa_pair_element])

    lume_dataset = LumeDataset('test', lumes=[train_lume, test_lume])

    with tempfile.TemporaryDirectory() as d:
        # Train the model
        _ = bento_service.train(
            lume_dataset,
            output_path=d,
            element_type="__elem__qa_pair",
            question_attribute="__attr__question",
            generate_negative_examples=False,
            validation_split=["test_lume"],
            random_state=1,
            num_train_epochs=1
        )

        # Get predictions with the trained model
        bento_service_finetuned = bentoml.load(d)

        pred_lume = bento_service_finetuned.predict(
            test_lume,
            element_type="__elem__candidate_span",
            output_element_type="__elem__qa_prediction",
            questions=["What is the fox's action?"],
            handle_impossible_answer=False
        )

    assert isinstance(pred_lume, Lume)
    assert set(["__attr__start_index", "__attr__end_index", "__attr__answer", "__attr__score"]) == set(pred_lume.elements[-1].attributes)
    assert pred_lume.elements[-1].element_type == "__elem__qa_prediction"

def test_train_model_no_validation_no_negatives():
    element = LumeElement("__elem__candidate_span", attributes = {
        "__attr__start_index": 0,
        "__attr__end_index": 1000
    })
    qa_pair_element = LumeElement("__elem__qa_pair", attributes = {
        "__attr__start_index": 20,
        "__attr__end_index": 25,
        "__attr__question": "What is the fox's action?"
    })
    train_lume = Lume(name='train_lume', data = "The quick brown fox jumps over the lazy dog.", elements = [element, qa_pair_element])
    test_lume = Lume(name='test_lume', data = "The quick brown fox jumps over the lazy dog.", elements = [element, qa_pair_element])

    lume_dataset = LumeDataset('test', lumes=[train_lume, test_lume])

    with tempfile.TemporaryDirectory() as d:
        # Train the model
        _ = bento_service.train(
            lume_dataset,
            output_path=d,
            element_type="__elem__qa_pair",
            question_attribute="__attr__question",
            generate_negative_examples=False,
            validation_split=0,
            random_state=1,
            num_train_epochs=1
        )

        # Get predictions with the trained model
        bento_service_finetuned = bentoml.load(d)

        pred_lume = bento_service_finetuned.predict(
            test_lume,
            element_type="__elem__candidate_span",
            output_element_type="__elem__qa_prediction",
            questions=["What is the fox's action?"],
            handle_impossible_answer=False
        )

    assert isinstance(pred_lume, Lume)
    assert set(["__attr__start_index", "__attr__end_index", "__attr__answer", "__attr__score"]) == set(pred_lume.elements[-1].attributes)
    assert pred_lume.elements[-1].element_type == "__elem__qa_prediction"

def test_train_model_with_negatives_random():
    element = LumeElement("__elem__candidate_span", attributes = {
        "__attr__start_index": 0,
        "__attr__end_index": 1000
    })
    qa_pair_element = LumeElement("__elem__qa_pair", attributes = {
        "__attr__start_index": 20,
        "__attr__end_index": 25,
        "__attr__question": "What is the fox's action?"
    })
    text = """The quick brown fox jumps over the lazy dog. This is another sentence in the Lume, which is unrelated to the first sentence."""
    train_lume = Lume(name='train_lume', data = text, elements = [element, qa_pair_element])
    test_lume = Lume(name='test_lume', data = text, elements = [element, qa_pair_element])

    lume_dataset = LumeDataset('test', lumes=[train_lume, test_lume])

    with tempfile.TemporaryDirectory() as d:
        # Train the model
        _ = bento_service.train(
            lume_dataset,
            output_path=d,
            element_type="__elem__qa_pair",
            question_attribute="__attr__question",
            context_size = 50,
            generate_negative_examples=True,
            negative_example_method="random",
            validation_split=0,
            random_state=1,
            num_train_epochs=1
        )

        # Get predictions with the trained model
        bento_service_finetuned = bentoml.load(d)

        pred_lume = bento_service_finetuned.predict(
            test_lume,
            element_type="__elem__candidate_span",
            output_element_type="__elem__qa_prediction",
            questions=["What is the fox's action?"],
            handle_impossible_answer=False
        )

    assert isinstance(pred_lume, Lume)
    assert set(["__attr__start_index", "__attr__end_index", "__attr__answer", "__attr__score"]) == set(pred_lume.elements[-1].attributes)
    assert pred_lume.elements[-1].element_type == "__elem__qa_prediction"

def test_train_model_with_negatives_overlap():
    element = LumeElement("__elem__candidate_span", attributes = {
        "__attr__start_index": 0,
        "__attr__end_index": 1000
    })
    qa_pair_element = LumeElement("__elem__qa_pair", attributes = {
        "__attr__start_index": 20,
        "__attr__end_index": 25,
        "__attr__question": "What is the fox's action?"
    })
    text = """The quick brown fox jumps over the lazy dog. This is another sentence in the Lume, which has some semantic overlap with the question due to the words fox, action, and what."""
    train_lume = Lume(name='train_lume', data = text, elements = [element, qa_pair_element])
    test_lume = Lume(name='test_lume', data = text, elements = [element, qa_pair_element])

    lume_dataset = LumeDataset('test', lumes=[train_lume, test_lume])

    with tempfile.TemporaryDirectory() as d:
        # Train the model
        _ = bento_service.train(
            lume_dataset,
            output_path=d,
            element_type="__elem__qa_pair",
            question_attribute="__attr__question",
            context_size = 50,
            generate_negative_examples=True,
            negative_example_method="question_word_overlap",
            validation_split=0,
            random_state=1,
            num_train_epochs=1
        )

        # Get predictions with the trained model
        bento_service_finetuned = bentoml.load(d)

        pred_lume = bento_service_finetuned.predict(
            test_lume,
            element_type="__elem__candidate_span",
            output_element_type="__elem__qa_prediction",
            questions=["What is the fox's action?"],
            handle_impossible_answer=False
        )

    assert isinstance(pred_lume, Lume)
    assert set(["__attr__start_index", "__attr__end_index", "__attr__answer", "__attr__score"]) == set(pred_lume.elements[-1].attributes)
    assert pred_lume.elements[-1].element_type == "__elem__qa_prediction"