# Readme

This is a ML Service bundle created with BentoML, containing a Question Answering model based on Google's Electra transformer architecture and pre-trained on Standford's SQuADv2 dataset.

# Usage

See the tutorials for this Bento for examples of usage within Ignite workflows. Also see the doc-strings of the individual endpoints in `bentoml.yml`

# Testing

Run `pytest --cov-report term-missing --cov QAModel tests` after cloning this repository and installing the packages from `requirements.txt` in your Ignite environment. Note that you may need to install the `pytest` and `pytest-cov` libraries in order to run the unit tests.

# Disclaimer
We make no representations or warranties regarding the license status and usage restrictions of any underlying code, training data, pre-trained model weights, or any other external resources that were used to build and train the model. Please confirm with appropriate KPMG leadership if you intend to use this model for any internal or external purposes.