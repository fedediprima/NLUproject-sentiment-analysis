# Natural Language Understanding Sentiment Analysis Project
This is the code for the final project of the "Natural Language Understanding" course - MSc in Artificial Intelligence Systems - University of Trento.
I recommend to read the project's report `NLU_project_report.pdf` before running the code.

## Environment
The code is written in Python 3.9.12 (with other version is not guaranteed to work) and has been developed using a conda environment:

```bash
$ conda env create -n nlu --file nlu_env.yml
```

I have also exported the requirements for a pip virtual environment in the file `requirements.txt` but i recommend to use conda.

## spigazione CARTELLE e FILE

## Weights


## Run the code

### Naive Bayes Classifier 

Both classifiers are trained by running 
```bash
$ python NB_baseline/baseline.py
```
Depending on the `subj_filter` flag (parameter of the main function), the polarity detector will be trained on a filtered or a not filtered dataset.
The results are printed on the standard output device.

### LSTM

To train the subjectivity classifier with lstm-based model, run:
```bash
$ python lstm/subj_lstm.py
```
For the polarity classifier, use:
```bash
$ python lstm/pol_lstm.py
```
Weights of the best achieved model are stored in `lstm/weigths` folder with the name `lstm_subj.pt` or `lstm_pol.pt` respectively. Also the "Word to Id" vocabulary is saved in the same folder with the name `w2id_subj_lstm.pkl` or `w2id_pol_lstm.pkl`. \

Set the `subj_filter` flag to True for filtering out objective sentences.

>Be sure to train the subjectivity detector first, and then to train the polarity classifier if you want to use the filter, otherwise it does not work.

### CNN

To train the subjectivity classifier with cnn-based model, run:
```bash
$ python cnn/subj_cnn.py
```
For the polarity classifier, use:
```bash
$ python cnn/pol_cnn.py
```
Weights of the best achieved model are stored in `cnn/weigths` folder with the name `cnn_subj.pt` or `cnn_pol.pt` respectively. Also the "Word to Id" vocabulary is saved in the same folder with the name `w2id_subj_cnn.pkl` or `w2id_pol_cnn.pkl`. \

Set the `subj_filter` flag to True for filtering out objective sentences.

>Be sure to train the subjectivity detector first, and then to train the polarity classifier if you want to use the filter, otherwise it does not work.

### Transformers

To train the subjectivity classifier with a transformer-based model, run:
```bash
$ python cnn/subj_trans.py
```
For the polarity classifier, use:
```bash
$ python cnn/pol_trans.py
```
Weights of the best achieved model are stored in `transformers/weigths` folder with the name `trans_subj.pt` or `trans_pol.pt` respectively. \

Set the `subj_filter` flag to True for filtering out objective sentences.

>Be sure to train the subjectivity detector first, and then to train the polarity classifier if you want to use the filter, otherwise it does not work.