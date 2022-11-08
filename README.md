# Natural Language Understanding Sentiment Analysis Project
This is the code for the final project of the "Natural Language Understanding" course - MSc in Artificial Intelligence Systems - University of Trento.
I suggest to read the project's report `NLU_project_report.pdf` before running the code.

## Environment
The code is written in Python 3.9.12 (with other version is not guaranteed to work) and has been developed using a conda environment:

```bash
$ conda env create -n nlu --file nlu_env.yml
```

I have also included the requirements for a pip virtual environment in the file `requirements.txt` but i recommend to use conda.

## Project structure
The project folder is organized in this way:
* `datasets_analysis.py` file, is used to analyze the main statistics of the two dataset used in this project, **subjetivity** and **movie_reviews**.
* `parameters.py` file in which you can find the most important parameters for all the models, i.e. Number of epochs, learning rates, batch sizes ecc..
* In the file `functions.py`, there are the utility functions used to train the models.
* Then we have a folder for each model: **NB_baseline**, **lstm**, **cnn** and **transformers** in which there are the real training code and the structure of the used architectures:
    * `model.py` describe the structure of the model used (not present in the transformers folder, because i used a pretrained model).
    * In the file `datasets.py` there are the costumized datasets objects ovewriting the pyTorch Dataset class.
    * `train_functions_*.py` files contains effectively the train and evaluation loops.
    * `subj_*.py` and `pol_*.py` are the files to run to start the training of the model, for subjectivity and polarity classifier respectively.
>**NB_baseline** folder has a completely different structure, it contains all the elements needed for training both classifiers in one single file `baseline.py`.


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
Weights of the best achieved model are stored in `lstm/weigths` folder with the name `lstm_subj.pt` or `lstm_pol.pt` respectively. Also the "Word to Id" vocabulary is saved in the same folder with the name `w2id_subj_lstm.pkl` or `w2id_pol_lstm.pkl`. 

Set `subj_filter=True` for filtering out objective sentences.

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
Weights of the best achieved model are stored in `cnn/weigths` folder with the name `cnn_subj.pt` or `cnn_pol.pt` respectively. Also the "Word to Id" vocabulary is saved in the same folder with the name `w2id_subj_cnn.pkl` or `w2id_pol_cnn.pkl`. 

Set `subj_filter=True` for filtering out objective sentences.


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
Weights of the best achieved model are stored in `transformers/weigths` folder with the name `trans_subj.pt` or `trans_pol.pt` respectively. 

Set `subj_filter=True` for filtering out objective sentences.

>Be sure to train the subjectivity detector first, and then to train the polarity classifier if you want to use the filter, otherwise it does not work.

## Saved weights
I decided not to include the weights here for memory reasons, anyway you can find the trained weights for all the models at this [link](https://drive.google.com/drive/u/1/folders/1GSa39jmwXNyAtqk9iuZ6C5PS9RDvDbAf.
Also the "Word to Id" vocabularies are available in the same folder.

## Images folder
In the **images** folder we can see the train and test accuracy evolution during training for each experiment and the structure of the proposed models.