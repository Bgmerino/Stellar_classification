# Galaxy Classification Model

This project trains a machine learning model to classify astronomical objects (STAR, GALAXY, QSO). It includes data preprocessing, model training, SHAP-based interpretability and csv files with the predictions.

Data dowloaded from: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

### 1. Project Structure

Stellar_classification
- configuration: YAML config files
  - config.yml  

- data: Raw data
  - star_classification.csv

- evaluation_figures: Shap figures for each of the classes
  - shap_summary_class0.png
  - shap_summary_class1.png
  - shap_summary_class1.png

- models: trained models that are saved 

- predictions: output file with the model predictions

- previous_study: 
  - jupyter notebook with the study of the data previous to the model development (.ipynb and .html)
  - Stellar_classification.pdf a presentation about the model development
  
- src: 
  - config.py
  - data_preparation
  - test_model.py
  - train_model.py

- requirements.txt


### 2. Run the model

The script has two input parameters:
- using_model: to select whether to use an existing saved model or to train it (reuse/train)
- mode: to decide whether to produce the predictions as outputs or to evaluate the model (production/evaluation)

python stellar_model_main.py --using_model <reuse/train> --mode <production/evaluation>

### 3. Considerations

- The model was developed using python 3.12.11
- There are two models available to make the predictions with: a Random Forest Classifier and a SVM. Its selection takes place in the conf.yml file
- The outputs of the model are: the shap value graphs and the file with the classifications. The model trained can be saved as well.