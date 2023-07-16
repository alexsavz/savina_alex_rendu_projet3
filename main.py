"""Customer churn prediction"""

# Functions files - modules

import src.data.import_data as imp
import src.data.export_data as exp
import src.features.build_features as bf
import src.models.train_evaluate as te

# PARAMETRES -------------------------------
config = imp.import_yaml_config("./configuration/config.yaml")

# IMPORT DES DONNEES --------------------------------
TRAIN_DEMOGRAPHICS = config["path"]["train"]["demographics"]
TRAIN_SERVICES = config["path"]["train"]["services"]
TRAIN_STATUS = config["path"]["train"]["status"]

TEST_DEMOGRAPHICS = config["path"]["test"]["demographics"]
TEST_SERVICES = config["path"]["test"]["services"]
TEST_STATUS = config["path"]["test"]["status"]

MODEL = config["path"]["model"]

TrainData1 = imp.import_data(TRAIN_DEMOGRAPHICS)
TrainData2 = imp.import_data(TRAIN_SERVICES)
TestData1 = imp.import_data(TEST_DEMOGRAPHICS)
TestData2 = imp.import_data(TEST_SERVICES)

X_train = imp.merge_data(TrainData1, TrainData2, "customer_id")
X_test = imp.merge_data(TestData1, TestData2, "customer_id ")
y_train = imp.import_data(TRAIN_STATUS)
y_test = imp.import_data(TEST_STATUS)

# Dataframe des données initiales, label inclu
df = imp.merge_data(X_train, y_train, "customer_id")

# FEATURE ENGINEERING --------------------------------

# Regularisation of the columns names
X_test = bf.columns_train_test(X_train, X_test)

# Create the new column "options_number"
X_train = bf.create_options_number(X_train)
X_test = bf.create_options_number(X_test)


# import base64
# import requests

# master = "https://raw.githubusercontent.com/alexsavz/projet3DataGong/main/lightGBM_model.txt"
# req = requests.get(master)
# req = req.text
# print(type(req))
# LGB_model = lgb.Booster(model_file="./models/lightGBM_model.txt")

X_train = te.data_features_selection(X_train)

# Features selections
X_test = te.data_features_selection(X_test)

# preprocess test set
X_test_prepro = te.preprocess_data(X_test)

# Object deserialization ---- Pipeline import with trained model

LGB_model = te.load_model(MODEL)

# MODELISATION: Light GBM ----------------------------

y_test_pred = LGB_model.predict(X_test_prepro)
print(f"Affichons les 5 premières prédictions : {y_test_pred[0:5]}")

# save csv
exp.save_data(y_test, y_test_pred)
