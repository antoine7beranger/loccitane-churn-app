
# DATA CLEANING
PATH_DATA = "raw_data/LE_WAGON_2026-03-05-1151.csv"


# DATA PREPROCESSING PARAMETERS

# FILTRER LES LIGNES D'UNE COLONNE DU DATASET
COLUMN_TO_FILTER = 'QUESTION_ID'
VALUES_TO_KEEP = ['NPS_Q1BIS','NPS_Q2_1','NPS_Q2_2','NPS_Q2_3',
    'NPS_Q5','NPS_Q1','NPS_Q2','NPS_SCORE']


# RENSEIGNER LES COLONNES DATES DU DATASET
DATE_COLUMNS = ['SURVEY_DAYID','SALES_DAYID']


## CONCATENATION DE COLONNES
NEW_COLUMN_NAME = 'QUESTION_CONTEXT'
COLUMNS_TO_CONCAT = ['NPS_TYPE','QUESTION_ID']


## MAPPING POUR LABELISER LES VALEURS D'UNE COLONNE
MAPPING_NPS_QUESTIONS = {'NPS_CS_NPS_Q1BIS':'SC_Note_Satisfaction_globale',
     'NPS_CS_NPS_Q2_1':'SC_Note_Facilite_Contact',
     'NPS_CS_NPS_Q2_2':'SC_Note_Expertise_Qualite_reponse',
     'NPS_CS_NPS_Q2_3':'SC_Note_Resolution_pb',
     'NPS_CS_NPS_Q5':'SC_Status_NPS',
     'NPS_EC_NPS_Q1':'EC_Status_NPS',
     'NPS_EC_NPS_Q1BIS':'EC_Note_Satisfaction_Online_Experience',
     'NPS_EC_NPS_Q2':'EC_Note_Intention_Reachat_Ecom',
     'NPS_RE_NPS_Q1BIS':'RE_Note_Qualite_accueil_magasin',
     'NPS_RE_NPS_Q2':'RE_Note_Expertise_Vendeuse',
     'NPS_RE_NPS_SCORE':'RE_Status_NPS'
    }

MAPPING_NPS_STATUS = {'Promoter': 10,
        'Detractor': 3,
        'Neutral': 7}


COLUMNS_RENAMING_DICT = {'VIPGROUP': 'RFM'}

NPS_TYPES = ['NPS_RE', 'NPS_EC', 'NPS_CS']


# PIPELINE
NUMERICAL_FEATURES = ['BATH',
 'BEAUTY OILS',
 'BODY SCRUB',
 'BODY SUN CARE',
 'CANDLE',
 'CLEANSER',
 'CONDITIONER',
 'DAILY CARE',
 'DAILY NEEDS',
 'EDP-EDT',
 'FACE SUN CARE',
 'FLORAL WATERS',
 'HAIR TREATMENT',
 'HAND CARE',
 'HOME PERFUME',
 'HOME PERFUME DIFFUSER',
 'HOME SACHET',
 'KITS_x',
 'LEG&FOOT CARE',
 'LIQUID SOAP',
 'MAKE-UP',
 'MASKS',
 'MOISTURIZING TREATMENT',
 'OTHER FRAGRANCE',
 'SERUM',
 'SHAMPOO AND 2IN1',
 'SHOWER',
 'SOLID SOAP',
 'TOOTHPASTE',
 'TREATMENT',
 'BODY CARE',
 'BODY PERFUME',
 'FACE CARE',
 'HAIR CARE',
 'HOME',
 'KITS_y',
 'TOILETRIES',
 'ACHATS_PRE_NPS',
 'ACHATS_DANS_FENETRE_PRE_NPS_12M']


NOMINAL_FEATURES=['HOME_STORE_MAIN_CHANNEL',
 'SURVEY_PERIOD']


ORDINAL_FEATURES_RE=['RE_Note_Expertise_Vendeuse',
 'RE_Note_Qualite_accueil_magasin',
 'RE_Status_NPS']

ORDINAL_FEATURES_CS=['SC_Note_Facilite_Contact',
 'SC_Note_Expertise_Qualite_reponse',
 'SC_Note_Resolution_pb','SC_Status_NPS']

ORDINAL_FEATURES_EC=['EC_Note_Intention_Reachat_Ecom',
 'EC_Note_Satisfaction_Online_Experience',
 'EC_Status_NPS']


####### chargement du dataset #######

import os
from dotenv import load_dotenv

load_dotenv(override=True)

DATASET_TYPE = os.getenv("DATASET_TYPE")
MODEL_TARGET = os.getenv("MODEL_TARGET")

###### Chemins data_set ####

BASE_DIR = os.getenv("BASE_DIR", os.path.dirname(os.path.abspath(__file__)))

#DATASET_PATHS = {
#  "EC": os.path.join(BASE_DIR, "cleaned_data/dataset_NPS_EC.csv"),
#  "RE": os.path.join(BASE_DIR, "cleaned_data/dataset_NPS_RE.csv"),
#  "CS": os.path.join(BASE_DIR, "cleaned_data/dataset_NPS_CS.csv"),
#}

DATASET_PATHS = {
    "EC": "data/clean_data/dataset_NPS_EC.csv",
    "RE": "data/clean_data/dataset_NPS_RE.csv",
    "CS": "data/clean_data/dataset_NPS_CS.csv",
}


###### chargement dataset ######

def get_dataset_type():
    if DATASET_TYPE not in DATASET_PATHS:
        raise ValueError(
            f"DATASET_TYPE invalide: {DATASET_TYPE}. "
            f"Valeurs attendues: {list(DATASET_PATHS.keys())}"
        )
    return DATASET_TYPE

def get_dataset_path():
    dataset_type = get_dataset_type()
    path = DATASET_PATHS[dataset_type]
    return path


# Liste des features commune pour l'entrainement X

LISTE_RAW_X = ['TOILETRIES', 'FACE CARE', 'BODY PERFUME', 'BODY CARE',
       'HAIR CARE', 'HOME']

LISTE_FEATURING_X = ['PURCHASE_INTENSITY', 'ACHATS_PRE_NPS','ACHATS_DANS_FENETRE_PRE_NPS_12M']

COMMONE_FEATURES = LISTE_RAW_X + LISTE_FEATURING_X

# Liste des features spécifiques à chaque data_set

LISTE_UNIQUE_EC = ['EC_Note_Intention_Reachat_Ecom','EC_Note_Satisfaction_Online_Experience','EC_Status_NPS']

LISTE_UNIQUE_RE = ['RE_Note_Expertise_Vendeuse', 'RE_Note_Qualite_accueil_magasin', 'RE_Status_NPS']

LISTE_UNIQUE_CS = ['SC_Note_Facilite_Contact', 'SC_Note_Expertise_Qualite_reponse', 'SC_Note_Resolution_pb','SC_Status_NPS']


DATASET_SPECIFIC_FEATURES = {
    "EC": LISTE_UNIQUE_EC,
    "RE": LISTE_UNIQUE_RE,
    "CS": LISTE_UNIQUE_CS,
}

# Logique  création des X

def get_features():
    dataset_type = get_dataset_type()
    specific_features = DATASET_SPECIFIC_FEATURES.get(dataset_type, [])
    return COMMONE_FEATURES + specific_features

###### save pickel ######
MODEL_DIR = "models"
