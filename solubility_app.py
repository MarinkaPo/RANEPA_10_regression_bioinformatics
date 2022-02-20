import numpy as np
import pandas as pd
import streamlit as st
import pickle
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Descriptors

######################
# Пользовательская функция
######################
## Считаем молекулярные дескрипторы:
def AromaticProportion(m):
  aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
  aa_count = []
  for i in aromatic_atoms:
    if i==True:
      aa_count.append(1)
  AromaticAtom = sum(aa_count)
  HeavyAtom = Descriptors.HeavyAtomCount(m)
  AR = AromaticAtom/HeavyAtom
  return AR

def generate(smiles, verbose=False):

    moldata= []
    for elem in smiles:
        mol=Chem.MolFromSmiles(elem)
        moldata.append(mol)

    baseData= np.arange(1,1)
    i=0
    for mol in moldata:

        desc_MolLogP = Descriptors.MolLogP(mol)
        desc_MolWt = Descriptors.MolWt(mol)
        desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
        desc_AromaticProportion = AromaticProportion(mol)

        row = np.array([desc_MolLogP,
                        desc_MolWt,
                        desc_NumRotatableBonds,
                        desc_AromaticProportion])

        if(i==0):
            baseData=row
        else:
            baseData=np.vstack([baseData, row])
        i=i+1

    columnNames=["MolLogP","MolWt","NumRotatableBonds","AromaticProportion"]
    descriptors = pd.DataFrame(data=baseData,columns=columnNames)

    return descriptors

######################
# Заголовок страницы
######################

image = Image.open('solubility-logo.jpg')

st.image(image, use_column_width=True)

st.write("""
# Веб-приложение для прогнозирования молекулярной растворимости

Предсказывает значения **растворимости (LogS)** молекул.

Данные взяты у John S. Delaney. [ESOL:  Estimating Aqueous Solubility Directly from Molecular Structure](https://pubs.acs.org/doi/10.1021/ci034243x). ***J. Chem. Inf. Comput. Sci.*** 2004, 44, 3, 1000-1005.

""")

# -----О методе-----
expander_bar = st.expander("Как это работает?")
expander_bar.markdown(
"""
**SMILES** (Simplified Molecular Input Line Entry System) — это химическая система обозначений, 
которая позволяет пользователю представить химическую структуру таким образом, который может быть использован компьютером. 

SMILES — это легко запоминающаяся и гибкая для работы конструкция. Запись в виде SMILES требует, чтобы вы выучили несколько правил. 
Не нужно беспокоиться о двусмысленных представлениях, потому что программное обеспечение автоматически изменит порядок ввода в строку SMILES, когда это необходимо.
"""
)
st.write("---")

######################
# Боковая панель: ввод молекул
######################

st.sidebar.header('Введите SMILES-последовательность:')

## Чтение SMILES-последовательности:
SMILES_input = "NCCCC\nCCC\nCN"

SMILES = st.sidebar.text_area("SMILES-последовательность:", SMILES_input)
SMILES = "C\n" + SMILES #Adds C as a dummy, first item
SMILES = SMILES.split('\n')

######################
# Центральная панель:
######################

st.header('Введённая последовательность:')
SMILES[1:] # Skips the dummy first item

## Calculate molecular descriptors
st.header('Вычисленные молекулярных дескриптороров:')
X = generate(SMILES)
X[1:] # Skips the dummy first item

######################
# Pre-built model
######################

# RБерём сохранённую модель
load_model = pickle.load(open('solubility_model.pkl', 'rb'))

# Предикт:
prediction = load_model.predict(X)
#prediction_proba = load_model.predict_proba(X)

st.header('Прогноз значения растворимости (LogS values):')
prediction[1:] # Skips the dummy first item
