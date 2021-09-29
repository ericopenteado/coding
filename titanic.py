import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


#funções para converter variáveis categoricas para inteiros
def converte_sex(val):
    if "male" == val:
        return 1
    return 0

def converte_embarking(val):
    if "C" == val:
        return 2
    elif "Q" == val:
        return 1
    else:
        return 0

#pré processamento dos dados

def preprocessamento_data(csv_file):
    df = pd.read_csv(csv_file)
    df = df.sample(frac = 1).reset_index(drop = True)

    # resolvendo o problema de missing values
    df["Age"] = df["Age"].fillna(value = df.Age.median())

    # tranfomação de categorico para inteiro
    df["Sex"] = df["Sex"].apply(converte_sex)
    df["Embarked"] = df["Embarked"].apply(converte_embarking)

    #Normalizando
    scaler = MinMaxScaler()
    df["Age"] = scaler.fit_transform(np.array(df["Age"]).reshape(-1,1))
    df["Fare"] = scaler.fit_transform(np.array(df["Fare"]).reshape(-1,1))

    #separação entre a coluna importante para a resposta e as secundárias
    target = ["Survived"]
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    y_train = df[target]
    X_train = df[features]

    return X_train, y_train

def fim_preprocesso ():
    X_train, y_train = preprocessamento_data("train.csv")

    #dividindo em 70% para treinamento e 30% para validação
    train_size = int(len(y_train)* 0.70)

fim_preprocesso()

