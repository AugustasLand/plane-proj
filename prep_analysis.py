import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize

def prep(df, drop_columns, encode_columns, id='id'):
    # Manual one-hot
    if id == 'hex':
        df.insert(1, 'is_military', df['dbFlags'] == 1, True)
        df.insert(1, 'is_interesting', df['dbFlags'] == 2, True)
        df.insert(1, 'LADD', df['dbFlags'] == 8, True) # Limiting Aircraft Data Displayed

        # auto one_hot
        for col in encode_columns:
            df = pd.concat([df, pd.get_dummies(df[col], dummy_na=True)], axis=1)
            df = df.rename(columns={np.nan: f"{col}_nan"})

    df = df.drop(columns=drop_columns)
    df = df.drop(columns=encode_columns)
    return df
    
def outlier_analysis(df, analysis_df):
    if_model = IsolationForest(random_state=42)
    gmm_model = GaussianMixture(2)
    if_anomalous = np.where(if_model.fit_predict(analysis_df) == -1)[0]
    gmm_pred = gmm_model.fit_predict(analysis_df)
    gmm_anomalous1 = np.where(gmm_pred != 0)[0]
    gmm_anomalous2 = np.where(gmm_pred == 0)[0]
    outliers1 = np.intersect1d(gmm_anomalous1, if_anomalous, assume_unique=True, return_indices=False)
    outliers2 = np.intersect1d(gmm_anomalous2, if_anomalous, assume_unique=True, return_indices=False)
    if len(outliers1) > len(outliers2):
        outliers = outliers1
    else: outliers = outliers2
    df["anomalous"] = (df.index).isin(outliers)

    return df

def train_supervised(data):
    y = data['anomalous']
    X = data[['turn1', 'turn2', 'turn3', 'deltaD1', 'deltaD2', 'deltaD3', 'time',
          'latitude', 'longitude', 'altitude', 'ground_speed']]
    rf = RandomForestClassifier(max_depth=5, random_state=0)
    gnb = GaussianNB()

    X[['altitude', 'ground_speed', 'time']] = normalize(X[['altitude', 'ground_speed', 'time']], copy=False)
    for d in range(3):
        X[f'deltaD{d+1}'] = normalize([X[f'deltaD{d+1}']], copy=False)[0]
        X[f'turn{d+1}'] = normalize([X[f'turn{d+1}']], copy=False)[0]

    gnb.fit(X, y)
    # print(f"NB accuracy :{(y_test == y_pred).mean()*100:.3f}%")
    # print(confusion_matrix(y_pred, y_test))

    rf.fit(X, y)
    # print(f"RF accuracy: {rf.score(X_test, y_test)*100:.3f}%")
    # print(confusion_matrix(rf.predict(X_test), y_test))

    return X, rf, gnb

def f24_analysis():
    f24planes = pd.read_csv("data/f24/new_full_data.csv")
    f24planes = f24planes[f24planes.columns[1:]]
    
    dropf24 = ["id", "icao_24bit", "registration", "flight_no", "airline_icao", "callsign", 'destination_airport_iata', 'origin_airport_iata', 'aircraft_code']
    # No encodable columns for small dataset
    encodef24 = []
    f24data = prep(f24planes, dropf24, encodef24, 'id')

    f24data[['altitude', 'ground_speed', 'time']] = normalize(f24data[['altitude', 'ground_speed', 'time']])#, 'duration']])
    for d in range(3):
        f24data[f'deltaD{d+1}'] = normalize([f24data[f'deltaD{d+1}']])[0]
        f24data[f'turn{d+1}'] = normalize([f24data[f'turn{d+1}']])[0]

    #Unsupervised
    # f24planes = outlier_analysis(f24planes, f24data)
    # anomalous = f24planes[f24planes['anomalous'] == 1]
    # print(f"There are {len(anomalous)} anomalies")
    # print(anomalous)
    
    # manual_anomalies = pd.read_csv("anomalous.csv")
    # false_positives = np.sum(planes["anomalous"] != manual_anomalies["anomalous"])
    # print(f"False positives: {false_positives/planes['anomalous'].sum()*100:.3f}%")
    

    #Supervised
    # traindata_full = pd.read_csv("old/anomalous.csv")
    traindata, rf, gnb = train_supervised(f24data)

    f24data = f24data[traindata.columns]
    rf_pred = rf.predict(f24data)
    gnb_pred = gnb.predict(f24data)

    # pd.concat([traindata_full, f24planes[traindata_full.columns]]).to_csv("new_full_data.csv")

    predicted_anomalies = f24planes[np.bool_(gnb_pred)]
    print(f"There are {len(predicted_anomalies)} outliers")
    print(predicted_anomalies.head())
    print("GNB confusion matrix")
    print(confusion_matrix(f24planes['anomalous'], gnb_pred))
    print("RF confusion matrix")
    print(confusion_matrix(f24planes['anomalous'], rf_pred))

def absd_analysis():
    absd_planes = pd.read_csv("data/absd/filtered_absd20240401.csv")
    drop_absd = ["plane_id", "flight_no", "squawk", "dbFlags", "plane_type", "longitude", "latitude"]
    encode_absd = ["type", "category"]

    # drop_absd = ["plane_id", "flight_no", "squawk", "dbFlags", "plane_type", "type", "category"]
    # encode_absd = []
    absd_data = prep(absd_planes, drop_absd, encode_absd, 'hex')

    #Mean instead of nan, replaces lack of data with normal data to not use it as discerning factor
    for col in absd_data.columns[absd_data.isna().any()].tolist():
        col_mean = absd_data[col].mean()
        absd_data.loc[pd.isna(absd_data[col]), col] = col_mean

    absd_data[['altitude', 'ground_speed', 'time', 'duration', 'messages', 'signal_power']]=\
    normalize(absd_data[['altitude', 'ground_speed', 'time', 'duration', 'messages', 'signal_power']])
    for d in range(4):
        absd_data[f'deltaD{d+1}'] = normalize([absd_data[f'deltaD{d+1}']])[0]
        absd_data[f'turn{d+1}'] = normalize([absd_data[f'turn{d+1}']])[0]

    absd_planes = outlier_analysis(absd_planes, absd_data)

    outliers = absd_planes[absd_planes['anomalous'] == 1]
    print(f"There are {len(outliers)} outliers")
    print(outliers.head())
    # outliers[['plane_id', 'anomalous']].to_csv('manual_review.csv')

if __name__ == "__main__":
    # absd_analysis()
    f24_analysis()

    
   

