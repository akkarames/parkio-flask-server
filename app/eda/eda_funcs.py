import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_model():
    with open('app/trained_models/model.pkl', 'rb') as f:
        classifier = pickle.load(f)
    return classifier


def data_scaling(dataset):
    # Feature Scaling
    sc = StandardScaler()
    return sc.fit_transform(np.array(dataset))



def preprocessing(df):
    input_array = []
    specific_columns=df[["gFx", "gFy", "gFz", "ax", "ay", "az", "wx", "wy", "wz", "Bx", "By", "Bz" ]]

    # print(specific_columns)
    count_row = specific_columns.shape[0]
    i=0
    while i < count_row:
        window=specific_columns.iloc[i:500+i, :]
        
        Mean_gFx = window["gFx"].mean()
        Mean_gFy = window["gFy"].mean()
        Mean_gFz = window["gFz"].mean()
        Mean_ax = window["ax"].mean()
        Mean_ay = window["ay"].mean()
        Mean_az = window["az"].mean()
        Mean_wx = window["wx"].mean()
        Mean_wy = window["wy"].mean()
        Mean_wz = window["wz"].mean()
        Mean_Bx = window["Bx"].mean()
        Mean_By = window["By"].mean()
        Mean_Bz = window["Bz"].mean()
        
        Median_gFx = window["gFx"].median()
        Median_gFy = window["gFy"].median()
        Median_gFz = window["gFz"].median()
        Median_ax = window["ax"].median()
        Median_ay = window["ay"].median()
        Median_az = window["az"].median()
        Median_wx = window["wx"].median()
        Median_wy = window["wy"].median()
        Median_wz = window["wz"].median()
        Median_Bx = window["Bx"].median()
        Median_By = window["By"].median()
        Median_Bz = window["Bz"].median()
        
        Std_gFx = window["gFx"].std()
        Std_gFy = window["gFy"].std()
        Std_gFz = window["gFz"].std()
        Std_ax = window["ax"].std()
        Std_ay = window["ay"].std()
        Std_az = window["az"].std()
        Std_wx = window["wx"].std()
        Std_wy = window["wy"].std()
        Std_wz = window["wz"].std()
        Std_Bx = window["Bx"].std()
        Std_By = window["By"].std()
        Std_Bz = window["Bz"].std()
        
        Var_gFx = window["gFx"].var()
        Var_gFy = window["gFy"].var()
        Var_gFz = window["gFz"].var()
        Var_ax = window["ax"].var()
        Var_ay = window["ay"].var()
        Var_az = window["az"].var()
        Var_wx = window["wx"].var()
        Var_wy = window["wy"].var()
        Var_wz = window["wz"].var()
        Var_Bx = window["Bx"].var()
        Var_By = window["By"].var()
        Var_Bz = window["Bz"].var()
        
        Max_gFx = window["gFx"].max()
        Max_gFy = window["gFy"].max()
        Max_gFz = window["gFz"].max()
        Max_ax = window["ax"].max()
        Max_ay = window["ay"].max()
        Max_az = window["az"].max()
        Max_wx = window["wx"].max()
        Max_wy = window["wy"].max()
        Max_wz = window["wz"].max()
        Max_Bx = window["Bx"].max()
        Max_By = window["By"].max()
        Max_Bz = window["Bz"].max()
        
        Min_gFx = window["gFx"].min()
        Min_gFy = window["gFy"].min()
        Min_gFz = window["gFz"].min()
        Min_ax = window["ax"].min()
        Min_ay = window["ay"].min()
        Min_az = window["az"].min()
        Min_wx = window["wx"].min()
        Min_wy = window["wy"].min()
        Min_wz = window["wz"].min()
        Min_Bx = window["Bx"].min()
        Min_By = window["By"].min()
        Min_Bz = window["Bz"].min()
        
        Kurt_gFx = window["gFx"].kurt()
        Kurt_gFy = window["gFy"].kurt()
        Kurt_gFz = window["gFz"].kurt()
        Kurt_ax = window["ax"].kurt()
        Kurt_ay = window["ay"].kurt()
        Kurt_az = window["az"].kurt()
        Kurt_wx = window["wx"].kurt()
        Kurt_wy = window["wy"].kurt()
        Kurt_wz = window["wz"].kurt()
        Kurt_Bx = window["Bx"].kurt()
        Kurt_By = window["By"].kurt()
        Kurt_Bz = window["Bz"].kurt()
        
        Skew_gFx = window["gFx"].skew()
        Skew_gFy = window["gFy"].skew()
        Skew_gFz = window["gFz"].skew()
        Skew_ax = window["ax"].skew()
        Skew_ay = window["ay"].skew()
        Skew_az = window["az"].skew()
        Skew_wx = window["wx"].skew()
        Skew_wy = window["wy"].skew()
        Skew_wz = window["wz"].skew()
        Skew_Bx = window["Bx"].skew()
        Skew_By = window["By"].skew()
        Skew_Bz = window["Bz"].skew()
        
        
        line = [Mean_gFx, Mean_gFy, Mean_gFz, Mean_ax, Mean_ay, Mean_az, Mean_wx, Mean_wy, Mean_wz, Mean_Bx, Mean_By, Mean_Bz, Median_gFx, Median_gFy, Median_gFz, Median_ax, Median_ay, Median_az, Median_wx, Median_wy, Median_wz, Median_Bx, Median_By, Median_Bz, Std_gFx, Std_gFy, Std_gFz, Std_ax, Std_ay, Std_az, Std_wx, Std_wy, Std_wz, Std_Bx, Std_By, Std_Bz, Var_gFx, Var_gFy, Var_gFz, Var_ax, Var_ay, Var_az, Var_wx, Var_wy, Var_wz, Var_Bx, Var_By, Var_Bz, Max_gFx, Max_gFy, Max_gFz, Max_ax, Max_ay, Max_az, Max_wx, Max_wy, Max_wz, Max_Bx, Max_By, Max_Bz, Min_gFx, Min_gFy, Min_gFz, Min_ax, Min_ay, Min_az, Min_wx, Min_wy, Min_wz, Min_Bx, Min_By, Min_Bz, Kurt_gFx, Kurt_gFy, Kurt_gFz, Kurt_ax, Kurt_ay, Kurt_az, Kurt_wx, Kurt_wy, Kurt_wz, Kurt_Bx, Kurt_By, Kurt_Bz, Skew_gFx, Skew_gFy, Skew_gFz, Skew_ax, Skew_ay, Skew_az, Skew_wx, Skew_wy, Skew_wz, Skew_Bx, Skew_By, Skew_Bz ]
        input_array.append(line)
        i=i+250
    
    return input_array
    

    
        			

