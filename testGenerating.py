from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from root_pandas import read_root
from keras.models import load_model
from plotting import plot_real_vs_gen
import os
import shutil

from main import sample_data_and_gen
from main import input_columns, output_columns, temp, swish, wasserstein_loss

low = 0.0
high = 5.0
def ptInRange(dataframe):
    return (dataframe.trk_pt>=low) & (dataframe.trk_pt<high)

###Recreate plots folders
folders_ = ['plots_testGenerator']
for dir in folders_:
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

scaler_inp = joblib.load('input_scaler.pkl')
scaler_out = joblib.load('output_scaler.pkl')
generator = load_model('generator.h5', custom_objects={'swish':swish})
print generator.summary()
input_file = "data/trackingNtuple_TTBarLeptons.root"
dataframe = read_root(input_file, columns=input_columns + output_columns + temp, flatten=True)[
    input_columns + output_columns + temp]
dataframe = dataframe[(dataframe.trk_isTrue == 1) & (dataframe.trk_algo == 4) & (dataframe.trk_pt >= low) & (dataframe.trk_pt < high)]

#Input to be generated
n_samples = dataframe.shape[0]
# pT = np.random.uniform(low, high, n_samples)
# eta = np.random.uniform(-2.1, 2.1, n_samples)
# phi = np.random.uniform(-math.pi, math.pi, n_samples)
pT = dataframe.trk_pt
eta = dataframe.trk_eta
phi = dataframe.trk_phi
inputs = np.column_stack((pT, eta, phi))

#Put into a dataframe
inputs_ = scaler_inp.transform(inputs)
inputs_ = generator.predict(inputs_)
gen_dataframe = scaler_out.inverse_transform(inputs_)
gen_dataframe = np.concatenate((inputs, gen_dataframe), axis=1)
gen_dataframe = pd.DataFrame(gen_dataframe, columns=input_columns + output_columns)

binning = np.linspace(-0.5,0.5,100)
for column in output_columns:
    plot_real_vs_gen(dataframe[column], gen_dataframe[column], column, title=column, xlabel="")
