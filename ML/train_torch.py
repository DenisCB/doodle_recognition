import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary

from helpers import im2arr
from torch_models import ConvNetSimple, InceptionBasedNet
from torch_trainer import Trainer

if torch.cuda.is_available():
    torch_dtype = torch.cuda.FloatTensor
else:
    torch_dtype = torch.FloatTensor

result_path = '../model/'
mean_global = np.load(result_path+'mean_global.npy')[0]
px, border_px = np.load(result_path+'processing_params.npy')

if os.environ.get('DEBUG') == '1':
    df = pd.read_csv('data/df_sample.csv')
    train = np.load('data/strokes_raw_sample.npy')
else:
    df = pd.read_csv('data/df.csv')
    train = np.load('data/strokes_raw.npy')
print(df.shape[0])


valid_ids = df[df.recognized == 1].index
train = train[valid_ids]
df = df.loc[valid_ids].reset_index(drop=True)
del (valid_ids)

le = LabelEncoder()
ytrain = le.fit_transform(df.target.values)


(train, valid,
 ytrain, yvalid,
 df, dfvalid) = train_test_split(train, ytrain, df, test_size=0.1, random_state=0)
num_classes = int(ytrain.max() + 1)
np.save(result_path+'classes.npy', le.classes_)


model = InceptionBasedNet(num_classes, (1, px, px)).type(torch_dtype)
loss_fn = nn.CrossEntropyLoss().type(torch_dtype)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
print(summary(model, (1, px, px)))

cpu_count = 6
epochs_num = 50
epoch_partion = 0.1
batch_size = 64
model_name = result_path+'InceptionBasedNet.t7'

trainer = Trainer(model, loss_fn, optimizer, cpu_count, mean_global, px, model_name, torch_dtype)
trainer.fit(train, ytrain, valid, yvalid, batch_size, epoch_partion, epochs_num, 4)
