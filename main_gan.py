from networks_gan import rganls
from utils import normalizer, slicing_window
import pandas as pd

####------------------------------------
# Training GAN for data augmentation 
####------------------------------------

for name in ['sp', 'bsh']:

       # read training data for GAN
       if name == 'sp':
              dataset = pd.read_csv('data/sp.csv', header=0, index_col=0, nrows=84)
              dataset = dataset.loc['2008-06':] 
       elif name == 'bsh':
              dataset = pd.read_csv('data/bsh.csv', header=0, index_col=0, nrows=125)
              dataset = dataset.loc['2008-06':] 

       dataset = dataset.reset_index(drop=True)
       dataset = dataset.dropna()
       data = dataset.values

       # normalisation
       normed_data, min_val, max_val = normalizer(data)

       # define time steps for prediction
       n_months = 12
       n_te_months = 24
       n_tr_months = len(dataset)-n_months-n_te_months+1

       # re-split the training set for gan (including the last feature)
       features_gan = slicing_window(normed_data, n_months)
       train_gan_x = features_gan[:n_tr_months, :, :]

       # training rnn gan
       params1  = {"epoch": 3000, "module_name":'gru', "batch_size": 5, 
                   "hidden_size": 16, "num_layer": 1, "z_dim": 5, "lr": 0.0001 }
       
       # list for hyper-parameter tuning, this list can be expanded to match the param grid
       params_list = [params1]

       for params in params_list:

              # experiment name
              expname = 'exp_original_batch5_hidden'+str(params["hidden_size"])+'_layer'+\
                            str(params["num_layer"])+'_zdim'+str(params["z_dim"])+'_lr'+\
                            str(params["lr"])+'_'+str(params["module_name"])
              
              # rnngan training
              rganls(train_gan_x, epochs=params["epoch"], batch_size=params["batch_size"], 
                     hidden_dim=params["hidden_size"], num_layers=params["num_layer"], 
                     module_name=params["module_name"], z_dim=params["z_dim"], 
                     learning_rate=params["lr"], dname=name, expname=expname)
