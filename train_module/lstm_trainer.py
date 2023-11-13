
import os

import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import random
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from multiprocessing import cpu_count
from sklearn.metrics import balanced_accuracy_score

class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_stacked_lstm, num_lin_layers, dropout_prob, use_relus, use_batchnormalization):
        super().__init__()
        self.num_lin_layers = num_lin_layers
        self.use_batchnormalization = use_batchnormalization
        self.use_relus = use_relus
        self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_stacked_lstm, batch_first=True)
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_dim,hidden_dim) for _ in range(num_lin_layers)])

        if self.use_batchnormalization:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(42) for _ in range(num_lin_layers + 1)])  # +1 for the LSTM layer output
        #self.relu_layers = nn.ModuleList([nn.ReLU() for _ in range(num_relu_layers)])
        self.dropout = nn.Dropout(dropout_prob)
        
        # Output Layer for next_day
        self.output_next_day = nn.Linear(hidden_dim, 1) 
        
        # Output Layer for time_to_end
        self.output_time_to_end = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        #print("before lstm:",x.shape)
        x, (hn, cn) = self.rnn(x)
        #print("after lstm:",x.shape)
        # Applying BatchNorm after LSTM layer
        if self.use_batchnormalization:
            x = self.bn_layers[0](x)
            #print("after bn:",x.shape)
        x = self.dropout(x)
        #print("after dropout:",x.shape)

        for i in range(0,self.num_lin_layers):
            x = self.linear_layers[i](x)
            # Applying BatchNorm after each Linear layer
            if self.use_batchnormalization:
                x = self.bn_layers[i + 1](x)
            if self.use_relus:
                x = F.relu(x)
            x = self.dropout(x)

        out_next_day = torch.sigmoid(self.output_next_day(x[:, -1, :]))
        out_time_to_end = self.output_time_to_end(x[:, -1, :])
        
        return out_next_day, out_time_to_end


class CustomLoss(nn.Module):
    def __init__(self, lamb=0.5, is_tuned=True):
        super(CustomLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.lamb = lamb
        self.is_tuned = is_tuned

    def forward(self, pred_next_day, pred_time_to_end, next_day, rest_lot_in_days, censored, next_day_tuned, rest_lot_in_days_tuned):
        if self.is_tuned:
            next_day = next_day_tuned.long()
            rest_lot_in_days = rest_lot_in_days_tuned.float()
        else:
            next_day = next_day.long()
            rest_lot_in_days = rest_lot_in_days.float()
        censored = censored.bool()
        pred_time_to_end = pred_time_to_end.squeeze()
        loss_next_day = self.cross_entropy(pred_next_day, next_day)

        mask = ~censored
        num_uncensored = mask.sum().float()
        total_samples = float(mask.numel())
        
        if num_uncensored > 0: 
            weighted_rmse = (num_uncensored / total_samples) * torch.sqrt(self.mse(pred_time_to_end[mask], rest_lot_in_days[mask]))
        else:
            weighted_rmse = 0.0
            
        total_loss = (1 - self.lamb) * loss_next_day + self.lamb * weighted_rmse
        
        return total_loss



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


ID_COLS = ['ID','series_id','measurement_number']

def create_grouped_array(data, group_col='series_id', drop_cols=ID_COLS):
    X_grouped = np.row_stack([group.drop(columns=drop_cols).values[None] for _, group in data.groupby(group_col)])
    return X_grouped

def create_dataset(X, y, dropcols=ID_COLS, seed:int=None):
    print('Using device:', device)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    next_day = y['next_day'].astype(int).values
    censored = y['censored'].astype(int).values
    next_day_tuned = y['next_day_tuned'].astype(int).values
    rest_lot_in_days = y['rest_lot_in_days'].astype(float).values
    rest_lot_in_days_tuned = y['rest_lot_in_days_tuned'].astype(float).values
    

    stacked_arrays = np.column_stack((next_day, censored, next_day_tuned, rest_lot_in_days, rest_lot_in_days_tuned))
    y = torch.from_numpy(stacked_arrays)
    y[:, :3] = y[:, :3].to(dtype=torch.long)
    X_grouped = create_grouped_array(X, drop_cols=dropcols)
    X = torch.tensor(X_grouped, dtype=torch.float32)
    ds = TensorDataset(X, y)
    return ds
 
def train_lstm(database:str = 'mimic', 
               fast = True, 
               seed:int = 42, 
               has_microbiology = True,
               use_censored= True,
               lookback = 7, 
               aggregated_hours = 4, 
               inc_ab=False, 
               num_lin_layers = 1, 
               num_stacked_lstm = 1,
               hidden_dim = 128,
               dropout_prob = 0.3, 
               lamb = 0, 
               is_tuned = False, 
               lr = 0.01,
               bs = 64,
               use_relus = False, 
               use_batchnormalization = False):
    path = "data/model_input/lstm/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/lookback_"+str(lookback)+"/aggregated_hours_"+str(aggregated_hours)+"/seed_"+str(seed)+"/"
    res_path = "data/results/lstm/"+database+"/microbiology_res_"+str(has_microbiology)+"/ab_"+str(inc_ab)+"/use_censored_"+str(use_censored)+"/lookback_"+str(lookback)+"/aggregated_hours_"+str(aggregated_hours)+"/seed_"+str(seed)+'/'+ \
               "dropout_"+str(dropout_prob).replace('.','-')+'/'+"lambda_"+str(lamb).replace('.','-')+'/'+"num_lin_layers_"+str(num_lin_layers)+'/' + \
               "num_stacked_lstm_"+str(num_stacked_lstm)+"/hidden_dim_"+str(hidden_dim)+"/lr_"+str(lr).replace('.','-')+"/bs_"+str(bs)+ \
               "/is_tuned_"+ str(is_tuned) + "/use_relus_"+ str(use_relus) + "/use_bn_"+ str(use_batchnormalization) + '/'


    if not os.path.exists(res_path):
        os.makedirs(res_path)

    print('Using device:', device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    X_lstm_train = pd.read_parquet(path+"X_lstm_train.parquet")
    y_lstm_train = pd.read_parquet(path+"y_lstm_train.parquet")
    
    X_lstm_validation = pd.read_parquet(path+"X_lstm_validation.parquet")
    y_lstm_validation = pd.read_parquet(path+"y_lstm_validation.parquet")

    if use_censored == False:
        print("censored excluded")
        print("X before: ", X_lstm_train.shape)
        uncensored_series_ids = y_lstm_train[y_lstm_train['censored'] == False]['series_id'].unique()
        X_lstm_train = X_lstm_train[X_lstm_train['series_id'].isin(uncensored_series_ids)]
        y_lstm_train = y_lstm_train[y_lstm_train['censored'] == False]

        uncensored_series_ids = y_lstm_validation[y_lstm_validation['censored'] == False]['series_id'].unique()
        X_lstm_validation = X_lstm_validation[X_lstm_validation['series_id'].isin(uncensored_series_ids)]
        y_lstm_validation = y_lstm_validation[y_lstm_validation['censored'] == False]
        print("X after: ", X_lstm_train.shape)

    if fast:
        print("ATTENTION: fast mode acitvated")
        sample = list(y_lstm_train.sample(200, random_state=seed)['series_id'])
        X_lstm_train = X_lstm_train[X_lstm_train['series_id'].isin(sample)]
        y_lstm_train = y_lstm_train[y_lstm_train['series_id'].isin(sample)]

        sample = list(y_lstm_validation.sample(200, random_state=seed)['series_id'])
        X_lstm_validation = X_lstm_validation[X_lstm_validation['series_id'].isin(sample)]
        y_lstm_validation = y_lstm_validation[y_lstm_validation['series_id'].isin(sample)]

        patience, trials = 10, 0
    else:
        patience, trials = 100, 0
    
    trn_ds = create_dataset(X_lstm_train, y_lstm_train, seed=seed)
    val_ds = create_dataset(X_lstm_validation, y_lstm_validation, seed=seed)
    

    trn_dl = DataLoader(trn_ds, batch_size = bs, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size = bs, shuffle=False, num_workers=4)
    
    n_epochs = 1000
    best_balanced_accuracy = 0
    
    #print("numebr of features ", X_lstm_train.shape[1]-3)
    model = LSTMClassifier(input_dim = X_lstm_train.shape[1]-3,  # number of features
                           hidden_dim = hidden_dim, 
                           num_stacked_lstm = num_stacked_lstm, 
                           num_lin_layers = num_lin_layers, 
                           dropout_prob = dropout_prob,
                           use_relus=use_relus,
                           use_batchnormalization=use_batchnormalization)

    #print(model)

    model = model.to(device)
    loss_function = CustomLoss(lamb=lamb, is_tuned=is_tuned)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=0.1, patience=10, verbose=True)

    print('Start model training')
    training_losses = []
    validation_losses = []

    for epoch in range(1, n_epochs + 1):
        epoch_train_loss = []
        epoch_valid_loss = 0

        for i, (x_batch, y_batch) in enumerate(trn_dl):
            #print("x_batch shape ",x_batch.shape)
            model.train()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            opt.zero_grad()
            pred_next_day, pred_time_to_end = model(x_batch.to(device))
            next_day = y_batch[:, 0]
            censored = y_batch[:, 1]
            next_day_tuned = y_batch[:, 2]
            rest_lot_in_days = y_batch[:, 3]
            rest_lot_in_days_tuned = y_batch[:, 4]
            loss = loss_function(pred_next_day, pred_time_to_end, next_day, rest_lot_in_days, censored, next_day_tuned, rest_lot_in_days_tuned)
            #epoch_train_loss += loss.item()
            epoch_train_loss.append(loss.item())
            loss.backward()
            opt.step()

        avg_loss = sum(epoch_train_loss) / len(epoch_train_loss)
        sched.step(avg_loss)

        training_losses.append(avg_loss)

        model.eval()
        correct, total = 0, 0
        predictions_list = []
        true_labels_list = []
        for x_val, y_val in val_dl:
            x_val = x_val.to(device)
            y_val = y_val.to(device)
            pred_next_day, pred_time_to_end = model(x_val)

            # loss for plot
            next_day = y_val[:, 0]
            censored = y_val[:, 1]
            next_day_tuned = y_val[:, 2]
            rest_lot_in_days = y_val[:, 3]
            rest_lot_in_days_tuned = y_val[:, 4]
            val_loss = loss_function(pred_next_day, pred_time_to_end, next_day, rest_lot_in_days, censored, next_day_tuned, rest_lot_in_days_tuned)
            epoch_valid_loss += val_loss.item()

            if is_tuned:
                y_val = y_val[:, 2]
            else:
                y_val = y_val[:, 0]
            preds = F.log_softmax(pred_next_day, dim=1).argmax(dim=1)
            total += y_val.size(0)
            correct += (preds == y_val).sum().item()
            predictions_list.append(preds.cpu().numpy())
            true_labels_list.append(y_val.cpu().numpy())

            

        validation_losses.append(epoch_valid_loss / len(val_dl))

        acc = correct / total

        balanced_accuracy = balanced_accuracy_score(np.concatenate(true_labels_list), np.concatenate(predictions_list))

        if epoch % 5 == 0:
            print(f'Epoch: {epoch:3d}. Loss: {loss.item():.4f}. Acc.: {acc:2.2%}. Bal. Acc.: {best_balanced_accuracy:2.2%}')


        if balanced_accuracy > best_balanced_accuracy:
            trials = 0
            best_balanced_accuracy = balanced_accuracy
            torch.save(model.state_dict(), res_path+'best.pth')
            print(f'Epoch {epoch} best model saved with balanced accuracy: {best_balanced_accuracy:2.2%}')
        else:
            trials += 1
            if trials >= patience:
                print(f'Early stopping on epoch {epoch}')
                break
    
    plt.plot(training_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Losses over Epochs")
    plt.legend()
    plt.savefig(res_path + "losses_over_epochs.png")
    plt.close()
