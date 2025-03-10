import pandas as pd
import sklearn
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import torch.nn as nn
import torch.optim as optim
import random

from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

import re

pd.set_option('mode.chained_assignment', None)
pd.options.display.max_rows = 999

import os
import numpy as np 
import pandas as pd 
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix
from sklearn.metrics import *

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeClassifierCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor

#-- Pytorch specific libraries import -----#
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from xgboost import XGBClassifier
from xgboost import XGBRegressor

import warnings
from testing_data_in import *
from helper_functions import *
from xgboost.callback import TrainingCallback

    
warnings.simplefilter(action='ignore', category=FutureWarning)
import random

def train_random_forest_with_progress_classifier(df_data, hyperparameters, model_num, progress_callback=None):
    n_estimators = hyperparameters['n_estimators']
    n_increment = 5  # Number of trees to add in each increment
    model = RandomForestClassifier(n_estimators=0, warm_start=True, criterion=hyperparameters['criterion'], max_depth=hyperparameters['max_depth'])

    X = df_data.iloc[:, :-1]
    y = df_data['Winner']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=random.randint(20, 150), test_size=0.2)

    for i in range(0, n_estimators, n_increment):
        model.n_estimators += n_increment
        model.fit(train_x, train_y)

        if progress_callback:
            base_progress = 20 + (model_num / (hyperparameters['num_models'] - 1)) * 70  # Where this model starts
            model_progress = (i + n_increment) / n_estimators  # Progress within this model
            current_progress = round(base_progress + (70 / hyperparameters['num_models']) * model_progress, 2)
            progress_callback(current_progress, f"Training Random Forest: {model.n_estimators} trees out of {n_estimators} for model {model_num + 1} of {hyperparameters['num_models']}")
    y_pred = model.predict(test_x)
    score = sklearn.metrics.accuracy_score(test_y, y_pred)

    return model, score

def train_random_forest_with_progress_regressor(df_data, hyperparameters, model_num, progress_callback=None):
    n_estimators = hyperparameters['n_estimators']
    n_increment = 5  # Number of trees to add in each increment
    model = RandomForestRegressor(n_estimators=0, warm_start=True, criterion=hyperparameters['criterion'], max_depth=hyperparameters['max_depth'])

    X = df_data.iloc[:, :-1]
    y = df_data['Score']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=random.randint(20, 150), test_size=0.2)

    for i in range(0, n_estimators, n_increment):
        model.n_estimators += n_increment
        model.fit(train_x, train_y)

        if progress_callback:
            base_progress = 20 + (model_num / (hyperparameters['num_models'] - 1)) * 70  # Where this model starts
            model_progress = (i + n_increment) / n_estimators  # Progress within this model
            current_progress = round(base_progress + (70 / hyperparameters['num_models']) * model_progress, 2)
            progress_callback(current_progress, f"Training Random Forest: {model.n_estimators} trees out of {n_estimators} for model {model_num + 1} of {hyperparameters['num_models']}")
    y_pred = model.predict(test_x)
    score = sklearn.metrics.mean_absolute_error(test_y, y_pred)

    return model, score

def train_gradient_boosting_with_progress_regression(df_data, hyperparameters, model_num, progress_callback=None):
    if(progress_callback):
        progress_callback(round(20 + 70 *((1 + model_num ) / hyperparameters['num_models'])), f"Training Gradient Boosting Model {model_num + 1} of {hyperparameters['num_models']}")
    n_estimators = hyperparameters['n_estimators']
    model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=hyperparameters['learning_rate'], max_depth=hyperparameters['max_depth'], loss=hyperparameters['loss_function'], criterion=hyperparameters['criterion'], subsample=hyperparameters['subsample'], min_samples_split=hyperparameters['min_samples_split'])

    X = df_data.iloc[:, :-1]
    y = df_data['Score']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=random.randint(20, 150), test_size=0.2)

    model.fit(train_x, train_y)
    print('Fit model')
    for i, y_pred in enumerate(model.staged_predict(test_x)):
        print('reached here?')
        if progress_callback:
            progress_callback(round(20 + (70 * (i + 1) / n_estimators) * ((1 + model_num) / hyperparameters['num_models']), 2), f"Training Gradient Boosting: {i + 1} trees out of {n_estimators} for model {model_num + 1} of {hyperparameters['num_models']}")
        score = sklearn.metrics.mean_absolute_error(test_y, y_pred)

    final_score = sklearn.metrics.mean_absolute_error(test_y, model.predict(test_x))
    return model, final_score

def train_gradient_boosting_with_progress_classification(df_data, hyperparameters, model_num, progress_callback=None):
    if(progress_callback):
        progress_callback(round(20 + 70 *((1 + model_num ) / hyperparameters['num_models'])), f"Training Gradient Boosting Model {model_num + 1} of {hyperparameters['num_models']}")
    n_estimators = hyperparameters['n_estimators']
    model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=hyperparameters['learning_rate'], max_depth=hyperparameters['max_depth'], loss=hyperparameters['loss_function'], criterion=hyperparameters['criterion'], subsample=hyperparameters['subsample'], min_samples_split=hyperparameters['min_samples_split'])

    X = df_data.iloc[:, :-1]
    y = df_data['Winner']

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=random.randint(20, 150), test_size=0.2)

    model.fit(train_x, train_y)

    for i, y_pred in enumerate(model.staged_predict(test_x)):
        if progress_callback:
            progress_callback(round(20 + (70 * (i + 1) / n_estimators) * ((1 + model_num) / hyperparameters['n_estimators']), 2), f"Training Gradient Boosting: {i + 1} trees out of {n_estimators} for model {model_num + 1} of {hyperparameters['num_models']}")
        score = sklearn.metrics.mean_absolute_error(test_y, y_pred)

    final_score = sklearn.metrics.mean_absolute_error(test_y, model.predict(test_x))

    return model, final_score


def main_distribution(hyperparameters, model_type, method, year, parameters, progress_callback=None):
    if progress_callback:
        progress_callback(5, "Processing data")
    print('yurrr')
    print(hyperparameters, model_type, method, year)
    parameters = fix_parameters(parameters)
    if(method == 'winner'):
        if progress_callback:
            progress_callback(10, "Creating Datasets")
        print(len(normal_win_games))
        training_games = normal_win_games[normal_win_games['Year'] != year].reset_index(drop = True)
        print(len(training_games))
        testing_games = normal_win_games[normal_win_games['Year'] == year].reset_index(drop = True).iloc[:32]
        #print(testing_games.shape)
        #print(testing_games.head())

        df_data = create_teams(training_games, 7, 'winner', 'initial')
        testing_data = create_teams(testing_games, 7, 'winner', 'initial').iloc[:, :-1]
        
        testing_data = testing_data[parameters]
        parameters.append('Winner')
        df_data = df_data[parameters]
        parameters.remove('Winner')
        if progress_callback:
            progress_callback(20, "Finished Creating Datasets")
        #print(testing_data.columns.to_list())
        #print(df_data.columns.to_list())

        if model_type == "neural_network":
            model = CNN_Classifier(df_data, testing_data, hyperparameters)
        elif model_type == 'decision_tree':
            model = DecisionTreeClassifier(max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split'], criterion= hyperparameters['criterion'])
        elif model_type == 'k_nearest':
            model = KNeighborsClassifier(n_neighbors=hyperparameters['n_neighbors'], weights=hyperparameters['k_weights'])
        elif model_type == 'linear_regression':
            model = RidgeClassifierCV(fit_intercept=np.bool(hyperparameters['fit_intercept']))
        best_score = 0
        best_model = None

        if(model_type != 'neural_network'):
            for i in range(0, hyperparameters['num_models']):
                if(model_type == 'random_forest'):
                    model, score = train_random_forest_with_progress_classifier(df_data, hyperparameters, i, progress_callback)
                elif model_type == 'xg_boost':
                    model, score = train_gradient_boosting_with_progress_classification(df_data, hyperparameters, i, progress_callback)
                else:
                    if progress_callback:
                        progress_callback(round(20 + 80 * (i + 1) / (hyperparameters['num_models'] + 1), 2), f"Creating Model {i + 1} out of {hyperparameters['num_models']}")

                    X= df_data.iloc[: , :-1]
                    y= df_data['Winner']

                    train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=random.randint(20, 150),test_size = .2)

                    model.fit(train_x, train_y)
                    y_pred = model.predict(test_x)

                    score = sklearn.metrics.accuracy_score(test_y, y_pred)
                if(score > best_score):
                    best_score = score
                    best_model = model
            progress_callback(90, "Simulating Games")
            winning_teams = sim_win_games(best_model, testing_data, year, model_type, parameters, progress_callback)
            progress_callback(98, 'Finishing Things Up')

            if('RandomForest' in str(model)):
                winning_teams[['prob']] = winning_teams[['prob']].round(2) * 100
            else:
                winning_teams[['prob']] = None
            output_dict = {'winners':list(winning_teams['Team']), 'probabilities':list(winning_teams['prob'])}
        return output_dict, best_model





    elif(method == 'score'):
        if progress_callback:
            progress_callback(15, "Creating Datasets")
        training_games = normal_score_games[normal_score_games['Year'] != year].reset_index(drop = True)
        testing_games = normal_score_games[normal_score_games['Year'] == year].reset_index(drop = True).iloc[:32]

        training_data = create_teams(training_games, 7, 'score', 'initial')
        testing_data = create_teams(testing_games, 7, 'score', 'initial').iloc[:, :-1]

        testing_data = testing_data[parameters]
        #print(testing_data.columns.to_list())
        parameters.append('Score')
        training_data = training_data[parameters]
        parameters.remove('Score')
        #print('Training Columns')
        #print(training_data.columns.to_list())
        if progress_callback:
            progress_callback(20, "Finished Creating Datasets")
        if model_type == 'decision_tree':
            model = DecisionTreeRegressor(max_depth=hyperparameters['max_depth'], min_samples_split=hyperparameters['min_samples_split'], criterion= hyperparameters['criterion'])
        elif model_type == 'linear_regression':
            model = LinearRegression(fit_intercept=np.bool(hyperparameters['fit_intercept']))
        elif model_type == 'k_nearest':
            model = KNeighborsRegressor(n_neighbors=hyperparameters['n_neighbors'], weights=hyperparameters['k_weights'])
        elif model_type == 'neural_network':
            model = CNN_Classifier(df_data, training_games, hyperparameters)
                
        best_score = 100
        best_model = None

        if(model_type != 'neural_network'):
            for i in range(0, hyperparameters['num_models']):
                if(model_type == 'random_forest'):
                    model, score = train_random_forest_with_progress_regressor(training_data, hyperparameters, i, progress_callback)
                    print('Random Forest fuck me')
                elif model_type == 'xg_boost':
                    model, score = train_gradient_boosting_with_progress_regression(training_data, hyperparameters, i, progress_callback)
                else:
                    if progress_callback:
                        progress_callback(round(20 + 80 * (i + 1) / (hyperparameters['num_models']), 0), f"Creating Model {i + 1} out of {hyperparameters['num_models']}")

                    X= training_data.iloc[: , :-1]
                    y= training_data['Score']

                    train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=random.randint(20, 150),test_size = .2)

                    model.fit(train_x, train_y)
                    y_pred = model.predict(test_x)

                    score = sklearn.metrics.mean_absolute_error(test_y, y_pred)
                if(score < best_score):
                    best_score = score
                    best_model = model
            progress_callback(90, "Simulating Games")
            winning_teams = sim_games_score(best_model, testing_data, year, model_type, parameters, progress_callback)
            progress_callback(98, 'Finishing Things Up')
        winning_teams[['S1', 'S2', 'Spread']] = winning_teams[['S1', 'S2', 'Spread']].round(1)
        if('RandomForest' in str(model)):
            winning_teams[['Win_Prob']] = winning_teams[['Win_Prob']].round(2) * 100
        else:
            winning_teams[['Win_Prob']] = None

        t1_id = np.arange(0, len(winning_teams) * 2, 2)
        t2_id = np.arange(1, len(winning_teams) * 2, 2)
        a1 = pd.DataFrame(winning_teams['S1'].values, t1_id).set_axis(labels = ['Score'], axis = 1)
        a2 = pd.DataFrame(winning_teams['S2'].values, t2_id).set_axis(labels = ['Score'], axis = 1)
        final = pd.concat([a1, a2], axis = 0).sort_index()
        #print(final)

        output_dict = {'winners':list(winning_teams['Team']), 'probabilities':list(winning_teams['Win_Prob']), 'Score':list(final.values), 'spread':list(winning_teams['Spread'])}
    
        return output_dict, best_model
    


def make_uniform_win(testing):
    def swap_teams(row):
        mid = (len(row) - 1) // 2  # Finding midpoint (excluding winner column)
        team1 = row[:mid]  # Team 1 data
        team2 = row[mid:-1]  # Team 2 data
        winner = int(row[-1])  # Winner column
    
        if random.random() > 0.5:
            # Keep the original order
            new_row = list(team1) + list(team2) + [winner]
        else:
            # Swap team order and invert winner
            new_row = list(team2) + list(team1) + [1 - winner]
    
        return pd.Series(new_row, index=testing.columns)
    
    # Apply the function to all rows
    return testing.apply(swap_teams, axis=1)

def CNN_Classifier(df_data, games_data, hyperparameters):
    #df_data = pd.read_csv('FINAL_INPUT_DATA_2025.csv').dropna()[:-1]
    epochs = int(hyperparameters['epochs'])
    learning_rate = float(hyperparameters['learning_rate'])
    t_games = games_data.to_numpy()
    #df_data = make_uniform_win(df_data)
    #t_games = pd.read_csv("FINAL_DATA_2024.csv").to_numpy()
    teams = []

    #Train & Test Set
    X= df_data.iloc[:, 1:]
    y= df_data['Winner']

    train_x,test_x,train_y,test_y = train_test_split(X,y,random_state=random.randint(20, 150), test_size= .1)
    #print("\n--Training data samples--")
    #print(train_x.shape)
    print(test_x.shape)
    scaler = preprocessing.MinMaxScaler()

    x_train = scaler.fit_transform(train_x.values)
    x_test =  scaler.fit_transform(test_x.values)
    x_tensor =  torch.from_numpy(x_train).float()
    y_tensor =  torch.from_numpy(train_y.values.ravel()).float()
    xtest_tensor =  torch.from_numpy(x_test).float()
    ytest_tensor =  torch.from_numpy(test_y.values.ravel()).float()

    scal_model = scaler.fit(train_x.values)

    #Define a batch size , 
    bs = 16
    #Both x_train and y_train can be combined in a single TensorDataset, which will be easier to iterate over and slice
    y_tensor = y_tensor.unsqueeze(1)
    train_ds = TensorDataset(x_tensor, y_tensor)
    #Pytorch’s DataLoader is responsible for managing batches. 
    #You can create a DataLoader from any Dataset. DataLoader makes it easier to iterate over batches
    train_dl = DataLoader(train_ds, batch_size=bs)


    #For the validation/test dataset
    ytest_tensor = ytest_tensor.unsqueeze(1)
    test_ds = TensorDataset(xtest_tensor, ytest_tensor)
    test_loader = DataLoader(test_ds, batch_size=16)

    n_dim = train_x.shape[1]

    #Layer size
    n_hidden1 = 512  # Number of hidden nodes
    n_hidden2 = 256
    n_hidden3 = 128
    n_hidden4 = 64
    n_output =  1   # Number of output nodes = for binary classifier


    class GamesModel(nn.Module):
        def __init__(self):
            super(GamesModel, self).__init__()
            
            self.layer_1 = nn.Linear(n_dim, n_hidden1) 
            self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
            self.layer_3 = nn.Linear(n_hidden2, n_hidden3)
            self.layer_4 = nn.Linear(n_hidden3, n_hidden4)
            self.layer_5 = nn.Linear(n_hidden4, n_hidden4)
            
            self.layer_out = nn.Linear(n_hidden4, n_output) 
            
            
            self.relu = nn.LeakyReLU(negative_slope = .03)
            self.sigmoid =  nn.Sigmoid()
            self.dropout = nn.Dropout(p=0.5)
            self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
            self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
            self.batchnorm3 = nn.BatchNorm1d(n_hidden3)
            self.batchnorm4 = nn.BatchNorm1d(n_hidden4)
            
            
        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.relu(self.layer_3(x))
            x = self.batchnorm3(x)
            x = self.dropout(x)
            x = self.relu(self.layer_4(x))
            x = self.sigmoid(self.layer_5(x))
            x = self.batchnorm4(x)
            x = self.dropout(x)
            x = self.batchnorm4(x)
            x = self.sigmoid(self.layer_out(x))
            
            return x
        

    model = GamesModel()


    loss_func = nn.BCELoss()
#Optimizer
    #learning_rate = 0.00003
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    #epochs = 5

    model.train()
    train_loss = []
    for epoch in range(epochs):
        #Within each epoch run the subsets of data = batch sizes.
        for xb, yb in train_dl:
            y_pred = model(xb)            # Forward Propagation
            loss = loss_func(y_pred, yb)  # Loss Computation
            optimizer.zero_grad()         # Clearing all previous gradients, setting to zero 
            loss.backward()               # Back Propagation
            optimizer.step()              # Updating the parameters 
        #print("Loss in iteration :"+str(epoch)+" is: "+str(loss.item()))
        train_loss.append(loss.item())
   #print('Last iteration loss value: '+str(loss.item()))

    y_pred_list = []
    y_test_list = []
    thresh_list = []
    model.eval()
    #Since we don't need model to back propagate the gradients in test set we use torch.no_grad()
    # reduces memory usage and speeds up computation
    m_s = 0
    b_t = 0
    for perc in range(10, 80):
        
        with torch.no_grad():
            for xb_test,yb_test  in test_loader:
                y_test_pred = model(xb_test)
                y_test_list.append(y_test_pred.detach().numpy())
                thresh = np.percentile(np.concatenate(y_test_list),perc)
                y_pred_tag = (y_test_pred>thresh).float()
                y_pred_list.append(y_pred_tag.detach().numpy())

        #Takes arrays and makes them list of list for each batch
        #print(y_pred_list)
        #print(len(y_pred_list))

        y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
        #print(len(y_pred_list))

        #flattens the lists in sequence
        ytest_pred = list(itertools.chain.from_iterable(y_pred_list))
        #print(ytest_pred)
        #print(len(test_y.values.ravel()))
        #print(len(ytest_pred))
        y_true_test = test_y.values.ravel()
        conf_matrix = confusion_matrix(y_true_test ,ytest_pred)
        s = (conf_matrix[0][0] + conf_matrix[1][1])/conf_matrix.sum()
        y_pred_list = []
        y_test_list = []
        if(s > m_s):
            m_s = s
            b_t = thresh
    def sim_games(best_model, upset_weight):
        winning_teams = []
        id_df = pd.read_csv("NCAAM_KEYS.csv")
        id_df.columns = ['id', 'team'] 
        t_games2 = t_games
        ids1 = t_games2['teamId_BPI_T1']
        ids2 = t_games2['teamId_BPI_T2']
        #t_games = pd.read_csv("2024_TEST_DATA.csv").to_numpy()
        scal_model = scaler.fit(train_x.values)
        scal_games = scal_model.transform(t_games)
        cur_games =  torch.from_numpy(scal_games).float()[:32]
        t_games3 = t_games2.copy()
        mid = int((len(t_games3.columns))/2)
        gpr = [32, 16, 8, 4, 2, 1, 1]
        winners = [1, 2]
        roundnum = 1

        while(len(winners) > 1):

            new_df =pd.DataFrame([[float('nan')]*len(t_games2.columns) for _ in range(gpr[roundnum])])
            #print("here?")
            if roundnum == 1:
                t_games2 = t_games3.copy()

            results = (best_model(cur_games)>b_t).float()
            r_b = [int(x[0]) for x in results.tolist()]
            r_b2 = [r ^ 1 if np.random.randint(0, 100)<(10-2*roundnum - upset_weight) else r for r in r_b]
            cur_games = []
            winners = []
            winner_ids = []
            games = pd.DataFrame(scal_games)
            for i in range(len(results)):
                if(r_b2[i] == 1):
                    winner = list(t_games2.iloc[i,:mid])
                    winner_ids.append(ids1[i])
                    winner_id = str(ids1[i])
                    loser_id = str(ids2[i])
                else:
                    winner = list(t_games2.iloc[i,mid:])
                    winner_ids.append(ids2[i])
                    winner_id = str(ids2[i])
                    loser_id = str(ids1[i])
                winning_teams.append(str(ids1[i]))
                winning_teams.append(str(ids2[i]))
                winners.append(winner)
                comb = winner_id + "_" + loser_id

            t_games2 = new_df
            if(len(winners) == 1):
                break
            for i in range(0, len(winners), 2):
                #print(i)
                cur_game = winners[i] + winners[i+1]
    
                cur_games.append(cur_game)
            ids1 = winner_ids[::2]
            ids2 = winner_ids[1::2]
            cur_games = np.array(cur_games, dtype = np.float32, ndmin = 2) 
            cur_games = scal_model.transform(cur_games)
            cur_games =  torch.from_numpy(cur_games).float()
            roundnum = roundnum+1
            #winning_teams = winning_teams + winner_ids
            #print(len(winning_teams))
        winning_teams = [id_df['team'][id_df['id'] == str(x)].iloc[0] for x in winning_teams]
        #print(winning_teams)
        return winning_teams
    teams = sim_games(model, 0)
    print(teams)
    return teams

    
def CNN_Regressor(df_data, games_data, hyperparameters):
    epochs = int(hyperparameters['epochs'])
    learning_rate = float(hyperparameters['learning_rate'])

    teams = []
    # Train & Test Set
    X = df_data.iloc[:, :-1]
    y = df_data.iloc[:, -1]

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=random.randint(20, 150), test_size=0.1)
    print(test_x.shape)

    scaler_x = preprocessing.MinMaxScaler()
    scaler_y = preprocessing.StandardScaler()

    x_train = scaler_x.fit_transform(train_x.values)
    x_test = scaler_x.transform(test_x.values)
    y_train_scaled = scaler_y.fit_transform(train_y.values.reshape(-1, 1))
    y_test_scaled = scaler_y.transform(test_y.values.reshape(-1, 1))

    x_tensor = torch.from_numpy(x_train).float()
    y_tensor = torch.from_numpy(y_train_scaled).float()
    xtest_tensor = torch.from_numpy(x_test).float()
    ytest_tensor = torch.from_numpy(y_test_scaled).float()

    # Define a batch size
    bs = 16
    # Both x_train and y_train can be combined in a single TensorDataset, which will be easier to iterate over and slice
    train_ds = TensorDataset(x_tensor, y_tensor)
    # Pytorch’s DataLoader is responsible for managing batches.
    # You can create a DataLoader from any Dataset. DataLoader makes it easier to iterate over batches
    train_dl = DataLoader(train_ds, batch_size=bs)

    # For the validation/test dataset
    test_ds = TensorDataset(xtest_tensor, ytest_tensor)
    test_loader = DataLoader(test_ds, batch_size=16)

    n_dim = train_x.shape[1]

    # Layer size
    n_hidden1 = 512  # Number of hidden nodes
    n_hidden2 = 256
    n_hidden3 = 128
    n_hidden4 = 64
    n_output = 1  # Number of output nodes = for regression

    class GamesModel(nn.Module):
        def __init__(self):
            super(GamesModel, self).__init__()
            self.layer_1 = nn.Linear(n_dim, n_hidden1)
            self.layer_2 = nn.Linear(n_hidden1, n_hidden2)
            self.layer_3 = nn.Linear(n_hidden2, n_hidden3)
            self.layer_4 = nn.Linear(n_hidden3, n_hidden4)
            self.layer_out = nn.Linear(n_hidden4, n_output)

            self.relu = nn.ELU()
            self.dropout = nn.Dropout(p=0.3)
            self.batchnorm1 = nn.BatchNorm1d(n_hidden1)
            self.batchnorm2 = nn.BatchNorm1d(n_hidden2)
            self.batchnorm3 = nn.BatchNorm1d(n_hidden3)
            self.batchnorm4 = nn.BatchNorm1d(n_hidden4)

        def forward(self, inputs):
            x = self.relu(self.layer_1(inputs))
            x = self.batchnorm1(x)
            x = self.relu(self.layer_2(x))
            x = self.batchnorm2(x)
            x = self.dropout(x)
            x = self.relu(self.layer_3(x))
            x = self.batchnorm3(x)
            x = self.relu(self.layer_4(x))
            x = self.batchnorm4(x)
            x = self.layer_out(x)  # No activation here!
            return x  # Raw output for regression

    model = GamesModel()

    loss_func = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    train_loss = []
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in train_dl:
            y_pred = model(xb)
            loss = loss_func(y_pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(train_dl)}")
        train_loss.append(epoch_loss / len(train_dl))

    print('Last iteration loss value: ' + str(train_loss[-1]))

    y_pred_list = []
    y_test_list = []
    model.eval()

    with torch.no_grad():
        for xb_test, yb_test in test_loader:
            y_test_pred = model(xb_test)
            y_pred_list.extend(y_test_pred.squeeze().tolist())
            y_test_list.extend(yb_test.squeeze().tolist())

    y_pred_real = scaler_y.inverse_transform(np.array(y_pred_list).reshape(-1, 1)).flatten()
    real_input = scaler_y.inverse_transform(np.array(y_test_list).reshape(-1, 1)).flatten()
    mae = sklearn.metrics.mean_absolute_error(y_pred_real, real_input)

    return y_pred_real, real_input