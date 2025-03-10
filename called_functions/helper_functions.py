import pandas as pd
import numpy as np
from testing_data_in import *
from scipy.stats import gaussian_kde

def rev_c(num):
    if(num < .5):
        return 1 - num
    return num

id_df =pd.read_csv('needed_data/NCAAM_KEYS.csv').set_axis(labels = ['id', 'team'], axis = 1)

def get_score(t1, t2):
    score = 0
    j = 0
    weight = 0
    round_cor = [] * 7
    ret_list = []
    for i in range(0, len(t1)):
        if(i < 32):
            weight = 10
            j = 0
        elif(i < 48):
            weight = 20
            j = 1
        elif(i < 56):
            weight = 40
            j = 2
        elif(i < 60):
            weight = 80
            j = 3
        elif(i < 62):
            weight = 160
            j = 4
        elif(i < 63):
            weight = 160
            j = 5
        else:
            weight = 160
            j = 6
        if(t1[i] == t2[i]):
            score += weight
            round_cor[j] += 1

        return [score] + round_cor

def sim_win_games(model, testing_data, year, model_type, parameters, progress_callback):
    import math
    winner_ids = pd.DataFrame()
    round_scores = pd.DataFrame()
    other_stats = pd.DataFrame()
    while(len(testing_data) >= 1):
        if(progress_callback):
            progress_callback(94, f'Round: {7 - math.log2(len(testing_data))}')
        #print(testing_data.columns.tolist())
        c = model.predict(testing_data)

        if('RandomForest' in str(model)):
            tree_preds3 = np.array([tree.predict(testing_data.values) for tree in model.estimators_])
            kdes = [gaussian_kde(tp.flatten()) for tp in tree_preds3.T]
            samples = [yur.resample(1000)[0] for yur in kdes]
            c = pd.DataFrame(samples).mean(axis = 1)
        
        b = pd.DataFrame(pd.concat([testing_data.id_COACH_T1[c > .5], testing_data.id_COACH_T2[c < .5]]).sort_index()).set_axis(labels = ['id'], axis = 1)
        c = pd.Series(c).apply(rev_c)
        other_stats = pd.concat([other_stats, c], axis = 0, ignore_index = True)
        winner_ids = pd.concat([winner_ids, b], axis = 0, ignore_index = True)
        #print(b)
        #break
        if(len(b) == 1):
            break
        if(len(b) > 1):
            g = pd.DataFrame(b.values.reshape(len(b)//2, 2)).set_axis(labels = ['Team1_id', 'Team2_id'], axis = 1)
        
            g['Year'] = year
        
            testing_data = create_teams(g, 7, 'winner', 'sim')[parameters]
            #print(testing_data['id_COACH_T1'])
    winning_teams = [id_df['team'][id_df['id'] == str(int(x))].iloc[0] for x in winner_ids.set_axis(labels = ['id'], axis = 1)['id']]
    retdf = pd.concat([other_stats, pd.Series(winning_teams)], axis = 1).set_axis(labels = ['prob', 'Team'], axis = 1)
    return retdf

def analyze_simulation_results(df):
    results = {}
    
    for index, row in df.iterrows():
        team1_samples = np.array(row['T1'])
        team2_samples = np.array(row['T2'])
        
        avg_team1 = np.mean(team1_samples)
        avg_team2 = np.mean(team2_samples)
        odds_t1_gt_t2 = np.mean(team1_samples > team2_samples)
        if(odds_t1_gt_t2 <= .5):
            odds_t1_gt_t2 = 1-odds_t1_gt_t2
        avg_difference = np.mean(team1_samples - team2_samples)
        
        results[index] = {
            "S1": avg_team1,
            "S2": avg_team2,
            "Win_Prob": odds_t1_gt_t2,
            "Spread": avg_difference
        }
    
    return pd.DataFrame.from_dict(results, orient='index')

def sim_games_score(model, testing_data, year, model_type, parameters, progress_callback):
    import math
    testing_data = pd.DataFrame({col: [val for pair in zip(testing_data[:len(testing_data)//2].reset_index(drop=True)[col], testing_data[len(testing_data)//2:].reset_index(drop=True)[col]) for val in pair] for col in testing_data.columns})#z = testing_data['id_COACH_T1']
    winner_ids = pd.DataFrame()
    round_scores = pd.DataFrame()
    other_stats = pd.DataFrame()
    while(len(testing_data) >= 2):
        if(progress_callback):
            progress_callback(95, f'Round: {8 - math.log2(len(testing_data))}')
        if('RandomForest' in str(model)):
            tree_preds3 = np.array([tree.predict(testing_data.values) for tree in model.estimators_])
            kdes = [gaussian_kde(tp.flatten()) for tp in tree_preds3.T]
            samples = [yur.resample(1000)[0] for yur in kdes]
            c = pd.DataFrame([samples[::2], samples[1::2]]).T.set_axis(labels = ['T1', 'T2'], axis = 1)
            c = analyze_simulation_results(c)
            t1 = testing_data['id_COACH_T1'].loc[::2].reset_index(drop = True)[c['S1'] > c['S2']].rename(index = 'id')
            t2 = testing_data['id_COACH_T1'].loc[1::2].reset_index(drop = True)[c['S1'] <= c['S2']].rename(index = 'id')
        else:
            a = model.predict(testing_data)
            b = a[::2]
            c = a[1::2]
            
            t1 = testing_data['id_COACH_T1'].loc[::2].reset_index(drop = True)[b > c].rename(index = 'id')
            t2 = testing_data['id_COACH_T1'].loc[1::2].reset_index(drop = True)[c >= b].rename(index = 'id')
            c = pd.DataFrame(data = [b, c]).T.set_axis(axis = 1, labels = ['S1', 'S2'])
            c['Spread'] = c['S1'] - c['S2']

        b = pd.concat([t1, t2], axis = 1).set_axis(labels = ['id1', 'id2'], axis =1)
        b['id1'] = b['id1'].where(b['id1'].notna(), b['id2'])
        b['id2'] = b['id2'].where(b['id2'].notna(), b['id1'])
        
        b = b.drop(columns = ['id2'], axis = 1).sort_index().reset_index(drop = True)
        other_stats = pd.concat([other_stats, c], axis = 0, ignore_index = True)
        winner_ids = pd.concat([winner_ids, b], axis = 0, ignore_index = True)
        #print(b)
        #break
        if(len(b) == 1):
            break
        if(len(b) > 1):
            g = pd.DataFrame(b.values.reshape(len(b)//2, 2)).set_axis(labels = ['Team1_id', 'Team2_id'], axis = 1)
        
            g['Year'] = year
        
            testing_data = create_teams(g, 7, 'score', 'sim')[parameters]
            testing_data = pd.DataFrame({col: [val for pair in zip(testing_data[:len(testing_data)//2].reset_index(drop=True)[col], testing_data[len(testing_data)//2:].reset_index(drop=True)[col]) for val in pair] for col in testing_data.columns})#z = testing_data['id_COACH_T1']
            #print(testing_data['id_COACH_T1'])
    winning_teams = [id_df['team'][id_df['id'] == str(int(x))].iloc[0] for x in winner_ids.set_axis(labels = ['id'], axis = 1)['id']]
    retdf = pd.concat([other_stats, pd.Series(winning_teams)], axis = 1)
    if('RandomForest' in str(model)):
        retdf.columns = ['S1', 'S2', 'Win_Prob', 'Spread', 'Team']
    else:
        retdf.columns = ['S1', 'S2', 'Spread', 'Team']

    return retdf

def fix_parameters(parameters):
    parameters = list(dict.fromkeys(parameters))
    parameters.append('id_COACH')
    output_param = []
    for param in parameters:
        if(param.split('_')[-1] == 'D'):
            output_param.append(param)
        elif(param.split('_')[-1] == 'SD'):
            output_param.append(param)
        else:
            output_param.append(param + '_T1')
            output_param.append(param + '_T2')
    return output_param


