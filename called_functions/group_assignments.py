from other_info import *
# print(position_dict)
# for dict in position_dict:

#     dict['group2'] = 'group2'
# print(position_dict)
#
import pandas as pd
pd.set_option('display.max_rows', None)
model_df2 = pd.read_csv('final_tourney_games.csv')
def assign_groups(predictions, year):
    
    winners = pd.read_csv('all_correct.csv')
    correct = {'Round of 32': 0, 'Sweet Sixteen': 0, 'Elite Eight': 0, 'Final Four': 0, 'Finals': 0, 'Winner': 0}
    cbs_scoring = {'Round of 32': 5, 'Sweet Sixteen': 8, 'Elite Eight': 13, 'Final Four': 21, 'Finals': 34, 'Winner': 55}
    seed_totals = 0
    winners = winners[winners['Year'] == year]

    for i, dict in enumerate(position_dict):
        if(i < 64):
            dict['Seed'] = int(winners['Seed'].iloc[i])
            dict['Team'] = winners['Team'].iloc[i]
            dict['group'] = 'group-1'
        elif i < 124:
            predictions[i-64] = predictions[i-64].replace('’', "'")
            #print(predictions[i-64])
            seed = winners['Seed'][winners['Team'] == predictions[i-64]].values[0]
            dict['Seed'] = int(seed)
            dict['Team'] = predictions[i-64]
            z = i-64

            if(predictions[z] in predictions[z+1:]):
                if(predictions[i-64] == winners['Team'].iloc[i]):
                    seed_totals += int(seed)
                    z = i-64
                    if(z < 32):
                        correct['Round of 32'] += 1
                    elif(z < 48):
                        correct['Sweet Sixteen'] += 1
                    elif(z < 56):
                        correct['Elite Eight'] += 1
                    elif(z < 60):
                        correct['Final Four'] += 1
                    dict['group'] = 'group-2'
                else:
                    dict['group'] = 'group-5'
            else:
                if(predictions[i-64] == winners['Team'].iloc[i]):
                    seed_totals += int(seed)
                    dict['group'] = 'group-4'
                    z = i-64
                    if(z < 32):
                        correct['Round of 32'] += 1
                    elif(z < 48):
                        correct['Sweet Sixteen'] += 1
                    elif(z < 56):
                        correct['Elite Eight'] += 1
                    elif(z < 60):
                        correct['Final Four'] += 1
                else:
                    dict['group'] = 'group-3'
        elif i < 126:
            z = i-64
            seed = winners['Seed'][winners['Team'] == predictions[i-64]].values[0]
            dict['Seed'] = int(seed)
            dict['Team'] = predictions[i-64]
            if(predictions[z] in predictions[z+1:]):
                if(predictions[i-64] == winners['Team'].iloc[i]):
                    if(z < 62):
                        seed_totals += int(seed)
                        correct['Finals'] += 1
                    dict['group'] = 'group-6'
                else:
                    dict['group'] = 'group-9'
            else:
                if(predictions[i-64] == winners['Team'].iloc[i]):
                    dict['group'] = 'group-8'
                    z = i-64
                    if(z < 62):
                        seed_totals += int(seed)
                        correct['Finals'] += 1
                else:
                    dict['group'] = 'group-7'
        else:
            seed = winners['Seed'][winners['Team'] == predictions[i-64]].values[0]
            dict['Seed'] = int(seed)
            dict['Team'] = predictions[i-64]
            if(predictions[i-64] == winners['Team'].iloc[i]):
                seed_totals += int(seed)
                correct['Winner'] += 1
                dict['group'] = 'group-10'
            else:
                dict['group'] = 'group-11'

    score = 5 * correct['Round of 32'] + 8 * correct['Sweet Sixteen'] + 13 * correct['Elite Eight'] + 21 * correct['Final Four'] + 34 * correct['Finals'] + 55 * correct['Winner']
    score += seed_totals
    
    return position_dict, correct, score
    
def assign_score(scores):
    score_positions2 = score_positions.copy()
    for i, dict in enumerate(score_positions2):
        #print(scores[i][0])
        dict['Score'] = scores[i][0]
        dict['left'] = dict['left'] - 2
    return score_positions2

def assign_prob(probs):
    for i, dict in enumerate(prob_positions):
        dict['Prob'] = probs[i]
    return prob_positions


from called_functions.prediction_functions import *

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

def messing_with_predictions(model, parameters, prediction_method, model_type):
    output_dict = {}
    parameters = fix_parameters(parameters)
    new_z = pd.read_csv('scraping_data/lunardi_projections.csv')

    ff = new_z.dropna().set_axis(labels = ['Team1_id', 'Team2_id'], axis =1).astype(int)
    ff['Year'] = 2025
    ff_input = create_teams(ff, 7, prediction_method, 'sim')
    ff_input = ff_input[parameters]
    winners = model.predict(ff_input)

    if(prediction_method == 'score'):
        scores = pd.DataFrame(winners.reshape(len(winners)//2, 2)).set_axis(labels = ['Score1', 'Score2'], axis = 1)
        score_winners = pd.DataFrame((scores['Score1'] > scores['Score2']).astype(int)).set_index(ff.index).set_axis(labels = ['Winner'], axis = 1)
        ff = pd.concat([ff, score_winners], axis = 1)
    else:
        ff['Winner'] = winners
    
    df_merged = new_z.merge(ff, left_on=['Team1', 'Team2'], right_on=['Team1_id', 'Team2_id'], how='left')
    all_ids = np.array(df_merged['Team1'])

    tourney_games = pd.DataFrame(all_ids.reshape((32, 2)), columns = ['Team1_id', 'Team2_id']).astype(int)
    tourney_games['Year'] = 2025
    tourney_data = create_teams(tourney_games, 7, prediction_method, 'sim')
    tourney_data = tourney_data[parameters]
    if(prediction_method == 'winner'):
        winners = sim_win_games(model, tourney_data, 2025, model_type, parameters)
        if('RandomForest' in str(model)):
            winners[['prob']] = winners[['prob']].round(2) * 100
            output_dict = {'probabilities':list(winners['prob'])}

    else:
        winning_teams = sim_games_score(model, tourney_data, 2025, model_type, parameters)
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
        if('RandomForest' in str(model)):
            output_dict = {'probabilities':list(winning_teams['Win_Prob']), 'Score':list(final.values)}
        else:
            output_dict = {'Score':list(final.values)}
        winners = winning_teams[['Win_Prob', 'Team']]

    seeds = pd.read_csv('scraping_data/lunardi_seeds.csv')
    seeds.columns = ['Seed', 'id']
    winners = winners.merge(id_df).merge(seeds)

    all_ids = pd.DataFrame(all_ids).set_axis(labels = ['id'], axis = 1).merge(seeds).merge(id_df)
    for i, dict in enumerate(position_dict):
        i2 = i - 64
        if(i < 64):
            dict['Seed'] = int(all_ids['Seed'].iloc[i])
            dict['Team'] = all_ids['Team'].iloc[i]
            dict['group'] = 'group-1'

        elif i < 124:
            dict['Seed'] = int(winners['Seed'].iloc[i2])
            dict['Team'] = winners['Team'].iloc[i2]
            dict['group'] = 'group-1'

            
        else:
            dict['Seed'] = int(winners['Seed'].iloc[i2])
            dict['Team'] = winners['Team'].iloc[i2]
            dict['group'] = 'group-12'
        
    return position_dict, output_dict


# all_correct = pd.DataFrame()
# for year in [i for i in range(2011, 2025) if i != 2020]:
#     all = model_df2[['Seed1', 'Team1', 'Seed2', 'Team2', 'Winner']][model_df2['Year'] == year].reset_index(drop=True)
#     all['Not_Winner'] = 1- all['Winner']
#     import numpy as np
#     t1_id = np.arange(0, len(all) * 2, 2)
#     t2_id = np.arange(1, len(all) * 2, 2)
#     a1 = pd.DataFrame(all[['Seed1', 'Team1', 'Winner']].values, t1_id).set_axis(labels = ['Seed', 'Team', 'Winner'], axis = 1)
#     a2 = pd.DataFrame(all[['Seed2', 'Team2', 'Not_Winner']].values, t2_id).set_axis(labels = ['Seed', 'Team', 'Winner'], axis = 1)

#     final = pd.concat([a1, a2], axis = 0).sort_index()
#     final.loc[126] = all[['Seed1', 'Team1']].iloc[-1].set_axis(labels = ['Seed', 'Team']) if all['Winner'].iloc[-1] else all[['Seed2', 'Team2']].iloc[-1].set_axis(labels = ['Seed', 'Team'])
#     final['Year'] = year

#     all_correct = pd.concat([all_correct, final], axis = 0)
# all_correct.to_csv('all_correct.csv', index = False)