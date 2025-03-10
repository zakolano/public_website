import pandas as pd
import requests
import os
#import Requests
import numpy as np
import time
import random
pd.options.display.max_rows = 5000
import re
import requests, io
import numpy as np
import datetime
from io import StringIO
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.ensemble import RandomForestRegressor

id_df =pd.read_csv('needed_data/NCAAM_KEYS.csv').set_axis(labels = ['id', 'team'], axis = 1)
cbbr_mapping = pd.read_csv('needed_data/cbbr_id_mapping.csv')
coachdf = pd.read_csv('needed_data/2025_COACHING_FINAL.csv')
cbbr_advanced = pd.read_csv('needed_data/2025_CBBR_ADVANCED.csv')
cbbr_normal = pd.read_csv('needed_data/2025_CBBR_NORMAL.csv')

bpi_df = pd.read_csv('needed_data/2025_BPI_DATA.csv')
bart_df = pd.read_csv('needed_data/2025_bart_stats.csv')
ken_df = pd.read_csv('needed_data/2025_kenpom_data.csv')

normal_win_games = pd.read_csv('needed_data/final_tourney_games.csv')
normal_score_games = pd.read_csv('needed_data/score_tourney_games.csv')
upset_win_games = pd.read_csv('needed_data/upset_tourney_winners.csv')
upset_score_games = pd.read_csv('needed_data/upset_tourney_score.csv')

cra = pd.read_csv("needed_data/2025_CBBRA_DATA.csv")
cbb2 = pd.read_csv("needed_data/2025_PLAYER_DATA.csv")
cbbrn = pd.read_csv("needed_data/2025_CBBR_Normal_Stats.csv")
coachdf = pd.read_csv('needed_data/2025_COACHING_FINAL.csv')



def get_coach(id, year):
    return coachdf[(coachdf['Year'] == year) & (coachdf['id'] == id)]

def get_CBBR_Normal(id, year, h_m = .25, a_m = .45, n_m = .3):
    new_df = cbbrn[(cbbrn.Year == year) & (cbbrn.id == id)].loc[-20:]
    a = new_df.groupby('Away')[['FGM_O', 'FGA_O', 'FG%_O', '3PM_O', '3PA_O','3P%_O', 'FTM_O', 'FTA_O', 'FT%_O', 'ORB_O', 'TRB_O', 'AST_O', 'STL_O','BLK_O', 'TOV_O', 'PF_O', 'FGM_D', 'FGA_D', 'FG%_D', '3PM_D', '3PA_D','3P%_D', 'FTM_D', 'FTA_D', 'FT%_D', 'ORB_D', 'TRB_D', 'AST_D', 'STL_D','BLK_D', 'TOV_D', 'PF_D']].mean()
    try:
        b = a.loc['H']*h_m + a.loc['A']*a_m + a.loc['N']*n_m
    except:
        b = a.loc['H']*(h_m + n_m* .3) + a.loc['A']*(a_m + n_m* .7)
    return pd.DataFrame(b.transpose()).transpose()
    
def get_CBB_2(id, year):
    return cbb2[(cbb2.ID == id) & (cbb2.Year == year)]

def get_cbb_ref_std(id, year):
    new_df = cra[(cra.Year == year) & (cra.ID == id)]
    a = pd.DataFrame(new_df[['ORtg','DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'OeFG%', 'OTOV%', 'ORB%', 'OFT/FGA', 'DeFG%', 'DTOV%', 'DRB%', 'DFT/FGA']].std()).T
    return a

def get_CBB_Ref_Stats(id, year, h_m = .25, a_m = .45, n_m = .3):
    new_df = cra[(cra.Year == year) & (cra.ID == id)]
    a = new_df.groupby('Away')[['ORtg','DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'OeFG%', 'OTOV%', 'ORB%', 'OFT/FGA', 'DeFG%', 'DTOV%', 'DRB%', 'DFT/FGA']].mean()
    try:
        b = a.loc['H']*h_m + a.loc['A']*a_m + a.loc['N']*n_m
    except:
        b = a.loc['H']*(h_m + n_m* .3) + a.loc['A']*(a_m + n_m* .7)
    return pd.DataFrame(b.transpose()).transpose()
    
def get_ken(id, year):
    return ken_df[(ken_df.Team_ID == id) & (ken_df.Year == year)]

def get_bart(id, year):
    return bart_df[(bart_df.Team_ID == id) & (bart_df.Year == year)]

def get_bpi(id, year):
    return bpi_df[(bpi_df.teamId == id) & (bpi_df.year == year)]

def get_all(id, year):
    #print(id, year)
    bpi = get_bpi(id, year).add_suffix('_BPI').reset_index(drop = True).head(1)
    bart = get_bart(id, year).add_suffix('_BART').reset_index(drop = True).head(1)
    ken = get_ken(id, year).add_suffix('_KEN').reset_index(drop = True).head(1)
    refa = get_CBB_Ref_Stats(id, year).add_suffix('_REFA').reset_index(drop = True).head(1)
    ref2 = get_CBB_2(id, year).add_suffix('_REFP').reset_index(drop = True).head(1)
    refn = get_CBBR_Normal(id, year).add_suffix('_REFN').reset_index(drop = True).head(1)
    coach = get_coach(id, year).add_suffix('_COACH').reset_index(drop = True).head(1)
    refstd = get_cbb_ref_std(id, year).add_suffix('_REFSTD').reset_index(drop = True).head(1)
    
    return pd.concat([bpi, bart, ken, refa, ref2, refn, refstd, coach], axis = 1)

def get_all_values(z):
    aa = pd.concat([z[['Team1_id', 'Year']].rename(columns = {'Team1_id':'Team'}), z[['Team2_id', 'Year']].rename(columns = {'Team2_id':'Team'})]).drop_duplicates()
    all_values = pd.DataFrame()
    for row in aa.itertuples():
        all_values = pd.concat([all_values, get_all(row.Team, row.Year)], ignore_index = True, axis = 0)

    return all_values

def create_teams(gamesdf, mergetype, predictor, use):
    #mergetype should be an int from 0 to 7 representing a permutation from the 3 "major" combinations
    #The first bit represents whether or not to have the basic data, i.e. have [t1_stats, t2_stats]
    #The second bit represents the direct combination, i.e. [t1_stats - t2_stats]
    #The third bit represents a fancier version of this, where stats are more indirectly compared, with t1_defense to t1_offense
    
    #Classification represents a 0 or 1
    #A 0 represents using regression by scores, while a 1 gives data for classification, i.e. win or loss.
    #Regression will give 2x the length as classification, with data = [[t1_stats, t2_stats, t1_score], [t2_stats, t1_stats, t2_score']]
    #While classification will give [t1_stats, t2_stats, t1_win]

    if(predictor == 'score'):
        classification = 0
    elif predictor == 'winner':
        classification = 1
    else:
        raise ValueError('Please Enter either "score" or "winner" for predictor value')

    if(use == 'sim'):
        simulation = 1
    elif use == 'initial':
        simulation = 0
    else:
        raise ValueError('Please Enter either "sim" or "initial" for use value')

    if(simulation == 0):
        if(classification):
            if('Winner' not in gamesdf.columns.to_list()):
                raise ValueError('Please make sure your gamesdf contains a Winner Column')
        else:
            if('Score1' not in gamesdf.columns.to_list()) and ('Score2' not in gamesdf.columns.to_list()):
                raise ValueError('Please make sure to include Score1 and Score2 Columns')
        
    def create_smart_comparison(statsdf):
        stat_mappings = {'OTOV%_REFA_T1': 'DTOV%_REFA_T2', 'DTOV%_REFA_T1': 'OTOV%_REFA_T2', 'OeFG%_REFA_T1': 'DeFG%_REFA_T2', 'DeFG%_REFA_T1': 'OeFG%_REFA_T2', 'ORB%_REFA_T1': 'DRB%_REFA_T2', 'DRB%_REFA_T1': 'ORB%_REFA_T2', 'OFT/FGA_REFA_T1': 'DFT/FGA_REFA_T2', 'DFT/FGA_REFA_T1': 'OFT/FGA_REFA_T2', 'ORtg_REFA_T1': 'DRtg_REFA_T2', 'DRtg_REFA_T1': 'ORtg_REFA_T2', 'AdjO_KEN_T1': 'AdjD_KEN_T2', 'AdjD_KEN_T1': 'AdjO_KEN_T2', 'OppO_KEN_T1': 'OppD_KEN_T2', 'OppD_KEN_T1': 'OppO_KEN_T2', 'Eff Ofg_BART_T1': 'Eff Dfg_BART_T2', 'Eff Dfg_BART_T1': 'Eff Ofg_BART_T2', 'TurnO_BART_T1': 'TurnD_BART_T2', 'TurnD_BART_T1': 'TurnO_BART_T2', 'RebO_BART_T1': 'RebD_BART_T2', 'RebD_BART_T1': 'RebO_BART_T2', 'FT%O_BART_T1': 'FT%D_BART_T2', 'FT%D_BART_T1': 'FT%O_BART_T2', '2%O_BART_T1': '2%D_BART_T2', '2%D_BART_T1': '2%O_BART_T2', '3%O_BART_T1': '3%D_BART_T2', '3%D_BART_T1': '3%O_BART_T2', '3PRO_BART_T1': '3PRD_BART_T2', '3PRD_BART_T1': '3PRO_BART_T2'}
        smart_comparison = pd.DataFrame()
        
        for stat_T1, stat_T2 in stat_mappings.items():
            diff_col_name = f"{stat_T1.replace('_O_', '_').replace('_D_', '_').replace('_T1', '')}_SD"
            smart_comparison[diff_col_name] = statsdf[stat_T1] - statsdf[stat_T2]
        
        return smart_comparison
    all_values = pd.read_csv('needed_data/2025_stats_by_id.csv')
    yuh = pd.DataFrame()
    if(simulation):
        si = 3
        td1 = ['Team1_id', 'Team2_id']
        td2 = ['Team1_id', 'Team2_id', 'year_BPI_T1', 'Year','year_BPI_T2']
    else:
        si = 9
        if(classification):
            si = 7
        td1 = ['Team1', 'Team1_id', 'Team2', 'Team2_id']
        td2 = ['Team1', 'Team1_id', 'Team2', 'Team2_id', 'Seed1', 'Seed2', 'year_BPI_T1', 'Year','year_BPI_T2', 'Winner', 'Winner_id']
        if(classification == 0):
            td2 += ['Score1', 'Score2']  
    
    base = gamesdf.merge(all_values, left_on = ['Team1_id', 'Year'], right_on = ['id_COACH', 'Year_COACH'], how = 'left').merge(all_values, left_on = ['Team2_id', 'Year'], right_on = ['id_COACH', 'Year_COACH'], how = 'left', suffixes = ['_T1', '_T2']).drop(columns = td1, axis = 1)
    base = base.iloc[:, si:].drop(columns = ['year_BPI_T2', 'teamId_BPI_T2'], axis =1)
    
    if(classification):
        #print(base.columns.to_list())
        diff = (base.iloc[:, : len(base.columns)//2].set_axis(axis = 1, labels = all_values.columns.to_list()[1:-2] + [all_values.columns.to_list()[-1]]) - base.iloc[:, len(base.columns)//2 : ].set_axis(axis = 1, labels = all_values.columns.to_list()[1:-2] + [all_values.columns.to_list()[-1]])).add_suffix('_D')
        sdiff = create_smart_comparison(base)
        
        yuh = pd.DataFrame()
        to_drop = []
        if(mergetype & 0x1):
            yuh = pd.concat([yuh, base], axis = 1)
            to_drop += ['Team_ID_BART_T1', 'Team_ID_BART_T2', 'Year_BART_T1', 'Year_BART_T2', 'Year_KEN_T1', 'Team_ID_KEN_T1', 'Year_COACH_T1', 'Year_REFP_T1', 'ID_REFP_T1', 'Year_KEN_T1', 'Team_ID_KEN_T2', 'Year_REFP_T2', 'ID_REFP_T2', 'Year_KEN_T2', 'Year_COACH_T2']
        if(mergetype >> 1 & 0x1):
            to_drop += ['teamId_BPI_D', 'Team_ID_BART_D', 'Year_BART_D', 'Year_KEN_D', 'Team_ID_KEN_D', 'Year_REFP_D', 'ID_REFP_D', 'id_COACH_D']
            yuh = pd.concat([yuh, diff], axis = 1)
        if(mergetype >> 2 & 0x1):
            yuh = pd.concat([yuh, sdiff], axis = 1)
            
        if(not simulation):
            yuh['Winner'] = gamesdf['Winner']

        yuh.drop(columns = to_drop, axis = 1, inplace = True)
        return yuh
    else:
        
        base2 = gamesdf.merge(all_values, left_on = ['Team1_id', 'Year'], right_on = ['id_COACH', 'Year_COACH'], how = 'left').merge(all_values, left_on = ['Team2_id', 'Year'], right_on = ['id_COACH', 'Year_COACH'], how = 'left', suffixes = ['_T2', '_T1']).drop(columns = td2, axis = 1)
        base = pd.concat([base, base2]).drop_duplicates().reset_index(drop = True).drop(['teamId_BPI_T2', 'teamId_BPI_T1'], axis = 1)
        #print(base.columns.to_list())
        diff = (base.iloc[:, : len(base.columns)//2].set_axis(axis = 1, labels = all_values.columns.to_list()[1:-2] + [all_values.columns.to_list()[-1]]) - base.iloc[:, len(base.columns)//2 : ].set_axis(axis = 1, labels = all_values.columns.to_list()[1:-2] + [all_values.columns.to_list()[-1]])).add_suffix('_D')

        sdiff = create_smart_comparison(base)
        to_drop = []

        if(mergetype & 0x1):
            yuh = pd.concat([yuh, base], axis = 1)
            to_drop += ['Team_ID_BART_T1', 'Team_ID_BART_T2', 'Year_BART_T1', 'Year_BART_T2', 'Year_KEN_T1', 'Team_ID_KEN_T1', 'Year_COACH_T1', 'Year_REFP_T1', 'ID_REFP_T1', 'Year_KEN_T1', 'Team_ID_KEN_T2', 'Year_REFP_T2', 'ID_REFP_T2', 'Year_KEN_T2', 'Year_COACH_T2']

        if(mergetype >> 1 & 0x1):
            yuh = pd.concat([yuh, diff], axis = 1)
            to_drop += ['teamId_BPI_D', 'Team_ID_BART_D', 'Year_BART_D', 'Year_KEN_D', 'Team_ID_KEN_D', 'Year_REFP_D', 'ID_REFP_D', 'id_COACH_D']
        if(mergetype >> 2 & 0x1):
            yuh = pd.concat([yuh, sdiff], axis = 1)
        yuh.drop(columns = to_drop, inplace = True, axis = 1)
        if(not simulation):
            yuh['Score'] = pd.concat([gamesdf['Score1'], gamesdf['Score2']], axis = 0, ignore_index = True)
        return yuh
def fix_cbbr_name(name):
    name = " ".join(name.split("-"))
    name = name.replace('southern illinois edwardsville', 'SIU Edwardsville').replace('northern colorado', 'N colorado').replace('Massachusetts lowell', 'Umass Lowell').replace('ETSU', 'Eastern Tennessee State').replace('virginia military institute', 'VMI').replace('texas el paso', 'UTEP').replace('dixie state', 'Utah Tech').replace('texas pan american university university', 'UT Rio Grande').replace('tennessee martin', 'UT Martin').replace('ipfw', 'Purdue Fort Wayne').replace('UPenn', 'Pennsylvania').replace('nebraska omaha', 'omaha').replace('N kentucky', 'Northern Kentucky').replace('N iowa', 'Northern Iowa').replace('N arizona', 'North Arizona').replace('N illinois', 'Northern Illinois').replace('NorthE', 'NorthEastern').replace('NC A&T', 'North Carolina A&T').replace('MTSU', 'Middle Tennesse State').replace('mcneese state', 'mcneese').replace('UMass', 'Massachusetts').replace('louisiana monroe', 'UL Monroe').replace('louisiana lafayette', 'louisiana').replace('iupui', 'iu indianapolis').replace('illinois chicago', 'UIC').replace('FAU', 'Florida Atlantic').replace('FGCU', 'Florida Gulf Coast').replace('e illinois', 'Eastern Illinois').replace('e kentucky', 'eastern kentucky').replace('e michigan', 'eastern michigan').replace('e washington', 'eastern washington').replace('texas am commerce', 'East Texas A&M').replace('ETSU', 'Eastern Tennesse State').replace('american', 'american university').replace('appalachian', 'app').replace("texas christian", 'TCU').replace('central florida', 'UCF').replace('loyola il', 'Loyola Chicago').replace('california davis', 'UC Davis').replace('southern california', 'USC').replace('middle tennessee', 'MTSU').replace('massachusetts', 'UMass').replace('louisiana state', 'LSU').replace('northeastern', 'Northeastern').replace('northwestern state', "N'Western St").replace("northern", "N").replace("nevada las vegas", 'UNLV').replace("eastern", 'E').replace("brigham young", 'BYU').replace('north carolina ash', 'UNC Asheville').replace("virginia commonwealth", 'VCU').replace('alabama birmingham', 'UAB').replace('nevada  las vegas', 'unlv').replace('florida gulf coast', 'FGCU').replace('north carolina at', 'NC A&T').replace('north carolina central', 'NC Central').replace('north dakota state', 'N Dakota St').replace("california irvine", 'UC Irvine').replace('southern methodist', 'SMU').replace('north carolina wilmington', 'UNC wilmington').replace('east tennessee state', 'ETSU').replace('mississippi', 'Ole Miss').replace("Ole Miss state", "Mississippi State").replace('north carolina greensboro', 'UNC greensboro').replace('florida atlantic', 'FAU').replace('pennsylvania', 'UPenn').replace("maryland baltimore county", 'UMBC').replace('texas san antonio', 'UTSA').replace("north carolina state", 'NC State')
    if(name == 'connecticut'):
        name = 'uconn'
    return name