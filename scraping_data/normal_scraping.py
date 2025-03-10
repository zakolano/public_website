import pandas as pd
import requests
import os
#import Requests
import numpy as np
import time
import random
pd.options.display.max_rows = 5000
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import re
import requests, io, cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import datetime
from io import StringIO
from fuzzywuzzy import fuzz, process

#from testing_data_in import *

def fun(text):

    text = text.replace('Sun ', '').replace('Great ', '').replace('Bowling Green', 'Bowling Greeen').replace("Ragin'", '').replace(' Mean', '').replace('Wolf', '').replace(' Big', '').replace(' Purple', '').replace('Screaming', '').replace(' Delta', '').replace('Demon', '').replace(' Tar', '').replace('Green', '').replace('Horned', '').replace('Thundering', '').replace('Black', '').replace('Crimson', '').replace("&amp;", '&').replace('Red', '').replace('Blue', '').replace('Yellow', '').replace('Fighting', '').replace('Golden', '').replace('Scarlet','').replace('Nittany', '').replace(r"Runnin'", '')
    text = " ".join(text.split(" ")[:-1])
    return text

link = 'https://www.espn.com/mens-college-basketball/standings'
r = Request(link, headers={"User-Agent": "Mozilla/5.0"})
c = urlopen(r).read()
soup = BeautifulSoup(c, 'html.parser')
a = soup.prettify().split(' ASUN Conference')[1]
match = r't:(\d*)" style="text-decoration:none" title="(.*)">'
x = re.findall(match, a)
y = list(x)
x = [(a[0], fun(a[1])) for a in y]
key_dict= dict(x)
key_dict = {v: k for k, v in key_dict.items()}

def find_closest_match(team_name, team_id_dict):
    match, score = process.extractOne(team_name, team_id_dict.keys(), scorer=fuzz.ratio)
    return team_id_dict[match]
def find_closest_match_str(team_name, team_id_dict):
    try:
        match, score = process.extractOne(team_name, team_id_dict.keys(), scorer=fuzz.ratio)
        return match
    except:
        return None
def find_closest_match_score(team_name, team_id_dict):
    match, score = process.extractOne(team_name, team_id_dict.keys(), scorer=fuzz.ratio)
    return score
def fix_cbbr_name(name):
    name = " ".join(name.split("-"))
    name = name.replace('southern illinois edwardsville', 'SIU Edwardsville').replace('Massachusetts lowell', 'Umass Lowell').replace('ETSU', 'Eastern Tennessee State').replace('virginia military institute', 'VMI').replace('texas el paso', 'UTEP').replace('dixie state', 'Utah Tech').replace('texas pan american university university', 'UT Rio Grande').replace('tennessee martin', 'UT Martin').replace('ipfw', 'Purdue Fort Wayne').replace('UPenn', 'Pennsylvania').replace('nebraska omaha', 'omaha').replace('N kentucky', 'Northern Kentucky').replace('N iowa', 'Northern Iowa').replace('N arizona', 'North Arizona').replace('N colorado', 'North Colorado').replace('N illinois', 'Northern Illinois').replace('NorthE', 'NorthEastern').replace('NC A&T', 'North Carolina A&T').replace('MTSU', 'Middle Tennesse State').replace('mcneese state', 'mcneese').replace('UMass', 'Massachusetts').replace('louisiana monroe', 'UL Monroe').replace('louisiana lafayette', 'lafayette').replace('iupui', 'iu indianapolis').replace('illinois chicago', 'UIC').replace('FAU', 'Florida Atlantic').replace('FGCU', 'Florida Gulf Coast').replace('e illinois', 'Eastern Illinois').replace('e kentucky', 'eastern kentucky').replace('e michigan', 'eastern michigan').replace('e washington', 'eastern washington').replace('texas am commerce', 'East Texas A&M').replace('ETSU', 'Eastern Tennesse State').replace('american', 'american university').replace('appalachian', 'app').replace("texas christian", 'TCU').replace('central florida', 'UCF').replace('loyola il', 'Loyola Chicago').replace('california davis', 'UC Davis').replace('southern california', 'USC').replace('middle tennessee', 'MTSU').replace('massachusetts', 'UMass').replace('louisiana state', 'LSU').replace('northeastern', 'Northeastern').replace('northwestern state', "N'Western St").replace("northern", "N").replace("nevada las vegas", 'UNLV').replace("eastern", 'E').replace("brigham young", 'BYU').replace('north carolina ash', 'UNC Asheville').replace("virginia commonwealth", 'VCU').replace('alabama birmingham', 'UAB').replace('nevada  las vegas', 'unlv').replace('florida gulf coast', 'FGCU').replace('north carolina at', 'NC A&T').replace('north carolina central', 'NC Central').replace('north dakota state', 'N Dakota St').replace("california irvine", 'UC Irvine').replace('southern methodist', 'SMU').replace('north carolina wilmington', 'UNC wilmington').replace('east tennessee state', 'ETSU').replace('mississippi', 'Ole Miss').replace("Ole Miss state", "Mississippi State").replace('north carolina greensboro', 'UNC greensboro').replace('florida atlantic', 'FAU').replace('pennsylvania', 'UPenn').replace("maryland baltimore county", 'UMBC').replace('texas san antonio', 'UTSA').replace("north carolina state", 'NC State')
    if(name == 'connecticut'):
        name = 'uconn'
    return name
def reference_gamelog(url):
    team = url.split("/")[5]
    year = url.split("/")[-1].split("-")[0]
    r = Request(url, headers={"User-Agent":"Mozilla/5.0"})
    c = urlopen(r).read()
    soup = BeautifulSoup(c, 'html.parser')
    table = soup.find_all('table')
    df = pd.read_html(StringIO(str(table[0])))[0]
    df.columns = [column[1] for column in df.columns]
    df = df.drop(columns = ['Gtm', 'Type', 'OT'], axis = 1)
    df.insert(loc = 17, column = 'temp', value = None)
    df.insert(loc = 22, column = 'temp1', value = None)
    
    #print(df.columns.to_list())
    df.columns = ['G', 'Date', 'Away', 'Opp', 'W/L', 'Tm', 'Opp2', 'ORtg',
           'DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%',
           'Unnamed: 17_level_1', 'OeFG%', 'OTOV%', 'ORB%', 'OFT/FGA',
           'Unnamed: 22_level_1', 'DeFG%', 'DTOV%', 'DRB%', 'DFT/FGA']
    #print(df)
    df['G'] = pd.to_numeric(df['G'], errors = 'coerce')
    df = df[['G', 'Date', 'Away', 'Opp', 'W/L', 'Tm', 'Opp2', 'ORtg','DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'OeFG%', 'OTOV%', 'ORB%', 'OFT/FGA', 'DeFG%', 'DTOV%', 'DRB%', 'DFT/FGA']]
    def fix_away(input):
        if(type(input) == float):
            return 'H'
        elif(input == '@'):
            return 'A'
        else:
            return input
    
    df['Away'] = df['Away'].apply(fix_away)
    
    df = df.convert_dtypes()
    df = df.dropna().copy()
    df[['ORtg','DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'OeFG%', 'OTOV%', 'ORB%', 'OFT/FGA', 'DeFG%', 'DTOV%', 'DRB%', 'DFT/FGA']] = df[['ORtg','DRtg', 'Pace', 'FTr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'OeFG%', 'OTOV%', 'ORB%', 'OFT/FGA', 'DeFG%', 'DTOV%', 'DRB%', 'DFT/FGA']].astype('float')
    df['Year'] = year
    df['Team'] = team
    return df
def get_pt(x):
    return float(x.split(" ")[0])
def conv_class_to_yr(ab):

    if ab == "FR":
        return 1
    if ab == "SO":
        return 2
    if ab == "JR":
        return 3
    if ab == "SR":
        return 4
    return 5
def get_cbb_player(url):
    year = int(url.split('/')[-1].split(".")[0])
    team = url.split("/")[-3]
    r = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    c = urlopen(r).read()
    soup = BeautifulSoup(c, 'html.parser')
    table = soup.find_all('table')
    df = pd.read_html(str(table))[0].drop(['RSCI Top 100', 'Hometown'], axis = 1).dropna()

    avg_height = df.Height.apply(lambda x: int(x.split('-')[0]) + float(x.split('-')[1])/12).mean()
    avg_weight = df.Weight.mean()
    avg_year = df.Class.loc[0:8].apply(conv_class_to_yr).mean()
    df['Pts'] = df.Summary.apply(get_pt)
    tot_points = df.Pts.sum()
    a = (df[['Pos', 'Pts']].groupby(['Pos']).sum())/tot_points
    try:
        c_p = round(a.loc['C'].Pts, 5)
    except:
        c_p = 0
    l_p = round(a.loc['F'].Pts, 5)
    g_p = round(a.loc['G'].Pts, 5)
    b_p = df.Pts.loc[5:].sum()/tot_points

    values = [avg_height, avg_weight, avg_year, c_p, l_p, g_p, b_p, year, team]
    a = pd.DataFrame(values).transpose()
    a.columns = ['Avg_Height', 'Avg_Weight', 'Avg_Year', 'CP', 'LP', 'GP', 'BP', 'Year', 'Team']
    
    return a
def get_pt(x):
    return float(x.split(" ")[0])
def conv_class_to_yr(ab):

    if ab == "FR":
        return 1
    if ab == "SO":
        return 2
    if ab == "JR":
        return 3
    if ab == "SR":
        return 4
    return 5
def get_cbb_player(url):
    year = int(url.split('/')[-1].split(".")[0])
    team = url.split("/")[-3]
    r = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    c = urlopen(r).read()
    soup = BeautifulSoup(c, 'html.parser')
    table = soup.find_all('table')
    df = pd.read_html(StringIO(str(table[0])))[0].drop(['RSCI Top 100', 'Hometown'], axis = 1).dropna()

    avg_height = df.Height.apply(lambda x: int(x.split('-')[0]) + float(x.split('-')[1])/12).mean()
    avg_weight = df.Weight.mean()
    avg_year = df.Class.loc[0:8].apply(conv_class_to_yr).mean()
    df['Pts'] = df.Summary.apply(get_pt)
    df['H_P'] =  df.Height.apply(lambda x: int(x.split('-')[0]) + float(x.split('-')[1])/12) * df['Pts']

    tot_points = df.Pts.sum()
    a = (df[['Pos', 'Pts']].groupby(['Pos']).sum())/tot_points
    try:
        c_p = round(a.loc['C'].Pts, 5)
    except:
        c_p = 0
    l_p = round(a.loc['F'].Pts, 5)
    g_p = round(a.loc['G'].Pts, 5)
    b_p = df.Pts.loc[5:].sum()/tot_points
    h_p = df['H_P'].sum()

    values = [avg_height, avg_weight, avg_year, c_p, l_p, g_p, b_p, h_p, year, team]
    a = pd.DataFrame(values).transpose()
    a.columns = ['Avg_Height', 'Avg_Weight', 'Avg_Year', 'CP', 'LP', 'GP', 'BP', 'HP', 'Year', 'Team']
    
    return a
def coach_stats(coach, year):
    link = 'https://www.sports-reference.com/cbb/coaches/' + coach + '.html'
    coach = link.split('/')[-1].split('.')[0]
    r = Request(link, headers={"User-Agent": "Mozilla/5.0"})
    c = urlopen(r).read()
    a = pd.read_html(c)[0]

    def teehee(year):
        if type(year) == float:
            return -1
        a = year.split('-')
        if len(a) == 2:
            if(int(a[1]) > 80):
                return '19' + a[1]
            else:
                return '20' + a[1]
        else:
            return -1

    a['Season'] = a['Season'].apply(teehee)
    a = a.loc[a['Season'] != -1].reset_index(drop = True)
    a['W'] = a['W'].astype(str)
    a['W'] = a['W'].str.replace('*', '')
    a['W'] = a['W'].astype(float)
    a['W'] = a['W'].astype(int)
    #a['W'] = a['W'].apply(lambda x: int("".join([y for y in str(x) if y.isnumeric()])))

    ls = a['School'][0]
    ct = 0
    yuh = []
    for school in a['School']:
        if(school == ls):
            ct += 1
            yuh.append(ct)
        else:
            ct = 1
            yuh.append(ct)
        ls = school
    a['School_Years'] = pd.Series(yuh)

    def find_tourney(text):
        if(type(text) == float):
            return 0
        if "NCAA Tournament" in text:
            return 1
        else: 
            return 0
    a['Tourney_Appearences'] = a['Notes'].apply(find_tourney)
    #print(a['W'])
    year_stats = {}
    ind = a['Season'].index[a['Season'] == str(year)][0]
    year_stats['Coach_Exp'] = [a['School_Years'][ind]]
    rel_data = a.loc[:ind, :][['W', 'SRS', 'SOS','Tourney_Appearences']]
    year_stats['Coach_Wins'] = [rel_data['W'].sum()]
    year_stats['Coach_Qual'] = [rel_data['SRS'].mean().round(3)]
    year_stats['Coach_SOS'] = [rel_data['SOS'].mean().round(3)]
    year_stats['Coach_T_Wins'] = [rel_data['Tourney_Appearences'][:-1].sum()]
    year_stats['Coach'] = [coach]

    return pd.DataFrame(year_stats)
def get_normal_gamelog(link):
    team = link.split("/")[5]
    year = link.split("/")[-1].split("-")[0]
    r = Request(link, headers={"User-Agent":"Mozilla/5.0"})
    c = urlopen(r).read()
    soup = BeautifulSoup(c, 'html.parser')
    table = soup.find_all('table')
    df = pd.read_html(StringIO(str(table[0])))[0]
    opp_ind = 3
    
    df.columns = [column[1] for column in df.columns]
    if(year == '2025'):
        to_drop = ['Gtm', 'Type', 'OT', '2P', '2PA', '2P%', 'eFG%', 'DRB']
        df.drop(columns = to_drop, axis = 1, inplace = True)
        df.insert(loc=23, column='Temp', value = None)
        opp_ind = 4
    df.columns = ['G', 'Date', 'Away', 'Opp', 'W/L', 'Tm', 'Opp', 'FGM_O', 'FGA_O', 'FG%_O', '3PM_O', '3PA_O', '3P%_O', 'FTM_O', 'FTA_O', 'FT%_O', 'ORB_O', 'TRB_O', 'AST_O', 'STL_O', 'BLK_O', 'TOV_O', 'PF_O', 'Temp', 'FGM_D', 'FGA_D', 'FG%_D', '3PM_D', '3PA_D', '3P%_D', 'FTM_D', 'FTA_D', 'FT%_D', 'ORB_D', 'TRB_D', 'AST_D', 'STL_D', 'BLK_D', 'TOV_D', 'PF_D']
    df['G'] = pd.to_numeric(df['G'], errors = 'coerce')
    df.drop(axis = 1, columns = ['Temp', 'Tm', 'Opp'], inplace = True)
    
    def fix_away(input):
        if(type(input) == float):
            return 'H'
        elif(input == '@'):
            return 'A'
        else:
            return input


    def get_opp_names(tup):
        try:
            return tup[1].split('/')[3]
        except:
            return None
    
    df2 = pd.read_html(StringIO(str(table[0])), extract_links = 'all')[0]
    opp_teams = df2.iloc[:, opp_ind].apply(get_opp_names).dropna()
    
    df['Away'] = df['Away'].apply(fix_away)
    df = df.convert_dtypes()
    df = df.dropna()
    df[['FGM_O', 'FGA_O', 'FG%_O', '3PM_O', '3PA_O', '3P%_O', 'FTM_O', 'FTA_O', 'FT%_O', 'ORB_O', 'TRB_O', 'AST_O', 'STL_O', 'BLK_O', 'TOV_O', 'PF_O','FGM_D', 'FGA_D', 'FG%_D', '3PM_D', '3PA_D', '3P%_D', 'FTM_D', 'FTA_D', 'FT%_D', 'ORB_D', 'TRB_D', 'AST_D', 'STL_D', 'BLK_D', 'TOV_D', 'PF_D']] = df[['FGM_O', 'FGA_O', 'FG%_O', '3PM_O', '3PA_O', '3P%_O', 'FTM_O', 'FTA_O', 'FT%_O', 'ORB_O', 'TRB_O', 'AST_O', 'STL_O', 'BLK_O', 'TOV_O', 'PF_O','FGM_D', 'FGA_D', 'FG%_D', '3PM_D', '3PA_D', '3P%_D', 'FTM_D', 'FTA_D', 'FT%_D', 'ORB_D', 'TRB_D', 'AST_D', 'STL_D', 'BLK_D', 'TOV_D', 'PF_D']].astype(float)
    df['Year'] = year
    df['Team'] = team
    
    pat = "/cbb/coaches/(.*).html"
    coach = re.findall(pat, soup.prettify())[0]
    df['Opp_Team'] = opp_teams
    df['Coach'] = coach
    a = coach_stats(coach, str(year))
    a['Team'] = team
    a['Year'] = str(year)
    a['Opp_Team'] = opp_teams
    a.to_csv('COACH_STATS_2025_INPUT_DATA.csv', mode = 'a+', index = False, header = False)

    return df
def get_ken_df():
    ken_df = pd.DataFrame()
    for year in range(2025, 2026):
        url = 'https://kenpom.com/index.php?y=' + str(year)
        r = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        c = urlopen(r).read()
        soup = BeautifulSoup(c, 'html.parser')

        table = soup.find_all('table')
        df = pd.read_html(StringIO(str(table)))[0]

        df.columns = ['Rk', 'Team','Conf', 'W-L', 'AdjEM', 'AdjO', 'AdjO.1', 'AdjD', 'AdjD.1', 'AdjT', 'AdjT.1', 'Luck', 'Luck.1', 'SoS', 'AdjEM.1', 'OppO', 'OppO.1', 'OppD', 'OppD.1', 'NSoS', 'NSoS.1']

        df = df[['Team', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SoS', 'OppO', 'OppD','NSoS']].dropna()

        seed_dict = {}
        for i in range(1, 17):
            seed_dict[i] = 0

        def fix_name(name):

            # if(any(char.isdigit() for char in name.split(" ")[-1])):
            #     num = int(re.search(r'(\d*)',name.split(" ")[-1]).group(0))
            #     seed_dict.update({num: seed_dict[num] + 1})
            #     name = " ".join(name.split(" ")[:-1])
                def fix_kp_name(text):
                    if(text.split(" ")[-1].isnumeric()):
                        return " ".join(text.split(" ")[:-1])
                    else:
                        return text
                
                name = fix_kp_name(name)
                
                name = name.replace('CSUN', 'Cal State Northridge').replace('Appalachian St.', 'App State').replace('Southeastern Louisiana', 'SE Louisiana').replace('Nebraska Omaha', 'Omaha').replace('San Jose St', 'San José State').replace('Queens', 'Queens University').replace('Green Bay', ' Bay').replace('SIUE', 'SIU Edwardsville').replace('Tennessee Martin', 'UT Martin').replace('Penn', 'Pennsylvania').replace('Pennsylvania St', 'Penn St').replace('Louisiana Monroe', 'UL Monroe').replace('USC Upstate', 'South Carolina Upstate').replace('Connecticut', 'Conn').replace('Appalachin St.', 'App State').replace('Illinois Chicago', 'UIC').replace('FIU', 'Florida International').replace('American', 'American University').replace('Texas A&M Commerce', 'East Texas A&M').replace('Saint Francis', 'St. Francis (PA)').replace('Mississippi', 'Ole Miss').replace('Ole Miss St.', 'Mississippi St').replace('LIU', 'Long Island University')
                return name    
            #name = name.replace("Northern Kentucky", "N Kentucky").replace("North Dakota St", 'N Dakota St').replace("East Tennessee St.", 'ETSU').replace('Middle Tennessee', 'MTSU').replace('LIU Brooklyn', 'Long Island').replace('Florida Gulf Coast', 'FGCU').replace('North Carolina A&T', 'NC A&T').replace('Massachusetts', 'UMass').replace('North Carolina Central', 'NC Central').replace('Mississippi', 'Ole Miss').replace("Ole Miss St", "Mississippi St").replace('Southeast Missouri St.', 'SE Missouri').replace("Florida Atlantic", "FAU").replace("Northwestern St", "N'Western St")
            #     if(seed_dict[num] <= 5):
            #         return name
            #     elif(((num == 11) or (num == 16)) and seed_dict[num] <= 6):
            #         return name
            #     else:
            #         return
            # else:
            #     return 

        df.loc["AdjEM":].apply(pd.to_numeric)
        df['Team'] = df['Team'].apply(fix_name)
        df = df[df.AdjO != 'ORtg']
        #print(df)
        df[[ 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SoS', 'OppO', 'OppD','NSoS']] = df[[ 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'SoS', 'OppO', 'OppD','NSoS']].astype(float)
        df['Year'] = year
        #df['Date'] = datetime.datetime.today()
        ken_df = pd.concat([ken_df, df], axis = 0, join = 'outer')
    ken_df.dropna(inplace = True)
    ken_df.reset_index(inplace = True, drop = True) 
    ken_df['Team_ID'] = ken_df['Team'].apply(lambda x: find_closest_match(x, key_dict))
    ken_df['Match_Name'] = ken_df['Team'].apply(lambda x: find_closest_match_str(x, key_dict))
    
    return ken_df
#ken_df = pd.read_csv('FINAL_kenpom_data.csv')
def get_bart2(year):
    link = 'https://barttorvik.com/trank.php?year=' + str(year) + '&sort=&top=0&conlimit=All#'
    response = requests.get(link)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    tables = soup.find('table')
    rows=list()
    for row in tables.find_all("tr"):
        rows.append(row)
    rows = rows[2:]
    values = []
    
    for row in rows:
        string = str(row)
        num_pat = r'((\d+\.*\d*)|(\d*\.*\d+))<br/'
        nums = [idx[0] for idx in re.findall(num_pat,string)]
        #print(row)
        if(len([float(num) for num in nums]) == 19):
            team_pat = r'>(\w+(( \w+.?)|(&\w+)?)*)</a'
            team_pat = r'<a\s.*?team=([^&]+)&amp.*?>(.*?)</a>'
            team_pat = r'style="text-decoration: none;">(.*)<span class'
    
            team = re.search(team_pat, string).group(1).replace(' Corpus Chris','-CC').replace('St.', 'St').replace('Ohio St', 'Ohio State').replace('Utah St', 'Utah State').replace('Louisiana Lafayette', 'Lafayette').replace('Detroit', 'Detroit Mercy').replace('St. Francis (PA)', 'Saint Francis').replace('Penn', 'Pennsylvania')
            team = team.replace('Connecticut','UConn').replace('Miami FL', 'Miami').replace('Central UConn', 'Central Connecticut').replace('Sam Houston St', 'Sam Houston').replace('College of Charleston', 'Charleston').replace('Coastal Carolina', 'Coast Car').replace('Southeastern Louisiana', 'SE Louisiana')
            team = team.replace('Western Kentucky', 'Western KY').replace('UC Santa Barbara', 'UCSB').replace('&amp;', '&').replace('Pennsylvania State', 'Penn State').replace('Iowa St', 'Iowa State').replace('Pittsburgh', 'Pitt').replace('Texas A&M-CC', 'Texas A&M-Corpus Christi')
            team = team.replace('Boston University', 'Boston Univ').replace('North Carolina St', 'NC State').replace('Fairleigh Dickinson', 'Fair Dickinson').replace('Mississippi Valley St', 'Miss Valley St').replace('Hawaii','Hawai''i').replace('Saint Francis', 'St. Francis (PA) ')
            team = team.replace('North Dakota St', 'N Dakota St').replace('South Dakota St', 'S Dakota St').replace('Massachusetts', 'Massachusetts').replace('Mount St Mary''s','Mount St Mary').replace('San Jose St', 'San José State').replace('Illinois Chicago', 'UIC')
            team = team.replace('Arkansas Pine Bluff', 'AR-Pine Bluff').replace('Cal St Bakersfield', 'Bakersfield').replace('North Carolina Central', 'NC Central').replace('Arizona St', 'Arizona State').replace('Queens', 'Queens University').replace('Green Bay', ' Bay')
            team = team.replace("Mississippi", 'Ole Miss').replace("Ole Miss St", 'Mississippi State').replace('UCSB', 'Santa Barbara').replace('LIU', 'Long Island  University').replace('USC Upstate', 'South Carolina Upstate').replace('Appalachian St', 'App State').replace('Texas A&M Commerce', 'East Texas A&M')
            team = team.replace('Nebraska Omaha', 'Omaha').replace('UMKC', 'Kansas City').replace('FIU', 'Florida International').replace('Bakersfield', 'Cal State Bakersfield').replace('American', 'American University').replace('Tennessee Martin', 'UT Martin').replace('Louisiana Monroe' , 'UL Monroe').replace('Pennsylvania St', 'Penn State')
            values.append([team] +  [float(num) for num in nums])
    
    bart_stats = pd.DataFrame(values, columns = ['Team', 'Off Eff', 'Def Eff', 'Bart', 'Eff Ofg', 'Eff Dfg', 'TurnO', 'TurnD', 'RebO', 'RebD','FT%O', 'FT%D','2%O', '2%D', '3%O', '3%D', '3PRO', '3PRD', 'AdjT', 'WAB'])
    bart_stats['Team_ID'] = bart_stats['Team'].apply(lambda x: find_closest_match(x, key_dict))
    bart_stats['Match_Name'] = bart_stats['Team'].apply(lambda x: find_closest_match_str(x, key_dict))
    pd.options.mode.chained_assignment = None
    
    bart_stats = bart_stats.drop(columns = ['Team', 'Match_Name'], axis = 1)
    #print(bart_stats['Team'][bart_stats['teamId'] == 'N/A'].unique)
    
    return bart_stats
def get_bip(year):
    import random
    import math

    link = 'https://www.espn.com/mens-college-basketball/bpi/_/season/' + str(year)
    r = Request(link, headers={"User-Agent":"Mozilla/5.0"})
    c = urlopen(r).read()
    soup = BeautifulSoup(c, 'html.parser')
    conferences = soup.prettify().split(r'All Conferences')[1].split("mens-college-basketball/team")[0]

    #print(conferences)
    values = []
    values2 = []
    values3 = []


    id_pat = r'data-param-value="(\d*)"'
    conf_ids = re.findall(id_pat, conferences)[1:]
   # print(conf_ids)
    for conf_id in conf_ids:
        link = 'https://www.espn.com/mens-college-basketball/bpi/_/season/' + str(year) + '/group/' + str(conf_id)

        #print(link)

        r = Request(link, headers={"User-Agent":"Mozilla/5.0"})
        c = urlopen(r).read()
        soup = BeautifulSoup(c, 'html.parser')
        teams = soup.prettify().split(r'team":{"')[1:-1]

        for team in teams:

            id_pat = r'id":"(\d+)'
            teamId = int(re.search(id_pat, team).group(1))
            #print(teamId)

            bpi_pat = r'"bpi","value":"(-?\d*\.?\d*)'
            bpi = float(re.search(bpi_pat, team).group(1))

            bpio_pat = r'"bpioffense","value":"(-?\d*\.?\d*)'
            bpio = float(re.search(bpio_pat, team).group(1))

            bpid_pat = r'"bpidefense","value":"(-?\d*\.?\d*)'
            bpid = float(re.search(bpid_pat, team).group(1))

            values.append([year, teamId, bpi, bpio, bpid])

            #if(teamId == 97):
               # print(values)
            #if(int(conf_id) == 23):
                #values3.append([teamId, bpi, bpio, bpid])
        #if(int(conf_id) == 23):
            #print(values3)


        time.sleep(5)
        link = 'https://www.espn.com/mens-college-basketball/bpi/_/view/resume/season/' + str(year) + '/group/' + str(conf_id)

        #print(link)

        r = Request(link, headers={"User-Agent":"Mozilla/5.0"})
        c = urlopen(r).read()
        soup = BeautifulSoup(c, 'html.parser')
        teams = soup.prettify().split(r'team":{"')[1:-1]


        for team in teams:
            id_pat = r'id":"(\d+)'
            teamId = int(re.search(id_pat, team).group(1))
            #print(teamId)

            t50_pat = r'"top50bpiwins","value":"(-?\d*)-(\d*)'
            t50_li = re.search(t50_pat, team)
            try:
                t50 = float(t50_li.group(1)) / (int(t50_li.group(1)) + int(t50_li.group(2))) 
            except:
                t50 = 0

            sos_pat = r'"sospastrank","value":"(-?\d*)'
            sos = int(re.search(sos_pat, team).group(1))

            nc_pat = r'"sosoutofconfpastrank","value":"(-?\d*\.?\d*)'
            try:
                nc = int(re.search(nc_pat, team).group(1))
            except:
                nc = sos + random.randint(math.ceil(-1 * sos / 5), math.ceil(sos/5))

            values2.append([year, teamId,  t50, sos, nc, conf_id])

        bpidf = pd.DataFrame(values, columns = ['year', 'teamId', 'BPI', 'BPIo', 'BPId'])
        bpidf2 = pd.DataFrame(values2, columns = ['year', 'teamId', 'T50', 'SOS', 'NC_SOS', 'Conf_ID'])
        final_bpo = bpidf.merge(bpidf2, how = 'inner')

    return final_bpo

bart2 = get_bart2(2025)

bart2['Year'] = 2025
bart_df = bart2.copy()
bart_df.to_csv('bart_current.csv', index = False)

kenpom2 = get_ken_df()#.drop(columns = ['Team', 'Match_Name'])
ken_df = kenpom2.copy()
ken_df.to_csv('current_kenpom.csv', index = False)

new_bpi = get_bip(2025).drop(columns = ['Conf_ID'], axis = 1)
bpi_df = new_bpi.copy()
bpi_df.to_csv('current_bpi.csv', index = False)