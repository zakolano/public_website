import pandas as pd
import re
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup



link = 'https://www.espn.com/espn/feature/story/_/page/bracketology/ncaa-bracketology-2025-march-madness-men-field-predictions'

r = Request(link, headers={"User-Agent": "Mozilla/5.0"})
c = urlopen(r).read()
soup = BeautifulSoup(c, 'html.parser')
div_by_id = soup.find('div', class_='bracket__region')


lis = div_by_id.find_all('li', class_ = 'bracket__item')

pat1 = r'https://www.espn.com/mens-college-basketball/team/_/id/(\d+)'
pat2 = r' bracket__seed">(\d+)</span>'

seeds = []
all_ids = []

j = 0
for i, li in enumerate(lis):
    z = re.findall(pat1, str(li))
    all_ids.append(z)
    seeds.append(re.findall(pat2, str(li)))
new_z = pd.DataFrame(all_ids).set_axis(labels = ['Team1', 'Team2'], axis = 1)
new_z['Team1'] = new_z['Team1'].astype(int)
new_z['Team2'] = new_z['Team2'].astype('Int64')   
new_z.to_csv('scraping_data/lunardi_projections.csv', index = False)
seeds = pd.DataFrame(seeds).set_axis(labels = ['Seed'], axis = 1)
seeds = pd.concat([pd.concat([seeds, new_z[new_z['Team2'].isna()]['Team1']], axis = 1), new_z.dropna().rename({"Team1": 'Temp1'}, axis = 1)], axis  = 1)

seeds = pd.concat([seeds[['Seed', 'Team1']].dropna().set_axis(labels = ['Seed', 'Team'], axis = 1), seeds[['Seed', 'Temp1']].dropna().set_axis(labels = ['Seed', 'Team'], axis = 1), seeds[['Seed', 'Team2']].dropna().set_axis(labels = ['Seed', 'Team'], axis = 1)], axis = 0)
seeds.columns = ['seed', 'id']
seeds.to_csv('scraping_data/lunardi_seeds.csv', index = False)