#!/usr/bin/env python
# coding: utf-8

# In[106]:


from baseball_reference import *
from beautiful_soup_helper import *
from draft_kings import *
from rotowire import *
from stat_miner import *
from team_dict import *
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
import pandas as pd
from selenium import webdriver
from datetime import datetime, date
from selenium.webdriver.chrome.service import Service
import string
import copy
import time
pd.options.mode.chained_assignment = None
AVERAGE_PITCHER = pd.DataFrame()
AVERAGE_BATTER = pd.DataFrame()
AVERAGE_PITCHER_OUTCOMES = pd.DataFrame()
AVERAGE_BATTER_OUTCOMES = pd.DataFrame()


# In[2]:


errors = .015


# In[3]:


def player_batting_stats(year):
    url = "https://www.baseball-reference.com/leagues/majors/"+str(year)+"-standard-batting.shtml"
    s = Service("/Users/PeterNavin/Desktop/MLBSim/chromedriver")
    driver = webdriver.Chrome(service=s)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', {"id":"players_standard_batting"})
    headers = [th.getText() for th in table.findAll('tr')[0].findAll("th")]
    headers = headers[1:]
    rows = table.findAll('tr')

    rows_data = [[td.getText() for td in rows[i].findAll('td')]
                    for i in range(len(rows))]
    rows_data = rows_data[1:]
    player_batting = pd.DataFrame(rows_data, columns = headers)
    return player_batting


# In[4]:


def simplify(text):
	import unicodedata
	try:
		text = unicode(text, 'utf-8')
	except NameError:
		pass
	text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
	return str(text)


# In[5]:


def pitcher_stats(year):
    url = "https://www.baseball-reference.com/leagues/majors/" +str(year) + "-batting-pitching.shtml"
    s = Service("/Users/PeterNavin/Desktop/MLBSim/chromedriver")
    driver = webdriver.Chrome(service = s)
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    table = soup.find('table', {"id":"players_batting_pitching"})
    headers = [th.getText() for th in table.findAll('tr')[0].findAll("th")]
    headers = headers[1:]
    rows = table.findAll('tr')
    rows_data = [[td.getText() for td in rows[i].findAll('td')]
                    for i in range(len(rows))]
    rows_data = rows_data[1:]
    pitching = pd.DataFrame(rows_data, columns = headers)
    table = soup.find('table', {"id":"teams_batting_pitching"})
    headers = [th.getText() for th in table.findAll('tr')[0].findAll("th")]
    headers = headers[1:]
    rows = table.findAll('tr')
    rows_data = [[td.getText() for td in rows[i].findAll('td')]
                    for i in range(len(rows))]
    rows_data = rows_data[-3:-2]
    rows_data[0].insert(0, "League Average")
    rows_data[0].insert(1, "")
    rows_data[0].insert(1, "")
    temp = pd.Series(rows_data[0], index = pitching.columns)
    pitching = pitching.append(temp, ignore_index = True)
    return pitching


# In[ ]:





# In[6]:


def clean_data(batting_stats):
    #Remove extra spaces
    for column in batting_stats.columns:
        batting_stats[column] = batting_stats[column].str.split().str.join(' ')

    #Remove the * and #
    punct = '!"#$%&\()*+,/:;<=>?@[\\]^_`{}~'   # `|` is not present here
    transtab = str.maketrans(dict.fromkeys(punct, ''))
    batting_stats["Name"]
    batting_stats = batting_stats.dropna()
    batting_stats["Name"] = '|'.join(batting_stats["Name"].tolist()).translate(transtab).split('|')

    #Fix Spanish Accents
    batting_stats["Name"] = batting_stats["Name"].apply(simplify)

    #Fix the Jr.
    batting_stats["Name"] = batting_stats["Name"].str.replace("( Jr.)","",regex=True)

    #Fix Michael A. Taylor
    batting_stats["Name"] = batting_stats["Name"].str.replace("(A. )","",regex=True)

    return batting_stats


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[111]:


def get_lineup_stats(games, batting_stats, pitching_stats):
    slate = {}
    global AVERAGE_PITCHER
    AVERAGE_PITCHER = pd.DataFrame(pitching_stats.loc[(pitching_stats["Name"] == "League Average")])
    global AVERAGE_BATTER
    AVERAGE_BATTER = pd.DataFrame(batting_stats.loc[(batting_stats["Name"] == "LgAvg per 600 PA")])
    get_average_outcomes()

    for game in games:
        home_team = None
        away_team = None
        home_lineup = game.home_lineup
        away_lineup = game.away_lineup
        home_players = list()
        away_players = list()
        away_pitcher = str(game.away_pitcher.name)
        home_pitcher = str(game.home_pitcher.name)

        for player in home_lineup:
            home_players.append(player.name)
            home_team = player.team
        for player in away_lineup:
            away_players.append(player.name)
            away_team = player.team
        temp_dict = {"Away":None, "Away Pitcher":None, "Home":None, "Home Pitcher": None}
        temp_away_df = pd.DataFrame()
        temp_home_df = pd.DataFrame()
        away_pitcher_df = pd.DataFrame()
        home_pitcher_df = pd.DataFrame()
        for name in away_players:
            if (batting_stats.loc[(batting_stats["Name"] == name)]).empty:
                temp_away_df = temp_away_df.append(batting_stats.loc[(batting_stats["Name"] == "LgAvg per 600 PA")])
            else:
                if len(batting_stats.loc[(batting_stats["Name"] == name)]) > 1:
                    if len(batting_stats.loc[(batting_stats["Name"] == name) & (batting_stats["Tm"] == "TOT")]) > 1:
                        temp_away_df = temp_away_df.append(batting_stats.loc[(batting_stats["Name"] == name) & (batting_stats["Tm"] == "TOT") & (batting_stats["Lg"] == "MLB")])
                    else:
                        temp_away_df = temp_away_df.append(batting_stats.loc[(batting_stats["Name"] == name) & (batting_stats["Tm"] == "TOT")])
                else:
                    temp_away_df = temp_away_df.append(batting_stats.loc[(batting_stats["Name"] == name)])
        for name in home_players:
            if (batting_stats.loc[(batting_stats["Name"] == name)]).empty:
                temp_home_df = temp_home_df.append(batting_stats.loc[(batting_stats["Name"] == "LgAvg per 600 PA")])
            else:
                if len(batting_stats.loc[(batting_stats["Name"] == name)]) > 1:
                    if len(batting_stats.loc[(batting_stats["Name"] == name) & (batting_stats["Tm"] == "TOT")]) > 1:
                        temp_away_df = temp_away_df.append(batting_stats.loc[(batting_stats["Name"] == name) & (batting_stats["Tm"] == "TOT") & (batting_stats["Lg"] == "MLB")])
                    else:
                        temp_home_df = temp_home_df.append(batting_stats.loc[(batting_stats["Name"] == name) & (batting_stats["Tm"] == "TOT")])
                else:
                    temp_home_df = temp_home_df.append(batting_stats.loc[(batting_stats["Name"] == name)])
        if len(pitching_stats.loc[(pitching_stats["Name"] == away_pitcher)]) > 0:
            away_pitcher_df = pd.DataFrame(pitching_stats.loc[(pitching_stats["Name"] == away_pitcher)])
        else:
            away_pitcher_df = pd.DataFrame(pitching_stats.loc[(pitching_stats["Name"] == "League Average")])
        if len(pitching_stats.loc[(pitching_stats["Name"] == home_pitcher)]) > 0:
            home_pitcher_df = pd.DataFrame(pitching_stats.loc[(pitching_stats["Name"] == home_pitcher)])
        else:
            home_pitcher_df = pd.DataFrame(pitching_stats.loc[(pitching_stats["Name"] == "League Average")])

        #temp_away_df = pd.concat(batting_stats.loc[(batting_stats["Name"] == name)]  for name in away_players)
        #temp_home_df = pd.concat(batting_stats.loc[(batting_stats["Name"] == name)] for name in home_players)
        order = [1,2,3,4,5,6,7,8,9]
        temp_away_df.insert(0, "Order",order)
        temp_home_df.insert(0, "Order", order)
        temp_dict["Away"] = temp_away_df
        temp_dict["Home"] = temp_home_df

        temp_dict["Away Pitcher"] = away_pitcher_df
        temp_dict["Home Pitcher"] = home_pitcher_df
        game_title = away_team + " @ " + home_team
        if game_title not in slate:
            slate[game_title] = None
        slate[game_title] = copy.deepcopy(temp_dict)
        print(slate[game_title])
    return slate


# In[ ]:





# In[141]:


def get_matchup_stats(team, pitcher):
    columns = ['Strikeout %', 'In Play Out %', 'Reach on Error %',
       'Walk/HBP %', 'Single %', 'Double %', 'Triple %', 'HR %', 'Total']
    pitcher_diff = pd.DataFrame(columns = columns)
    pitcher_outcomes = pitcher[['Strikeout %', 'In Play Out %', 'Reach on Error %',
       'Walk/HBP %', 'Single %', 'Double %', 'Triple %', 'HR %', 'Total']]
    ave_pitcher_outcomes = AVERAGE_PITCHER_OUTCOMES[['Strikeout %', 'In Play Out %', 'Reach on Error %',
       'Walk/HBP %', 'Single %', 'Double %', 'Triple %', 'HR %', 'Total']]
    pitcher_diff = pitcher_outcomes - ave_pitcher_outcomes

    batter_diff = pd.DataFrame(columns = columns)
    batter_outcomes = team[['Strikeout %', 'In Play Out %', 'Reach on Error %',
       'Walk/HBP %', 'Single %', 'Double %', 'Triple %', 'HR %', 'Total']]
    ave_batter_outcomes = AVERAGE_BATTER_OUTCOMES[['Strikeout %', 'In Play Out %', 'Reach on Error %',
       'Walk/HBP %', 'Single %', 'Double %', 'Triple %', 'HR %', 'Total']]
    batter_diff = batter_outcomes - ave_batter_outcomes

    matchup_adj_outcomes = batter_diff - pitcher_diff + ave_batter_outcomes
    matchup_adj_outcomes.insert(0, "Name", team["Name"])
    matchup_adj_outcomes.insert(1, "Order", team["Order"])
    return matchup_adj_outcomes


# In[ ]:





# In[121]:


def get_average_outcomes():
    average_pitcher_stats = AVERAGE_PITCHER[["Name","PA","AB", "BA", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SH", "SF"]]
    for (columnName, columnData) in average_pitcher_stats.iteritems():
        if columnName != "Name":
            average_pitcher_stats[columnName] = pd.to_numeric(average_pitcher_stats[columnName])
    average_pitcher_stats["Strikeout %"] = (average_pitcher_stats["SO"]/average_pitcher_stats["PA"])*(1-errors)
    average_pitcher_stats["In Play Out %"] = ((average_pitcher_stats["PA"]-average_pitcher_stats["H"]-average_pitcher_stats["HBP"]-average_pitcher_stats["BB"]-average_pitcher_stats["SO"])/average_pitcher_stats["PA"])*(1-errors)
    average_pitcher_stats["Reach on Error %"] = errors
    average_pitcher_stats["Walk/HBP %"] = ((average_pitcher_stats["BB"]+average_pitcher_stats["HBP"])/average_pitcher_stats["PA"])*(1-errors)
    average_pitcher_stats["Single %"] = ((average_pitcher_stats["H"]-average_pitcher_stats["2B"]-average_pitcher_stats["3B"]-average_pitcher_stats["HR"])/average_pitcher_stats["PA"])*(1-errors)
    average_pitcher_stats["Double %"] = (average_pitcher_stats["2B"]/average_pitcher_stats["PA"])*(1-errors)
    average_pitcher_stats["Triple %"] = (average_pitcher_stats["3B"]/average_pitcher_stats["PA"])*(1-errors)
    average_pitcher_stats["HR %"] = (average_pitcher_stats["HR"]/average_pitcher_stats["PA"])*(1-errors)
    average_pitcher_stats["Total"] = average_pitcher_stats["Strikeout %"] + average_pitcher_stats["In Play Out %"] + average_pitcher_stats["Reach on Error %"] + average_pitcher_stats["Walk/HBP %"] + average_pitcher_stats["Single %"] + average_pitcher_stats["Double %"] + average_pitcher_stats["Triple %"] + average_pitcher_stats["HR %"]
    global AVERAGE_PITCHER_OUTCOMES
    AVERAGE_PITCHER_OUTCOMES = average_pitcher_stats

    average_batter_stats = AVERAGE_BATTER[["Name","PA","AB", "BA", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SH", "SF"]]
    for (columnName, columnData) in average_batter_stats.iteritems():
        if columnName != "Name":
            average_batter_stats[columnName] = pd.to_numeric(average_batter_stats[columnName])
    average_batter_stats["Strikeout %"] = (average_batter_stats["SO"]/average_batter_stats["PA"])*(1-errors)
    average_batter_stats["In Play Out %"] = ((average_batter_stats["PA"]-average_batter_stats["H"]-average_batter_stats["HBP"]-average_batter_stats["BB"]-average_batter_stats["SO"])/average_batter_stats["PA"])*(1-errors)
    average_batter_stats["Reach on Error %"] = errors
    average_batter_stats["Walk/HBP %"] = ((average_batter_stats["BB"]+average_batter_stats["HBP"])/average_batter_stats["PA"])*(1-errors)
    average_batter_stats["Single %"] = ((average_batter_stats["H"]-average_batter_stats["2B"]-average_batter_stats["3B"]-average_batter_stats["HR"])/average_batter_stats["PA"])*(1-errors)
    average_batter_stats["Double %"] = (average_batter_stats["2B"]/average_batter_stats["PA"])*(1-errors)
    average_batter_stats["Triple %"] = (average_batter_stats["3B"]/average_batter_stats["PA"])*(1-errors)
    average_batter_stats["HR %"] = (average_batter_stats["HR"]/average_batter_stats["PA"])*(1-errors)
    average_batter_stats["Total"] = average_batter_stats["Strikeout %"] + average_batter_stats["In Play Out %"] + average_batter_stats["Reach on Error %"] + average_batter_stats["Walk/HBP %"] + average_batter_stats["Single %"] + average_batter_stats["Double %"] + average_batter_stats["Triple %"] + average_batter_stats["HR %"]
    global AVERAGE_BATTER_OUTCOMES
    AVERAGE_BATTER_OUTCOMES = average_batter_stats


# In[ ]:





# In[142]:


def get_outcomes(slate_final):

    for game in slate_final:
        away_pitcher = pd.DataFrame()
        home_pitcher = pd.DataFrame()

        if len(slate_final[game]["Away Pitcher"]) > 0:
            away_pitcher = slate_final[game]["Away Pitcher"][["Name", "PA","AB", "BA", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SH", "SF"]]
        if len(slate_final[game]["Home Pitcher"]) > 0:
            home_pitcher = slate_final[game]["Home Pitcher"][["Name", "PA","AB", "BA", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SH", "SF"]]
            slate_final[game]["Away Pitcher"] = away_pitcher
            slate_final[game]["Home Pitcher"] = home_pitcher
        for (columnName, columnData) in away_pitcher.iteritems():
            if columnName != "Name":
                away_pitcher[columnName] = pd.to_numeric(temp[columnName])
        for (columnName, columnData) in home_pitcher.iteritems():
            if columnName != "Name":
                home_pitcher[columnName] = pd.to_numeric(temp[columnName])

        for team in slate_final[game]:
            if team == "Away":
                temp = pd.DataFrame()
                temp = slate_final[game][team][["Name", "Order", "PA","AB", "BA", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SH", "SF"]]
                for (columnName, columnData) in temp.iteritems():
                    if columnName != "Name":
                        temp[columnName] = pd.to_numeric(temp[columnName])
                temp = get_matchup_stats(temp, home_pitcher)
                slate_final[game][team] = temp
            elif team == "Home":
                temp = pd.DataFrame()
                temp = slate_final[game][team][["Name", "Order", "PA","AB", "BA", "H", "2B", "3B", "HR", "BB", "SO", "HBP", "SH", "SF"]]
                for (columnName, columnData) in temp.iteritems():
                    if columnName != "Name":
                        temp[columnName] = pd.to_numeric(temp[columnName])
                temp = get_matchup_stats(temp, away_pitcher)
                slate_final[game][team] = temp
    return slate_final


# In[143]:


def run_scraper(year):
    batting_stats = player_batting_stats(year)
    pitching_stats = pitcher_stats(year)
    games = get_game_lineups("https://www.rotowire.com/baseball/daily-lineups.php?date=tomorrow")
    batting_stats = clean_data(batting_stats)
    pitching_stats = clean_data(pitching_stats)
    test = get_lineup_stats(games, batting_stats, pitching_stats)
    outcomes = get_outcomes(test)
    return outcomes


# In[ ]:





# In[144]:


outcomes = run_scraper(2022)


# In[145]:


outcomes


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




