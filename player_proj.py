#!/usr/bin/env python
# coding: utf-8

# In[63]:


import csv
import pandas as pd
from decimal import Decimal

# In[51]:


file = open('standard-projections-the-bat-x-hitters-3372512-2.csv')
csvreader = csv.reader(file)
header = []
header = next(csvreader)
rows = []
for row in csvreader:
    rows.append(row)

file = open('standard-projections-the-bat-x-3372510-2.csv')
csvreader = csv.reader(file)
pitcher_header = []
pitcher_header = next(csvreader)
pitcher_rows = []
for row in csvreader:
    pitcher_rows.append(row)


# In[49]:


# In[19]:


def column_index(column):
    return header.index(column)


# In[20]:


def pitcher_column_index(column):
    return pitcher_header.index(column)


# In[29]:


error = 0.015


# In[88]:


def get_at_bat_stats(df):
    # # Batting vs starting pitcher projections / Plate appearances vs Starting Pitcher
    # df["SP Single %"] = df["SP1B"].astype(float)/df["PAVSSP"].astype(float)*(1-error)
    # df["SP Double %"] = df["SP2B"].astype(float)/df["PAVSSP"].astype(float)*(1-error)
    # df["SP Triple %"] = df["SP3B"].astype(float)/df["PAVSSP"].astype(float)*(1-error)
    # df["SP HR %"] = df["SPHR"].astype(float)/df["PAVSSP"].astype(float)*(1-error)
    # df["SP Walk/HBP %"] = df["SPBB"].astype(float)/df["PAVSSP"].astype(float)*(1-error)
    # df["SP Strikeout %"] = df["SPK"].astype(float)/df["PAVSSP"].astype(float)*(1-error)
    # df["SP Reach on Error %"] = 0
    # df["SP In Play Out %"] = (df["SPIPO"].astype(float)/df["PAVSSP"].astype(float))*(1-error)
    # #df["SP In Play Out %"] = 1 - df["SP Single %"] - df["SP Double %"] - df["SP Triple %"] - df["SP HR %"] - df["SP Walk/HBP %"] - df["SP Strikeout %"] - df["SP Reach on Error %"]
    # df["SP Total"] = (df["SP In Play Out %"]+df["SP Single %"]+df["SP Double %"]+df["SP Triple %"]+df["SP HR %"]+df["SP Walk/HBP %"]+df["SP Strikeout %"]+df["SP Reach on Error %"])

    df["SP Single %"] = (df["SP1B"].astype(float) / df["PA"].astype(float)) * (1 - error)
    df["SP Double %"] = (df["SP2B"].astype(float) / df["PA"].astype(float)) * (1 - error)
    df["SP Triple %"] = (df["SP3B"].astype(float) / df["PA"].astype(float)) * (1 - error)
    df["SP HR %"] = (df["SPHR"].astype(float) / df["PA"].astype(float)) * (1 - error)
    df["SP Walk/HBP %"] = (df["SPBB"].astype(float) / df["PA"].astype(float)) * (1 - error)
    df["SP Strikeout %"] = (df["SPK"].astype(float) / df["PA"].astype(float)) * (1 - error)
    df["SP Reach on Error %"] = 0.015
    df["SP In Play Out %"] = (df["SPIPO"].astype(float) / df["PA"].astype(float)) * (1 - error)
    df["SP Total"] = (
            df["SP In Play Out %"] + df["SP Single %"] + df["SP Double %"] + df["SP Triple %"] + df["SP HR %"] + df[
        "SP Walk/HBP %"] + df["SP Strikeout %"] + df["SP Reach on Error %"])
    #Batting vs bullpen projections / plate appearances vs bullpen
    df["BP Single %"] = df["BP1B"].astype(float) / (df["PA"].astype(float) - df["PAVSSP"].astype(float)) * (1 - error)
    df["BP Double %"] = df["BP2B"].astype(float) / (df["PA"].astype(float) - df["PAVSSP"].astype(float)) * (1 - error)
    df["BP Triple %"] = df["BP3B"].astype(float) / (df["PA"].astype(float) - df["PAVSSP"].astype(float)) * (1 - error)
    df["BP HR %"] = df["BPHR"].astype(float) / (df["PA"].astype(float) - df["PAVSSP"].astype(float)) * (1 - error)
    df["BP Walk/HBP %"] = df["BPBB"].astype(float) / (df["PA"].astype(float) - df["PAVSSP"].astype(float)) * (1 - error)
    df["BP Strikeout %"] = df["BPK"].astype(float) / (df["PA"].astype(float) - df["PAVSSP"].astype(float)) * (1 - error)
    df["BP Reach on Error %"] = 0
    df["BP In Play Out %"] = df["BPIPO"].astype(float) / (df["PA"].astype(float) - df["PAVSSP"].astype(float)) * (
            1 - error)
    df["BP Total"] = (
            df["BP In Play Out %"] + df["BP Single %"] + df["BP Double %"] + df["BP Triple %"] + df["BP HR %"] + df[
        "BP Walk/HBP %"] + df["BP Strikeout %"] + df["BP Reach on Error %"])

    # Remove negatives (shouldn't be any after updates that were made)
    temp = df._get_numeric_data()
    temp[temp < 0] = 0
    temp.round(4)
    df = df.reset_index(drop=True)
    return df


# In[90]:


def create_sp_stats(df):
    # Cell numbers correspond to the cells in Data_example sheet on google drive
    # get the percentage of plate appearances vs start pitcher
    sp_factor = df["PAVSSP"].astype(float) / df["PA"].astype(float)
    # For now, setting the sp_factor to one to use just raw stats
    sp_factor = 1
    # Multiplies the raw stats by the PAVSSP%
    # (M3:S11)
    df["SPH"] = df["H"].astype(float) * sp_factor

    df["SP1B"] = df["1B"].astype(float) * sp_factor
    df["SP2B"] = df["2B"].astype(float) * sp_factor
    df["SP3B"] = df["3B"].astype(float) * sp_factor
    df["SPHR"] = df["HR"].astype(float) * sp_factor
    df["SPK"] = df["K"].astype(float) * sp_factor
    df["SPBB"] = (df["BB"].astype(float) + df["HBP"].astype(float)) * sp_factor

    # IPO == In Play Out, must be calculated as projections do not include it
    # df["SPIPO"] = df["PAVSSP"].astype(float)-df["SP1B"].astype(float)-df["SP2B"].astype(float)-df["SP3B"].astype(float)-df["SPHR"].astype(float)-df["SPK"].astype(float)-df["SPBB"].astype(float)

    df["SPIPO"] = df["PA"].astype(float) - df["1B"].astype(float) - df["2B"].astype(float) - df["3B"].astype(float) - df["HR"].astype(float) - df["BB"].astype(float) - df["K"].astype(float)
    df = df.reset_index(drop=True)
    return df


# In[91]:


def create_bp_stats(df):
    # get the percentage of plate appearances vs  bullpen
    bp_factor = 1 - (df["PAVSSP"].astype(float) / df["PA"].astype(float))

    # Multiplies the raw stats by the PAVBP%
    df["BPH"] = df["H"].astype(float) * bp_factor
    df["BP1B"] = df["1B"].astype(float) * bp_factor
    df["BP2B"] = df["2B"].astype(float) * bp_factor
    df["BP3B"] = df["3B"].astype(float) * bp_factor
    df["BPHR"] = df["HR"].astype(float) * bp_factor
    df["BPK"] = df["K"].astype(float) * bp_factor
    df["BPBB"] = (df["BB"].astype(float) + df["HBP"].astype(float)) * bp_factor
    # IPO == In Play Out, must be calculated as projections do not include it
    df["BPIPO"] = (df["PA"].astype(float) - df["PAVSSP"].astype(float)) - df["BPH"].astype(float) - df["BP1B"].astype(
        float) - df["BP2B"].astype(float) - df["BP3B"].astype(float) - df["BPHR"].astype(float) - df["BPK"].astype(
        float) - df["BPBB"].astype(float)
    df = df.reset_index(drop=True)
    return df


# In[92]:


def weight_sp_stats(pitcher_df, batter_df):
    # Cell numbers correspond to the cells in Data_example sheet on google drive
    # Number of hits the starting pitcher is projected to give up (M14)
    sp_hits = float(pitcher_df["H"].item())
    # Number of hits the batters are expected to hit against the starting pitcher (M18)
    batter_sp_hits = batter_df["SPH"].sum()
    # Scaling factor for SP (M20)
    sp_factor = sp_hits / batter_sp_hits
    # For now, setting the sp_factor to one to use just raw stats
    sp_factor = 1
    batter_df["SPH"] = batter_df["SPH"] * sp_factor
    batter_df["SP1B"] = batter_df["SP1B"] * sp_factor
    batter_df["SP2B"] = batter_df["SP2B"] * sp_factor
    batter_df["SP3B"] = batter_df["SP3B"] * sp_factor
    batter_df["SPHR"] = batter_df["SPHR"] * sp_factor
    batter_df["SPK"] = batter_df["SPK"] * sp_factor
    batter_df["SPBB"] = batter_df["SPBB"] * sp_factor
    batter_df["SPIPO"] = batter_df["PA"].astype(float) - batter_df["SP1B"] - batter_df["SP2B"] - batter_df["SP3B"] - \
                         batter_df["SPHR"] - batter_df["SPK"] - batter_df["SPBB"]
    batter_df = batter_df.reset_index(drop=True)
    return batter_df


# In[86]:


def weight_bp_stats(pitcher_df, batter_df):
    # Cell numbers correspond to the cells in Data_example sheet on google drive
    # Number of hits the starting pitcher is projected to give up (M14)
    sp_hits = float(pitcher_df["H"].item())
    # Starting pitcher innings pitched (M15)
    sp_ip = float(pitcher_df["IP"].item())
    # Number of hits the batters are expected to hit against the starting pitcher (M18)
    batter_sp_hits = batter_df["SPH"].sum()
    # Number of hits the batters are expected to hit total (M19)
    batter_total_hits = batter_df["H"].astype(float).sum()
    # Not sure what this number really represents (Q16)
    bp_ratio = (9 - sp_ip) / 9
    # Estimated hits vs bullpen, total game hits *  bullpen ratio (Q17)
    total_xbp_hits = batter_total_hits * bp_ratio
    # Actual hits vs bullpen, total game hits - starting pitcher hits (Q18)
    actual_bp_hits = batter_total_hits - sp_hits
    # Actual hits / estimated hits (Q19)
    bp_factor = actual_bp_hits / total_xbp_hits
    batter_df["BPH"] = batter_df["BPH"] * bp_factor
    batter_df["BP1B"] = batter_df["BP1B"] * bp_factor
    batter_df["BP2B"] = batter_df["BP2B"] * bp_factor
    batter_df["BP3B"] = batter_df["BP3B"] * bp_factor
    batter_df["BPHR"] = batter_df["BPHR"] * bp_factor
    batter_df["BPK"] = batter_df["BPK"] * bp_factor
    batter_df["BPBB"] = batter_df["BPBB"] * bp_factor
    batter_df["BPIPO"] = (batter_df["PA"].astype(float) - batter_df["PAVSSP"].astype(float)) - batter_df["BP1B"] - \
                         batter_df["BP2B"] - batter_df["BP3B"] - batter_df["BPHR"] - batter_df["BPK"] - batter_df[
                             "BPBB"]
    batter_df = batter_df.reset_index(drop=True)
    return batter_df


# In[35]:


def run_player_proj():
    slate = {}

    for row in rows:
        game = None
        away = None
        home = None
        away_lineup = list()
        home_lineup = list()
        if row[column_index("PARK")] == row[column_index("TEAM")]:
            game = row[column_index("OPP")] + " @ " + row[column_index("TEAM")]
            away = row[column_index("OPP")]
            home = row[column_index("TEAM")]
        elif row[column_index("PARK")] == row[column_index("OPP")]:
            game = row[column_index("TEAM")] + " @ " + row[column_index("OPP")]
            away = row[column_index("TEAM")]
            home = row[column_index("OPP")]
        if game != None:
            teams = {"Away": away, "Home": home, "Away Lineup": away_lineup, "Home Lineup": home_lineup,
                     "Away SP": None, "Home SP": None}
            if game not in slate:
                slate[game] = teams
    for row in rows:
        player_team = row[column_index("TEAM")]
        for key in slate:
            if player_team in key:
                if player_team == slate[key]["Away"]:
                    slate[key]["Away Lineup"].append(row)
                elif player_team == slate[key]["Home"]:
                    slate[key]["Home Lineup"].append(row)
                continue
    for key in slate:
        away_lineup = pd.DataFrame(slate[key]["Away Lineup"], columns=header)
        away_lineup = away_lineup.sort_values(by=["LP"])
        away_lineup = away_lineup[["NAME", "LP", "PAVSSP", "PA", "H", "1B", "2B", "3B", "HR", "BB", "HBP", "K"]]
        away_lineup = create_sp_stats(away_lineup)
        away_lineup = create_bp_stats(away_lineup)

        slate[key]["Away Lineup"] = away_lineup
        home_lineup = pd.DataFrame(slate[key]["Home Lineup"], columns=header)
        home_lineup = home_lineup.sort_values(by=["LP"])
        home_lineup = home_lineup[["NAME", "LP", "PAVSSP", "PA", "H", "1B", "2B", "3B", "HR", "BB", "HBP", "K"]]
        home_lineup = create_sp_stats(home_lineup)
        home_lineup = create_bp_stats(home_lineup)

        slate[key]["Home Lineup"] = home_lineup

    for key in slate:
        for player in pitcher_rows:
            if player[column_index("TEAM")] in key:
                if player[column_index("TEAM")] == slate[key]["Away"]:
                    away_sp = pd.DataFrame([player], columns=pitcher_header)
                    slate[key]["Away SP"] = away_sp
                elif player[column_index("TEAM")] == slate[key]["Home"]:
                    home_sp = pd.DataFrame([player], columns=pitcher_header)
                    slate[key]["Home SP"] = home_sp

    for key in slate:
        for key2 in slate[key]:
            if key2 == "Away Lineup":
                slate[key][key2] = weight_sp_stats(slate[key]["Home SP"], slate[key][key2])
                slate[key][key2] = weight_bp_stats(slate[key]["Home SP"], slate[key][key2])
                slate[key][key2] = get_at_bat_stats(slate[key][key2])
                slate[key][key2].rename(columns={'LP': 'Order', 'NAME': 'Name'}, inplace=True)
            elif key2 == "Home Lineup":
                slate[key][key2] = weight_sp_stats(slate[key]["Away SP"], slate[key][key2])
                slate[key][key2] = weight_bp_stats(slate[key]["Away SP"], slate[key][key2])
                slate[key][key2] = get_at_bat_stats(slate[key][key2])
                slate[key][key2].rename(columns={'LP': 'Order', 'NAME': 'Name'}, inplace=True)
    return slate


# In[93]:


#slate = run_player_proj()

# In[82]:


# In[94]:


# In[ ]:


# In[ ]:


# In[ ]:
