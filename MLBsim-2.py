#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import copy
import numpy as np
import pandas as pd
import time
from datetime import datetime
import multiprocessing
import csv
from player_proj import run_player_proj

init_bases = {"base1": 0, "base2": 0, "base3": 0, "home": 0}
playProb = {"strikeout": .2318, "inPlayOut": .4338, "error": .015, "walk": .1023, "single": .1375, "double": .0432,
            "triple": .0037, "homer": .0327}
score_from_second = {"Yes": .45, "No": .55}
first_to_third = {"Yes": .45, "No": .55}
total_runs = list()
away_batter = 1
home_batter = 1


# In[28]:


def get_odds(line):
    # Returns breakeven odds for a given line

    odds = 0
    if x < 0:
        odds = (-x) / (-x + 100)
    else:
        odds = 100 / (100 + x)
    return odds


# In[29]:


def impliedOdds(line1, line2):
    # Returns implied odds for 2 lines

    odds_line1 = get_odds(line1)
    odds_line2 = get_odds(line2)

    implied_line1 = odds_line1 / (odds_line1 + odds_line2)
    implied_line2 = odds_lin2 / (odds_lin2 + odds_lin1)
    return (implied_line1, implied_line2)


# In[30]:


def get_line(win_pct):
    # Returns line for win pct

    line = 0
    if win_pct > .5:
        try:
            line = ((-100 * win_pct) / (-win_pct + 1))
        except:
            print("div by 0")
    else:
        try:
            line = (100 / win_pct - 100)
        except:
            print("div by 0")
    return line


# In[31]:


def advance_2_bases(t):
    global score_from_second
    global first_to_third
    two_bases = None
    if t == "score_from_second":
        two_bases = np.random.choice(list(score_from_second.keys()), p=list(score_from_second.values()))
    elif t == "first_to_third":
        two_bases = np.random.choice(list(first_to_third.keys()), p=list(first_to_third.values()))
    if two_bases == "Yes":
        return True
    else:
        return False


# In[32]:


def hit_advance_runner(hit, bases):
    # Advances base runners on hit

    base1, base2, base3, home = bases.values()
    batter = 1
    runs = 0
    while hit:
        if base3 == 1:
            base3 = 0
            home += 1
            runs += 1
        if base2 == 1:
            if advance_2_bases("score_from_second") and batter == 1:
                base2 = 0
                home += 1
                runs += 1
            else:
                base2 = 0
                base3 = 1
        if base1 == 1:
            if advance_2_bases("first_to_third") and base3 == 0 and batter == 1 and base2 == 0:
                base3 = 1
                batter = 0
                base1 = 1
            else:
                if batter:
                    base2 = 1
                    base1 = 1
                    batter = 0
                else:
                    base1 = 0
                    base2 = 1
        else:
            if batter:
                base1 = 1
                batter = 0
        hit -= 1

    bases["base1"] = base1
    bases["base2"] = base2
    bases["base3"] = base3
    bases["home"] = home
    # print(bases)
    return (runs, bases)


# In[33]:


def walk_batter(bases):
    # Advances base runners on walk

    base1, base2, base3, home = bases.values()
    runs = 0
    if base1 == 1:
        if base3 == 1 and base2 == 1:
            base3 = 1
            base2 = 1
            base1 = 1
            home += 1
            runs += 1
        elif base3 == 1 and base2 == 0:
            base2 = 1
            base1 = 1
        elif base3 == 0 and base2 == 1:
            base3 = 1
            base2 = 1
            base1 = 1
        if base3 == 0 and base2 == 0:
            base2 = 1
            base1 = 1
    if base1 == 0:
        base1 = 1

    bases["base1"] = base1
    bases["base2"] = base2
    bases["base3"] = base3
    bases["home"] = home
    # print(bases)
    return (runs, bases)


# In[34]:


def get_at_bat_result(batter, team, inning, outs, ip):
    # Returns at bat result based off the given averages

    # Checks what inning and out it is and choses batter vs starter or batter vs bullpen stats
    # Currently setting it to use just the raw stats
    curr = inning + (outs / 3)
    #team = team.reset_index(drop=True)
    # if curr <= ip:
    #     try:
    #         probs = team.iloc[batter - 1][
    #             ["SP Strikeout %", "SP In Play Out %", "SP Reach on Error %", "SP Walk/HBP %", "SP Single %", "SP Double %",
    #             "SP Triple %", "SP HR %"]].to_numpy()
    #     except:
    #         print(batter)
    #         print(team)
    #     sum_of_probs = sum(probs)
    #     temp = 1 / sum_of_probs
    #     probs_scaled = [e * temp for e in probs]
    #     randChoice = np.random.choice(list(playProb.keys()), p=probs_scaled)
    # else:
    #     probs = team.iloc[batter - 1][
    #         ["BP Strikeout %", "BP In Play Out %", "BP Reach on Error %", "BP Walk/HBP %", "BP Single %", "BP Double %",
    #          "BP Triple %", "BP HR %"]].to_numpy()
    #     sum_of_probs = sum(probs)
    #     temp = 1 / sum_of_probs
    #     probs_scaled = [e * temp for e in probs]
    #     randChoice = np.random.choice(list(playProb.keys()), p=probs_scaled)
    probs = team.iloc[batter - 1][
        ["SP Strikeout %", "SP In Play Out %", "SP Reach on Error %", "SP Walk/HBP %", "SP Single %", "SP Double %",
         "SP Triple %", "SP HR %"]].to_numpy()
    sum_of_probs = sum(probs)
    temp = 1 / sum_of_probs
    probs_scaled = [e * temp for e in probs]
    randChoice = np.random.choice(list(playProb.keys()), p=probs_scaled)
    return randChoice


# In[35]:


def double_play(outs, bases):
    dp = False
    temp_outs = outs
    temp_bases = bases

    if temp_outs == 0 or temp_outs == 1:
        if bases["base1"] == 1:
            temp_outs += 2
            temp_bases["base1"] = 0
            temp_bases["base2"] = 0
            dp = True
    return (dp, temp_outs, temp_bases)


# In[36]:


def process_at_bat(batter, team, bases, inning, outs, ip, box_score):
    # Processes an at bat, getting runs scored, outs, and base runners
    # print("Batter: " + str(batter))
    curr = (0, bases)
    made_outs = 0
    runs = 0
    res = get_at_bat_result(batter, team, inning, outs, ip)
    if batter == 1 and inning == 1:
        box_score["Name"] = team["Name"]
        box_score = box_score.fillna(0)
    # print(res)
    if res == "strikeout":
        made_outs += 1
        box_score.at[batter-1, "SO"] += 1
        box_score.at[batter-1, "PA"] += 1
        box_score.at[batter-1, "AB"] += 1
    elif res == "inPlayOut":
        doubleplay = np.random.choice(["Yes", "No"], p=[.02, .98])
        dp = [False]
        if doubleplay == "Yes":
            dp = double_play(outs, bases)
        if dp[0]:
            made_outs = dp[1]
            curr = (0, dp[2])
        else:
            made_outs += 1
        if curr[1]["base3"] == 1 and outs < 2:
            temp = curr[1]
            temp["base3"] = 0
            curr = (1, temp)
        box_score.at[batter-1, "PA"] += 1
        box_score.at[batter-1, "AB"] += 1
        box_score.at[batter-1, "IPO"] += 1
    elif res == "error":
        curr = walk_batter(bases)
        box_score.at[batter-1, "PA"] += 1

    elif res == "walk":
        curr = walk_batter(bases)
        box_score.at[batter-1, "PA"] += 1
        box_score.at[batter - 1, "BB/HBP"] += 1
    elif res == "single":
        curr = hit_advance_runner(1, bases)
        box_score.at[batter-1, "H"] += 1
        box_score.at[batter-1, "PA"] += 1
        box_score.at[batter-1, "AB"] += 1
        box_score.at[batter-1, "1B"] += 1
    elif res == "double":
        curr = hit_advance_runner(2, bases)
        box_score.at[batter - 1, "H"] += 1
        box_score.at[batter-1, "PA"] += 1
        box_score.at[batter-1, "AB"] += 1
        box_score.at[batter-1, "2B"] += 1
    elif res == "triple":
        curr = hit_advance_runner(3, bases)
        box_score.at[batter - 1, "H"] += 1
        box_score.at[batter-1, "PA"] += 1
        box_score.at[batter-1, "AB"] += 1
        box_score.at[batter-1, "3B"] += 1
    elif res == "homer":
        curr = hit_advance_runner(4, bases)
        box_score.at[batter - 1, "H"] += 1
        box_score.at[batter-1, "PA"] += 1
        box_score.at[batter-1, "AB"] += 1
        box_score.at[batter-1, "HR"] += 1
    runs = curr[0]
    game_state = (made_outs, runs, curr[1])
    return game_state, box_score


# In[ ]:


# In[37]:


def process_half_inning(batter, team, HA, inning, ip, box_score, bases=copy.copy(init_bases)):
    # Processes at bats until 3 outs

    cur_bases = copy.copy(bases)
    outs = 0
    runs = 0
    while outs < 3:
        # print("Outs: " + str(outs))
        temp = process_at_bat(batter, team, cur_bases, inning, outs, ip, box_score)
        box_score = temp[1]
        if HA == "Away":
            global away_batter
            if away_batter == 9:
                away_batter = 1
            else:
                away_batter += 1
            batter = away_batter
        else:
            global home_batter
            if home_batter == 9:
                home_batter = 1
            else:
                home_batter += 1
            batter = home_batter
        outs += temp[0][0]
        runs += temp[0][1]
        cur_bases = temp[0][2]
    # print("Runs: " + str(runs))
    return runs, temp[1]


# In[54]:


def process_full_inning(game, inning, away_box_score, home_box_score):
    # Processes 2 half innings
    tempa = away_batter
    #     if tempa%9 == 0:
    #         tempa = 9
    #     else:
    #         tempa = tempa%9
    temph = home_batter
    #     if temph%9 == 0:
    #         temph = 9
    #     else:
    #         temph = temph%9
    # print("Inning: " + str(inning))
    away = process_half_inning(tempa, game["Away Lineup"], "Away", inning, float(game["Home SP"]["IP"].item()), away_box_score)
    home = process_half_inning(temph, game["Home Lineup"], "Home", inning, float(game["Away SP"]["IP"].item()), home_box_score)
    return away[0], home[0], away[1], home[1]


# In[39]:


def process_game(game):
    # Processes full games for 9 innings, if tied after 9 innings, will run until 30 extra innings
    # Returns scores for first, first 5, full game
    cols = ["Name", "PA", "AB", "H", "SO", "IPO", "1B", "2B", "3B", "HR", "BB/HBP"]
    away_box_score = pd.DataFrame(columns=cols)
    home_box_score = pd.DataFrame(columns=cols)
    away = 0
    home = 0
    first_inning_away_runs = 0
    first_inning_home_runs = 0
    first_5_away_runs = 0
    first_5_home_runs = 0
    extras = 0
    global away_batter
    away_batter = 1
    global home_batter
    home_batter = 1
    for i in range(1, 10):

        tempa = away_batter
        #         if tempa%9 == 0:
        #             tempa = 9
        #         else:
        #             tempa = tempa%9

        temph = home_batter
        #         if temph%9 == 0:
        #             temph = 9
        #         else:
        #             temph = temph%9
        if i == 9:
            top_9_inning_score = process_half_inning(tempa, game["Away Lineup"], "Away", i,
                                                     float(game["Home SP"]["IP"].item()), away_box_score)
            away += top_9_inning_score[0]
            away_box_score = top_9_inning_score[1]
            if away >= home:
                bottom_9_inning_score = process_half_inning(temph, game["Home Lineup"], "Home", i,
                                                            float(game["Away SP"]["IP"].item()), home_box_score)
                home += bottom_9_inning_score[0]
                home_box_score = bottom_9_inning_score[1]
        else:
            inning_score = process_full_inning(game, i, away_box_score, home_box_score)
            away += inning_score[0]
            home += inning_score[1]
            away_box_score = inning_score[2]
            home_box_score = inning_score[3]
        if i == 1:
            first_inning_away_runs += inning_score[0]
            first_inning_home_runs += inning_score[1]
        if i < 6:
            first_5_away_runs += inning_score[0]
            first_5_home_runs += inning_score[1]
    if away == home:
        while away == home and extras < 30:
            inning_score = process_full_inning(game, i, away_box_score, home_box_score)
            away += inning_score[0]
            home += inning_score[1]
            away_box_score = inning_score[2]
            home_box_score = inning_score[3]
            extras += 1

    # df = pd.DataFrame(np.array([[first_inning_away_runs, first_5_away_runs, away, first_inning_home_runs, first_5_home_runs, home]]), columns = ["Away First Inning", "Away First 5", "Away Total", "Home First Inning", "Home First 5", "Home Total"])
    df = list([first_inning_away_runs, first_5_away_runs, away, first_inning_home_runs, first_5_home_runs, home])
    return df, away_box_score, home_box_score


# In[ ]:


# In[40]:


def run_sims(n, game):
    # Runs n number of games
    # Returns dataframe of game scores
    cols = ["Name", "PA", "AB", "H", "SO", "IPO", "1B", "2B", "3B", "HR", "BB/HBP"]
    away_box_score = list()
    home_box_score = list()
    away_wins = 0
    away_runs = list()
    home_wins = 0
    home_runs = list()
    df_list = list()
    df = pd.DataFrame()
    ties = 0
    away_names = pd.Series()
    home_names = pd.Series()
    for i in range(0, n):
        results = process_game(game)
        score = results[0]
        away_box_score.append(results[1])
        if away_names.empty:
            away_names = results[1]["Name"]
        home_box_score.append(results[2])
        if home_names.empty:
            home_names = results[2]["Name"]
        df_list.append(score)
        temp_df = pd.DataFrame([score], columns=["Away First Inning", "Away First 5", "Away Total", "Home First Inning",
                                                 "Home First 5", "Home Total"])

        away_score_total = temp_df["Away Total"].values
        away_score_first_inning = temp_df["Away First Inning"].values
        away_score_first_5 = temp_df["Away First 5"].values

        home_score_total = temp_df["Home Total"].values
        home_score_first_inning = temp_df["Home First Inning"].values
        home_score_first_5 = temp_df["Home First 5"].values

        away_runs.append(away_score_total)
        home_runs.append(home_score_total)
        total_runs.append(away_score_total + home_score_total)

        if away_score_total > home_score_total:
            away_wins += 1
        elif home_score_total > away_score_total:
            home_wins += 1
        else:
            ties += 1
    ave_away_box_score = pd.DataFrame(columns=cols)
    ave_home_box_score = pd.DataFrame(columns=cols)
    for box_score in away_box_score:
        ave_away_box_score = pd.concat((ave_away_box_score, box_score))
    for box_score in home_box_score:
        ave_home_box_score = pd.concat((ave_home_box_score, box_score))
    ave_away_box_score = ave_away_box_score.drop("Name", axis=1).apply(pd.to_numeric)
    ave_home_box_score = ave_home_box_score.drop("Name", axis=1).apply(pd.to_numeric)
    ave_away_box_score = ave_away_box_score.groupby(ave_away_box_score.index).mean()
    ave_home_box_score = ave_home_box_score.groupby(ave_home_box_score.index).mean()
    ave_away_box_score.insert(0, "Name", away_names)
    ave_home_box_score.insert(0, "Name", home_names)
    date = datetime.today().strftime('%Y%m%d')
    name = str(date + "_boxscore")
    filename = "%s.csv" % name
    ave_away_box_score.to_csv(filename, mode='a', index=False, header=True)
    ave_home_box_score.to_csv(filename, mode='a', index=False, header=True)
    df = pd.DataFrame(df_list,
                      columns=["Away First Inning", "Away First 5", "Away Total", "Home First Inning", "Home First 5",
                               "Home Total"])
    return (df)

def NRFI_line(df):
    NRFI = len(df.loc[(df["Away First Inning"] == 0) & (df["Home First Inning"] == 0)])
    YRFI = len(df) - NRFI

    NRFI_line = round(get_line(NRFI / (NRFI + YRFI)))
    YRFI_line = round(get_line(YRFI / (NRFI + YRFI)))

    YNRFI = (str(NRFI_line), str(YRFI_line))
    return YNRFI

def moneyline(df):
    away_wins = len(df.loc[(df["Away Total"] > df["Home Total"])])
    home_wins = len(df) - away_wins

    away_ml = round(get_line(away_wins / (away_wins + home_wins)))
    home_ml = round(get_line(home_wins / (home_wins + away_wins)))

    ml = (away_ml, home_ml)
    return ml

def pythagorean_ml(df):
    home_median = df["Home Total"].mean()
    away_median = df["Away Total"].mean()

    away_ml = round(get_line(1 / (1 + ((home_median / away_median) ** 2))))
    home_ml = round(get_line(1 / (1 + ((away_median / home_median) ** 2))))

    ml = (away_ml, home_ml)
    return ml

def runline(df):
    away_ml = moneyline(df)[0]
    home_ml = moneyline(df)[1]
    away_wins = 0
    home_wins = 0
    line = None
    if away_ml < 0:
        away_wins = len(df.loc[(df["Away Total"] > (1.5 + df["Home Total"]))])
        home_wins = len(df) - away_wins
        line = "Away -1.5: "
        line2 = " Home +1.5: "
    else:
        home_wins = len(df.loc[(df["Home Total"] > (1.5 + df["Away Total"]))])
        away_wins = len(df) - home_wins
        line = "Home -1.5: "
        line2 = " Away +1.5: "

    away_rl = round(get_line(away_wins / (away_wins + home_wins)))
    home_rl = round(get_line(home_wins / (home_wins + away_wins)))

    if away_ml < 0:
        rl = (line, str(away_rl), line2, str(home_rl))
    else:
        rl = (line2, str(away_rl), line, str(home_rl))
    return rl

def total(df):
    temp = copy.copy(df)
    cols = ["Away Total", "Home Total"]
    temp["Total"] = temp[cols].sum(axis=1)
    med_runs = round(temp["Total"].median() * 2) / 2
    over = len(df.loc[(df["Away Total"] + df["Home Total"]) > med_runs])
    under = len(df) - over

    over_line = round(get_line(over / (over + under)))
    under_line = round(get_line(under / (under + over)))

    total_line = (med_runs, over_line, under_line)
    return total_line

def total2(df):
    temp = copy.copy(df)
    cols = ["Away Total", "Home Total"]
    temp["Total"] = temp[cols].sum(axis=1)

    med_away = round(temp["Away Total"].median() * 2) / 2
    med_home = round(temp["Home Total"].median() * 2) / 2

    med_runs = med_away + med_home
    over = len(df.loc[(df["Away Total"] + df["Home Total"]) > med_runs])
    under = len(df) - over

    over_line = round(get_line(over / (over + under)))
    under_line = round(get_line(under / (under + over)))

    total_line = (med_runs, over_line, under_line)
    return total_line

def score_first_inning(df):
    away_score = len(df.loc[(df["Away First Inning"] > 0)])
    away_no_score = len(df) - away_score
    home_score = len(df.loc[(df["Home First Inning"] > 0)])
    home_no_score = len(df) - home_score

    away_score_line = round(get_line(away_score / (away_score + away_no_score)))
    away_no_score_line = round(get_line(away_no_score / (away_score + away_no_score)))
    home_score_line = round(get_line(home_score / (home_score + home_no_score)))
    home_no_score_line = round(get_line(home_no_score / (home_score + home_no_score)))

    away_score_2 = len(df.loc[(df["Away First Inning"] > 1)])
    away_no_score_2 = len(df) - away_score
    home_score_2 = len(df.loc[(df["Home First Inning"] > 1)])
    home_no_score_2 = len(df) - home_score

    away_score_line_2 = round(get_line(away_score_2 / (away_score_2 + away_no_score_2)))
    away_no_score_line_2 = round(get_line(away_no_score_2 / (away_score_2 + away_no_score_2)))
    home_score_line_2 = round(get_line(home_score_2 / (home_score_2 + home_no_score_2)))
    home_no_score_line_2 = round(get_line(home_no_score_2 / (home_score_2 + home_no_score_2)))
    return ((away_score_line, away_no_score_line), (home_score_line, home_no_score_line),
            (away_score_line_2, away_no_score_line_2), (home_score_line_2, home_no_score_line_2))

def moneyline_first_5(df):
    away_wins = len(df.loc[(df["Away First 5"] > df["Home First 5"])])
    home_wins = len(df.loc[(df["Away First 5"] < df["Home First 5"])])

    away_ml = round(get_line(away_wins / (away_wins + home_wins)))
    home_ml = round(get_line(home_wins / (home_wins + away_wins)))

    f5_ml = (away_ml, home_ml)
    return f5_ml

def pyth_moneyline_first_5(df):
    home_median = df["Home First 5"].mean()
    away_median = df["Away First 5"].mean()

    away_ml = round(get_line(1 / (1 + ((home_median / away_median) ** 2))))
    home_ml = round(get_line(1 / (1 + ((away_median / home_median) ** 2))))

    ml = (away_ml, home_ml)
    return ml

def runline_first_5(df):
    away_ml = moneyline_first_5(df)[0]
    home_ml = moneyline_first_5(df)[1]
    away_wins = 0
    home_wins = 0
    line = None
    if away_ml < 0:
        away_wins = len(df.loc[(df["Away First 5"] > (.5 + df["Home First 5"]))])
        home_wins = len(df) - away_wins
        line = "Away F5 -.5: "
        line2 = "Home F5 +.5: "
    else:
        home_wins = len(df.loc[(df["Home First 5"] > (.5 + df["Away First 5"]))])
        away_wins = len(df) - home_wins

        line = "Home F5 -.5: "
        line2 = "Away F5 +.5: "

    away_rl = round(get_line(away_wins / (away_wins + home_wins)))
    home_rl = round(get_line(home_wins / (home_wins + away_wins)))

    if away_ml < 0:
        rl = (line, str(away_rl), line2, str(home_rl))
    else:
        rl = (line2, str(away_rl), line, str(home_rl))
    return rl

def total_first_5(df):
    temp = copy.copy(df)
    cols = ["Away First 5", "Home First 5"]
    temp["First 5 Total"] = temp[cols].sum(axis=1)
    med_runs = temp["First 5 Total"].median()
    over = len(df.loc[(df["Away First 5"] + df["Home First 5"]) > med_runs])
    under = len(df) - over

    over_line = round(get_line(over / (over + under)))
    under_line = round(get_line(under / (under + over)))
    f5_total = (med_runs, over_line, under_line)
    return f5_total


def game_odds(n, game):
    # Prints out the odds for n games
    df = run_sims(n, game)
    date = datetime.today().strftime('%Y%m%d')
    name = str(date + "_MLB_sims")
    filename = "%s.csv" % name
    f = open(filename, "a")
    writer = csv.writer(f)

    game_ml = moneyline(df)
    game_total = total(df)
    game_rl = runline(df)
    game_pyth_ml = pythagorean_ml(df)
    game_total_2 = total2(df)

    f5_ml = moneyline_first_5(df)
    f5_pyth_ml = pyth_moneyline_first_5(df)
    f5_total = total_first_5(df)
    f5_rl = runline_first_5(df)

    YNRFI = NRFI_line(df)
    first_inning = score_first_inning(df)
    writer.writerow(
        [game["Away"], game["Home"], game_ml[0], game_ml[1], game_pyth_ml[0], game_pyth_ml[1], game_total[0],
         game_total[1], game_total[2], game_rl[1], game_rl[3], f5_ml[0], f5_ml[1], f5_pyth_ml[0], f5_pyth_ml[1], f5_total[0], f5_total[1], f5_total[2], f5_rl[1], f5_rl[3], YNRFI[1], YNRFI[0]])
    f.close()
    # writer.writerow([str(game["Away"] + " @ " + str(game["Home"]))
    # print("\n")
    # print("Away ML: " + str(game_ml[0]) + "\t" + " Home ML: " + str(game_ml[1]))
    # print("Away Pyth. ML: " + str(game_pyth_ml[0]) + "\t" + " Home Pyth. ML: " + str(game_pyth_ml[1]))
    # print("Over " + str((game_total[0])) + ": " + str(game_total[1]) + "\t" + " Under " + str(
    # (game_total[0])) + ": " + str(game_total[2]))
    # print("Alt Over " + str(game_total_2[0]) + ": " + str(game_total_2[1]) + "\t" + " Alt Under " + str(
    # game_total_2[0]) + ": " + str(game_total_2[2]))
    # print(game_rl[0] + game_rl[1] + "\t" + game_rl[2] + game_rl[3])
    # print("\n")
    # print("Away F5 ML: " + str(f5_ml[0]) + "\t " + " Home F5 ML: " + str(f5_ml[1]))
    # print("Away F5 Pyth. ML: " + str(f5_pyth_ml[0]) + "\t" + " Home F5 Pyth. ML: " + str(f5_pyth_ml[1]))
    # print("F5 Over " + str(int(f5_total[0])) + ": " + str(f5_total[1]) + "\t" + " F5 Under " + str(
    #int(f5_total[0])) + ": " + str(f5_total[2]))
   # print(f5_rl[0] + f5_rl[1] + "\t" + f5_rl[2] + f5_rl[3])
   #  print("\n")
   #  print("YRFI: " + YNRFI[1] + "\t" + "NRFI: " + YNRFI[0])
   #  print(
   #      "Away over 0.5 Run First Inning: " + str(first_inning[0][0]) + "\t" + "Away Under 0.5 Run First Inning: " + str(
   #  first_inning[0][1]))
   #  print(
   #      "Away over 1.5 Run First Inning: " + str(first_inning[2][0]) + "\t" + "Away Under 1.5 Run First Inning: " + str(
   #  first_inning[2][1]))
   #  print(
   #      "Home over 0.5 Run First Inning: " + str(first_inning[1][0]) + "\t" + "Home Under 0.5 Run First Inning: " + str(
   #  first_inning[1][1]))
   #  print(
   #      "Home over 1.5 Run First Inning: " + str(first_inning[3][0]) + "\t" + "Home Under 1.5 Run First Inning: " + str(
   #  first_inning[3][1]))
   #  print("\n")


# return df

# In[52]:


# In[55]:

outcomes = run_player_proj()
start = time.time()
# for game in outcomes:
#     start = time.time()
#     print(game)
#     test = outcomes[game]
#     data = game_odds(100, test)
#     print("\n")
#     end = time.time()
#     print(end - start)

if __name__ == "__main__":
    date = datetime.today().strftime('%Y%m%d')
    name = str(date + "_MLB_sims")
    filename = "%s.csv" % name
    f = open(filename, "a")
    writer = csv.writer(f)
    writer.writerow(
        ["Away", "Home", "Away ML", "Home ML", "Away Pyth ML", "Home Pyth ML", "Total", "Over", "Under", "Away RL",
         "Home RL", "Away F5 ML", "Home F5 ML", "Away F5 Pyth ML", "Home F5 Pyth ML", "Total F5", "Over F5", "Under F5",
         "Away F5 RL", "Home F5 RL", "YRFI", "NRFI"])
    f.close()

    date = datetime.today().strftime('%Y%m%d')
    name = str(date + "_boxscore")
    filename = "%s.csv" % name
    f = open(filename, "a")
    writer = csv.writer(f)
    writer.writerow(["Name", "PA", "AB", "H", "SO", "IPO", "1B", "2B", "3B", "HR", "BB/HBP"])
    f.close()

    pool = multiprocessing.Pool()
    processes = [multiprocessing.Process(target=game_odds, args=(2000, outcomes[game])) for game in outcomes]
    #result = [p.get() for p in processes]
    [p.start() for p in processes]
    for p in processes:
        p.join()
    end = time.time()
    print(end - start)
    date = datetime.today().strftime('%Y%m%d')
    name = str(date + "_MLB_sims")
    filename = "%s.csv" % name
    f = open(filename, "a")
    writer = csv.writer(f)
    writer.writerow([(end-start)])
    f.close()
