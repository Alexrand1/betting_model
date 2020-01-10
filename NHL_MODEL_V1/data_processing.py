import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error

files = [
    'C:\\Users\\alexrand\\PycharmProjects\\BettingModel\\nhl_data\\game.csv',
    'C:\\Users\\alexrand\\PycharmProjects\\BettingModel\\nhl_data\\game_goalie_stats.csv',
    'C:\\Users\\alexrand\\PycharmProjects\\BettingModel\\nhl_data\\game_teams_stats.csv'
         ]

input_files = [
'C:\\Users\\alexrand\\PycharmProjects\\BettingModel\\nhl_data\\Miscellaneous.xlsx',
'C:\\Users\\alexrand\\PycharmProjects\\BettingModel\\nhl_data\\SAT Counts (5v5, since 2009-10).xlsx',
'C:\\Users\\alexrand\\PycharmProjects\\BettingModel\\nhl_data\\SAT Percentages (5v5, since 2009-10).xlsx'
]


team_info = {'New Jersey Devils': 1, 'Philadelphia Flyers': 4, 'Los Angeles Kings': 26,
             'Tampa Bay Lightning': 14, 'Boston Bruins': 6, 'New York Rangers': 3,
             'Pittsburgh Penguins': 5, 'Detroit Red Wings': 17, 'San Jose Sharks': 28,
             'Nashville Predators': 18, 'Vancouver Canucks': 23, 'Chicago Blackhawks': 16,
             'Ottawa Senators': 9, 'Montreal Canadiens': 8, 'Minnesota Wild': 30,
             'Washington Capitals': 15, 'St. Louis Blues': 19, 'Anaheim Ducks': 24,
             'Phoenix Coyotes': 27, 'New York Islanders': 2, 'Toronto Maple Leafs': 10,
             'Florida Panthers': 13, 'Buffalo Sabres': 7, 'Calgary Flames': 20,
             'Colorado Avalanche': 21, 'Dallas Stars': 25, 'Columbus Blue Jackets': 29,
             'Winnipeg Jets': 52, 'Edmonton Oilers': 22, 'Vegas Golden Knights': 54,
             'Carolina Hurricanes': 12, 'Arizona Coyotes': 53}


def data_processing(filenames):
    X = pd.read_csv(filenames[0])
    X2 = pd.read_csv(filenames[1])
    X3 = pd.read_csv(filenames[2])
    X['total_goals'] = X['away_goals'] + X['home_goals']
    X_4 = pd.merge(X, X2, on='game_id')
    X_merge = pd.merge(X_4, X3, on='game_id')
    test_data = X_merge.drop('total_goals', axis=1)
    test_data = test_data.select_dtypes(include=[np.number])
    test_data.fillna(test_data.mean(), inplace=True)
    X_num = X_merge.select_dtypes(include=[np.number])
    X_num[X_num.replace([np.inf, -np.inf], np.nan).notnull().all(axis=1)]
    X_num.dropna()
    imputer = SimpleImputer(strategy='median')
    X_impute = imputer.fit(X_num)
    imputer.transform(X_num)
    X_num
    X_num.fillna(X_num.mean(), inplace=True)
    corr_matrix = X_num.corr()
    return X_num, test_data


def create_model(X, test):
    y = X['total_goals']
    features = ['away_team_id', 'home_team_id', 'savePercentage', 'shots_y', 'takeaways']
    X = pd.get_dummies(X[features])
    X_test = pd.get_dummies(test[features])

    model = RandomForestClassifier(n_estimators=500, max_depth=100, random_state=42)
    model.fit(X, y)

    return model


def model_metrics(files):
    misc = pd.read_excel(input_files[0])
    savc = pd.read_excel(input_files[1])
    savp = pd.read_excel(input_files[2])
    merge1 = pd.merge(misc, savc, on='Team')
    merge2 = pd.merge(merge1, savp, on='Team')
    #merge2['TkA'] = merge2['TkA'].str.replace(',', '').astype(float)
    merge2['tka_per_game'] = merge2['TkA'] / merge2['GP']
    merge2['SAT For'] = merge2['SAT For'].str.replace(',', '').astype(float)
    merge2['shots_per_game'] = merge2['SAT For'] / merge2['GP']
    return merge2


def game_info(team1, team2):
    stats = model_metrics(input_files)
    team1_id = team_info[team1]
    team2_id = team_info[team2]
    team1_df = stats.loc[stats['Team'] == team1]
    team2_df = stats.loc[stats['Team'] == team2]
    tka_avg = (team1_df['tka_per_game'].values[0] + team2_df['tka_per_game'].values[0])/2
    shots_avg = (team1_df['shots_per_game'].values[0] + team2_df['shots_per_game'].values[0])/2
    savp_avg = (team1_df['5v5 Sv%'].values[0] + team2_df['5v5 Sv%'].values[0])/2
    return [[team1_id, team2_id, savp_avg, shots_avg, tka_avg]]


if __name__ == '__main__':
    X, test = data_processing(files)
    model = create_model(X, test)
    print('Arizona Coyotes', 'Carolina Hurricanes', model.predict(game_info('Arizona Coyotes', 'Carolina Hurricanes')))
    print('Ottawa Senators', 'Detroit Red Wings', model.predict(game_info('Ottawa Senators', 'Detroit Red Wings')))
    print('Pittsburgh Penguins', 'Colorado Avalanche', model.predict(game_info('Pittsburgh Penguins', 'Colorado Avalanche')))
    #print('New Jersey Devils', 'New York Rangers', model.predict(game_info('New Jersey Devils', 'New York Rangers')))
    #print('Buffalo Sabres', 'St. Louis Blues', model.predict(game_info('Buffalo Sabres', 'St. Louis Blues')))
    #print('Minnesota Wild', 'Calgary Flames',
          #model.predict(game_info('Minnesota Wild', 'Calgary Flames')))
    #print('Los Angeles Kings', 'Vegas Golden Knights',
          #model.predict(game_info('Los Angeles Kings', 'Vegas Golden Knights')))
    #print('Dallas Stars', 'Anaheim Ducks',
          #model.predict(game_info('Dallas Stars', 'Anaheim Ducks')))
    #print('Columbus Blue Jackets', 'San Jose Sharks',
          #model.predict(game_info('Columbus Blue Jackets', 'San Jose Sharks')))
    #print('Nashville Predators', 'Chicago Blackhawks',
          #model.predict(game_info('Nashville Predators', 'Chicago Blackhawks')))
    #print('Pittsburgh Penguins', 'Vegas Golden Knights',
     #     model.predict(game_info('Pittsburgh Penguins', 'Vegas Golden Knights')))




