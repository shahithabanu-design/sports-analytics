import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import os

# ==========================================
# Simulating Random Player Performance Data
# ==========================================
def player_performance_optimization():
    # Randomly generate player performance data
    players = [f'Player{i+1}' for i in range(10)]
    distance_covered = [random.uniform(5, 15) for _ in range(10)]  # Random km between 5 and 15
    average_speed = [random.uniform(4, 6) for _ in range(10)]  # Random km/h between 4 and 6
    pass_accuracy = [random.uniform(70, 95) for _ in range(10)]  # Random pass accuracy between 70% and 95%
    stamina = [random.uniform(75, 100) for _ in range(10)]  # Random stamina between 75% and 100%

    data = {
        'Player': players,
        'Distance_Covered': distance_covered,
        'Average_Speed': average_speed,
        'Pass_Accuracy': pass_accuracy,
        'Stamina': stamina
    }

    df = pd.DataFrame(data)
    features = df[['Distance_Covered', 'Average_Speed', 'Pass_Accuracy', 'Stamina']]

    # KMeans clustering for performance grouping
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    df['Performance_Cluster'] = kmeans.fit_predict(features)

    # Visualize clusters
    plt.figure(figsize=(8, 6))
    for cluster in df['Performance_Cluster'].unique():
        cluster_data = df[df['Performance_Cluster'] == cluster]
        plt.scatter(cluster_data['Distance_Covered'], cluster_data['Average_Speed'], label=f'Cluster {cluster}')
    plt.title("Player Performance Clusters")
    plt.xlabel("Distance Covered (km)")
    plt.ylabel("Average Speed (km/h)")
    plt.legend()
    plt.grid()
    plt.show()

    return df

# ==========================================
# Simulating Injury Prediction Data
# ==========================================
def injury_prediction():
    # Randomly generate injury prediction data
    players = [f'Player{i+1}' for i in range(10)]
    workload = [random.randint(60, 100) for _ in range(10)]  # Random workload between 60 and 100
    rest_periods = [random.randint(3, 15) for _ in range(10)]  # Random rest period between 3 and 15 days
    training_intensity = [random.randint(5, 10) for _ in range(10)]  # Random intensity between 5 and 10
    injury = [random.randint(0, 1) for _ in range(10)]  # Random injury status: 1 = injured, 0 = not injured

    data_injury = {
        'Player': players,
        'Workload': workload,
        'Rest_Periods': rest_periods,
        'Training_Intensity': training_intensity,
        'Injury': injury
    }

    df_injury = pd.DataFrame(data_injury)

    X = df_injury[['Workload', 'Rest_Periods', 'Training_Intensity']]
    y = df_injury['Injury']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Injury Prediction Accuracy: {accuracy * 100:.2f}%")

    return df_injury

# ==========================================
# Simulating Tactical Analysis Using Random Positions
# ==========================================
def tactical_analysis():
    # Randomly generate player positions on the field (0-100 for both X and Y coordinates)
    players = [f'Player{i+1}' for i in range(10)]
    x_positions = [random.randint(0, 100) for _ in range(10)]
    y_positions = [random.randint(0, 100) for _ in range(10)]

    df_positions = pd.DataFrame({
        'Player': players,
        'X_Pos': x_positions,
        'Y_Pos': y_positions
    })

    # Create a pitch heatmap based on player positions
    pitch = np.zeros((100, 100))  # Create a 100x100 field grid
    for _, row in df_positions.iterrows():
        x, y = row['X_Pos'], row['Y_Pos']
        pitch[x, y] += 1

    plt.figure(figsize=(10, 7))
    sns.heatmap(pitch, cmap="YlGnBu", linewidths=0.5)
    plt.title("Tactical Analysis: Player Positions Heatmap")
    plt.xlabel("Field X")
    plt.ylabel("Field Y")
    plt.show()

    return df_positions

# ==========================================
# Simulating Game Strategy Using Random Player Formations
# ==========================================
def game_strategy_improvement():
    # Randomly generate player formation positions on the field
    players = [f'Player{i+1}' for i in range(10)]
    x_positions = [random.randint(0, 100) for _ in range(10)]
    y_positions = [random.randint(0, 100) for _ in range(10)]

    df_formation = pd.DataFrame({
        'Player': players,
        'X_Pos': x_positions,
        'Y_Pos': y_positions
    })

    positions = df_formation[['X_Pos', 'Y_Pos']]
    dbscan = DBSCAN(eps=20, min_samples=2)
    df_formation['Formation_Cluster'] = dbscan.fit_predict(positions)

    # Visualize player formation clusters
    plt.figure(figsize=(8, 6))
    for cluster in df_formation['Formation_Cluster'].unique():
        cluster_data = df_formation[df_formation['Formation_Cluster'] == cluster]
        plt.scatter(cluster_data['X_Pos'], cluster_data['Y_Pos'], label=f'Cluster {cluster}')
    plt.title("Game Strategy: Player Formations")
    plt.xlabel("Field X")
    plt.ylabel("Field Y")
    plt.legend()
    plt.grid()
    plt.show()

    return df_formation

# ==========================================
# Full Pipeline Using Random Data and Simulated Video Paths
# ==========================================
def sports_analytics_pipeline():
    # Simulate Video Path (This is just an example, you can replace with actual video data)
    video_path = "path/to/soccer_game_video.mp4"  # Simulated video path for game footage
    print(f"Using video data from: {video_path}")

    player_df = player_performance_optimization()
    injury_df = injury_prediction()
    tactical_df = tactical_analysis()  # Simulating tactical analysis
    strategy_df = game_strategy_improvement()

    return player_df, injury_df, tactical_df, strategy_df

# Run the full pipeline
output = sports_analytics_pipeline()
