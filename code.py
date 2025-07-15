# this Python script performs the following tasks:
#     - Loads and preprocesses a CSV dataset containing episode and character information.
#     - Calculates total character appearances and lists the most frequently appearing characters.
#     - Reshapes the data to analyze which characters appear in which episodes and maps episodes to plot groups.
#     - Visualizes the appearance arc of a specific character (e.g., "Rose") using a bar plot.
#     - Identifies top characters for each plot group.
#     - Computes co-occurrence statistics for character pairs (i.e., how often two characters appear in the same episode).
#     - Displays the most common character pair co-occurrences.
#     - Creates and visualizes a heatmap of co-occurrence counts among the top N most connected characters.

import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === Load data ===
file_path = "your_dataset_path_here.csv"  # Replace with actual file path or use a file dialog if needed
df = pd.read_csv(file_path)

# === Preprocessing ===
episode_cols = df.columns[1:]
df_characters = df[df['episode'] != 'plot'].copy()
df_characters[episode_cols] = df_characters[episode_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
df_characters['total_appearances'] = df_characters[episode_cols].sum(axis=1)

# === Reshape for analysis ===
df_long = df_characters.set_index('episode')[episode_cols].stack().reset_index()
df_long.columns = ['character', 'episode_id', 'appeared']
df_long = df_long[df_long['appeared'] > 0]

# === Add plot ID mapping ===
plot_mapping_series = df[df['episode'] == 'plot'].iloc[0, 1:].astype(float)
plot_mapping = plot_mapping_series.to_dict()
df_long['episode_id'] = pd.to_numeric(df_long['episode_id'])
df_long['plot_id'] = df_long['episode_id'].map(plot_mapping)

# === Character Popularity ===
character_appearances = df_characters[['episode', 'total_appearances']].sort_values(by='total_appearances', ascending=False)
print("\nTop 10 Characters by Total Appearances:")
print(character_appearances.head(10))

# === Character Arc Plot (example: Rose) ===
if 'Rose' in df_characters['episode'].values:
    rose_appearances = df_characters[df_characters['episode'] == 'Rose'][episode_cols].iloc[0]
    plt.figure(figsize=(15, 6))
    rose_appearances.plot(kind='bar')
    plt.title("Rose's Appearances Per Episode")
    plt.xlabel('Episode ID')
    plt.ylabel('Appearance')
    plt.xticks(rotation=90, fontsize=6)
    plt.tight_layout()
    plt.show()
else:
    print("\n'Rose' character not found in the dataset.")

# === Plot Dominance: Top Characters Per Plot ===
plot_character_counts = df_long.groupby(['plot_id', 'character']).size().reset_index(name='appearances_in_plot')
print("\nTop Characters per Plot Group:")
for plot_id in plot_character_counts['plot_id'].unique():
    top_chars = plot_character_counts[plot_character_counts['plot_id'] == plot_id].nlargest(3, 'appearances_in_plot')
    print(f"Plot {int(plot_id)}: {', '.join(top_chars['character'])}")

# === Co-occurrence Analysis ===
episode_character_map = df_long.groupby('episode_id')['character'].apply(list).to_dict()
co_occurrence_counts = {}
for characters in episode_character_map.values():
    for char1, char2 in itertools.combinations(sorted(characters), 2):
        pair = tuple(sorted((char1, char2)))
        co_occurrence_counts[pair] = co_occurrence_counts.get(pair, 0) + 1

co_occurrence_df = pd.DataFrame(co_occurrence_counts.items(), columns=['Character_Pair', 'Co_occurrence_Count'])
co_occurrence_df[['Character_A', 'Character_B']] = pd.DataFrame(co_occurrence_df['Character_Pair'].tolist(), index=co_occurrence_df.index)
co_occurrence_df.drop(columns=['Character_Pair'], inplace=True)

print("\nTop 10 Character Pair Co-occurrences:")
print(co_occurrence_df.sort_values(by='Co_occurrence_Count', ascending=False).head(10))

# === Heatmap of Top N Character Co-occurrences ===
N_TOP_CHARACTERS = 15
co_sum = {}

for _, row in co_occurrence_df.iterrows():
    co_sum[row['Character_A']] = co_sum.get(row['Character_A'], 0) + row['Co_occurrence_Count']
    co_sum[row['Character_B']] = co_sum.get(row['Character_B'], 0) + row['Co_occurrence_Count']

top_characters = sorted(co_sum.items(), key=lambda x: x[1], reverse=True)[:N_TOP_CHARACTERS]
top_character_names = [char for char, _ in top_characters]

print(f"\nTop {N_TOP_CHARACTERS} Characters by Total Co-occurrence:")
for char, count in top_characters:
    print(f"- {char}: {count}")

# Filter for heatmap
filtered_df = co_occurrence_df[
    co_occurrence_df['Character_A'].isin(top_character_names) &
    co_occurrence_df['Character_B'].isin(top_character_names)
]

# Pivot for symmetric matrix
matrix = filtered_df.pivot_table(
    values='Co_occurrence_Count',
    index='Character_A',
    columns='Character_B',
    fill_value=0
)

ordered_chars = sorted(top_character_names)
matrix = matrix.reindex(index=ordered_chars, columns=ordered_chars, fill_value=0)
matrix = matrix + matrix.T

# Upper triangle mask
mask = np.triu(np.ones_like(matrix, dtype=bool))

# Plot heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(matrix, mask=mask, annot=True, cmap='viridis', fmt='g',
            linewidths=.5, linecolor='lightgray',
            cbar_kws={'label': 'Co-occurrence Count'})
plt.title(f'Character Co-occurrence Heatmap (Top {N_TOP_CHARACTERS})', size=18)
plt.xlabel('Character', size=14)
plt.ylabel('Character', size=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
