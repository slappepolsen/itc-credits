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
print("\ntop 10 characters by total appearances:")
print(character_appearances.head(10))

# === Plot dominance: top characters per plot ===
plot_character_counts = df_long.groupby(['plot_id', 'character']).size().reset_index(name='appearances_in_plot')
print("\ntop characters per arc:")
for plot_id in plot_character_counts['plot_id'].unique():
    top_chars = plot_character_counts[plot_character_counts['plot_id'] == plot_id].nlargest(3, 'appearances_in_plot')
    print(f"plot {int(plot_id)}: {', '.join(top_chars['character'])}")

# === Co-occurrence Analysis ===
episode_character_map = df_long.groupby('episode_id')['character'].apply(list).to_dict()
co_occurrence_counts = {}
for characters in episode_character_map.values():
    for char1, char2 in itertools.combinations(sorted(characters), 2):
        pair = tuple(sorted((char1, char2)))
        co_occurrence_counts[pair] = co_occurrence_counts.get(pair, 0) + 1

co_occurrence_df = pd.DataFrame(co_occurrence_counts.items(), columns=['character_pair', 'co_occurrence_count'])
co_occurrence_df[['character_A', 'character_B']] = pd.DataFrame(co_occurrence_df['character_pair'].tolist(), index=co_occurrence_df.index)
co_occurrence_df.drop(columns=['character_pair'], inplace=True)

print("\ntop 10 character pair co-occurrences:")
print(co_occurrence_df.sort_values(by='co_occurrence_count', ascending=False).head(10))

# === Heatmap of top N character co-occurrences ===
N_TOP_CHARACTERS = 15
co_sum = {}

for _, row in co_occurrence_df.iterrows():
    co_sum[row['character_A']] = co_sum.get(row['character_A'], 0) + row['co_occurrence_count']
    co_sum[row['character_B']] = co_sum.get(row['character_B'], 0) + row['co_occurrence_count']

top_characters = sorted(co_sum.items(), key=lambda x: x[1], reverse=True)[:N_TOP_CHARACTERS]
top_character_names = [char for char, _ in top_characters]

print(f"\ntop {N_TOP_CHARACTERS} characters by total co-occurrence:")
for char, count in top_characters:
    print(f"- {char}: {count}")

# Filter for heatmap
filtered_df = co_occurrence_df[
    co_occurrence_df['character_A'].isin(top_character_names) &
    co_occurrence_df['character_B'].isin(top_character_names)
]

# Pivot for symmetric matrix
matrix = filtered_df.pivot_table(
    values='co_occurrence_count',
    index='character_A',
    columns='character_B',
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
            cbar_kws={'label': 'co-occurrence count'})
plt.title(f'character co-occurrence heatmap (top {N_TOP_CHARACTERS})', size=18)
plt.xlabel('character', size=14)
plt.ylabel('character', size=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
