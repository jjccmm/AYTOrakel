import numpy as np
from itertools import permutations
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sympy.utilities.iterables import multiset_permutations
import time
import math

male_participants = ['Eti', 'Gerrit', 'Kevin', 'Martin', 'Paddy', 'Paul', 'Ryan', 'Sandro', 'Sidar', 'Wilson']

female_participants = ['Afra', 'Edda', 'Jana', 'Julia', 'Lina', 'Lisa-Marie', 'Maja',  'Pia',
                       'Shelly', 'Sina', 'Tais'] # 'Melanie',

ayto_events = [{'type': 'box', 'couple': ['Paul', 'Jana'], 'match': 'no'},
               {'type': 'night', 'lights': 2,
                'matching': ['Lisa-Marie', 'Pia', 'Sina', 'Julia', 'Maja', 'Tais', 'Lina', 'Edda', 'Jana', 'Shelly']},
               {'type': 'box', 'couple': ['Ryan', 'Lina'], 'match': 'no'},
               {'type': 'night', 'lights': 2,
                'matching': ['Afra', 'Tais', 'Sina', 'Lina', 'Shelly', 'Lisa-Marie', 'Pia', 'Edda', 'Jana', 'Maja']},
               {'type': 'box', 'couple': ['Kevin', 'Maja'], 'match': 'no'},
               ]


def faster_permutations(n):
    # From https://stackoverflow.com/questions/64291076/generating-all-permutations-efficiently
    # empty() is fast because it does not initialize the values of the array
    # order='F' uses Fortran ordering, which makes accessing elements in the same column fast
    perms = np.empty((math.factorial(n), n), dtype=np.uint8, order='F')
    perms[0, 0] = 0

    rows_to_copy = 1
    for i in range(1, n):
        perms[:rows_to_copy, i] = i
        for j in range(1, i + 1):
            start_row = rows_to_copy * j
            end_row = rows_to_copy * (j + 1)
            splitter = i - j
            perms[start_row: end_row, splitter] = i
            perms[start_row: end_row, :splitter] = perms[:rows_to_copy, :splitter]  # left side
            perms[start_row: end_row, splitter + 1:i + 1] = perms[:rows_to_copy, splitter:i]  # right side

        rows_to_copy *= i + 1

    return perms


def save_match_probabilities(event_number, event_name, remaining_combinations):
    df_cm = pd.DataFrame(match_probabilities, index=female_participants, columns=male_participants + ['No Match'])
    fig = plt.figure(figsize=(10, 8))
    plot = sn.heatmap(df_cm, annot=True, vmin=0, vmax=100, square=True, linewidth=.5)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    fig.suptitle(f'AYTO Match Probabilities after {event_name} - {remaining_combinations} Combinations')
    plot.figure.savefig(f'{event_number}_ayto_match_probabilities_{event_name}.png', dpi=300)
    #plt.show()


def save_light_probabilities(event_number, event_name):
    df_bar = pd.DataFrame(possible_lights)
    print(df_bar.value_counts())
    print(df_bar.value_counts(normalize=True))
    fig = plt.figure(figsize=(10, 8))
    plot = sn.histplot(df_bar, bins=matching_goal+1, binrange=(-0.5, matching_goal+0.5), stat='probability', shrink=0.9)
    # plot.bar_label(plot.containers[0], fmt='%.2f')
    for el in plot.containers[0]:
        print(el)
    plt.legend([], [], frameon=False)
    plt.xlabel("Number of Lights")
    plt.ylim(0, 1)
    plot.set_xticks(range(matching_goal))
    plot.set_xticklabels(range(matching_goal))
    fig.suptitle(f'AYTO Light Probabilities in {event_name}')
    plot.figure.savefig(f'{event_number}_ayto_light_probabilities_{event_name}.png', dpi=300)
    #plt.show()


# Generate all possible matches
matching_goal = 10
no_matches = 2
couples = 10

print(f'AYTO - Match Probabilities')
print('Generate all Permutations')
possible_matches = faster_permutations(len(female_participants))
print('Remove No Matches from Permutations')
possible_matches = np.delete(possible_matches, [10], axis=1)
print('Remove Duplicates')

#possible_matches = np.unique(possible_matches, axis=0)
#possible_matches = list(permutations(list(range(12)), 10))
#print('to numpy')
#possible_matches = np.array(possible_matches, dtype='u1')
#print('done')



# Initial Probability
match_probabilities = np.full((len(female_participants), len(male_participants)+1), 1/len(female_participants)) * 100
for girl_index, girl_probs in enumerate(match_probabilities):
    prob_sum = sum(girl_probs) - (100/len(female_participants))
    match_probabilities[girl_index][len(male_participants)] = 100 - prob_sum


save_match_probabilities(0, 'first Meeting', len(possible_matches))


print(f'Initially possible combinations: {len(possible_matches)}')

match_box_count = 0
match_night_count = 0
event_title = ''

for i, event in enumerate(ayto_events):
    if event['type'] == 'night':
        match_night_count += 1
        event_title = f'Matching Night #{match_night_count}'
        night_matches = [female_participants.index(girl) for girl in event['matching']]
        possible_lights = np.sum(possible_matches == night_matches, axis=1)
        # Light Prediction
        save_light_probabilities(i+1, event_title)
        # Update Match Probabilities
        correct_count_mask = (possible_lights == event['lights'])
        possible_matches = possible_matches[correct_count_mask]
    elif event['type'] == 'box':
        match_box_count += 1
        event_title = f'Match Box #{match_box_count}'
        male_index = male_participants.index(event['couple'][0])
        female_index = female_participants.index(event['couple'][1])
        box_match = np.full(matching_goal, -1)
        box_match[male_index] = female_index
        matching_elements_count = np.sum(possible_matches == box_match, axis=1)
        correct_match_count = 1 if event['match'] == 'yes' else 0
        correct_count_mask = (matching_elements_count == correct_match_count)
        possible_matches = possible_matches[correct_count_mask]

    remaining_possibilities = len(possible_matches)
    print(f'Possible combinations after {i+1} events: {remaining_possibilities}')



    df = pd.DataFrame(possible_matches, columns=male_participants)
    counts = np.zeros((len(female_participants), len(male_participants)+1), dtype=int)
    for boy_index, boy in enumerate(male_participants):
        girl_counts = df[boy].value_counts()
        for (girl_index, count) in list(girl_counts.items()):
            counts[girl_index][boy_index] = count
    #for girl_index, girl_probs in enumerate(counts):
    #    prob_sum = sum(girl_probs)
    #    counts[girl_index][len(male_participants)] = 100-prob_sum
    match_probabilities = np.divide(counts, remaining_possibilities) * 100
    for girl_index, girl_probs in enumerate(match_probabilities):
        prob_sum = sum(girl_probs)
        match_probabilities[girl_index][len(male_participants)] = 100 - prob_sum
    save_match_probabilities(i+1, event_title, remaining_possibilities)


