import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import json
import os
from PIL import Image, ImageDraw, ImageFont
import subprocess
import shutil


def main():
    season = 's5vip'
    save_reel = False
    
    data = read_season_data(season=season)
    create_folders(season)
    df = generate_all_possible_matches(data)

    logs = {'light_map': np.zeros((11, 10)),
            'light_history': [],
            'confirmed_matches': [[], []],
            'confirmed_no_matches': [[], []],
            'sold_matches': [[], []],
            'event_history': [],
            'night_matches': [],
            'nights_together': np.zeros((len(data['group_of_more']), len(data['group_of_ten'])), dtype=np.int8),
            'match_box_count': 0,
            'match_night_count': 0,
            'week_with_perfect_match': [],
            'frame_index': 0}
    
    # Initialize Match Probabilities
    # Start with all 0 just for visualisation
    match_probabilities = np.full((len(data['group_of_more']), len(data['group_of_ten'])), 0, dtype=np.float64)
    # Initial Probabilities
    new_match_probabilities = np.full((len(data['group_of_more']), len(data['group_of_ten'])),
                                      100 / len(data['group_of_ten']), dtype=np.float64)
    logs['event_history'].append('>')
    if save_reel:
        save_reel_frames(data, logs, match_probabilities, new_match_probabilities, {'type': 'intro'})
    match_probabilities = new_match_probabilities
    save_match_probabilities(data, logs, match_probabilities, "1-0", 'Einzug', len(df))

    for week in data['weeks']:
        week_number = week['number']
        generate_week_cover(data, week_number)
        for event_number, event in enumerate(week['events'], start=1):
            event_name = f'Week {week_number}, Event {event_number},  Type {event["type"]}'
            event_number = f'{week_number}-{event_number}'

            if event['type'] == 'night':
                df = update_after_night_event(data, logs, df, event, event_number)

            elif event['type'] == 'box':
                df = update_after_box_event(data, logs, df, event, week_number)

            possible_matches = df[data['group_of_ten']].to_numpy().astype(np.uint8)
            remaining_possibilities = len(possible_matches)
            print(f'Possible combinations after Week {week_number} Event {event_number}: {remaining_possibilities}')

            counts = np.zeros((len(data['group_of_more']), len(data['group_of_ten'])), dtype=int)
            for got_member_index, got_member in enumerate(data['group_of_ten']):
                gom_counts = df[got_member].value_counts()
                for (gom_member_index, count) in list(gom_counts.items()):
                    counts[gom_member_index][got_member_index] = count

            # Sum up all possibilities in girl row
            total_possibilities_per_girl = np.sum(counts, axis=1)
            # Convert counts to percentage.
            # Sum of girl row is = 100%,
            # Sum of Boy column can be > 100% if double/triple matches exist
            new_match_probabilities = (counts / total_possibilities_per_girl[:, None]) * 100
            new_match_probabilities = np.nan_to_num(new_match_probabilities)
            if save_reel:
                save_reel_frames(data, logs, match_probabilities, new_match_probabilities, event)
            match_probabilities = new_match_probabilities
            save_match_probabilities(data, logs, match_probabilities, event_number, event_name, len(df))
            if len(df) < 40:
                save_insta_combinations(data, df, event_number)
                

    save_light_map(data, logs)

    if len(df) < 500:
        df.to_csv(f'{season}/remaining_combinations_dm.csv', index=False)
    
    if save_reel:
        merge_reel_frames(data)
        delte_folder(f'{season}/reel_raw')
        delte_folder(f'{season}/reel_frames')
            

def generate_week_cover(data, week_number):
    season = data['season']
    font_90 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=90)
    font_18 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=18)
    img = Image.open(f'insta_styles/image_backgrounds/ayto_{season}.png')
    
    ayto='AYTO'
    season_number = data['season'].replace('vip','').replace('s','')
    season_vip = 'VIP' if 'vip' in data['season'] else ''
    season=f'S{season_number} {season_vip}'
    week=f'W{week_number}'
    insta = '@AYTOrakel'
    
    d = ImageDraw.Draw(img)
    d.text((img.width/2, img.height/2-120), ayto, fill='white', font=font_90, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, img.height/2), season, fill='white', font=font_90, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, img.height/2+120), week, fill='red', font=font_90, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((10, img.height-30), insta, fill='white', font=font_18, stroke_width=2, stroke_fill='black', anchor='la')
    
    img.save(f'{data["season"]}/insta/{data["season"]}_{week_number}_0_insta_cover.png')


def delte_folder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))


def update_after_box_event(data, logs, df, event, week_number):
    # save meta data
    logs['match_box_count'] += 1
    logs['night_matches'] = []  # For a box event we dont want to visualize the last matching night pairs
    # convert to index
    got = data['group_of_ten']
    gom = data['group_of_more']
    got_idx = got.index(event['couple'][0])
    gom_idx = gom.index(event['couple'][1])

    if event['match'] == 'sold':
        logs['sold_matches'][0].append(got_idx)
        logs['sold_matches'][1].append(gom_idx)
        logs['event_history'].append(f'd')
        return df

    # Create matching vector for match box pair
    box_match = np.full(len(got), -1)
    box_match[got_idx] = gom_idx

    # Check for which of the possibilities this is a match
    possible_matches = df[got].to_numpy().astype(np.uint8)
    matching_elements_count = np.sum(possible_matches == box_match, axis=1)
    match_mask = (matching_elements_count == 1)
    match_ids = df[['id']].to_numpy()[match_mask].flatten()

    if event['match'] == 'yes':
        # When it is a perfect Match: keep only the ids that match
        df = df[df['id'].isin(match_ids)]
        logs['confirmed_matches'][0].append(got_idx)
        logs['confirmed_matches'][1].append(gom_idx)
        logs['event_history'].append(f'P')
        logs['week_with_perfect_match'].append(week_number)

        if len(event['multi_match']) > 0:
            # The Matchbox revealed the multi match
            # Collect all the indexes of the multi match members
            multi_match_index = [gom.index(multi_match_member) for multi_match_member in event['multi_match']]
            multi_match_index.append(gom_idx)
            multi_match_index = sorted(multi_match_index)  # Sort them to have a fixed order

            # Compare to the multi match index of the possibilities and only keep when equal
            multi_match_columns = [f'mm{i + 1}' for i in range(data['multi_match_size'])]
            multi_match_mask = df[multi_match_columns].apply(lambda row: sorted(row) == multi_match_index, axis=1)
            df = df[multi_match_mask]

            # All members of the multi match array end with no match and no money
            # We can delete all possibilities where they still occur in the matching night
            for mm_member in event['multi_match']:
                mm_index = gom.index(mm_member)
                logs['confirmed_matches'][0].append(got_idx)
                logs['confirmed_matches'][1].append(mm_index)
                df = df[~df.iloc[:, :10].isin([mm_index]).any(axis=1)]

        else:
            # We have a perfect match but no nobody from the pair belongs to the multi match
            # Therefore we can delete all possibilities where a member ob the perfect match is in the multi match
            for i in range(data['multi_match_size']):
                df = df[df[f'mm{i + 1}'] != gom_idx]

    else:
        # When it is a no Match: keep only the ids that dont match
        df = df[~df['id'].isin(match_ids)]
        logs['confirmed_no_matches'][0].append(got_idx)
        logs['confirmed_no_matches'][1].append(gom_idx)
        logs['event_history'].append(f'X')

    return df


def update_after_night_event(data, logs, df, event, event_number):
    # save meta data
    logs['match_night_count'] += 1
    light_number = event['lights']
    logs['light_history'].append(light_number)
    logs['event_history'].append(f'${light_number}$')

    # convert name to index
    got = data['group_of_ten']
    gom = data['group_of_more']
    logs['night_matches'] = [gom.index(gom_member) if gom_member in gom else -1 for gom_member in event['matching']]
    for boy, girl in enumerate(logs['night_matches']):
        logs['nights_together'][girl][boy] += 1

    # Check for each possibility how many light would go on for the pairs of the matching night
    possible_matches = df[got].to_numpy().astype(np.uint8)
    possible_lights = np.sum(possible_matches == logs['night_matches'], axis=1)
    df[f'lights'] = possible_lights

    # Due to the Multi Matches we have several possibilities for each multi match:
    # 1 2 3 ID MM1 MM2
    # A B C  1   C   D
    # A B D  1   C   D
    # A B C  2   B   D
    # A D C  2   B   D
    # These are the two options A B (C/D) and A (B/D) C
    # If we now assume a pairing in the matching night of A B C with 2 lights
    # 1 2 3 ID MM1 MM2 -> lights -> lights corrected
    # A B C  1   C   D    3         3
    # A B D  1   C   D    2         3
    # A B C  2   B   D    3         3
    # A D C  2   B   D    2         3
    # Then ABD with CD as Multi Match and ADC with BD as multi match are also valid options
    # Thats why the lights are corrected to the max lights of each group (same id)
    max_values = df.groupby('id')['lights'].transform('max')
    df['lights'] = max_values

    save_light_probabilities(data, logs, df, event_number, event)

    # TODO Fix here if the wrong count mask is needed
    # correct_count_mask = (possible_lights == event['lights'])
    # wrong_count_mask = (possible_lights == (event['lights'] + 1))
    correct_count_mask = (max_values == event['lights'])
    wrong_count_mask = (max_values == (event['lights'] + 1))

    correct_ids = df[['id']].to_numpy()[correct_count_mask].flatten()
    wrong_ids = df[['id']].to_numpy()[wrong_count_mask].flatten()

    c_ids = set(correct_ids)
    w_ids = set(wrong_ids)
    intersect = c_ids.intersection(w_ids)
    if len(intersect) > 0:
        print('Wrong ids found!')

    df = df[df['id'].isin(correct_ids) & ~df['id'].isin(wrong_ids)]
    # possible_matches = df[data['group_of_ten']].to_numpy().astype(np.uint8)

    return df


def create_folders(season):
    season_root = os.path.join(os.getcwd(), season)
    for sub_folder in ['reel_raw', 'reel_frames', 'matches', 'matches_tight', 'lights', 'lights_tight', 'insta']:
        sub_folder_path = os.path.join(season_root, sub_folder)
        os.makedirs(sub_folder_path, exist_ok=True)


def save_reel_frames(data, logs, match_probabilities, new_match_probabilities, event, event_frames=15, transition_frames=30):
    got = data['group_of_ten']
    gom = data['group_of_more']
    diff = new_match_probabilities - match_probabilities
    diff_step = diff / transition_frames
    match_lw_max = 6
    match_lw_step = (match_lw_max-1)/transition_frames
    box_marker_max = 1500
    box_marker_step = (box_marker_max-40)/transition_frames
    box = [0]

    if event['type'] == 'box':
        got_indexs = [got.index(event['couple'][0])]
        gom_indexs = [gom.index(event['couple'][1])]

        if event['match'] == 'yes':
            for girl in event['multi_match']:
                got_indexs.append(got.index(event['couple'][0]))
                gom_indexs.append(gom.index(girl))

            result = 'P'
            c = 'green'
        elif event['match'] == 'no':
            result = 'X'
            c = 'red'
        else:
            result = 'd'
            c = 'lightblue'

        box = [got_indexs, gom_indexs, result, c, box_marker_max]

    for j in range(transition_frames):
        match_probabilities += diff_step
        match_probabilities = np.clip(match_probabilities, a_min=0, a_max=None)  # Avoid negative values resulting in "-0"
        box[-1] -= box_marker_step
        save_match_probabilities_reel(data, logs, match_probabilities, match_lw_max-match_lw_step*j, box, transition_frames, event_frames)
        logs['frame_index'] += 1

    match_probabilities = new_match_probabilities

    for _ in range(event_frames):
        save_match_probabilities_reel(data, logs, match_probabilities, 1, [], transition_frames, event_frames)
        logs['frame_index'] += 1


def save_match_probabilities_reel(data, logs, match_probabilities, match_lw, box, transition_frames, event_frames):
    got = data['group_of_ten']
    gom = data['group_of_more']
    df_cm = pd.DataFrame(match_probabilities, index=gom, columns=got)
    image_size = (len(got) * 0.6 * 0.51, (len(gom) + 0.55) * 0.6 * 0.51)
    fig, (bar, ax) = plt.subplots(2, 1, height_ratios=[1, 24], figsize=image_size)

    bar.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    bar.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
    bar.set_xlim(0, 21 * (transition_frames + event_frames)-1)
    bar.set_ylim(0, 1)
    for e in range(21):
        bar.axvline(e * (transition_frames + event_frames), lw=0.5, color='white')

    for x, event_type in enumerate(logs['event_history']):
        size = 35
        e_color = 'white'
        if event_type == '>':
            color = 'white'
        elif event_type == 'P':
            color = 'green'
        elif event_type == 'X':
            color = 'red'
        elif event_type == 'd':
            color = 'lightblue'
        else:
            color = 'yellow'
            size = 60
            e_color = 'yellow'
        x = (x + 0.5) * (transition_frames + event_frames)
        bar.scatter([x], [0.5], marker=event_type, s=size, color=color, edgecolor=e_color, zorder=50)
    bar.add_patch(patch.Rectangle((0, 0), logs['frame_index'], 1, color='#222222', fill=True))
    plot = sn.heatmap(df_cm, annot=True, fmt='.0f', vmin=0, vmax=100, square=True, linewidth=.5, cbar=False,
                      annot_kws={"size": 6}, ax=ax)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plot.set_yticklabels(plot.get_yticklabels(), rotation=45, verticalalignment='top')

    night_count_xs = []
    night_count_ys = []
    for x, _ in enumerate(logs['nights_together'][0]):
        for y, _ in enumerate(logs['nights_together']):
            for nights in range(logs['nights_together'][y][x]):
                night_count_xs.append(x + (nights + 1) * 0.1)
                night_count_ys.append(y + 0.1)
    ax.scatter(night_count_xs, night_count_ys, marker='o', s=5, color='yellow', edgecolor='black')

    for boy, girl in enumerate(logs['night_matches']):
        ax.plot([boy, boy+1, boy+1, boy, boy], [girl, girl, girl+1, girl+1, girl], color='yellow', lw=match_lw)

    ax.scatter([x+0.78 for x in logs['confirmed_matches'][0]], [x+0.78 for x in logs['confirmed_matches'][1]], marker='P', s=35, color='green', edgecolor='black')
    ax.scatter([x+0.78 for x in logs['confirmed_no_matches'][0]], [x+0.78 for x in logs['confirmed_no_matches'][1]], marker='X', s=40, color='red', edgecolor='black')
    ax.scatter([x+0.78 for x in logs['sold_matches'][0]], [x+0.78 for x in logs['sold_matches'][1]], marker='d', s=30, color='lightblue', edgecolor='black')

    if len(box) > 1:
        ax.scatter([x + 0.78 for x in box[0]], [y + 0.78 for y in box[1]], marker=box[2], s=box[4], color=box[3], edgecolor='black')

    plt.subplots_adjust(left=0.01, bottom=0.00, right=0.99, top=0.99, hspace=0)
    season = data['season']
    plot.figure.savefig(f'{season}/reel_raw/{season}_ayto_summary_{logs["frame_index"]}.png', dpi=300)
    plt.close()


def save_match_probabilities(data, logs, match_probabilities, event_number, event_name, remaining_combinations):
    got = data['group_of_ten']
    gom = data['group_of_more']
    df_cm = pd.DataFrame(match_probabilities, index=gom, columns=got)
    fig = plt.figure(figsize=(len(got) * 0.6, len(gom) * 0.6))
    ax = fig.add_subplot(111)
    plot = sn.heatmap(df_cm, annot=True, fmt='.0f', vmin=0, vmax=100, square=True, linewidth=.5, cbar=False,
                      annot_kws={"size": 10}, ax=ax)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45, horizontalalignment='right')
    plot.set_yticklabels(plot.get_yticklabels(), rotation=45, verticalalignment='top')
    fig.suptitle(f'AYTO Match Probabilities after {event_name} - {remaining_combinations} Combinations')
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.988, top=0.957)

    night_count_xs = []
    night_count_ys = []
    for x,_ in enumerate(logs['nights_together'][0]):
        for y,_ in enumerate(logs['nights_together']):
            for nights in range(logs['nights_together'][y][x]):
                night_count_xs.append(x+(nights+1)*0.1)
                night_count_ys.append(y+0.1)
    ax.scatter(night_count_xs, night_count_ys, marker='o', s=10, color='yellow', edgecolor='black')

    for boy, girl in enumerate(logs['night_matches']):
        ax.plot([boy, boy+1, boy+1, boy, boy], [girl, girl, girl+1, girl+1, girl], color='yellow', lw=1)

    ax.scatter([x+0.85 for x in logs['confirmed_matches'][0]], [x+0.85 for x in logs['confirmed_matches'][1]], marker='P', s=70, color='green', edgecolor='black')
    ax.scatter([x+0.85 for x in logs['confirmed_no_matches'][0]], [x+0.85 for x in logs['confirmed_no_matches'][1]], marker='X', s=80, color='red', edgecolor='black')
    ax.scatter([x+0.85 for x in logs['sold_matches'][0]], [x+0.85 for x in logs['sold_matches'][1]], marker='d', s=60, color='lightblue', edgecolor='black')

    season = data['season']

    plot.figure.savefig(f'{season}/matches/{season}_{event_number}_ayto_match_probabilities_{event_name}.png', dpi=300)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    fig.suptitle(f'')
    tight_image_name = f'{season}_{event_number}_ayto_match_probabilities_{event_name}_tight.png'
    plot.figure.savefig(f'{season}/matches_tight/{tight_image_name}', dpi=300)
    plt.close()
    save_insta_probabilities(data, tight_image_name, event_number, event_name, remaining_combinations)


def save_insta_probabilities(data, tight_image_name, event_number, event_name, remaining_combinations):
    font_18 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=18)
    font_30 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=30)
    font_85 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=85)
    
    season = data['season']
    img = Image.open(f'insta_styles/image_backgrounds/ayto_{season}.png')
    face_layer = Image.open(f'insta_styles/image_face_layers/ayto_{season}.png')
    
    img_match_probs = Image.open(f'{season}/matches_tight/{tight_image_name}')
    img_match_probs.thumbnail((810, 891), Image.LANCZOS)
    frame = 7
    new_size = (img_match_probs.width + frame*2, img_match_probs.height + frame*2)
    img_match_probs_border = Image.new("RGB", new_size)
    img_match_probs_border.paste(img_match_probs, (frame, frame))
    
    season_number = season.replace('vip','').replace('s','')
    season_vip = 'VIP' if 'vip' in season else ''
    week_number = event_number.split('-')[0]
    event_number = event_number.split('-')[1]
    ayto=f'AYTO S{season_number} {season_vip} W{week_number}'
    if 'box' in event_name:
        event='Match Box'
    elif 'night' in event_name:
        event=f'Matching Night'
    elif 'Einzug' in event_name:
        event='Einzug'
    combinations=f'Mögliche Kombinationen:  {remaining_combinations:,}'.replace(',','.')
    aytorakel = '@AYTOrakel'
    
    d = ImageDraw.Draw(img)
    d.text((img.width/2, 60), ayto, fill='white', font=font_85, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, 155), event, fill='red', font=font_85, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, 240), combinations, fill='white', font=font_30, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((10, img.height-30), aytorakel, fill='white', font=font_18, stroke_width=2, stroke_fill='black', anchor='la')
    
    img.paste(img_match_probs_border, (220, 270))
    img.paste(face_layer, (0, 0), face_layer)
    
    img.save(f'{season}/insta/{season}_{week_number}_{event_number}_insta_{event}.png')


def read_season_data(season: str) -> dict:
    data_file = os.path.join(os.getcwd(), 'ayto_data.json')
    if not os.path.isfile(data_file):
        print(f'The AYTO Data file not found: {data_file}')
        raise SystemExit

    with open(data_file, 'r') as file:
        data = json.load(file)

    if season not in data.keys():
        print(f'Season "{season}" is not in the AYTO Data File')
        print(f'Available Seasons are {list(data.keys())}')
        raise SystemExit
    season_data = data[season]
    season_data['season'] = season

    return season_data


def calculate_multi_match_count(group_size: int, multi_match_size: int, known_multi_match_member: str) -> int:
    # Calculates how many multi matches are possible
    # Assuming one member of the multi match is known
    # Is calculated with n over k with
    # n: size of the group of possible multi match people
    # k: size of the subgroup required to complete the multi match
    
    if known_multi_match_member == "":
        known_mmm_count = 0
    else: 
        # One multi match member is known
        known_mmm_count = 1
    
    n = group_size - known_mmm_count 
    k = multi_match_size - known_mmm_count 
    return math.factorial(n) // (math.factorial(k) * math.factorial(n - k))


def generate_all_possible_matches(data):
    got = data['group_of_ten']
    gom = data['group_of_more']

    # Calculate how many options for multi matches exists
    multi_match_count = calculate_multi_match_count(len(gom), data['multi_match_size'], data['known_multi_match_member']) 
    # With no multi matches the total amount of combinations is 10!
    # With Multi matches we have to multiply that by the options we have for multi matches
    total_combinations = int(multi_match_count * math.factorial(len(got)))

    # Create empty array in which we fill batch-wise the possible matches
    # First 10 columns are index of matched person, then match id, then member of multi match
    initial_combinations = np.ones((total_combinations * data['multi_match_size'], 11 + data['multi_match_size']),
                                     dtype=np.uint32)

    # Each possible match gets an unique id
    match_id = 0
    # The combinations are generated in batches for each possible multi match
    completed_mm = 0
    for multi_match in itertools.combinations(gom, data['multi_match_size']):
        if data['known_multi_match_member'] != "":
            # If we have the name of one multi match member, we need to ensure that that person in in the generated multi match
            if data['known_multi_match_member'] not in multi_match:
                continue
        
        # Get the Indexes of the remaining members of the group of more (i.e. which are not in the multi match)
        remaining_gom_idx = [[gom.index(gom_member)] for gom_member in gom if gom_member not in multi_match]
        # Get the index array if the multi match members
        multi_match_idx = [gom.index(mm_member) for mm_member in multi_match]
        # Combine the Index arrays to the
        available_options = remaining_gom_idx + [multi_match_idx]

        combinations_for_mm = []
        for perm in tqdm(itertools.permutations(available_options), total=math.factorial(len(available_options)),
                         desc=f'{completed_mm + 1}/{int(multi_match_count)} Generating with Multi Match {multi_match}'):

            index_of_multi_match = perm.index(multi_match_idx)

            # row is what we will save as one possible combination
            row = [member[0] for member in perm]  # first 10 are index of person,
            row += [match_id]  # Then match_id
            row += multi_match_idx  # Then multi match member indexes

            # Generate and append a row where each of the multi match member is within the first 10 entries
            for multi_match_member in multi_match_idx:
                row_option = row.copy()
                row_option[index_of_multi_match] = multi_match_member
                combinations_for_mm.append(row_option)
            match_id = match_id + 1
        combinations_for_mm = np.array(combinations_for_mm, dtype=np.uint32)

        # Each member of the multi match (mm) can end up in the perfect match
        matches_per_mm = math.factorial(len(got)) * data['multi_match_size']

        # Save the generated batch of combinations in our array
        initial_combinations[completed_mm * matches_per_mm:(completed_mm + 1) * matches_per_mm] = combinations_for_mm
        completed_mm += 1

    column_names = got + ['id'] + [f'mm{i+1}' for i in range(data['multi_match_size'])]
    df = pd.DataFrame(initial_combinations, columns=column_names)

    for col in [col for col in df.columns if col != 'id']:
        df[col] = df[col].astype('uint8')

    print(f'All {len(df)} Initial Combinations generated.')
    return df


def save_light_probabilities(data, logs, df, event_number, event):
    for light, prob in df[f'lights'].value_counts(normalize=True).items():
        logs['light_map'][10-light][logs['match_night_count']-1] = prob*100
        print(f'{light}: {prob*100:.5f}')

    matching_goal = len(data['group_of_ten'])
    fig = plt.figure(figsize=(11 * 0.6, 5 * 0.6))
    plot = sn.histplot(df[f'lights'], bins=matching_goal + 1, binrange=(-0.5, matching_goal + 0.5), stat='probability',
                       shrink=0.9, color='yellow', edgecolor='black')
    plt.legend([], [], frameon=False)
    plt.xlabel("Number of Lights")
    plt.ylim(0, 1)
    plt.axis('off')
    plot.set_xticks(range(matching_goal + 1))
    plot.set_xticklabels(range(matching_goal + 1))
    fig.suptitle(f'AYTO Light Probabilities in Week {logs["match_night_count"]}')
    plt.subplots_adjust(left=0.095, bottom=0.148, right=0.986, top=0.895)
    season = data['season']
    plot.figure.savefig(f'{season}/lights/{season}_{logs["match_night_count"]}_ayto_light_probabilities.png', dpi=300)
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    fig.suptitle(f'')
    tight_image_name = f'{season}_{logs["match_night_count"]}_ayto_light_probabilities_tight.png'
    plot.figure.savefig(f'{season}/lights_tight/{tight_image_name}', dpi=300)
    plt.close()
    save_insta_light_probabilities(data, logs, tight_image_name, event_number, event['lights'], )


def save_insta_light_probabilities(data, logs, tight_image_name, event_number, number_of_lights):
    font_18 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=18)
    font_30 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=30)
    font_50 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=50)
    font_85 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=85)
    
    season = data['season']

    img = Image.open(f'insta_styles/image_backgrounds/ayto_{season}.png')
    img_light_probs = Image.open(f'{season}/lights_tight/{tight_image_name}')
    
    # Replace white with transparency
    img_light_probs = img_light_probs.convert('RGBA')
    data = img_light_probs.getdata()
    new_data = []
    for item in data:
        # change all white (also shades of whites)
        if item[2] in list(range(255, 256)):
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
    img_light_probs.putdata(new_data)
    
    img_light_probs = img_light_probs.resize((int(img_light_probs.width*0.5), int(img_light_probs.height*0.8)))
    
    season_number = season.replace('vip','').replace('s','')
    season_vip = 'VIP' if 'vip' in season else ''
    week_number = event_number.split('-')[0]
    event_number = event_number.split('-')[1]
    ayto=f'AYTO S{season_number} {season_vip} W{week_number}'
    event=f'Matching Night'
    aytorakel = '@AYTOrakel'
    lights='Wahrscheinlichkeiten für Lichter'
    
    d = ImageDraw.Draw(img)
    d.text((img.width/2, 60), ayto, fill='white', font=font_85, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, 155), event, fill='red', font=font_85, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, 260), lights, fill='white', font=font_50, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((10, img.height-30), aytorakel, fill='white', font=font_18, stroke_width=2, stroke_fill='black', anchor='la')
    
    # Draw X Axis with Light Number
    for i in range(11):
        if i==number_of_lights:
            color='yellow'
        else:
            color='white'
        d.text((172+81*i, 1120), f'{i}', fill=color, font=font_50, stroke_width=7, stroke_fill='black', anchor='mm')
        if i == len(logs['week_with_perfect_match']):
            d.text((172+81*i, 1170), f'Black', fill='white', font=font_18, stroke_width=3, stroke_fill='black', anchor='mm')
            d.text((172+81*i, 1190), f'Out', fill='white', font=font_18, stroke_width=3, stroke_fill='black', anchor='mm')

    # Draw Y Axis with Probability
    for i in range(6):
        start= 1092
        step = 141
        d.text((65, start-i*step), f'{i*20}%', fill='white', font=font_30, stroke_width=7, stroke_fill='black', anchor='mm')
        d.line([(65+60, start-i*step), (1030, start-i*step)], fill='white', width=2)

    img.paste(img_light_probs, (80, 380), img_light_probs)

    img.save(f'{season}/insta/{season}_{week_number}_{event_number}_insta_lights.png')

    
def save_light_map(data, logs):
    df_lights = pd.DataFrame(logs['light_map'])
    fig = plt.figure(figsize=(10*0.6, 11*0.6))
    ax = fig.add_subplot(111)
    plot = sn.heatmap(df_lights, annot=True, fmt='.0f', vmin=0, vmax=100, square=True, linewidth=.5, cbar=False,
                      annot_kws={"size": 10}, ax=ax)
    # Make yellow edge around actual lights
    for night, lights in enumerate(logs['light_history']):
        lights = 10 - lights
        ax.plot([night, night + 1, night + 1, night, night], [lights, lights, lights + 1, lights + 1, lights], color='yellow', lw=3)

    # Make line for expected lights
    expected_lights = []
    for i in range(10):
        night_i = logs['light_map'][:, i]
        expected_lights_i = sum((night_i / 100) * np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]))
        expected_lights.append(10.5-expected_lights_i)

    sn.lineplot(x=[x+0.5 for x in range(10)], y=expected_lights, lw=2, color='yellow')

    # Mark Confirmed Matches with green Plus
    for i, week in enumerate(logs['week_with_perfect_match']):
        ax.scatter([x+0.5 for x in range(week-1, 10)], [10.5-i for x in range(week-1, 10)], marker='P', s=450, color='white', edgecolor='green')

    season = data['season']
    fig.suptitle(f'AYTO {season} Light Summary')
    plt.xlabel("Matching Night")
    plt.ylabel("Number of Lights")
    plot.set_xticklabels(range(1, 11))
    plot.set_yticklabels(range(10, -1, -1))
    plt.subplots_adjust(left=0.1, bottom=0.08, right=0.99, top=0.94)
    plot.figure.savefig(f'{season}/{season}_light_history.png', dpi=300)
    fig.suptitle(f'')
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99)
    tight_image_name = f'{season}_light_history_tight.png'
    plot.figure.savefig(f'{season}/{tight_image_name}', dpi=300)
    save_insta_light_map(data, logs, tight_image_name)


def save_insta_light_map(data, logs, tight_image_name):
    font_18 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=18)
    font_30 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=30)
    font_50 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=50)
    font_85 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=85)

    season = data['season']
    
    img = Image.open(f'insta_styles/image_backgrounds/ayto_{season}.png')
    img_light_map = Image.open(f'{season}/{tight_image_name}')
    img_light_map.thumbnail((810, 891), Image.LANCZOS)

    ayto='AYTO S4 VIP'
    event='Zusammenfassung'
    aytorakel = '@AYTOrakel'

    d = ImageDraw.Draw(img)
    d.text((img.width/2, 60), ayto, fill='white', font=font_85, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, 155), event, fill='red', font=font_85, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, 260), 'Wahrscheinlichkeiten für Lichter', fill='white', font=font_50, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((10, img.height-30), aytorakel, fill='white', font=font_18, stroke_width=2, stroke_fill='black', anchor='la')
    d.text((img.width/2, 1280), 'Woche', fill='white', font=font_50, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((78, 330), 'Lichter', fill='yellow', font=font_30, stroke_width=2, stroke_fill='black', anchor='mm')

    for i in range(1,11):
        color='white'
        d.text((122+79*i, 1230), f'{i}', fill=color, font=font_30, stroke_width=7, stroke_fill='black', anchor='mm')
        
    for i in range(11):
        start= 1160
        step = 79
        d.text((115, start-i*step), f'{i}', fill='yellow', font=font_30, stroke_width=7, stroke_fill='black', anchor='mm')

    img.paste(img_light_map, (150, 320), img_light_map)
    img.save(f'{season}/insta/{season}_{11}_{0}_insta_summary.png')


def merge_reel_frames(data):
    
    season = data['season']

    num_core_images = 900

    # Laden des Hintergrundbildes
    background = Image.open(f"insta_styles/reel_backgrounds/ayto_{season}.png")
    top_layer = Image.open(f"insta_styles/reel_face_layers/ayto_{season}.png")

    # Anzahl der Core-Bilder
    # number of images in folder
    num_core_images = len(os.listdir(f"{season}/reel_raw"))
    
    # Breite und Höhe der Core-Bilder
    core_width = 918
    core_height = 1152

    # Breite und Höhe des Top-Layer-Bildes
    top_layer_width = 1080
    top_layer_height = 1920

    # Iteriere über alle Core-Bilder
    for i in range(num_core_images):
        # Kopiere das Hintergrundbild für jeden Frame
        frame = background.copy()

        # Extrahiere den Ausschnitt aus dem Hintergrundbild
        bg_crop = background.crop((0, i, 1080, i + 1920))

        # Lade das Core-Bild
        core_image = Image.open(f"{season}/reel_raw/{season}_ayto_summary_{i}.png")

        # Position des Core-Bildes auf dem neuen Bild
        core_position = (156, 365)

        # Füge das Core-Bild über den Ausschnitt des Hintergrundbildes ein
        #frame.paste(bg_crop, (0, 0))
        bg_crop.paste(core_image, core_position, core_image)

        # Füge das Top-Layer-Bild ein
        bg_crop.paste(top_layer, (0, 0), top_layer)

        # Speichere das Bild als frame_0, frame_1, usw.
        bg_crop.save(f"{season}/reel_frames/frame_{i}.png")

    print("Frames wurden erstellt und gespeichert.")

    ffmpeg_command = fr"C:\ffmpeg\bin\ffmpeg.exe -framerate 30 -i C:\Users\User\Code\AYTOrakel\{season}\reel_frames\frame_%d.png -c:v libx264 -r 30 -pix_fmt yuv420p C:\Users\User\Code\AYTOrakel\{season}\insta\{season}_summary_reel.mp4"
    subprocess.run(ffmpeg_command, shell=True)


def save_insta_combinations(data, df, event_number):
    season = data['season']
    got = data['group_of_ten']
    gom = data['group_of_more']
    
    colors = {0: '#FFC0CB', 1: '#FF8000', 2: '#FFFF00', 3: '#80FF00', 4: '#00FF00', 5: '#00FF80', 6: '#00FFFF', 7: '#0080FF', 8: '#FF6FFF', 9: '#C291A4', 10: '#FF00FF', 11: '#FF9A8A'}

    img = Image.open(f'insta_styles/image_backgrounds/ayto_{season}.png')
    font_10 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=10)
    font_18 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=18)
    font_30 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=30)
    font_85 = ImageFont.truetype('insta_styles/DelaGothicOne-Regular.ttf', size=85)
    
    season_number = season.replace('vip','').replace('s','')
    season_vip = 'VIP' if 'vip' in season else ''
    week_number = event_number.split('-')[0]
    event_number = event_number.split('-')[1]
    ayto=f'AYTO S{season_number} {season_vip} W{week_number}'
    event=f'Matching Night'
    combinations=f'Verbleibende Kombinationen'
    aytorakel = '@AYTOrakel'
    
    d = ImageDraw.Draw(img)
    d.text((img.width/2, 60), ayto, fill='white', font=font_85, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, 155), event, fill='red', font=font_85, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((img.width/2, 240), combinations, fill='white', font=font_30, stroke_width=7, stroke_fill='black', anchor='mm')
    d.text((10, img.height-30), aytorakel, fill='white', font=font_18, stroke_width=2, stroke_fill='black', anchor='la')


    for i, name in enumerate(got):
        d.text((90+i*(img.width/11), 290), name, fill='red', font=font_18, stroke_width=2, stroke_fill='black', anchor='mm')

    df_no_duplicates = df.drop_duplicates(subset='id', keep='first')
    df_no_duplicates.reset_index(drop=True, inplace=True)
    for j, row in df_no_duplicates.iterrows():
        for i, name in enumerate(got):
            gom_index = row[name]
            gom_name = gom[gom_index]
            gom_name_2 = None
            if gom_index == row['mm1']:
                gom_index_2 = row['mm2']
                gom_name_2 = gom[gom_index_2]
            elif gom_index == row['mm2']:
                gom_index_2 = row['mm1']
                gom_name_2 = gom[gom_index_2]
                
            if gom_name_2:
                d.text((90+i*(img.width/11), 330+j*40-6), gom_name, fill=colors[gom_index], font=font_10, stroke_width=2, stroke_fill='black', anchor='mm')
                d.text((90+i*(img.width/11), 330+j*40+6), gom_name_2, fill=colors[gom_index_2], font=font_10, stroke_width=2, stroke_fill='black', anchor='mm')

            else: 
                d.text((90+i*(img.width/11), 330+j*40), gom_name, fill=colors[gom_index], font=font_18, stroke_width=2, stroke_fill='black', anchor='mm')

    img.save(f'{season}/insta/{season}_{week_number}_{event_number}_insta_remaining.png')


if __name__ == '__main__':
    main()





