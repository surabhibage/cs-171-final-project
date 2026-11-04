import pandas as pd

genre_map = {
    # Rock / Punk / Metal
    'alt-rock': 'rock_metal', 'alternative': 'rock_metal', 'grunge': 'rock_metal',
    'hard-rock': 'rock_metal', 'psych-rock': 'rock_metal', 'rock': 'rock_metal',
    'rock-n-roll': 'rock_metal', 'rockabilly': 'rock_metal', 'j-rock': 'rock_metal',
    'guitar': 'rock_metal', 'british': 'rock_metal',
    'punk': 'rock_metal', 'punk-rock': 'rock_metal', 'emo': 'rock_metal',
    'hardcore': 'rock_metal', 'goth': 'rock_metal',
    'black-metal': 'rock_metal', 'death-metal': 'rock_metal', 'grindcore': 'rock_metal',
    'heavy-metal': 'rock_metal', 'metal': 'rock_metal', 'metalcore': 'rock_metal',
    'industrial': 'rock_metal',

    # Electronic / Bass / Dance
    'chicago-house': 'electronic_dance', 'deep-house': 'electronic_dance',
    'house': 'electronic_dance', 'progressive-house': 'electronic_dance',
    'detroit-techno': 'electronic_dance', 'minimal-techno': 'electronic_dance',
    'techno': 'electronic_dance', 'trance': 'electronic_dance',
    'hardstyle': 'electronic_dance', 'electro': 'electronic_dance',
    'electronic': 'electronic_dance', 'edm': 'electronic_dance',
    'idm': 'electronic_dance', 'garage': 'electronic_dance',
    'breakbeat': 'electronic_dance', 'drum-and-bass': 'electronic_dance',
    'dub': 'electronic_dance', 'dubstep': 'electronic_dance',
    'trip-hop': 'electronic_dance',
    'dance': 'electronic_dance', 'club': 'electronic_dance',
    'disco': 'electronic_dance', 'j-dance': 'electronic_dance',
    'party': 'electronic_dance',

    # Chill / Jazz / Classical
    'ambient': 'chill_jazz_classical', 'chill': 'chill_jazz_classical',
    'new-age': 'chill_jazz_classical', 'sleep': 'chill_jazz_classical',
    'study': 'chill_jazz_classical', 'sad': 'chill_jazz_classical',
    'piano': 'chill_jazz_classical',
    'jazz': 'chill_jazz_classical', 'blues': 'chill_jazz_classical',
    'classical': 'chill_jazz_classical', 'opera': 'chill_jazz_classical',
    'show-tunes': 'chill_jazz_classical',

    # Pop
    'pop': 'pop', 'pop-film': 'pop', 'power-pop': 'pop',
    'synth-pop': 'pop', 'indie-pop': 'pop', 'j-pop': 'pop',
    'k-pop': 'pop', 'cantopop': 'pop', 'mandopop': 'pop',
    'j-idol': 'pop', 'anime': 'pop', 'romance': 'pop', 'happy': 'pop',

    # Indie / Folk / Singer-songwriter
    'indie': 'indie_folk', 'singer-songwriter': 'indie_folk',
    'songwriter': 'indie_folk', 'folk': 'indie_folk', 'acoustic': 'indie_folk',

    # Hip-hop / R&B / Soul / Funk
    'hip-hop': 'hiphop_rnb', 'r-n-b': 'hiphop_rnb', 'soul': 'hiphop_rnb',
    'funk': 'hiphop_rnb', 'groove': 'hiphop_rnb', 'gospel': 'hiphop_rnb',

    # Country
    'country': 'country', 'bluegrass': 'country', 'honky-tonk': 'country',

    # Latin
    'latin': 'latin', 'latino': 'latin', 'spanish': 'latin',
    'salsa': 'latin', 'tango': 'latin', 'reggaeton': 'latin',
    'brazil': 'latin', 'forro': 'latin', 'mpb': 'latin',
    'pagode': 'latin', 'samba': 'latin', 'sertanejo': 'latin',

    # Reggae / Caribbean
    'reggae': 'reggae', 'dancehall': 'reggae', 'ska': 'reggae',

    # World
    'afrobeat': 'world', 'indian': 'world', 'iranian': 'world',
    'malay': 'world', 'turkish': 'world', 'world-music': 'world',
    'french': 'world', 'german': 'world', 'swedish': 'world',

    # Kids / Novelty
    'children': 'kids', 'kids': 'kids', 'disney': 'kids', 'comedy': 'kids',
}

def clean_data(df):
    df = df.drop(columns=['Unnamed: 0'])
    df['genre_group'] = df['track_genre'].map(genre_map)
    df = df.dropna()
    df = df.drop_duplicates('track_id')

    return df

if __name__ == '__main__':
    df = pd.read_csv('data/spotify-tracks-dataset.csv')
    df = clean_data(df)
    print(df.head())
