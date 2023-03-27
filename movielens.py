from urllib.request import urlretrieve
import zipfile
import pandas as pd
import os.path as path

if path.isfile('ml-100k/u.user') is False:
	print('Downloading...')
	urlretrieve(
		'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
		'movielens.zip'
	)
	zip_ref = zipfile.ZipFile('movielens.zip', 'r')
	zip_ref.extractall()

user_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')

rating_cals = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='|', names=rating_cals, encoding='latin-1')

genre_cols = [
    "genre_unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
movie_cols = [
	'movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'
	] + genre_cols

movies = pd.read_csv('ml-100k/u.item', sep='|', names=movie_cols, encoding='latin-1')

genre_occurences = movies[genre_cols].sum().to_dict()


print(list(
	zip(*[movies[:5][genre] for genre in genre_cols[:2]])
))

