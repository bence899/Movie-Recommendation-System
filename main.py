# Using all these Libraries to Process the dataset and webscrapping from IMDB
import pandas as pd
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import urllib.request
import numpy as np


# load the natural Language Processing model and tfidf vectorizer from the local files
filename = 'naturalLanguageProcessing_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('transform.pkl', 'rb'))


def produce_similarity():
    # Reading the main_data.csv File
    movieInfo = pd.read_csv('main_data.csv')
    # Creating a count matrix
    cv = CountVectorizer()
    # Each value in the matrix represents the number of reads in a cell originating from the corresponding gene. Using the count matrix, I can filter the data, keeping only the higher quality cells.
    countMatrix = cv.fit_transform(movieInfo['comb'])
    # Creating a similarity score matrix from the count_matrix
    similarity = cosine_similarity(countMatrix)
    return movieInfo, similarity


def recommendation(chosenMovie):
    # Based on the chosen movie the user selects from the list of recommendation or Search Query
    # Converting the string to lowercase
    chosenMovie = chosenMovie.lower()
    try:
        # Viewing the first 5 values which is the deafault value
        movieInfo.head()
        similarity.shape
    except:
        # Reading the CSV file and creating a count matrix then a similarity matrix and returning the Read CSV file and the Similarity matrix
        movieInfo, similarity = produce_similarity()
        # If the chosen movie selected from the user is not in the list of movies presentred in the CSV file Then let the user know that this movie is not in the database
    if chosenMovie not in movieInfo['movie_title'].unique():
        return ('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        # If the movie is present in the CSV file then record the index i of the movie within the CSV file (movieInfo) searching in the column 'movie_title'
        i = movieInfo.loc[movieInfo['movie_title'] == chosenMovie].index[0]
        # Create a list of enumerations for the similarity score [(movie_id, similarity score),(...)]
        score = list(enumerate(similarity[i]))
        # Sort the list
        score = sorted(score, key=lambda x: x[1], reverse=True)
        # I'm excluding first item since it is the requested movie itself from the user
        score = score[1:11]
        recommendedMovies = []

        # Creating a loop to show the similar movies
        for i in range(len(score)):
            a = score[i][0]
            # add each movie to the recommendedMovies Array
            recommendedMovies.append(movieInfo['movie_title'][a])
            # Return the recommendedMovies
        return recommendedMovies

# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])


def convert_to_list(myList):
    myList = myList.split('","')
    myList[0] = myList[0].replace('["', '')
    myList[-1] = myList[-1].replace('"]', '')
    return myList

# This function retrieves auto complete answers to Capital letters and returns them


def get_suggestions():
    movieInfo = pd.read_csv('main_data.csv')
    return list(movieInfo['movie_title'].str.capitalize())


app = Flask(__name__)


@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

# Transfering information using the POST Method


@app.route("/similarity", methods=["POST"])
# Data retrieved from the form is recorded as movie and fed to the recommendation(chosenMovie) function
def similarity():
    movie = request.form['name']
    rc = recommendation(movie)
    # Data Validation
    if type(rc) == type('string'):
        return rc
    else:
        m_str = "---".join(rc)
        return m_str

# Transfering information using the POST Method


@app.route("/recommend", methods=["POST"])
def recommend():
    # getting data from AJAX request
    movieTitle = request.form['title']
    castIds = request.form['cast_ids']
    castChars = request.form['cast_chars']
    castNames = request.form['cast_names']
    castDOB = request.form['cast_bdays']
    castInfo = request.form['cast_bios']
    castPlaces = request.form['cast_places']
    castProfiles = request.form['cast_profiles']
    imdbID = request.form['imdb_id']
    genres = request.form['genres']
    overview = request.form['overview']
    poster = request.form['poster']
    ratingAverage = request.form['rating']
    voteCount = request.form['vote_count']
    releaseDate = request.form['release_date']
    movieRuntime = request.form['runtime']
    status = request.form['status']
    recMovies = request.form['rec_movies']
    recPosters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # Every string that needs toi be converted to a list call the convert_to_list function
    recMovies = convert_to_list(recMovies)
    recPosters = convert_to_list(recPosters)
    castNames = convert_to_list(castNames)
    castChars = convert_to_list(castChars)
    castProfiles = convert_to_list(castProfiles)
    castDOB = convert_to_list(castDOB)
    castInfo = convert_to_list(castInfo)
    castPlaces = convert_to_list(castPlaces)

    # Convert string to list (eg. "[1,2,3]" to [1,2,3])
    castIds = castIds.split(',')
    castIds[0] = castIds[0].replace("[", "")
    castIds[-1] = castIds[-1].replace("]", "")

    # Rendering the string to python string
    for i in range(len(castInfo)):
        castInfo[i] = castInfo[i].replace(r'\n', '\n').replace(r'\"', '\"')

    # Combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movieCards = {recPosters[i]: recMovies[i] for i in range(len(recPosters))}

    casts = {castNames[i]: [castIds[i], castChars[i], castProfiles[i]]
             for i in range(len(castProfiles))}

    castDetails = {castNames[i]: [castIds[i], castProfiles[i], castDOB[i],
                                  castPlaces[i], castInfo[i]] for i in range(len(castPlaces))}

    # Web scraping to Retrieve user reviews from IMDB site
    target = urllib.request.urlopen(
        'https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdbID)).read()
    userReview = bs.BeautifulSoup(target, 'lxml')
    userReview_result = userReview.find_all(
        "div", {"class": "text show-more__control"})

    reviews_list = []  # list of reviews
    reviews_status = []  # list of comments (good or bad)
    for reviews in userReview_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            # Sentiment Validation
            reviews_status.append('Good Movie' if pred else 'Bad Movie')

    # Combining reviews and comments into a dictionary
    movieReviews = {reviews_list[i]: reviews_status[i]
                    for i in range(len(reviews_list))}

    # Passing all the data to the recommendation html file
    return render_template('recommend.html', title=movieTitle, overview=overview, poster=poster, vote_average=ratingAverage,
                           vote_count=voteCount, release_date=releaseDate, status=status, runtime=movieRuntime, genres=genres,
                           movie_cards=movieCards, reviews=movieReviews, casts=casts, cast_details=castDetails)


if __name__ == '__main__':
    app.run(debug=True)
