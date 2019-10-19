from flask import Flask, request
from flask_sqlalchemy import SQLAlchemy
import json
import os
import random
import spotipy
import spotipy.oauth2 as oauth2
from decouple import config


def create_app():
    client_id = config('CLIENT_ID')
    client_secret = config('CLIENT_SECRET')

    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL')
    DB = SQLAlchemy(app)

    spotify = spotipy.Spotify()
    credentials = oauth2.SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret)

    class Track(DB.Model):
        track_id = DB.Column(DB.String(50), primary_key=True)
        track_name = DB.Column(DB.String(250))
        artist_name = DB.Column(DB.String(100))
        acousticness = DB.Column(DB.Float)
        danceability = DB.Column(DB.Float)
        duration_ms = DB.Column(DB.Integer)
        energy = DB.Column(DB.Float)
        instrumentalness = DB.Column(DB.Float)
        key = DB.Column(DB.SmallInteger)
        liveness = DB.Column(DB.Float)
        loudness = DB.Column(DB.Float)
        mode = DB.Column(DB.Boolean)
        speechiness = DB.Column(DB.Float)
        tempo = DB.Column(DB.Float)
        time_signature = DB.Column(DB.SmallInteger)
        valence = DB.Column(DB.Float)
        popularity = DB.Column(DB.Integer)

        def to_dict(self):
            return {'track_id': self.track_id,
                    'track_name': self.track_name,
                    'artist_name': self.artist_name,
                    'acousticness': self.acousticness,
                    'danceability': self.danceability,
                    'duration_ms': self.duration_ms,
                    'energy': self.energy,
                    'instrumentalness': self.instrumentalness,
                    'key': self.key,
                    'liveness': self.liveness,
                    'loudness': self.loudness,
                    'mode': self.mode,
                    'speechiness': self.speechiness,
                    'tempo': self.tempo,
                    'time_signature': self.time_signature,
                    'valence': self.valence,
                    'popularity': self.popularity}

        def __repr__(self):
            return json.dumps(self.to_dict())

    @app.route('/')
    def root():
        """Base view."""
        return 'Hello, world!'

    @app.route('/track')
    def track():
        """Return basic track info."""
        token = credentials.get_access_token()
        spotify = spotipy.Spotify(auth=token)
        track_id = request.args.get('track',
                                    default='4uLU6hMCjMI75M1A2tKUQC',
                                    type=str)
        results = spotify.track(track_id)
        return results

    @app.route('/audio_features')
    def audio_features():
        """Return audio features for track."""
        token = credentials.get_access_token()
        spotify = spotipy.Spotify(auth=token)
        track_id = request.args.get('track',
                                    default='4uLU6hMCjMI75M1A2tKUQC',
                                    type=str)
        results = spotify.audio_features(track_id)
        return json.dumps(results)

    @app.route('/getlike')
    def getlike():
        """Return info for seed track and n random tracks."""
        seed = request.args.get('seed',
                                default='0DpOKHtemH6UMhVGKXY6DJ',
                                type=str)
        num = request.args.get('num', default=10, type=int)

        q1 = Track.query.filter(Track.track_id == seed).first()

        q2 = Track.query.filter(Track.track_id != seed)
        rowCount = int(q2.count())
        randomRows = q2.offset(int(rowCount*random.random())).limit(num)
        similar_tracks = [(row) for row in randomRows]

        return f'{{"seed": {q1}, "results": {similar_tracks}}}'

    return app
