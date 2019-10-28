"""The data processing and prediction backend for the Songsight app.

Routes returning track information, given a Spotify track id:
    track (basic info)
    audio_features (audio details)

Routes returning tracks:
    get_range (given min & max values for a particular feature)
    get_random (from the top or bottom n tracks for a specified feature)
    get_like (given a seed track for similarity measurement)

Routes returning visualizations:
    compare (given two track ids)
"""

import io
import json
import numpy as np
import pandas as pd
import pickle
import random
import spotipy
import spotipy.oauth2 as oauth2

from decouple import config
from flask import Flask, Response, request
from flask_sqlalchemy import SQLAlchemy
from math import pi
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure


def create_app():

    app = Flask(__name__)

    # Spotify API authentication details.
    client_id = config('CLIENT_ID')
    client_secret = config('CLIENT_SECRET')

    credentials = oauth2.SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret)

    # Backend database location and associated metadata.
    app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL')
    DB = SQLAlchemy(app)
    DB.Model.metadata.reflect(DB.engine)

    # KDTree for finding 'k nearest neighbor' tracks. Loaded from file.
    tree = pickle.load(open('tree.p', 'rb'))

    class Track(DB.Model):
        __table__ = DB.Model.metadata.tables['track']

        def to_array(self):
            """
            Returns a numpy array with only those features used by the KDTree
            to evaluate track similarity.
            """
            return np.array([self.acousticness_scaled,
                             self.danceability_scaled,
                             self.duration_ms_scaled,
                             self.energy_scaled,
                             self.instrumentalness_scaled,
                             self.key_scaled,
                             self.liveness_scaled,
                             self.loudness_scaled,
                             self.mode_scaled,
                             self.speechiness_scaled,
                             self.tempo_scaled,
                             self.time_signature_scaled,
                             self.valence_scaled,
                             self.popularity_scaled,
                             self.alternative_rnb,
                             self.atl_hip_hop,
                             self.banda,
                             self.baroque,
                             self.big_room,
                             self.brostep,
                             self.cali_rap,
                             self.ccm,
                             self.chamber_pop,
                             self.chillhop,
                             self.classical,
                             self.classical_era,
                             self.contemporary_country,
                             self.dance_pop,
                             self.early_music,
                             self.early_romantic_era,
                             self.edm,
                             self.electro_house,
                             self.electropop,
                             self.emo_rap,
                             self.folk_pop,
                             self.gangster_rap,
                             self.german_baroque,
                             self.grupera,
                             self.hip_hop,
                             self.indie_folk,
                             self.indie_pop,
                             self.indie_poptimism,
                             self.indie_rnb,
                             self.indie_rock,
                             self.indie_soul,
                             self.indietronica,
                             self.k_pop,
                             self.latin,
                             self.lo_fi_beats,
                             self.mellow_gold,
                             self.melodic_rap,
                             self.modern_rock,
                             self.neo_mellow,
                             self.norteno,
                             self.pop,
                             self.pop_edm,
                             self.pop_rap,
                             self.pop_rock,
                             self.post_teen_pop,
                             self.progressive_house,
                             self.progressive_trance,
                             self.ranchera,
                             self.rap,
                             self.regional_mexican,
                             self.regional_mexican_pop,
                             self.rock,
                             self.sleep,
                             self.soft_rock,
                             self.southern_hip_hop,
                             self.stomp_and_holler,
                             self.trance,
                             self.trap_music,
                             self.tropical_house,
                             self.underground_hip_hop,
                             self.uplifting_trance,
                             self.vapor_trap,
                             self.classical_super,
                             self.country_super,
                             self.folk_super,
                             self.house_super,
                             self.indian_super,
                             self.indie_super,
                             self.jazz_super,
                             self.latin_super,
                             self.metal_super,
                             self.rap_super,
                             self.reggae_super,
                             self.rock_super,
                             self.worship_super])

        def to_dict(self):
            """
            Returns a dictionary with only the fields from the original
            Kaggle dataset. Primarily for display purposes.
            """
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
        return 'Welcome to the data science backend for the SongSight app!'

    @app.route('/track')
    def track():
        """
        A wrapper for the Spotify tracks endpoint. Returns basic track info.

        Args:
            track: The Spotify track id whose information is sought.

        Returns:
            On success - a json object containing the relevant details (album,
            artists, available markets, etc.).
            On failure - a server error.
        """
        token = credentials.get_access_token()
        spotify = spotipy.Spotify(auth=token)
        track_id = request.args.get('track',
                                    default='4uLU6hMCjMI75M1A2tKUQC',
                                    type=str)
        results = spotify.track(track_id)
        return results

    @app.route('/audio_features')
    def audio_features():
        """
        A wrapper for the Spotify audio-features endpoint. Returns audio
        features for track.

        Args:
            track: The Spotify track id whose features are sought.

        Returns:
            On success - the string representation of a list containing a
            single json object with the track's audio features.
            On failure - the string representation of a list of one or more
            null objects.

        Note:
            The underlying Spotipy method used here will take a list of track
            ids as an argument; this route could be extended to take
            advantage of that.
        """
        token = credentials.get_access_token()
        spotify = spotipy.Spotify(auth=token)
        track_id = request.args.get('track',
                                    default='4uLU6hMCjMI75M1A2tKUQC',
                                    type=str)
        results = spotify.audio_features(track_id)
        return json.dumps(results)

    @app.route('/get_range')
    def get_range():
        """
        Retrieves tracks matching a specified range of values for a single
        audio (or other) feature.

        Args:
            audio_feature: the feature on which to filter
            min: the minimum desired value for that audio feature
            max: the maximum desired value for the same audio feature
            limit: the maximum number of tracks to return

        Returns:
            On success: The string representation of a list of dictionaries
            containing the track info for each matching track found.
            On failure: The string representation of an empty dictionary.

        Note:
            The successful execution of a query with 0 results is essentially
            indistinguishable from an error here.
        """
        audio_feature = request.args.get('audio_feature',
                                         default='acousticness',
                                         type=str)
        min_range = request.args.get('min', type=float)
        max_range = request.args.get('max', type=float)
        max_limit = request.args.get('limit', default=200, type=int)

        tracks = {}
        if min_range and max_range:
            condition = (f'(Track.{audio_feature} >= min_range) & '
                         f'(Track.{audio_feature} <= max_range)')
            tracks = Track.query.filter(eval(condition)).limit(max_limit).all()
        elif min_range:
            condition = f'Track.{audio_feature} >= min_range'
            tracks = Track.query.filter(eval(condition)).limit(max_limit).all()
        elif max_range:
            condition = f'Track.{audio_feature} <= max_range'
            tracks = Track.query.filter(eval(condition)).limit(max_limit).all()

        return str(tracks)

    @app.route('/get_like')
    def get_like():
        """
        Returns info for a seed track and its n nearest neighbors.

        Args:
            seed: the Spotify track_id of the seed track
            num: the number of neighbor tracks to find

        Returns:
            On success: a json string with the track info for the seed track
            and the results list of similar tracks.
            On failure: a server error.
        """
        seed = request.args.get('seed',
                                default='0DpOKHtemH6UMhVGKXY6DJ',
                                type=str)
        num = request.args.get('num', default=10, type=int)

        q1 = Track.query.filter(Track.track_id == seed).first()
        dist, ind = tree.query(q1.to_array().reshape(1, -1), k=num+1)
        indices = [val.item() for val in ind[0]]

        q2 = Track.query.filter(Track.id.in_(indices))
        q2 = q2.filter(Track.track_id != seed).all()

        return f'{{"seed": {q1}, "results": {q2}}}'

    @app.route('/get_random')
    def get_random():
        """
        Returns info for n random tracks from top m by feature.

        Args:
            feature: the feature on which tracks will be sorted
            num: the number of tracks to return (n)
            top: the number of tracks from which the returned tracks should be
            picked (m)
            order: 'asc' for ascending sort, to return a selection from the
            bottom m tracks by feature. Otherwise defaults to descending order
            and the top m tracks.

        Returns:
            On success: a json string with the track info for the query
            results.
            On failure: a server error.
        """
        feature = request.args.get('feature', default='popularity', type=str)
        num = request.args.get('num', default=10, type=int)
        top = request.args.get('top', default=100, type=int)
        order = request.args.get('order', default='desc', type=str)

        if order.lower() == 'asc':
            q = Track.query.order_by(eval(f'Track.{feature}')).limit(top)
        else:
            q = Track.query.order_by(eval(f'Track.{feature}.desc()'))
            q = q.limit(top)

        rowCount = int(q.count())
        randomRows = q.offset(int(rowCount*random.random())).limit(num)
        tracks = [(row) for row in randomRows]

        return f'{{"results": {tracks}}}'

    @app.route('/compare')
    def compare():
        """
        Returns a visual comparison of two tracks.

        Args:
            label_a: a label for the first track (optional, defaults to
            <artist> - <title>).
            label_b: a label for the second track (optional, defaults to
            <artist> - <title>).
            track_a: the Spotify id for the first track.
            track_b: the Spotify id for the second track.

        Returns:
            On success: an svg image comparing the two tracks selected.
            On failure: a server error.
        """
        label_a = request.args.get('label_a', default='', type=str)
        label_b = request.args.get('label_b', default='', type=str)
        track_a = request.args.get('track_a',
                                   default='1IVJDJy9rWFAynjhta7l2J',
                                   type=str)
        track_b = request.args.get('track_b',
                                   default='5w9c2J52mkdntKOmRLeM2m',
                                   type=str)

        q1 = Track.query.filter(Track.track_id == track_a).first()
        q2 = Track.query.filter(Track.track_id == track_b).first()

        df = pd.DataFrame([q1.to_dict(), q2.to_dict()])

        # Re-order columns.
        df = df[['track_id', 'track_name', 'artist_name', 'mode',
                 'danceability', 'energy', 'instrumentalness', 'liveness',
                 'speechiness', 'valence']]

        if(label_a == ''):
            label_a = '{} - {}'.format(df.loc[0]['artist_name'],
                                       df.loc[0]['track_name'])
        if(label_b == ''):
            label_b = '{} - {}'.format(df.loc[1]['artist_name'],
                                       df.loc[1]['track_name'])

        # ------- PART 1: Create background

        # number of variables
        categories = list(df)[4:]
        N = len(categories)

        # What will be the angle of each axis in the plot?
        # Divide the plot according to the number of variables.
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        # Initialise the radar plot.
        fig = Figure()
        ax = fig.add_subplot(111, polar=True)

        # Put the first axis to top.
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)

        # Draw one axis per variable and label each.
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)

        # Draw ylabels
        ax.set_rlabel_position(0)
        ax.set_yticks([0.25, 0.50, 0.75])
        ax.set_yticklabels(['0.25', '0.50', '0.75'])
        ax.set_ylim(0, 1)

        # ------- PART 2: Add polygons.

        # Plot each track (each row of the dataframe).
        drop_cols = ['track_id', 'track_name', 'artist_name', 'mode']

        # Track 1
        values = df.loc[0].drop(drop_cols).values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=label_a)
        ax.fill(angles, values, 'b', alpha=0.1)

        # Track 2
        values = df.loc[1].drop(drop_cols).values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=label_b)
        ax.fill(angles, values, 'r', alpha=0.1)

        # Add legend
        ax.legend(loc='best')

        output = io.BytesIO()
        FigureCanvasSVG(fig).print_svg(output)
        return Response(output.getvalue(), mimetype="image/svg+xml")

    return app
