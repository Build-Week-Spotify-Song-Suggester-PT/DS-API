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
from matplotlib.backends.backend_svg import FigureCanvasSVG
from matplotlib.figure import Figure


def create_app():
    client_id = config('CLIENT_ID')
    client_secret = config('CLIENT_SECRET')

    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL')
    DB = SQLAlchemy(app)
    DB.Model.metadata.reflect(DB.engine)

    credentials = oauth2.SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret)

    tree = pickle.load(open('tree.p', 'rb'))

    class Track(DB.Model):
        __table__ = DB.Model.metadata.tables['track']

        def to_array(self):
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

    @app.route('/get_range')
    def get_range():
        audio_feature = request.args.get('audio_feature', type=str)
        min_range = request.args.get('min', type=float)
        max_range = request.args.get('max', type=float)

        tracks = {}
        if min_range and max_range:
            condition = (f'(Track.{audio_feature} >= min_range) & '
                         f'(Track.{audio_feature} <= max_range)')
            tracks = Track.query.filter(eval(condition)).all()
        elif min_range:
            condition = f'Track.{audio_feature} >= min_range'
            tracks = Track.query.filter(eval(condition)).all()
        elif max_range:
            condition = f'Track.{audio_feature} <= max_range'
            tracks = Track.query.filter(eval(condition)).all()

        return str(tracks)

    @app.route('/getlike')
    def getlike():
        """Return info for seed track and n nearest neighbors."""
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

    @app.route('/getrandom')
    def getrandom():
        """Return info for n random tracks from top m by feature."""
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
        """Return a visual comparison of two tracks."""
        label_a = request.args.get('label_a', default='', type=str)
        label_b = request.args.get('label_b', default='', type=str)
        track_a = request.args.get('track_a',
                                   default='5w9c2J52mkdntKOmRLeM2m',
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
            label_b = '{} - {}'.format(df.loc[0]['artist_name'],
                                       df.loc[0]['track_name'])

        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)
        x_points = range(5)
        axis.plot(x_points, [random.randint(1, 30) for x in x_points])

        output = io.BytesIO()
        FigureCanvasSVG(fig).print_svg(output)
        return Response(output.getvalue(), mimetype="image/svg+xml")

    return app
