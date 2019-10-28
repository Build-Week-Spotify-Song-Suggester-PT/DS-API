# DS-API

The data processing and prediction backend for the SongSight app, designed to find song suggestions and display related visualizations.

## Routes returning track information, given a Spotify track id:
    track (basic info)
    audio_features (audio details)

## Routes returning tracks:
    get_range (given min & max values for a particular feature)
    get_random (from the top or bottom n tracks for a specified feature)
    get_like (given a seed track for similarity measurement)

## Routes returning visualizations:
    compare (given two track ids)
