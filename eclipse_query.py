import satkit as sk
import numpy as np
import math as m
from flask import jsonify, request, Flask

import googlemaps
from datetime import datetime
from zoneinfo import ZoneInfo

import os

astro_dir = os.environ.get('ASTRO_DIR')
if astro_dir is None:
    sk.utils.update_datafiles()

api_key = os.environ.get('GOOGLE_API_KEY')

gmaps = googlemaps.Client(key=api_key)

# Eclipse happens on April 8, 2024
time0 = sk.time(2024, 4, 8, 12, 0, 0)
timearr = np.array(
    time0 + [sk.duration.from_days(x) for x in np.linspace(0, 0.5, 43200)]
)

# Get exact JPL ephemeris for sun & moon
sun_light_travel_time = sk.duration.from_seconds(sk.consts.au / sk.consts.c)
sun_gcrf = sk.jplephem.geocentric_pos(
    sk.solarsystem.Sun, timearr - sun_light_travel_time
)
moon_gcrf = sk.jplephem.geocentric_pos(sk.solarsystem.Moon, timearr)
# Rotation to Earth-fixed frame
qitrf2gcrf = sk.frametransform.qitrf2gcrf(timearr)


def eclipse_stats(loc: sk.itrfcoord):
    qitrf2ned = loc.qned2itrf.conj

    # Location in GCRF
    loc_gcrf = np.array([x * loc.vector for x in qitrf2gcrf])

    # Compute angle between sun and moon at location
    sun_diff = sun_gcrf - loc_gcrf
    moon_diff = moon_gcrf - loc_gcrf
    sun_norm = np.sqrt(np.sum(sun_diff**2, axis=1))
    moon_norm = np.sqrt(np.sum(moon_diff**2, axis=1))
    theta = np.arccos(np.sum(sun_diff * moon_diff, axis=1) / sun_norm / moon_norm)

    # Compute angular extent of sun & moon
    moon_dist = np.mean(moon_norm)
    moon_extent_rad = sk.consts.moon_radius / moon_dist
    sun_extent_rad = sk.consts.sun_radius / sk.consts.au
    # How far off can they be while still having total eclipse?
    max_eclipse_offset_rad = moon_extent_rad - sun_extent_rad

    timearr_datetime = [x.datetime() for x in timearr]
    idx = np.argwhere(theta == np.min(theta))[0][0]
    # Look for times where there is total eclipse
    eidx = np.argwhere(theta < max_eclipse_offset_rad)
    # Look for times of partial eclipse
    pidx = np.argwhere(theta < (sun_extent_rad + moon_extent_rad))

    data = {"latitude": loc.latitude_deg, "longitude": loc.longitude_deg}

    if len(eidx) > 0:
        data["total"] = {
            "start": timearr[eidx[0][0]].datetime(),
            "stop": timearr[eidx[-1][0]].datetime(),
            "duration_seconds": (timearr[eidx[-1][0]] - timearr[eidx[0][0]]).seconds(),
        }
        data["partial"] = {
            "start": timearr[pidx[0][0]].datetime(),
            "stop": timearr[pidx[-1][0]].datetime(),
            "duration_seconds": (timearr[pidx[-1][0]] - timearr[pidx[0][0]]).seconds(),
        }
    elif np.min(theta) < (sun_extent_rad + moon_extent_rad):
        durp = timearr[pidx[-1]][0] - timearr[pidx[0]][0]
        mintheta = np.min(theta)
        # Derived via traingles & law of cosines
        theta_a = m.acos(
            (sun_extent_rad**2 + mintheta**2 - moon_extent_rad**2)
            / (2 * sun_extent_rad * mintheta)
        )
        theta_b = m.acos(
            (moon_extent_rad**2 + mintheta**2 - sun_extent_rad**2)
            / (2 * moon_extent_rad * mintheta)
        )
        h = sun_extent_rad * m.sin(theta_a)
        Lb = h / m.tan(theta_b)
        La = h / m.tan(theta_a)
        # Area of right side of overlapping "lens"
        aright = m.pi * moon_extent_rad**2 * theta_b / m.pi - Lb * h
        # Area of left side of overlapping "lens"
        aleft = m.pi * sun_extent_rad**2 * theta_a / m.pi - La * h
        ashown = m.pi * sun_extent_rad**2 - aright - aleft
        max_frac_area_occluded = 1 - ashown / (m.pi * sun_extent_rad**2)
        max_frac_diam_occluded = 1 - (sun_extent_rad + mintheta - moon_extent_rad) / (
            2 * sun_extent_rad
        )
        data["partial"] = {
            "start": timearr[pidx[0][0]].datetime(),
            "stop": timearr[pidx[-1][0]].datetime(),
            "duration_seconds": (timearr[pidx[-1][0]] - timearr[pidx[0][0]]).seconds(),
            "minangle_deg": np.min(theta) * 180.0 / m.pi,
            "max_area_occlusion": max_frac_area_occluded,
            "max_diam_occlusion": max_frac_diam_occluded,
        }
        data["total"] = None
    else:
        data["total"] = None
        data["partial"] = None

    return data


app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/api/")
def return_eclipse_stats():

    location = request.args.get('location')
    if not location:
        # Get latitude and longitude from URL parameters
        latitude = request.args.get('latitude')
        longitude = request.args.get('longitude')

        # Check for valid keys
        if not latitude or not longitude:
            return jsonify({"error": "Must specify latitude and longitude in degrees"}), 400
    else:
        geocode_result = gmaps.geocode(location)
        try:
            latitude = geocode_result[0]['geometry']['location']['lat']
            longitude = geocode_result[0]['geometry']['location']['lng']
        except Exception:
            return jsonify({"error": "Could not find location"}), 400

    loc = sk.itrfcoord(
        latitude_deg=float(latitude),
        longitude_deg=float(longitude))        

    stats = eclipse_stats(loc)
    
    tz = None
    if stats['partial'] is not None and 'start' in stats['partial'].keys():
        tz = gmaps.timezone({'latitude': latitude, 'longitude': longitude}, timestamp = stats['partial']['start'])
    elif stats['total'] is not None and 'start' in stats['total'].keys():
        tz = gmaps.timezone({'latitude': latitude, 'longitude': longitude}, timestamp = stats['total']['start'])
    
    if tz is not None and tz['status'] == 'OK':
        localtz = ZoneInfo(tz['timeZoneId'])
        print(localtz)
        if stats['partial'] is not None and 'start' in stats['partial'].keys():
            stats['partial']['start'] = str(stats['partial']['start'].astimezone(localtz))
            stats['partial']['stop'] = str(stats['partial']['stop'].astimezone(localtz))
        if stats['total'] is not None and 'start' in stats['total'].keys():
            stats['total']['start'] = str(stats['total']['start'].astimezone(localtz))
            stats['total']['stop'] = str(stats['total']['stop'].astimezone(localtz))

    if location:
        stats['location'] = location
    print(stats)
    print(tz)
    return jsonify(stats), 200

if __name__ == "__main__":
    app.run(debug=True)
