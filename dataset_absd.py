import requests
import pandas as pd

date = "2024/04/01"

for t in range(0, 60*24, 15):
    h = str(t//60)
    if len(h) == 1:
        h = "0" + h
    m = str(int(t%60))
    if len(m) == 1:
        m = "0" + m
    url = f"https://samples.adsbexchange.com/readsb-hist/{date}/{h+m}00Z.json.gz"
    resp = requests.get(url=url)
    data = resp.json()
    new_aircraft = pd.DataFrame(data['aircraft'])
    new_aircraft.insert(1, 'autopilot' , int(h)+int(m)/60, True)

    new_aircraft = new_aircraft[['hex', 'type', 'flight', 't', 'alt_geom', 'gs',   
       'track', 'squawk', 'category', 'lat', 'lon','messages', 'rssi', 'dbFlags']]
    
    new_aircraft = new_aircraft.rename(columns={'t': 'plane_type','alt_geom': 'altitude',
                                                 'gs': 'ground_speed', 'track':'heading',
                                                   'rssi':'signal_power', 'flight':'flight_no',
                                                   'lat':'latitude', 'lon':'longitude'})
    
    new_aircraft = new_aircraft[~(pd.isna(new_aircraft['latitude']) | pd.isna(new_aircraft['longitude']))]
    new_aircraft.insert(1, 'time' , int(h)+int(m)/60, True)
    
    if t == 0:
        aircraft = new_aircraft
    else:
        aircraft = pd.concat([aircraft, new_aircraft])

aircraft.to_csv(f"absd_{date.replace('/', '')}.csv")