import numpy as np
from geopy import distance
import pandas as pd
from datetime import datetime

def haversine(s_lat, s_lon, e_lat, e_lon):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    R = 6371.0 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    # convert decimal degrees to radians
    s_lat = np.deg2rad(s_lat)
    s_lon = np.deg2rad(s_lon)
    e_lat = np.deg2rad(e_lat)
    e_lon = np.deg2rad(e_lon)

    # haversine formula 
    dlat = e_lat - s_lat
    dlon = e_lon - s_lon
    d = np.sin(dlon/2)**2 + np.cos(s_lat) * np.cos(e_lat) * np.sin(dlat/2)**2
    
    return 2 * R * np.arcsin(np.sqrt(d))

def new_data(data, subset, id):
    planes = subset[pd.isna(subset.flight_no)][id]
    flights = subset[~pd.isna(subset.flight_no)]['flight_no']
    data = data[data[id].isin(planes) | data['flight_no'].isin(flights)]
    return data


def get_subset(data, id='id', time=False, radius=False, origin=False):
    if time[0]:
        subset = data[(data['time'] >= time[0]) & (data['time'] <= time[1])]
        data = new_data(data, subset, id)

    if radius:
        distances = haversine(origin[0], origin[1], data['latitude'], data['longitude'])
        subset = data[distances <= radius]
        data = new_data(data, subset, id)

    return data

def tracked_flights(data, min_appearances = 4, id='id'):
    # id for flight24, hex for absd
    return pd.concat(g for _, g in data.groupby(id) if len(g) >= min_appearances) # Flight was tracked for >1h15m

def flight_resume(data, id='id', min_appearances=4):
    unique_ids = data[id].unique()
    print(f"Unique tracked flights in dataset: {len(unique_ids)}") # Number of unknown airline and private flights

    flights = []
    for plane_id in unique_ids:
        flight = data[data[id] == plane_id]
        if id=='hex':
            if len(flight['flight_no'].unique()) > 1:
                continue
        elif np.all(flight["on_ground"]==1):
            continue

        if (np.all(flight["ground_speed"]==0 | pd.isna(flight["ground_speed"])) or
             np.all(flight["heading"]==0 | pd.isna(flight["heading"]))):
            continue

        duration = flight.iloc[-1].time - flight.iloc[0].time
        if id=='hex':
            time = flight['time'].mean()
            signal_power = flight.signal_power.mean()
            messages = flight.messages.mean()

            speeds = flight.ground_speed.values

            times = flight.time.values
            time1 = times[0]

            locations = np.array([flight['latitude'].values, flight['longitude'].values]).T
            loc1 = locations[0]
        else:
            duration = duration//3600 + (duration%3600)/60
            time = np.mean([datetime.fromtimestamp(time).hour + datetime.fromtimestamp(time).minute/60 for time in flight["time"]])

            d_airport = flight.d_airport.values
            d1 = d_airport[0]

        lat = flight["latitude"].median()
        lon = flight["longitude"].median()
        altitude = flight["altitude"].mean()
        ground_speed = flight["ground_speed"].mean()
        
        headings = flight.heading.values
        turn = []
        deltaD = [] # Distance between theoretical prediction and actual distance traveled or change for distance to airport
        h1 = headings[0]

        for h in range(1, min_appearances+1):
            if h == min_appearances:
                h = -1
                speed = (speeds[h] + speeds[min_appearances-1])/2
            else:
                speed = (speeds[h] + speeds[h-1])/2
            
            h2 = headings[h]
            turn.append(np.abs(h2-h1))
            h1 = h2
            
            if id == 'hex':
                time2 = times[h]
                loc2 = locations[h]
                tdist = 1.852*speed*(time2-time1) #Knots to kmh, theoretical distance
                adist = distance.distance(loc2, loc1).km #Actual distance travelled
                if adist > 30:
                    deltaD.append(tdist/adist)
                else:
                    deltaD.append(1) # Default is that predicted and actual match
                time1 = time2
                loc1 = loc2

            else:
                d2 = d_airport[h]
                deltaD.append(d2-d1)
                d1=d2

        if id == 'hex':
            flight_data = [plane_id, time, flight.iloc[0]["type"], flight.iloc[0]["flight_no"], flight.iloc[0]["plane_type"],
                flight.iloc[0]["squawk"], flight.iloc[0].category, turn, deltaD, lat, lon, altitude, ground_speed,
                signal_power, flight.iloc[0].dbFlags, duration, messages]
        else:
            flight_data = [plane_id, flight["icao_24bit"].iloc[0], flight["aircraft_code"].iloc[0], flight["registration"].iloc[0], 
                flight["origin_airport_iata"].iloc[0], flight["destination_airport_iata"].iloc[0], flight["number"].iloc[0],
                flight["airline_icao"].iloc[0], flight["callsign"].iloc[0], turn, deltaD, duration,
                time, lat, lon, altitude, ground_speed]
        
        flights.append(flight_data)

    return flights

def to_df(predf, colnames):
    df = pd.DataFrame(predf, columns=colnames)

    df[[f"turn{i+1}" for i in range(len(df['turn'].iloc[0]))]] = pd.DataFrame(df.turn.tolist(), index=df.index)
    df[[f"deltaD{i+1}" for i in range(len(df['deltaD'].iloc[0]))]] = pd.DataFrame(df.deltaD.tolist(), index=df.index)

    return df.drop(columns=['turn', 'deltaD'])
    

if __name__ == "__main__":
    try:
        t1 = np.float32(input("Please input decimal hour start time for flights of interest: "))
        t2 = np.float32(input("Please input decimal hour end time for flights of interest: "))
    except:
        t1 = 0
        t2 = 0
    time = [t1, t2]
    try:
        origin = np.float32(input("Please input center of region of interest (lat, lon separated by comma): ").split(','))
        radius = np.float32(input("Please input radius in km of region of interest: "))
    except:
        origin = 0
        radius = 0

    absd_data = pd.read_csv("data/absd/absd20240401.csv")
    absd_data = absd_data[absd_data.columns[1:]] # Remove index column
    print(f"Tracked airplane count absd: {len(absd_data['hex'].unique())}")
    print(f"Tracked flight count absd: {len(absd_data['flight_no'].unique())}")
    absd_data = get_subset(absd_data, 'hex', time, radius, origin)
    absd_colnames = ["plane_id", "time", "type", "flight_no", "plane_type", "squawk", "category", "turn", "deltaD", "latitude",
                 "longitude", "altitude", "ground_speed", "signal_power", "dbFlags", "duration", "messages"]
    
    absd_data = tracked_flights(absd_data, id='hex')
    absd_flights = flight_resume(absd_data, id='hex')
    absd_flights = to_df(absd_flights, absd_colnames)
    absd_flights.to_csv('absd/data/filtered_absd20240401.csv', index=False) 
    
    # f24_data = pd.read_csv("flight24/flight_data_23_16_31_23_22_17.csv")
    # f24_data = f24_data.sort_values(by=["time"])
    # f24_colnames = ["id", "icao_24bit", "aircraft_code", "registration", "origin_airport_iata",
    #              "destination_airport_iata", "flight_no", "airline_icao", "callsign", "turn", "deltaD", "time", "duration",
    #                "latitude", "longitude", "altitude", "ground_speed"]
    # print(f"Tracked airplane count fr24: {len(f24_data['id'].unique())}")
    # print(f"Tracked flight count fr24: {len(f24_data['flight_no'].unique())}")
    
    # f24_data = tracked_flights(f24_data, id='id')
    # f24_flights = to_df(flight_resume(f24_data), f24_colnames)
    # f24_flights.to_csv('flight24/data/full_data.csv', index=False)  

            

            
            