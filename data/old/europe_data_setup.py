import pandas as pd
import os
from datetime import datetime
import numpy as np
from geopy import distance
from FlightRadar24 import FlightRadar24API

fr_api = FlightRadar24API()

for i, data_file in enumerate(os.listdir("europe_data")):
    file = f"europe_data/{data_file}"
    new_planes = pd.read_csv(file)
    # new_planes = new_planes.assign(date=os.path.getctime(file))
    if i == 0:
        planes = new_planes
    else:
        planes = pd.concat([planes, new_planes], ignore_index=True)

planes = planes.sort_values(by=["time"])

# tracked_planes = planes[planes.id.duplicated(keep=False)]
# tracked_planes = planes[pd.isna(planes["airline_icao"])]
tracked_planes = planes[planes["ground_speed"]!=0]
tracked_planes = pd.concat(g for _, g in tracked_planes.groupby("id") if len(g) > 3) # Flight was tracked for more than an hour

airlines = tracked_planes[~tracked_planes["id"].duplicated()]["airline_icao"].value_counts()
unpopular_airlines = airlines[airlines<4] # Airline cannot have flown more than three times
# No popular airline selected
tracked_planes = tracked_planes[tracked_planes["airline_icao"].isin(unpopular_airlines.index).values | pd.isna(tracked_planes["airline_icao"]).values]  
tracked_planes = tracked_planes.reset_index()
unique_ids = tracked_planes["id"].unique()
print(f"Unique tracked flights in dataset: {len(unique_ids)}") # Number of unknown airline and private flights

flights = []

#latitude,longitude,id,icao_24bit,heading,altitude,ground_speed,squawk,aircraft_code,registration,time,origin_airport_iata,destination_airport_iata,number,airline_iata,on_ground,callsign,airline_icao,d_airport

for flight_id in unique_ids:
    flight = tracked_planes[tracked_planes["id"] == flight_id]
    if np.all(flight["on_ground"]==1):
        continue
    
    airport = flight["destination_airport_iata"].values[0]
    d_airport = []
    if not pd.isna(airport):
        details = fr_api.get_airport_details(airport)
        airport = details["airport"]["pluginData"]["details"]["position"]
        airpos = (airport["latitude"], airport["longitude"])
        for index, ping in flight.iterrows():
            flight_pos = (ping["latitude"], ping["longitude"])
            d_airport.append(distance.distance(flight_pos, airpos).km) 
    else:
        d_airport = np.zeros(len(flight))

    latitude = flight["latitude"].median()
    longitude = flight["longitude"].median()
    altitude = flight["altitude"].mean()
    ground_speed = flight["ground_speed"].mean()
    hour = np.mean([datetime.fromtimestamp(time).hour for time in flight["time"]])
    #Squawk is NA for all data points

    turn = []
    deltaD = []
    headings = flight.heading.values
    h1 = headings[0]
    d1 = d_airport[0]
    for h in range(1, 4):
        if h == 4:
            h = -1
        h2 = headings[h]
        d2 = d_airport[h]
        turn.append(np.abs(h2-h1))
        deltaD.append(d2-d1)
        h1 = h2
        d1 = d2

    flight_data = [flight_id, flight["icao_24bit"].values[0], flight["aircraft_code"].values[0], flight["registration"].values[0], 
                   flight["origin_airport_iata"].values[0], flight["destination_airport_iata"].values[0], flight["number"].values[0],
                    flight["airline_icao"].values[0], flight["callsign"].values[0], turn[0], turn[1], turn[2],
                    deltaD[0], deltaD[1], deltaD[2], hour, latitude, longitude, altitude, ground_speed]
    flights.append(flight_data)


column_names = ["id", "icao_24bit", "aircraft_code", "registration", "origin_airport_iata",
                 "destination_airport_iata", "number", "airline_icao", "callsign", "turn1", "turn2", "turn3",
                   "deltaD1", "deltaD2", "deltaD3", "hour", "latitude", "longitude", "altitude", "ground_speed"]


flights = pd.DataFrame(flights, columns=column_names)
flights.to_csv('full_data.csv')  