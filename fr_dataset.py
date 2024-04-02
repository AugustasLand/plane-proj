import numpy as np
from apscheduler.schedulers.blocking import BlockingScheduler
from datetime import datetime
from FlightRadar24 import FlightRadar24API

import pandas as pd
from time import sleep
from geopy import distance
fr_api = FlightRadar24API()
sched = BlockingScheduler()

zone = fr_api.get_zones()['europe']
bounds = fr_api.get_bounds(zone)

airports = fr_api.get_airports()
airport_location = {}

for airport in airports:
    airport_location[airport.iata] = (airport.latitude, airport.longitude)

def get_data(df_flights=pd.DataFrame()):
    flights = fr_api.get_flights(bounds = bounds)
    new_flights = []
    for flight in flights:
        new_flights.append(list(flight.__dict__.values()))

    new_flights = pd.DataFrame(new_flights, columns = flight.__dict__.keys())

    d_airport = np.zeros(len(new_flights))
    for index, row in df_flights.iterrows():
        aloc = airport_location[row.destination_airport_iata]
        ploc = (row.latitude, row.longitude)
        d_airport[index] = distance.distance(aloc, ploc).km

    new_flights.insert(1, 'd_airport', d_airport, allow_duplicates=True)

    if not df_flights.empty:
        new_flights = new_flights[(new_flights['d_airport'] < 50 | # transcontinental flight did not enter zone randomly
                                new_flights['id'].isin(df_flights['id'])) &
                                new_flights['on_ground'] == False]

        new_flights = new_flights[~new_flights['airline_icao'].duplicated(False) | # Airline cannot have 2 active flights
                                pd.isna(new_flights['airline_icao']) |
                                    new_flights['id'].isin(df_flights['id'])]
    
        df_flights = pd.concat([df_flights, new_flights])

    else:
        new_flights = new_flights[(new_flights['d_airport'] < 50) &
                                new_flights['on_ground'] == False]

        new_flights = new_flights[~new_flights['airline_icao'].duplicated(False) |
                                pd.isna(new_flights['airline_icao'])]
    
        df_flights = new_flights
        
    return df_flights

df_flights = get_data()
start = datetime.now().strftime("%d_%H_%M")
print(start)
while datetime.now() < datetime(2024, 3, 24, 17, 45):
    sleep(15*60)
    df_flights = get_data(df_flights)
    end = datetime.now().strftime("%d_%H_%M")
    df_flights.to_csv(f"data/f24/flight_data_{start}_{end}.csv")


# sched.add_job(get_data(df_flights), 'interval', hours=0.25, start_date=datetime.now(), end_date=datetime(2024, 3, 24, 17, 45))
# sched.start()