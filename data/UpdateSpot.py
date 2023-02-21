from   nordpool import elspot, elbas
from   datetime import datetime, timedelta, timezone
import pytz
import tzlocal
import pickle
import sys
import matplotlib.pyplot as plt
from os.path import exists
import numpy as np

Tax  = 1.25

local_timezone=pytz.timezone('Europe/Oslo')
# Zone = 'Tr.heim'
Zone = 'Oslo'
Now  = datetime.now(tz=local_timezone)

Year = 2022
# ZoneData = 'Trheim'
ZoneData = 'Oslo'
filename = 'SpotData'+str(Year)+'_'+ZoneData+'.pkl'
file_exists = exists(filename)

if file_exists:
    print('File exists')
    f = open(filename,"rb")
    Spot = pickle.load(f)
    f.close()
    LastData = Spot['Time_end'][-1]

else:
    print('Create new file')
    Spot = {'Time_start'  : [],
            'Time_end'    : [],
            'Price'       : []}
    LastData = datetime(Year,1,1,0,0).astimezone(local_timezone)


# Number of days to pull
Ndays = int( (Now-LastData).total_seconds()/3600/24 )

# Initialize Nordpool spot prices
prices_spot = elspot.Prices(currency='NOK')

LastDay = LastData.replace( microsecond = 0, second = 0, minute = 0, hour = 0)
Now     = Now.replace(      microsecond = 0, second = 0, minute = 0, hour = 0)



if Ndays > 0:
    day = LastDay
    while day < Now and day.year == Year:
        print('Pull Date : '+str(day))
        data = prices_spot.hourly(areas=[Zone],end_date=day)
        
        for item in data['areas'][Zone]['values']:
            Spot['Time_start'].append(item['start'])
            Spot['Time_end'].append(  item['end'])
            Spot['Price'].append(    Tax * item['value']/1e1)  # convert to Øre/kWh
        day = day + timedelta(days=1)
        
plt.figure(1)
plt.step(Spot['Time_start'],Spot['Price'],where='post')
plt.show(block=False)

f = open('SpotData'+str(Year)+'_'+Zone+'.pkl',"wb")
pickle.dump(Spot,f, protocol=2)
f.close()
