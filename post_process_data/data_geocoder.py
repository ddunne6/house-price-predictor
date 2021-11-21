from geopy.geocoders import Nominatim
from numpy import number
from shapely.geometry import shape, Point

import csv
import json

with open('secrets.json') as json_file:
    data = json.load(json_file)
    GOOGLE_API_KEY = data.get('GOOGLE_API_KEY')

def latlong_geocoder(filename):
    with open(filename, encoding="utf-8") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        results = []

        index = 0
        for row in reader:
            if index != 0:
                address = str(row[2])[1:]
                import requests

                payload = {'address': address,
                           'key': GOOGLE_API_KEY}
                response = requests.get(
                    'https://maps.googleapis.com/maps/api/geocode/json?', params=payload)

                resp_json_payload = response.json()
                try:
                    print(resp_json_payload['results'][0]
                        ['geometry']['location']['lat'])
                    print(resp_json_payload['results'][0]
                        ['geometry']['location']['lng'])
                    results.append({
                        "address": address,
                        "id": row[0],
                        "lat": resp_json_payload['results'][0]['geometry']['location']['lat'],
                        "lng": resp_json_payload['results'][0]['geometry']['location']['lng']
                    })
                except:
                    results.append({
                        "address": address,
                        "id": row[0],
                        "lat": "ERROR",
                        "lng": "ERROR"
                    })
                    print("error with parsing")
            index += 1

        with open('geocoded.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

# Reference: https://stackoverflow.com/questions/20776205/point-in-polygon-with-geojson-in-python
def find_electoral_region():
    # Load GeoJSON file
    with open('Electoral_Divisions_-_OSi_National_Statutory_Boundaries.geojson', encoding="utf-8") as f:
        geojson = json.load(f)

    with open('geocoded.json', encoding="utf-8") as f:
        addresses = json.load(f)

    address_regions_list = []
    index = 0
    for address in addresses:
        # if index == 10:
        #     break
        if address.get('lng') == "ERROR" or address.get('lat') == "ERROR":
            continue
        point = Point(address.get('lng'), address.get('lat'))
        # Check each polygon to see if it contains the point
        for feature in geojson['features']:
            polygon = shape(feature['geometry'])
            if polygon.contains(point):
                print(f"{index} Found containing polygon: {feature['properties']['ENGLISH']}")
                address['electoral_division'] = feature['properties']['ENGLISH']
                address_regions_list.append(address)
                break
        index += 1

    with open('geocoded-with-regions.json', 'w', encoding='utf-8') as f:
            json.dump(address_regions_list, f, ensure_ascii=False, indent=4)

import pandas as pd

def assign_median_income():
    divison_mappings = {
        "TULLOKYNE": "Maigh Cuilinn",
        "SAINT KEVIN'S": "St. Kevin's",
        "ATHY  WEST URBAN": "Athy West Urban",
        "CRUMPAUN": "An Crompán",
        "SLIEVEANEENA": "Sliabh an Aonaigh",
        "Dún Laoghaire SALLYNOGGIN EAST": "Dún Laoghaire-Sallynoggin East",
        "MUINEBEAG RURAL": "Muinebeag (Bagenalstown) Rural",
        "St. NICHOLAS": "Paróiste San Nicoláis",
        "KNOCKNACARRAGH": "Salthill",
        "BALLINCHALLA": "Baile an Chalaidh",
        'CAHERDANIEL': "Cathair Dónall",
        'SPIDDLE': "An Spidéal",
        'CARROWKEEL': "An Cheathrú Chaol",
        'CEANANNAS MÓR URBAN': "Ceannanus Mór \(Kells\) Urban",
        'INISHMORE': "Inishbofin",
        'OVOCA': "Avoca",
        'DUNURLIN': "Dún Urlann",
        'CREESLOUGH': "An Craoslach",
        'CARROWBROWNE': "Ceathrú an Bhrúnaigh",
        'CASTLEGAR': "An Caisleán Gearr",
        "Droichead Nua (Newbridge) Urban": "Droichead Nua \(Newbridge\) Urban",
        "DROICHEAD NUA RURAL": "Droichead Nua \(Newbridge\) Rural",
        "CEANANNAS MÓR RURAL": "Ceannanus Mór \(Kells\) Rural \(Part Urban\)",
        "St. MARY'S (PART)": "St. Mary's \(Part Urban\)",
        "SUNDAY'S WELL B": "Sundays Well B",
        "PORTLAOIGHISE RURAL": "Portlaoighise \(Maryborough\) Rural",
        "PORTLAOIGHISE URBAN": "Portlaoighise \(Maryborough\) Urban"
    }

    with open('geocoded-with-regions.json', encoding="utf-8") as f:
        addresses = json.load(f)
    
    incomes = pd.read_csv('IIA01 - Household median gross income.csv')
    results = []

    num_issues = 0
    for address in addresses:
        address['electoral_division'] = divison_mappings.get(address['electoral_division'], address['electoral_division'])

        income = incomes[incomes["Electoral Division"].str.contains(address.get('electoral_division'), case=False)]
        if not income.empty:
            print(income.VALUE.values[0])
            address["median_income"] = int(income.VALUE.values[0])
            results.append(address)
        else:
            num_issues += 1
    print(f"{num_issues} unmapped Electoral Regions")
    with open('geocoded-with-regions-incomes.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    #latlong_geocoder('properties-decorated.csv')
    #find_electoral_region()
    assign_median_income()
