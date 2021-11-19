from bs4 import BeautifulSoup
from requests import get
import pandas as pd
import time
from random import randint
import re

def findByIcon(icon_name):
        value = info.find(attrs={"icon": icon_name})
        if value != None:
            return True
        else:
            return False





headers = ({'User-Agent':
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/95.0.4638.69 Safari/537.36'})



arr_prices = []
arr_addr = []
arr_bed = []
arr_bath = []
arr_area = []
arr_home = []
i = 0
for p in range(0, 600):
    url = "https://www.myhome.ie/residential/ireland/property-for-sale?page=" + str(p)
    response = get(url, headers=headers)
    print(response)
    soup = BeautifulSoup(response.text, 'html.parser')
    if soup != None:
        property_cards = soup.findAll('div', class_="PropertyListingCard__PropertyInfo")
        for card in property_cards:
            price = card.find('div', class_="PropertyListingCard__Price").text
            price_formatted = price.replace('€','').replace(',','').replace(' ', '')
            arr_prices.append(price_formatted)

            address = card.find('a', class_="PropertyListingCard__Address").text
            arr_addr.append(address)

            info_strip = card.find('div', class_="PropertyInfoStrip")
            if info_strip != None:
                info_strip = info_strip.findAll('span')
                for info in info_strip:
                    if findByIcon("bed"):
                        arr_bed.append(info.text)
                    if findByIcon("bath"):
                        arr_bath.append(info.text)
                    if findByIcon("cube"):
                        arr_area.append(info.text)
                    if findByIcon("home"):
                        arr_home.append(info.text)

                    ##energy_rating = info.find(attrs={"alt": "Energy Rating"})            
                    ##if energy_rating != None:
                    ##    print(energy_rating)
                    ##    if energy_rating.has_attr('src'):
                    ##        print(energy_rating['src'])
                

            if len(arr_bed)-1 < i:
                arr_bed.append("")
            if len(arr_bath)-1 < i:
                arr_bath.append("")
            if len(arr_area)-1 < i:
                arr_area.append("")
            if len(arr_home)-1 < i:
                arr_home.append("")
            if len(arr_prices)-1 < i:
                arr_prices.append("")
            if len(arr_addr)-1 < i:
                arr_addr.append("")

            i = i+1
    time.sleep(randint(1,3))


print(len(arr_prices))
print(len(arr_addr))
print(len(arr_bed))
print(len(arr_bath))
print(len(arr_area))
print(len(arr_home))

properties = pd.DataFrame({'Price' : arr_prices,
                            'Address' : arr_addr,
                            'Beds' : arr_bed,
                            'Bathrooms' : arr_bath,
                            'Area' : arr_area,
                            'Type' : arr_home
})

properties.to_csv('properties.csv')