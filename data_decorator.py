import csv
import re

def data_decorator(filename):
    # id -> add column title
    # price not in number, e.g. POA,GuidePrice,AMV
    # beds -> show only number e.g. NOT 3 beds
    # bathrooms -> show only number
    # area -> convert all to m^2, remove units e.g. NOT 45 m 2
    
    # remove rows with blank entries
    
    decorated_data = []

    with open(filename) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        data = [row for row in reader]
        index = 0
        for row in data:
            row_is_clean = True
            # Add column title for ID
            if index == 0:
                row[0] = 'ID'
            else:
                # Price
                try:
                    row[1] = int(row[1])
                except:
                    price = re.search(r'\d+', row[1])
                    if price is not None:
                        row[1] = int(price.group(0))
                    else:
                        row_is_clean = False

                # Beds
                beds = re.search(r'\d+', row[3])
                if beds is not None:
                    row[3] = int(beds.group(0))
                else:
                    row_is_clean = False

                # Bathrooms
                bathrooms = re.search(r'\d+', row[4])
                if bathrooms is not None:
                    row[4] = int(bathrooms.group(0))
                else:
                    row_is_clean = False
                
                # Area
                area_m2_int = re.search(r'\d+m 2', row[5]) # e.g. 29 m 2
                area_m2_float = re.search(r'\d+.\d+m 2', row[5]) # e.g. 45.204 m 2
                area_ft2_int = re.search(r'\d+ft 2', row[5]) # e.g. 29 f 2
                area_ft2_float = re.search(r'\d+.\d+ft 2', row[5]) # e.g. 45.204 f 2

                if area_m2_float:
                    temp_area = re.search(r'\d+.\d+', area_m2_float.group(0))
                    row[5] = float(temp_area.group(0))
                elif area_m2_int:
                    temp_area = re.search(r'\d+', area_m2_int.group(0))
                    row[5] = int(temp_area.group(0))
                elif area_ft2_int:
                    temp_area = re.search(r'\d+', area_ft2_int.group(0))
                    temp_area = int(temp_area.group(0))
                    row[5] = temp_area / 10.764
                elif area_ft2_float:
                    temp_area = re.search(r'\d+.\d+', area_ft2_float.group(0))
                    temp_area = float(temp_area.group(0))
                    row[5] = temp_area / 10.764
                elif row[5] != '':
                    print(row[5])
                else:
                    row_is_clean = False

                # House Type
                if row[6] == '':
                    row_is_clean = False

            if row_is_clean: decorated_data.append(row) 
            index += 1

    with open("properties-decorated.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(decorated_data)
            

if __name__ == "__main__":
    data_decorator('properties.csv')