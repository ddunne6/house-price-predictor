import csv
import json

def main():
    with open('post_process_data/geocoded-with-regions-incomes.json', encoding="utf-8") as f:
        incomes = json.load(f)

    results = []
    with open('post_process_data/properties-decorated.csv', encoding="utf-8") as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        index = 0
        for row in reader:
            if index != 0:
                for i in range(0, len(incomes)):
                    if row[0] == incomes[i].get('id'):
                        row[2] = incomes[i].get('median_income')
                        results.append(row)
                        break
            else:
                row[2] = "Address Median Income"
                results.append(row)
            index += 1

    with open("ml-dataset.csv", "w", newline='', encoding='UTF8') as fw:
        writer = csv.writer(fw)
        writer.writerows(results)

if __name__ == "__main__":
    main()
