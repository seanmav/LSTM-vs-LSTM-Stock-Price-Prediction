import csv

csvFile = open('MICROSOFT_CLOSE.csv', encoding='utf8')
reader = csv.reader(csvFile)


# get data
data = [row for row in reader]
# get headers and remove from data
headers = data.pop(0)
# reverse the data
data_reversed = data[::-1]

print(data_reversed)

with open("MICROSOFT_CLOSE_copy.csv", "wt") as fp:
    writer = csv.writer(fp, delimiter=",")
    # writer.writerow(["your", "header", "foo"])  # write header
    writer.writerows(data_reversed)
    

# =============================================================================
# with open("MICROSOFT.csv") as csvFile:
#     reader = csv.reader(csvFile)
# 
# 
#     # get data
#     data = [row for row in reader]
#     # get headers and remove from data
#     headers = data.pop(0)
#     # reverse the data
#     data_reversed = data[::-1]
#     
#     print(data_reversed)
#     
#     writer = csv.writer(csvFile, delimiter=",")
#     writer.writerows(data_reversed)
#     csvFile.close()
# 
# =============================================================================




