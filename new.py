from unstructured.partition.auto import partition

filename = "./VTS/G1110-Ed2.1-Use-of-Decision-Support-Tools-for-VTS-Personnel-January-2022.pdf"

elements = partition(filename=filename,
                     strategy='hi_res',
           )

tables = [el for el in elements if el.category == "Table"]

# print(tables[0].text)
# print(tables[0].metadata.text_as_html)
for table in tables:
    print(table)
    print("\n---------------\n")