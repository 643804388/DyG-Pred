import xlrd


def read_ontology_excel(path):
    # xlrd version: 1.2.0
    ontology = {
        "events": {},
        "entities": [],
        "relations": [],
        "arguments": []
    }
    wb = xlrd.open_workbook(path)
    sheet1 = wb.sheet_by_index(0)
    rows = sheet1.nrows
    for i in range(1, rows):
        event_type = ".".join([
            sheet1.cell(i, 1).value,
            sheet1.cell(i, 3).value,
            sheet1.cell(i, 5).value
        ])
        event_argument = []
        for j in range(9, 25, 3):
            argument = sheet1.cell(i, j).value
            arg_type = sheet1.cell(i, j + 2).value
            if argument != "":
                event_argument.append((argument, arg_type.split(",")))
                if argument not in ontology["arguments"]:
                    ontology["arguments"].append(argument)
        ontology["events"][event_type] = event_argument
    sheet2 = wb.sheet_by_index(1)
    rows = sheet2.nrows
    for i in range(1, rows):
        ontology["entities"].append(sheet2.cell(i, 1).value)
    sheet3 = wb.sheet_by_index(2)
    rows = sheet3.nrows
    for i in range(1, rows):
        relation = ".".join([
            sheet3.cell(i, 1).value,
            sheet3.cell(i, 3).value,
        ])
        if relation not in ontology["relations"]:
            ontology["relations"].append(relation)
    # for k in ontology:
    #     print(k)
    #     for i in ontology[k]:
    #         print(i)
    return ontology
