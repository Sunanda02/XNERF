import csv
import sys
import re
from collections import Counter

path = sys.argv[1]
hex_re = re.compile(r'^[0-9a-fA-F]{64}$')

total = 0
missing = 0
bad_sha = 0
bad_date = 0
families = Counter()
types = Counter()

with open(path, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    try:
        header = next(reader)
    except StopIteration:
        print('EMPTY')
        sys.exit(0)

    for i,row in enumerate(reader, start=2):
        total += 1
        if len(row) != len(header):
            print(f'MALFORMED_LINE:{i}:COLS={len(row)}')
        if any(cell.strip()=='' for cell in row):
            missing += 1
        sha = row[0].strip() if len(row)>0 else ''
        if not hex_re.match(sha):
            bad_sha += 1
        date = row[3].strip() if len(row)>3 else ''
        # simple YYYY-MM-DD check
        if not re.match(r'^\d{4}-\d{2}-\d{2}$', date):
            bad_date += 1
        if len(row) > 1:
            families[row[1].strip()] += 1
        if len(row) > 2:
            types[row[2].strip()] += 1

print('HEADER:', header)
print('TOTAL_ROWS:', total)
print('ROWS_WITH_EMPTY_CELLS:', missing)
print('BAD_SHA256_COUNT:', bad_sha)
print('BAD_DATE_COUNT:', bad_date)
print('TOP_FAMILIES:', families.most_common(10))
print('TOP_TYPES:', types.most_common(10))
