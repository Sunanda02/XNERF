import sys, zipfile

if len(sys.argv) < 2:
    print('Usage: list_zip_names.py <zip>')
    sys.exit(1)

for p in sys.argv[1:]:
    print('ZIP:', p)
    try:
        with zipfile.ZipFile(p) as z:
            names = z.namelist()
            csvs = [n for n in names if n.lower().endswith('.csv')]
            if not csvs:
                print('  No CSVs found')
            else:
                for n in csvs:
                    print('  ', n)
    except Exception as e:
        print('  ERROR:', e)
