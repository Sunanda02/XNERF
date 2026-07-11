import sys, zipfile, io, csv

if len(sys.argv) < 2:
    print('Usage: inspect_zip_headers.py <zip1> [zip2 ...]')
    sys.exit(1)

for p in sys.argv[1:]:
    print('ZIP:', p)
    try:
        with zipfile.ZipFile(p) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if not csvs:
                print('  No CSVs')
                continue
            for n in csvs:
                try:
                    with z.open(n) as f:
                        text = io.TextIOWrapper(f, encoding='utf-8', errors='replace')
                        reader = csv.reader(text)
                        try:
                            header = next(reader)
                        except StopIteration:
                            header = []
                        print('  ', n, '-> header:', header)
                except Exception as e:
                    print('   ERROR reading', n, e)
    except Exception as e:
        print('  ERROR opening zip:', e)
