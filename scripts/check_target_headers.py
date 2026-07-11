import zipfile, io, csv

zips = [
    'data/archives/AndMal2020/static/CCCS-CIC-Benign-CSVs.zip',
    'data/archives/AndMal2020/static/CCCS-CIC-Malicious-CSVs.zip',
    'data/archives/AndMal2020/dynamic/AndMal2020-Dynamic-BeforeAndAfterReboot.zip',
    'data/archives/cicmaldroid2020/CSV.zip',
]

for p in zips:
    print('ZIP:', p)
    try:
        with zipfile.ZipFile(p) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if not csvs:
                print('  No CSVs')
                continue
            for n in csvs[:10]:  # limit per-zip output
                try:
                    with z.open(n) as f:
                        text = io.TextIOWrapper(f, encoding='utf-8', errors='replace')
                        reader = csv.reader(text)
                        try:
                            header = next(reader)
                        except StopIteration:
                            header = []
                        if header and any(cell.strip() for cell in header):
                            print('  ', n, '-> header:', header)
                        else:
                            print('  ', n, '-> header: (none) first row preview:', header if header else '(empty)')
                except Exception as e:
                    print('   ERROR reading', n, e)
    except Exception as e:
        print('  ERROR opening zip:', e)
    print()
