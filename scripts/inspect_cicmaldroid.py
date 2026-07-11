import zipfile, io, csv
p='data/archives/cicmaldroid2020/CSV.zip'
with zipfile.ZipFile(p) as z:
    csvs=[n for n in z.namelist() if n.lower().endswith('.csv')]
    print('FILES:')
    for n in csvs:
        print(n)
    print('---')
    for n in csvs[:5]:
        with z.open(n) as f:
            text=io.TextIOWrapper(f,encoding='utf-8',errors='replace')
            reader=csv.reader(text)
            header=next(reader, None)
            print(n,'->', header)
