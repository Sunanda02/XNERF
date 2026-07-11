import sys
import zipfile
import io
import csv

def inspect_zip(path, preview_lines=3):
    print('ZIP:', path)
    try:
        z = zipfile.ZipFile(path)
    except Exception as e:
        print('  ERROR opening zip:', e)
        return
    csv_files = [n for n in z.namelist() if n.lower().endswith('.csv')]
    if not csv_files:
        print('  No CSV files found inside.')
        return
    for name in csv_files:
        print('  CSV:', name)
        try:
            with z.open(name) as f:
                text = io.TextIOWrapper(f, encoding='utf-8', errors='replace')
                reader = csv.reader(text)
                try:
                    header = next(reader)
                except StopIteration:
                    print('    EMPTY CSV')
                    continue
                print('    Header:', header)
                for i,row in enumerate(reader):
                    print('    Row sample:', row)
                    if i+1 >= preview_lines:
                        break
        except Exception as e:
            print('    ERROR reading CSV:', e)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: inspect_zips.py <zip1> [zip2 ...]')
        sys.exit(1)
    for p in sys.argv[1:]:
        inspect_zip(p)
