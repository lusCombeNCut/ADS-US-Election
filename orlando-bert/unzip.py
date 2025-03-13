import gzip
import shutil

def unzip_gz_file(input_file, output_file):
    with gzip.open(input_file, 'rb') as f_in:
        with open(output_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

if __name__ == "__main__":
    input_file = r'C:\Users\Orlan\Documents\Applied-Data-Science\orlando-bert\topics.csv.gz'
    output_file = r'file.csv'
    unzip_gz_file(input_file, output_file)