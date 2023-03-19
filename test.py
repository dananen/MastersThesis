def write(file_writer):
        for i in range(10):
            file_writer.write(str(i) + '\n')

with open('/Users/dansvenonius/Desktop/big_text_file_test.txt', 'w') as f:

    write(f)
    write(f)
