import glob, sys, os, shutil

src, trg = sys.argv[1:]
SRC, TRG = './{}-{}'.format(src, trg), './{}-{}'.format(trg, src)

if not os.path.exists(SRC):
    raise NotADirectoryError
if not os.path.exists(TRG):
    os.mkdir(TRG)

files = glob.glob(SRC + '/*')

for file in files:
    file = file.split('/')[-1]
    print(file)

    if file.split('.')[-1] == 'src':
        shutil.copy(SRC + '/' + file, TRG + '/' + file.replace('src', 'trg'))
    elif file.split('.')[-1] == 'trg':
        shutil.copy(SRC + '/' + file, TRG + '/' + file.replace('trg', 'src'))
    else:
        shutil.copy(SRC + '/' + file, TRG + '/' + file)

print('done')