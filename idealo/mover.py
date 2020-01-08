import os
import shutil

dest = 'imgs_all'
src_dir = 'imgs'

def use_walk():
    if not os.path.isdir(dest):
        raise OSError('not a dir!')
    for p, _, files in os.walk(src_dir, topdown = True, onerror = None, followlinks = False):
        for f in files:
            if os.path.isfile(os.path.join(dest,f)):
                continue
            full = os.path.join(p,f)
            shutil.move(full,dest)

if __name__ == '__main__':
    use_walk()
