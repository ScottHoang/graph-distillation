import os
import shutil
import sys

if __name__ == "__main__":
    _, dst = sys.argv
    sub_folders = ['results', 'configs', 'logs']
    folder2remove = set()
    for sub_folder in sub_folders:
        sympath = os.path.join(dst, sub_folder, "latest")
        if os.path.islink(sympath):
            os.remove(sympath)

        path = [
            os.path.join(dst, sub_folder, i)
            for i in os.listdir(os.path.join(dst, sub_folder))
        ]
        path2remove = list(filter(lambda x: len(os.listdir(x)) == 0, path))
        path2sort = list(filter(lambda x: len(os.listdir(x)) > 0, path))
        path2sort.sort(key=lambda x: os.path.getmtime(x))
        os.symlink(os.path.basename(path2sort[-1]), sympath)
        for p in path2remove:
            folder2remove.add(os.path.basename(p))

    for sub_folder in sub_folders:
        for folder in folder2remove:
            path = os.path.join(dst, sub_folder, folder)
            if os.path.isdir(path):
                shutil.rmtree(path)
