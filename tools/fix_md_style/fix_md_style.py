
import re
import os

# re-write all .md files in current directory tree to not have <style scoped> blocks.


def main():
    targets = [os.path.join(root, file) for root, dirs, files in os.walk(".") for file in files if file.endswith(".md")]
    for ti in targets:
        with open(ti, 'r') as fi:
            txt = ''.join(fi.readlines()) + os.linesep
            txt2 = re.sub('<style scoped>[^<]*</style>', '', txt)
        if txt != txt2:
            print("re-writing " + ti)
            with open(ti, 'w') as fo:
                fo.write(txt2)


if __name__ == "__main__":
    main()
