

from ast import main
import os


def convert2jpg(filename):
    f = open(filename, "r")
    filename_out = str(name_gen(filename))
    f_out = open(filename_out,"w")
    for line in f:
        if "tiff" in line.lower():
            temp = line.replace("tiff","jpg").replace("_0","")
            line = temp
        f_out.write(line)
    f.close()
    f_out.close()

def name_gen(filename):
    temp = filename.split(".")
    new_name = temp[0] + "_new.json"
    return new_name

def labelme_json_to_dataset(json_path):
    os.system("labelme_json_to_dataset "+json_path+" -o "+json_path.replace(".","_"))

if __name__ == '__main__':
    cwd = os.getcwd()
    print("cwd = " + cwd)
    for filename in os.listdir(cwd):
        print("file name = " + str(filename))
        if "new" not in filename.lower() and "json" in  filename.lower() :
           f = os.path.join(cwd, filename)
           print(f)
           convert2jpg(f)
    print("change all the jsons :) ")
    print("convert to data set now")
    for filename in os.listdir(cwd):
        print("file name = " + str(filename))
        if "new"  in filename.lower() and "json" in  filename.lower() :
           f = os.path.join(cwd, filename)
           labelme_json_to_dataset(f)