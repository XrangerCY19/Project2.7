#Made By XrangerCY19
import os
import time
import random
import cv2
from skimage.measure import compare_ssim as ssim
from PIL import Image
import numpy
import sys
import shutil

# global variables
classifier_file_name = 'haarcascade_frontalface_default.xml'

#Structural similarity 
min_ssim_index_val = 0.44

#source of input 0 for primary camera
source = ''

scale_factor = 1.1
min_neighbors = 6
min_size = (100, 100)
flags = cv2.CASCADE_SCALE_IMAGE
cwd = os.getcwd()
pathname = os.path.join(cwd + "/detected_faces/")
temp_path = os.path.join(cwd + "/temp/")
folder_name = 0


#Made By XrangerCY19
def detect_face(gray):
    """
    Detect faces in a frame

    """
    face_cascade = cv2.CascadeClassifier(classifier_file_name)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor,minNeighbors=min_neighbors,minSize=min_size,flags=flags)
    return faces


def crop_and_save(gray, faces):
    """
    Crop and save faces in a frame

    """
    count = 0
    for (x, y, w, h) in faces:
        name = int(time.time())
        name = str(name)
        while os.path.isfile("temp/" + name + ".png"):
            count += 1
            name = name + "+" + str(count)
        cv2.imwrite(os.path.join("temp/" + name + ".png"), gray[y:y + h, x:x + w])

#Made By XrangerCY19
def get_ssim(img1, img2):
    """
    comparing two image files.
    """
    img1 = cv2.cvtColor(numpy.array(img1), cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(numpy.array(img2), cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    s_value = ssim(img1, img2)
    return s_value


def get_immediate_subdirectories():
    """
    Returns immediate subdirectories
    """
    return [name for name in os.listdir("detected_faces")
            if os.path.isdir(os.path.join("detected_faces", name))]


def save_unique_image():
    """
    Detect and save unique images in different folders
    """
    global folder_name
    filelist = [file for file in os.listdir('temp') if file.endswith('.png')]

    if filelist:
            for image_path in filelist:
                found = 0
                img_to_del = Image.open("temp/" + image_path)
                if not get_immediate_subdirectories():
                        found = 1
                        os.makedirs('detected_faces/1/')
                        img_to_del.save('detected_faces/1/'+ image_path)
                        os.remove(os.path.join(temp_path, image_path))
                        folder_name = 1
                else:
                    for folder in get_immediate_subdirectories():
                        folder_filelist = [file for file in os.listdir("detected_faces/" + folder) if
                                           file.endswith('.png')]
                        count = len(folder_filelist)
                        file = folder_filelist[0]
                        img_to_compare = Image.open("detected_faces/" + folder + "/" + file)
                        if img_to_del.size > img_to_compare.size:
                            temp_image_resized = img_to_del.resize(img_to_compare.size, Image.ANTIALIAS)
                            index = get_ssim(temp_image_resized, img_to_compare)
                        elif img_to_del.size < img_to_compare.size:
                            img_to_compare = img_to_compare.resize(img_to_del.size, Image.ANTIALIAS)
                            index = get_ssim(img_to_del, img_to_compare)
                        else:
                            index = get_ssim(img_to_del, img_to_compare)
                        if index > min_ssim_index_val:
                            found = 1
                            if count < 5:
                                img_to_del.save(pathname + "/" + folder + "/" + image_path)
                            print image_path
                            if os.path.isfile(os.path.join(temp_path, image_path)):
                                os.remove(os.path.join(temp_path, image_path))
                if found == 0:
                    folder_name += 1
                    os.makedirs('detected_faces/' + str(folder_name))
                    img_to_del.save(pathname + "/" + str(folder_name) + "/" + image_path)
                    if os.path.isfile(os.path.join(temp_path, image_path)):
                        os.remove(os.path.join(temp_path, image_path))



def get_check_folder():
    """
    Checks folders if faces exists

    """
    filelist = [file for file in os.listdir('temp') if file.endswith('.png')]
    image_count = len(filelist)
    if image_count == 0:
        with open('file.txt', 'a') as f:
            f.write('No faces detected in image. \n')
        print"No faces detected in image."
        exit()
    print "Detected "+str(image_count)+" faces in the image."
    with open('file.txt', 'a') as f:
        f.write("Detected "+str(image_count)+" faces in the image.\n")


    if filelist:
        for image_path in filelist:
            target = cv2.imread("temp/" + image_path)
            cv2.imshow("detected face", target)
            k = cv2.waitKey(1) & 0xFF
            img_to_del = Image.open("temp/" + image_path)
            for folder in get_immediate_subdirectories():
                count = 0
                val = 0
                folder_filelist = [file for file in os.listdir("detected_faces/" + folder) if
                                   file.endswith('.png')]
                for file in folder_filelist:
                    img_to_compare = Image.open("detected_faces/" + folder + "/" + file)
                    if img_to_del.size > img_to_compare.size:
                        temp_image_resized = img_to_del.resize(img_to_compare.size, Image.ANTIALIAS)
                        index = get_ssim(temp_image_resized, img_to_compare)
                    elif img_to_del.size < img_to_compare.size:
                        img_to_compare = img_to_compare.resize(img_to_del.size, Image.ANTIALIAS)
                        index = get_ssim(img_to_del, img_to_compare)
                    else:
                        index = get_ssim(img_to_del, img_to_compare)
                    val += index
                    count += 1
                if count > 0:
                    index = val/count
                    if index > min_ssim_index_val:
                        with open('file.txt', 'a') as f:
                            f.write("   Detected same face in DB folder  "+ folder +" \n")
                        print "     Detected same face in DB folder "+ folder

                        num = 6     #6th line in the text file generated by exiftool
                        rfilename = random.choice(os.listdir(cwd+"\\detected_faces\\"+folder))
                        #print rfilename
                        os.system('cd '+ cwd +'\\detected_faces\\'+ folder +'&& exiftool -w txt '+ rfilename) #Execute exiftool
                        tfile = rfilename[:-4] #Remove last 4 characters from the filename, ie to remove ".png"
                        #print tfile
                        path2txt = cwd+'\\detected_faces\\'+ folder +'\\'+ tfile +'.txt' #Change path & add .txt extension
                        with open(path2txt) as f: #block to open corresponding path & read only the 6th line(contain timestamp)
                            for i, line in enumerate(f, 1):
                                if i == num:
                                    break
                        #print line
                        ltime = '       Last seen : '+line[-26:] #Remove first 26 character
                        #print ltime
                        tfilepath = cwd+'\\file.txt'
                        with open(tfilepath, 'a') as f:     #Write the timestamp to file.txt
                            f.write(ltime)
                        os.remove(path2txt) #remove the generated txt file from exiftool


                if os.path.isfile(os.path.join(temp_path, image_path)):
                    os.remove(os.path.join(temp_path, image_path))




#Made By XrangerCY19
def checkifexists():
    """
    Get input images and detect faces in them

   """
    i = 0
    for image in sys.argv:
        if i == 0:
            i += 1
        else:
            if os.path.exists("temp"):
                shutil.rmtree("temp")
            os.makedirs("temp")
            target = cv2.imread(image)
            gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            faces = detect_face(gray)
            crop_and_save(gray, faces)
            get_check_folder()




if __name__ == '__main__':
    if not len(sys.argv) > 1:
        pass
    else:
        if os.path.exists("file.txt"):
            os.remove("file.txt")
        checkifexists()

#Made By XrangerCY19