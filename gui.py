#imports for gui
import cv2
import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from subprocess import call
from tkinter import *
from tkinter import messagebox
import tkinter.font as font

#====================================

def close_window():  # to close the window when you click on the Quit button
    import sys
    sys.exit()



def retake():
    Title = ttk.Label(root, text="|| Paw Patrol ||  ", background="#8080ff", font=('times', 20)).place(x=240, y=8)
    btn1 = ttk.Button(root, text="Select your file", command=c_open_file_old, image=openIcon, compound=LEFT).place(
        x=235, y=60)
    location = ttk.Label(root, text="PATH : ", background="#8080ff", font=('times', 15)).place(x=20, y=120)


def c_open_file_old():
    rep = filedialog.askopenfilenames(
        parent=root,
        initialdir='shell:MyComputerFolder',
        initialfile='file',
        filetypes=[
            ("All files", "*"),
            ("JPEG", "*.jpg"),
            ("PNG", ".png")])
    # print ("Repairs where the file is:", dir) // just to display the directory in console
    # display a label that puts the path where the image is located

    list = root.grid_slaves()
    # destroy the button that contains the path each time when wanted to reuse
    for l in list:
        l.destroy()

    for i in range(1, 8):
        Label(root, background="#8080ff").grid(row=i)

    # change the font of the PATH button
    s = ttk.Style()
    s.configure('my.TButton', font=('Times', 12))

    chemin = ttk.Button(root, text="  " + rep[0], style='my.TButton').grid(row=8)
    retake()  # this function is just to display the starting content again on empty lables

    try:

        # os.startfile(rep[0])
        # Load the cascade
        face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')

        # Read the input image
        img = cv2.imread(rep[0])  # rep [0] is not rep alone because rep is an argument array
        scale_percent = 50  # the percentage to be taken from the original image, i.e. we want to resize the original image by 25%

        width = int(img.shape[1] * scale_percent / 100)  # length
        height = int(img.shape[0] * scale_percent / 100)  # height
        dim = (width, height)
        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        # Convert into grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # Detect dog
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the dog
        for (x, y, w, h) in faces:
            cv2.rectangle(resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display the output
        cv2.imshow('Detected faces', resized)
        cv2.waitKey()

    except IndexError:
        print("No file selected")

# ----------------------------------------------------
#            GUI
# ----------------------------------------------------
root = tk.Tk()
root.geometry("600x250")
root.title("Dog Breed Recognition")
root.resizable(width=False, height=False)
root.configure(bg='#0A10E7')  # change the background color
style = ttk.Style(root)
style.theme_use("clam")
img = PhotoImage(file= "C:/Users/liamo/Soft_Engr/dogbkgrd.png")
#add downloaded image into an img file so the code can access it
# -----------------------------------------------------
#          GUI Position
# -----------------------------------------------------
# Retrieve the values of the length and width of the window
windowWidth = root.winfo_reqwidth()
windowHeight = root.winfo_reqheight()
#Retrieve the width / length value of the screen and the current tab
positionRight = int(root.winfo_screenwidth() / 2 - windowWidth / 2 - 50)
positionDown = int(root.winfo_screenheight() / 3 - windowHeight / 3 + 50)

# Position the tab in the center of the screen
root.geometry("+{}+{}".format(positionRight, positionDown))
# ==================================================================================================
icon1 = PhotoImage(file=r"C:/Users/liamo/Soft_Engr/exit.png")
photoExit = icon1.subsample(8, 8)
icon2 = PhotoImage(file=r"C:/Users/liamo/Soft_Engr/open.png")
openIcon = icon2.subsample(17, 17)

# ==================================================================================================
# ==================================================================================================


Title = ttk.Label(root, text="|| Paw Patrol ||  ", background="#0A10E7", font=('times', 20)).place(x=240, y=8)
btn1 = ttk.Button(root, text="Select your file", command=c_open_file_old, image=openIcon, compound=LEFT).place(x=235,
                                                                                                               y=60)

exitButton = ttk.Button(root, text="Exit ", command=close_window, compound=LEFT, image=photoExit)
exitButton.place(x=450, y=195)

location = ttk.Label(root, text="PATH : ", background="#0A10E7", font=('times', 15)).place(x=20, y=120)
#original background color #8080ff

root.mainloop()