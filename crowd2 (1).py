import numpy as np
import cv2
import os
from tkinter import *
from tkinter import messagebox
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance as distance
import cmath
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog


 
root = Tk()
root.title(" People Density Estimation ")
root.configure(background="lightgreen")

labelpath ='coco.names'
file = open(labelpath)
label = file.read().strip().split("\n")  
label[0]

weightspath ='yolov3.weights'
configpath ='yolov3.cfg'

net = cv2.dnn.readNetFromDarknet(configpath, weightspath)  


layer_names = net.getLayerNames()  
ln = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]    


def videocheck():
    i=0
    fln=filedialog.askopenfilename(initialdir=os.getcwd(),title="Open file",filetypes=(("MP4","*.mp4"),("All File","*.*")))
    videopath =fln

    video = cv2.VideoCapture(videopath)
    ret = video
    init = time.time()
    sample_time = 5
    if sample_time < 1:
        sample_time = 1

    data=[]
    while(True):
        ret, frame = video.read()  #returns a boolean value
        if ret == False:
            print('Error running the file :(')
        frame = cv2.resize(frame, (640, 440), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(           
            frame, 1/255.0, (416, 416), swapRB=True, crop=False) 
        r = blob[0, 0, :, :]                        
        net.setInput(blob)  #First, we have to set the input blob to our neural network model that we have loaded from the disk.
 
        t0 = time.time()   
        outputs = net.forward(ln) 

        t = time.time()    

        boxes = []
        confidences = []
        classIDs = []
        center = []
        output = []
        count = 0
        results = []
        breach = set()

        h, w = frame.shape[:2]
        for output in outputs: #detecting in output model
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)   #mapping the class id with class name

                confidence = scores[classID]      

                if confidence > 0.5: #checking if confidence> threshold
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    center.append((centerX, centerY))
                    box = [x, y, int(width), int(height)] #width nd height of bounding boxes
                    boxes.append(box)
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # color = [int(c) for c in colors[classIDs[i]]]
                if(label[classIDs[i]] == 'person'):
                    # people()
                    cX = (int)(x+(y/2))
                    cY = (int)(w+(h/2))
                    center.append((cX, cY))
                    res = ((x, y, x+w, y+h), center[i])
                    results.append(res)
                    dist = cmath.sqrt(
                        ((center[i][0]-center[i+1][0])**2)+((center[i][1]-center[i+1][1])**2)) #finding the eucleadian dist
                    if(dist.real < 100):
                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (0, 0, 255), 2)
                        cv2.circle(frame, center[i], 4, (0, 0, 255), -1)
                        # cv2.line(frame, (center[i][0], center[i][1]), (center[i+1][0], center[i+1][1]), (0,0, 255), thickness=3, lineType=8)
                        count = count+1

                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (0, 255, 0), 2)
                        cv2.circle(frame, center[i], 4, (0, 255, 0), -1)
                        count = count+1
            
           

            cv2.putText(frame, "Count: {}".format(
                count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
        cv2.imshow('Frame', frame)
        # plt.show()
        print(count)
        current_time =i
        data.append((count,current_time))
        i=i+1
        if cv2.waitKey(25) & 0xFF == ord('q'): #all windows will be closed on pressing q
            break
    print(1)
    
    video.release()
    cv2.destroyAllWindows()
    if count >0:
        t3.delete("1.0", END)
        t3.insert(END, count)
    
    
    
    def Sort(sub_li):
    
       
        sub_li.sort(key = lambda x: x[1])
        return sub_li
    
    # Driver Code

    print(Sort(data))
    print(data)
    x = []
    y=[]
    for i in data:
        x.append(i[1])
    for i in data:
        y.append(i[0])
    # corresponding y axis values

    
    # plotting the points 
    plt.plot(x, y)
    
    # naming the x axis
    plt.xlabel('x - axis')
    # naming the y axis
    plt.ylabel('y - axis')
    
    # giving a title to my graph
    plt.title('My first graph!')
    
    # function to show the plot
    plt.show()

def photo():

    ret = True
    f_types= [('Image Files', '*.jpg')]
    filename = filedialog.askopenfilename(filetypes=f_types)
    img=cv2.imread(filename) 
    frame=img
    cv2.imshow('Frame', frame) 
    if ret == False:
        print('Error running the file :(')
    frame = cv2.resize(frame, (640, 440), interpolation=cv2.INTER_AREA) 
    blob = cv2.dnn.blobFromImage(
        frame, 1/255.0, (416, 416), swapRB=True, crop=False) 
    r = blob[0, 0, :, :]
    net.setInput(blob) 
    t0 = time.time()
    outputs = net.forward(ln) 
    t = time.time()

    boxes = []
    confidences = []
    classIDs = []
    center = []
    output = []
    count = 0
    results = []

    h, w = frame.shape[:2]
    for output in outputs:
        for detection in output:
            scores = detection[5:] 
            classID = np.argmax(scores)

            confidence = scores[classID]

            if confidence > 0.5: 
                box = detection[0:4] * np.array([w, h, w, h]) 
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2)) 
                y = int(centerY - (height / 2))
                center.append((centerX, centerY))
                box = [x, y, int(width), int(height)]
                boxes.append(box)
                confidences.append(float(confidence))
                classIDs.append(classID)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) 

    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # color = [int(c) for c in colors[classIDs[i]]]
            if(label[classIDs[i]] == 'person'): 
                # people()
                cX = (int)(x+(y/2))
                cY = (int)(w+(h/2))
                center.append((cX, cY))
                res = ((x, y, x+w, y+h), center[i])
                results.append(res)
                dist = cmath.sqrt(
                    ((center[i][0]-center[i+1][0])**2)+((center[i][1]-center[i+1][1])**2))
                if(dist.real < 100):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.circle(frame, center[i], 4, (0, 0, 255), -1)
                    # cv2.line(frame, (center[i][0], center[i][1]), (center[i+1][0], center[i+1][1]), (0,0, 255), thickness=3, lineType=8)
                    count = count+1

                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.circle(frame, center[i], 4, (0, 255, 0), -1)
                    count = count+1
        
        cv2.putText(frame, "Count: {}".format(
            count), (20, frame.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    
    cv2.imshow('Frame', frame)
   
    if count >0:
        t4.delete("1.0", END)
        t4.insert(END, count)
    cv2.waitKey()
    cv2.destroyAllWindows()



def Sort(sub_li):
  
   
    sub_li.sort(key = lambda x: x[1])
    return sub_li
 
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)

w2 = Label(root,justify=LEFT, text=" People Density Estimation Using Machine Learning ")
w2.config(font=("Elephant", 30),background="lightblue")
w2.grid(row=1, column=0, columnspan=2, padx=100,pady=40)

NameLb1 = Label(root, text="Please Select the Options  ")
NameLb1.config(font=("Elephant", 12),background="lightblue")
NameLb1.grid(row=5, column=0, pady=10)

S1Lb = Label(root,  text="Video")
S1Lb.config(font=("Elephant", 14))
S1Lb.grid(row=7, column=0, pady=10 )

S2Lb = Label(root,  text="Upload photo")
S2Lb.config(font=("Elephant", 14))
S2Lb.grid(row=8, column=0,pady=10)

lr = Button(root, text="VIDEO",height=2, width=10, command=videocheck)
lr.config(font=("Elephant", 14),background="green")
lr.grid(row=15, column=0,pady=20)
lr = Button(root, text="PHOTO",height=2, width=10, command=photo)
lr.config(font=("Elephant", 14),background="green")
lr.grid(row=16, column=0,pady=20)

NameLb = Label(root, text="Predict Using:")
NameLb.config(font=("Elephant", 15),background="lightblue")
NameLb.grid(row=13, column=0, pady=20)

t3 = Text(root, height=2, width=15)
t3.config(font=("Elephant", 15))
t3.grid(row=15, column=1 ,padx=60)
t4 = Text(root, height=2, width=15)
t4.config(font=("Elephant", 15))
t4.grid(row=16, column=1 ,padx=60)

root.mainloop()
