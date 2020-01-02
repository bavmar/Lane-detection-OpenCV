#Imports
import cv2
import numpy as np

#My versions where
#OpenCV: 3.4.3
#Numpy: 1.15.4
#Python 3.7

#Shows Opencv/Numpy version you're using.
print ("OpenCV version: " + cv2. __version__)
print ("Numpy version: " + np.version.version)

#VARIABLE SETTINGS

#Display different windows based off this value
#normal window = 1
#edges window  = 2
#all windows   = 3 
display = 3

#Window sizes
window_size = (1980,1040)

#Use maxlinegap to determine the total length a single display line can have (play around a bit)
max_line_gap = 20

#Color and boldness for drawn lines
line_color = (0, 0, 255)
line_boldness = 10

#Threshhold for yellow lanes
low_yellow = np.array([18, 94, 140], dtype=np.uint8)
up_yellow = np.array([48, 255, 255], dtype=np.uint8)

#Threshold for white lanes
low_white = np.array([0,0,165], dtype=np.uint8)
up_white = np.array([255,255,255], dtype=np.uint8)

#Region of interest points
vertices = np.array([[200,650],[450,500],[450,500],[600,425],[750,425],[1250,650],], np.int32)
display_vertices = True


#Videocapture
#Change to 0 if you want to use your first attached webcam / change path where video is located
#Download the example video for best use.
#If you decide to use your own video make sure to change the values of the region of interest points (vertices)
video = cv2.VideoCapture("../path-tofile/example.mp4")

#Define region of interest
def roi(video, vertices):
	maskz = np.zeros_like(video)
	cv2.fillPoly(maskz, vertices, 255)
	masked = cv2.bitwise_and(video, maskz)
	return masked

while True:
    #Read
    ret, orig_frame = video.read()
    
    #Create Draw layer
    line_image = np.copy(orig_frame) * 0  
	
    #Gassianblur
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    
    #Color to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    #Inrages
    mask1 = cv2.inRange(hsv, low_yellow, up_yellow)
    mask2 = cv2.inRange(hsv, low_white, up_white)
    
    #Put these in a single mask
    mask = cv2.bitwise_or(mask1, mask2)
    
    #Check
    target = cv2.bitwise_and(frame,frame, mask=mask)
    
    #Canny
    canny = cv2.Canny(target, 75, 150)
        
    edges = roi(canny,[vertices])
    
    #The lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=max_line_gap)
    
    if display_vertices:
	cv2.polylines(frame,[vertices],True,(0,255,255))
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            #Draw lines
            cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_boldness) 			
            
            #Overlay
            lines_edges = cv2.addWeighted(frame, 1, line_image, 1, 0)
    
    #Open normal frames
    if (display == 1) or (display == 3):
	    #Give name to window
	    cv2.namedWindow("frameOutput", cv2.WINDOW_NORMAL)
	    
	    #Resize
	    frameR = cv2.resize(lines_edges, window_size)
	    
	    #Show frame ouput with overlay
	    cv2.imshow("frameOutput", frameR)
	
    #Open edges frames
    if (display == 2) or (display == 3):
	    #Give name to window
	    cv2.namedWindow("edgesOutput", cv2.WINDOW_NORMAL)
	    
	    #Resize
	    edgesR = cv2.resize(edges, window_size)
	    
	    #Show canny output
	    cv2.imshow("edgesOutput", edgesR)
	
    #Press 'q' to quit.
    if cv2.waitKey(1) & 0xFF == ord('q'):
	    break

#Break all after quit		
video.release()
cv2.destroyAllWindows()
