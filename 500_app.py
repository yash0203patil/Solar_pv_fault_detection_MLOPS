import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pandas as pd
import math
import torch
from datetime import datetime , timezone 
import os




# Models Initialization
# seg_model = YOLO('seg/best.pt')
# # det_model = YOLO('obj/best.pt')
det_model = YOLO("models/best.pt")


# Set up Streamlit sidebar for user controls
st.sidebar.title(":red[Controls]")




# 2. Toggle Buttons to switch between original and processed images
show_original = st.sidebar.checkbox("Show Original Image", value=True)
show_processed = st.sidebar.checkbox("Show Processed Image", value=True)



# 4. Checkboxes for displaying masks and detection boxes
show_segmentation_mask = st.sidebar.checkbox("Show Segmentation Mask", value=True)
# show_detection_boxes = st.sidebar.checkbox("Show Detection Boxes", value=True)

# Functions and main app logic remains largely unchanged
# Functions
def box_label(image, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
    lw = max(round(sum(image.shape) / 2 * 0.003), 2)
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
    if label:
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        p1n = (p1[0]+25, p1[1])
        p2n = (p2[0]+25, p2[1])
        cv2.rectangle(image, p1n, p2n, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (p1[0]+25,  p1[1] - 2 if outside else p1[1] + h + 2), 0, lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)

def plot_bboxes(image, boxes, labels=[], colors=[], score=False, conf=None):
    if labels == []:
        labels = {0: u'__background__', 1: u'Normal', 2: u'Crack' , 3: u'Blackspot'}
    if colors == []:
        colors = [(0, 255, 0),(255, 0, 0),(89, 161, 197)]
    for box in boxes:
        if score:
            label = labels[int(box[-1])+1] + " " + str(round(100 * float(box[-2]),1)) + "%"
        else:
            label = labels[int(box[-1])+1]
        if conf:
            if box[-2] > conf:
                color = colors[int(box[-1])]
                box_label(image, box, label, color)
        else:
            color = colors[int(box[-1])]
            box_label(image, box, label, color)

# calculate number of rows in panel , panel height
def calculate_nrows(data):
    # Calculate and print the distances for each point
    distances_from_origin = []
    for point in data:
        x, y ,w, h, conf, c = point
        distance = math.sqrt(x**2 + y**2)
        distances_from_origin.append([x, y ,w, h, conf, c, distance])

    df = pd.DataFrame(distances_from_origin).sort_values(6,axis=0).reset_index().drop("index", axis=1)
    
    total_height = math.dist((df.iloc[0,0], df.iloc[0,1]), (df.iloc[0,0], df.iloc[-1, 3]))
    height_rect = math.dist((df.iloc[0,0], df.iloc[0,1]), (df.iloc[0,0], df.iloc[0,3]))
    nrows = int(math.ceil(total_height / height_rect))
    panel_height = int(df.iloc[-1, 3])

    return nrows, panel_height

def sequence(image, data, nrows, panel_height):
    centers = np.array(data)
    d = panel_height / nrows
    for i in range(nrows):
        f = centers[:, 1] - d * i
        a = centers[(f < d) & (f > 0)]
        rows = a[a.argsort(0)[:, 0]]
        yield rows


def sorted_bbox(image, data, nrows, panel_height):
    sorted_array = []
    count = 0
    for row in sequence(image, data, nrows,  panel_height):
        #cv2.polylines(image, np.int32([row[0], row[1]]), False, (255, 0, 255), 2)

        for x1, y1, x2, y2, conf, clas  in row:
            count +=1
            #cv2.circle(image, np.int32((x1+100, y1+100)), 50, (0, 0, 255), -1)  
            cv2.putText(image, str(count), np.int32((x1 + 100 - 50, y1 +100 + 25)), 2, cv2.FONT_HERSHEY_PLAIN, (0, 255, 255), 2)
            sorted_array.append([x1, y1, x2, y2, conf, clas,count])
    return sorted_array, image

# def total_defected_area(img ,det_data , seg_data):
    
    

#     # det_data  =  det_results[0].boxes.data.cpu().data.numpy() 
#     distance = []
#     for points in det_data:
#         x,y,w,h,conf,c = points
#         dist = math.sqrt(x**2 + y**2)
#         distance.append([ x,y,w,h,conf,c,dist])
#     df = pd.DataFrame(distance).sort_values(6,axis=0).reset_index().drop("index",axis=1)

#     cord1 = (df.iloc[0,0] , df.iloc[0,1])
#     cordh = (df.iloc[0,0] , df.iloc[-1,3])
#     cordw = (df.iloc[-1,2] , df.iloc[0,1])
#     total_area = math.dist(cord1,cordh)*math.dist(cord1,cordw)
#     if len(seg_data) != 0:
#         binary_mask = torch.any(seg_data,dim=0).int()*255
#         binary_mask= binary_mask.cpu().numpy()
#         binary_count = np.unique(binary_mask,return_counts=True)
#         blackspot_pixel_count  = binary_count[1][1]
#         blackspot_per = round((blackspot_pixel_count / total_area)*100 , 2)
#         return blackspot_per, binary_mask,total_area 
#     else: 
#         blackspot_per = 0
#         binary_mask = []
#         return blackspot_per, binary_mask ,total_area





# Sidebar
# st.sidebar.title("Settings")
# st.sidebar.title("Confidence Threshold")
# st.sidebar.write("1. A confidence threshold is a predetermined score that determines whether a detected object is accepted or rejected ")
# st.sidebar.write("2. Keep the Confidence Threshold value close to 50% for better results")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)



# Main
csv_data = []

# st.title("")
st.title('PV Solar Module Fault Detection')


st.subheader(":red[Description]")
st.write("1. Upload the solar panel Images (Maximum 10 images) ")
st.write("2. The Image size should be less than 2000 x 2000 ")
st.write("3. The Processed Image indicates the Defected cell and the Normal Cell present in solar panel")
st.write("4. The Segmented Mask Image indicates the total defected area of the solar panel")
st.write("5. Download Report will give the detailed analysis of solar panel image ")

uploaded_files = st.file_uploader("Upload images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if len(uploaded_files) < 11:
    for uploaded_file in uploaded_files:
        file_buffer = uploaded_file.read()
        if not file_buffer:
            st.warning(f"File {uploaded_file.name} seems to be empty or unreadable. Skipping...")
            continue
        image = cv2.imdecode(np.frombuffer(file_buffer, np.uint8), -1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # results = det_model.predict(image)
        resized_img = cv2.resize(image,(640,640))
        det_results = det_model.predict(resized_img , hide_conf=True , hide_labels=True)
        # seg_results = seg_model.predict(image)
        if show_original:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)

        det_data = det_results[0].boxes.data.cpu().data.numpy()
        # if hasattr(seg_results[0].masks, 'masks'):
        #     seg_data  =  seg_results[0].masks.masks
        # else:
        #     seg_data = []

        
        # Extracting data for CSV
        nrows, panel_height = calculate_nrows(det_data)
        sorted_array, image = sorted_bbox(image, det_data, nrows, panel_height)
        defected_cell_num = []
        for  x1, y1,x2,y2,conf,c,sort_ind in sorted_array:
            if c == 1:
                defected_cell_num.append(sort_ind)
            else:
                pass


        total_boxes = len(det_results[0].boxes.data)
        defect_boxes = sum([1 for box in det_results[0].boxes.data if int(box[-1]) == 1]) # Assuming class "defect" has index 1
        # defected_area_percentage,binary_mask,total_area = total_defected_area(image , det_data , seg_data)
        csv_data.append([uploaded_file.name, total_boxes, defect_boxes, defected_cell_num])

        # Process and display processed image
        if show_processed:
            plot_bboxes(resized_img, det_results[0].boxes.data.cpu().data.numpy(), conf=confidence_threshold)
            st.subheader("Processed Image")
            st.image(resized_img, use_column_width=True)
        # if show_segmentation_mask:
        #     if len(binary_mask) != 0 :
        #         st.subheader("Segmented mask")
        #         st.image(binary_mask, use_column_width=True)
        #     else:
        #         st.warning("Panel does not conatin any black spot")

        


    # Saving CSV
   
    df = pd.DataFrame(csv_data, columns=["Image Name", "Total Cells", "Total Defected Cells", "Defeceted Cell No."])
    csv_file = df.to_csv(index=False)
    date_time = datetime.now().isoformat(timespec="seconds")

    st.sidebar.download_button(label="Download Report", data=csv_file, file_name=f"{date_time}.csv", mime="text/csv")
    st.sidebar.title(":red[Confidence Threshold]")
    st.sidebar.write("1. A confidence threshold is a predetermined score that determines whether a detected object is accepted or rejected ")
    st.sidebar.write("2. Keep the Confidence Threshold value close to 50% for better results")
    # confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
    if st.button('Train'):
        os.system("python app.py")
    else:
        pass


else:
    st.warning("WARNING: Please upload upto 10 images at a time")