import cv2
import numpy as np


def preprocess_frame(img):
    # -- 1.1 convert to grayscale --
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.namedWindow('img_gray',cv2.WINDOW_NORMAL)    
    cv2.imshow('img_gray',img_gray)
    
    # -- 1.2 Binarize image --
    _, thresh = cv2.threshold(img_gray,10,255,0)
    cv2.namedWindow('thresh',cv2.WINDOW_NORMAL)    
    cv2.imshow('thresh',thresh)
    return thresh


def veh_detection_processing(img):
    
    '''Processing the image '''
    # 1.1 sobel edge detections
    sobelxy = cv2.Sobel(src=img, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) 
    
    cv2.namedWindow('sobelxy',cv2.WINDOW_NORMAL)    
    cv2.imshow('sobelxy',sobelxy)
    
    ## 1.2 Dialation
    dialation = cv2.dilate(sobelxy,(3,3),iterations=2)
    cv2.namedWindow('dialation',cv2.WINDOW_NORMAL)    
    cv2.imshow('dialation',dialation)
    
    ## 1.3 Closing
    closing_kern_odd_l = 35
    closing_kern_odd_b = 5
    kernel_closing = np.ones((closing_kern_odd_b,closing_kern_odd_l),np.uint8)
    closing = cv2.morphologyEx(dialation,cv2.MORPH_CLOSE,kernel_closing)
    cv2.namedWindow('closing',cv2.WINDOW_NORMAL)    
    cv2.imshow('closing',closing)

    ## 1.4 Opening
    Opening_kern_odd_l = 7
    Opening_kern_odd_b = 3

    kernel_opening = np.ones((Opening_kern_odd_b,Opening_kern_odd_l),np.uint8)
    opening = cv2.morphologyEx(closing,cv2.MORPH_OPEN,kernel_opening)
    cv2.namedWindow('opening',cv2.WINDOW_NORMAL)    
    cv2.imshow('opening',opening)

    return opening

def get_vehicle_from_area(area):
    '''Funtion to return vehicle type according to contour area'''
    if area <100:
        vehicle_type = "Two-wheeler"
    elif area < 400:
        vehicle_type = "Small Car"
    elif area < 600:
        vehicle_type = "Big Car"
    elif area < 700:
        vehicle_type = "Truck"
    elif area >= 700:
        vehicle_type = "Long Truck"

    return vehicle_type



def check_if_point_lies_in_ROI(point,region):
    '''Function to check if the point is within ROI or not'''
    veh_cnt_center = point
    x1,y1 = region[0]
    x2,y2 = region[1]
    x_veh,y_veh = veh_cnt_center
    
    vehicle_in_ROI =  False
    if x_veh >= x1 and x_veh <=x2:
        if y_veh >=y1 and y_veh <=y2:
            vehicle_in_ROI = True

    return vehicle_in_ROI



def count_vehicle(binary_img, main_img, dict_roi_region):
    '''Funtion to count the vehicles '''
    # print('binary_img.shape',binary_img.shape)
    binary_img = binary_img.astype(np.uint8)
    contours, heirarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cnt_centers_list = []
    cnt_area_list = []
    cnt_veh_type_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        center = (int(x + w/2),int(y +h/2))
        vehicle_type = get_vehicle_from_area(area)

        cnt_centers_list.append(center)
        cnt_area_list.append(area) 
        cnt_veh_type_list.append(vehicle_type)


    roi_list = dict_roi_region['Roi_regions']
    to_count_status_list = dict_roi_region['to_Count_status']
    list_region_contious_black_cnt = dict_roi_region['roi_continuous_blacked_fr_cnt']
    cropped_list = []
    roi_wise_veh_type = [None,None,None,None,None,None,None,None,None,None]
    
    ## Check through each ROI whether contour exist in it or not
    for i in range(len(roi_list)):
        region = roi_list[i]
        x1,y1 = region[0]
        x2,y2 = region[1]
        cropped = binary_img[y1:y2,x1:x2]
        whites = np.sum(cropped==255)
        cropped_area = cropped.shape[0]* cropped.shape[1]
        percentage = whites/cropped_area * 100
        main_img = cv2.rectangle(main_img, (x1,y1),(x2,y2),(0,255,0),3)
        to_Count_status = to_count_status_list[i]
        continous_black_region_count = list_region_contious_black_cnt[i]
        # print("Percentage",percentage)
        if percentage >0 and to_Count_status==True:
            for j, cnt_center in enumerate(cnt_centers_list):
                cnt_in_ROI = check_if_point_lies_in_ROI(point=cnt_center,region=region)
                # print('cnt_in_ROI',cnt_in_ROI)
                if cnt_in_ROI:
                    ## If veh_contour_center is in ROI                        
                    vehicle_found = cnt_veh_type_list[j]
                    # print('vehicle_found', vehicle_found)
                    roi_wise_veh_type[i] = vehicle_found
                    to_count_status_list[i]= False  # Reset to False
        
        elif percentage==0 and continous_black_region_count>5:
            to_count_status_list[i]= True  # Reset to True for next count

        
        elif percentage==0:
            list_region_contious_black_cnt[i] = continous_black_region_count+1    
        elif percentage>0:
            list_region_contious_black_cnt[i]= 0

    updated_dictionary = dict_roi_region
    updated_dictionary['roi_continuous_blacked_fr_cnt'] = list_region_contious_black_cnt
    updated_dictionary['to_Count_status'] = to_count_status_list

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)        
        main_img = cv2.rectangle(main_img,(int(x),int(y)),(int(x+w),int(y+h)),(255,0,0),1)
        

    return main_img, roi_wise_veh_type, updated_dictionary


vid_path = 'Input\production ID_4626414.mp4'

cap = cv2.VideoCapture(vid_path)


ret, backgr_img = cap.read()

## Resizing the images to process faster
backgr_img = cv2.resize(backgr_img,None,fx=0.25,fy=0.25)

# Variables to store the count of vehicles
veh_count_bike = 0
veh_count_sm_cars = 0
veh_count_lg_cars = 0
veh_count_sm_truck = 0
veh_count_lg_trucks = 0

# Dictionary to store ROI region details
dict_roi_region = {'Roi_regions':[[(740,60),(760,75)],[(740,76),(760,90)], [(740,91),(760,105)],
                        [(430,207),(467,224)],[(430,225),(467,241)],[(430,242),(467,258)],
                        [(420,277),(460,297)],[(420,298),(460,317)],[(420,318),(460,338)],
                        [(250,435),(286,483)]],
                'to_Count_status':[True,True,True,
                                    True,True,True,
                                    True,True,True,
                                    True],
                'roi_continuous_blacked_fr_cnt':[0,0,0,0,0,0,0,0,0,0]
                }  
frameNo = 0
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,None,fx=0.25,fy=0.25)

    ## Subtracting current image from previous
    subtracted = cv2.subtract(frame,backgr_img)
    (h, w) = subtracted.shape[:2]
    (cX, cY) = (w // 2, h // 2) 

    cv2.namedWindow('subtracted',cv2.WINDOW_NORMAL)    
    cv2.imshow('subtracted',subtracted)
    ## Rotating the image as roads are at 50 degree, to make roads horizontal
    M = cv2.getRotationMatrix2D((cX, cY), 50, 1.0)
    rotated = cv2.warpAffine(subtracted, M, (w, h))
    rotated_main = cv2.warpAffine(frame.copy(), M, (w, h))
    cv2.namedWindow('rotated',cv2.WINDOW_NORMAL)    
    cv2.imshow('rotated',rotated)
    cv2.namedWindow('rotated_main',cv2.WINDOW_NORMAL)    
    cv2.imshow('rotated_main',rotated_main)
    ## Preprocessing and applying median filter
    processed_subtr = preprocess_frame(rotated)
    median_filter = cv2.medianBlur(processed_subtr,5)
    cv2.namedWindow('median_filter',cv2.WINDOW_NORMAL)    
    cv2.imshow('median_filter',median_filter)

    ## Post processing for contour detection and area finding
    processed_binary = veh_detection_processing(median_filter)

    ## Getting Output as per the postprocessed result
    Output, roi_wise_veh_type, updated_dictionary = count_vehicle(processed_binary,rotated_main,dict_roi_region)
    # print('roi_wise_veh_type',roi_wise_veh_type)

    dict_roi_region = updated_dictionary

    for veh_found in roi_wise_veh_type:
        if veh_found is not None:
            if veh_found  == "Two-wheeler":
                veh_count_bike+=1
            elif veh_found == "Small Car":
                veh_count_sm_cars+=1
            elif veh_found == "Big Car":
                veh_count_lg_cars+=1
            elif veh_found == "Truck":
                veh_count_sm_truck+=1
            elif veh_found == "Long Truck":
                veh_count_lg_trucks+=1

    
    ## Showing output on the Result image
    
    output_text1 = f"Two-wheeler Count : {veh_count_bike}"
    output_text2 = f"Small Car Count : {veh_count_sm_cars}"
    output_text3 = f"Big Car Count : {veh_count_lg_cars}"
    output_text4 = f"Heavy Vehicle Count : {veh_count_sm_truck}"
    output_text5 = f"Long Heavy Vehicle Count : {veh_count_lg_trucks}"


    Output = cv2.putText(Output,output_text1,(10,10),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
    Output = cv2.putText(Output,output_text2,(10,25),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
    Output = cv2.putText(Output,output_text3,(10,40),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
    Output = cv2.putText(Output,output_text4,(10,55),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
    Output = cv2.putText(Output,output_text5,(10,70),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)


    
    
    # For DEBUGGING
    # print("Output.shape",Output.shape)
    # # cv2.namedWindow('processed_back',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('subtracted',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('processed_subtr',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('median_filter',cv2.WINDOW_NORMAL)
    # cv2.namedWindow('Output',cv2.WINDOW_NORMAL)
    # # cv2.imshow('processed_back',processed_back)
    # cv2.imshow('subtracted',subtracted)
    # cv2.imshow('processed_subtr',processed_subtr)
    # cv2.imshow('median_filter',median_filter)
    
    cv2.namedWindow('frame',cv2.WINDOW_NORMAL)
    cv2.imshow('frame',frame)
    cv2.imshow('Output',Output)
    cv2.imwrite('Output.jpg',Output)


    frameNo+=1
    
    backgr_img = frame
    key = cv2.waitKey(1)
    if key == ord('q'):
        break



cv2.destroyAllWindows()