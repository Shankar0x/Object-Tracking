import numpy as np
import cv2
import os

def predict(A,B,x,u,P,Q):   
    # Update states           
    x = np.dot(A,x) + np.dot(B, u)  #x_k =Ax_(k-1) + Bu_(k-1)       
    # Calculate error covariance
    P = np.dot(np.dot(A,P), A.T) + Q  # P= A*P*A' + Q     
    return x
def load_annotations(annotation_file):
    with open(annotation_file, 'r') as file:
        annotations = [list(map(int, line.strip().split(','))) for line in file.readlines()]
    return annotations

def update(H,x,P,R, z):  
    S = np.dot(H, np.dot(P, H.T)) + R   # S = H*P*H'+R
    K = np.dot(np.dot(P, H.T), np.linalg.inv(S)) # K = P * H'* inv(H*P*H'+R)   
    x = np.round(x + np.dot(K,(z - np.dot(H, x))))  # Estimate x
    x=([x[0,0],x[1,1],0,0])    
    I = np.eye(H.shape[1])
    # Update error covariance matrix
    P = (I - (K * H)) * P     
    return x

def find_center(frame_idx, annotations):
    # Get the annotation for the current frame
    if frame_idx < len(annotations):
        x, y, w, h = annotations[frame_idx] 
        center = [x + w // 2, y + h // 2] 
        return center
    else:
        print("Warning: No annotation available for frame", frame_idx)
        return None  
 
def kalman_filter():
    # Define sampling time
    dt = 1
    u_x = 1
    u_y = 1
    std_acc = 1
    x_std_meas=0.1
    y_std_meas=0.1
    # Define the  control input variables
    u = np.matrix([[u_x],[u_y]])
    # Intial State
    x = np.matrix([[0],[0],[0],[0]])

    # Define the State Transition Matrix A
    A = np.matrix([[1, 0, dt, 0],
                        [0, 1, 0, dt],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    # Define the Control Input Matrix B
    B = np.matrix([[(dt**2)/2, 0],
                        [0, (dt**2)/2],
                        [dt,0],
                        [0,dt]])
    # Define Measurement Mapping Matrix
    H = np.matrix([[1, 0, 0, 0],
                        [0, 1, 0, 0]])
    #Initial Process Noise Covariance
    Q = np.matrix([[(dt**4)/4, 0, (dt**3)/2, 0],
                        [0, (dt**4)/4, 0, (dt**3)/2],
                        [(dt**3)/2, 0, dt**2, 0],
                        [0, (dt**3)/2, 0, dt**2]]) * std_acc**2
    #Initial Measurement Noise Covariance
    R = np.matrix([[x_std_meas**2,0],[0, y_std_meas**2]])
    #Initial Covariance Matrix
    P = np.eye(A.shape[1]) 
    return A, B, H, Q, P, R, u, x


def object_tracking(frames, annotations, output_path):
    A, B, H, Q, P, R, u, x = kalman_filter()
    VideoCap = cv2.VideoCapture(output_path)

    # loading the frames

    frames_folder = frames
    frame_files = sorted(os.listdir(frames_folder))

    # starting frame
    f_ind = 0
    while(True):

        if f_ind < len(frame_files):
                frame_path = os.path.join(frames_folder, frame_files[f_ind])
                frame = cv2.imread(frame_path)
        else:
            break  # Break the loop if there are no more frames
        
        # load the annotations

        annotation_file = annotations
        annots = load_annotations(annotation_file)
        center = find_center(f_ind, annots)
        dev=np.random.randint(-20,30) # Large error in observation is added
        z=[center[0]+dev,center[1]-dev]
        cv2.circle(frame,center, 10, (0,255, 255), 2)  
        # Predict
        x=predict(A,B,x,u,P,Q)
        (x1,y1)=(x[0],x[1])
        # Draw a rectangle as the predicted object position        
        cv2.rectangle(frame, (int(x1 - 5), int(y1 - 5)), (int(x1 + 5), int(y1 + 5)), (0, 0, 255), 2)
        x=update(H,x,P,R,z)
        (x2,y2) = (x[0],x[1])  
        cv2.rectangle(frame, (int(x2 - 5), int(y2 - 5)), (int(x2 + 5), int(y2 + 5)), (255, 0, 255), 2)   

        # Draw the bounding boxes 
        x, y, w, h = map(int, annots[f_ind])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        x=np.matrix([[x2],[y2],[0],[0]])
        cv2.imshow("image", frame) 
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        f_ind += 1
    VideoCap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    frames = "basketball/basketball-3/img" 
    annotations = "basketball/basketball-3/groundtruth.txt"
    output_path = "output_vids/bb_3.mp4"
    object_tracking(frames, annotations, output_path)