
'''
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
'''

import numpy as np
import cv2
import sys
from utils import ARUCO_DICT
import argparse
import time
import os
from datetime import datetime

def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, frame_number, folder_path, pose_file):
    '''
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera
    return:
    frame - The frame with the axis drawn on it
    '''

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters_create()

    corners, ids, rejected_img_points = cv2.aruco.detectMarkers(
        gray, cv2.aruco_dict, parameters=parameters,
        cameraMatrix=matrix_coefficients,
        distCoeff=distortion_coefficients
    )

    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                corners[i], 0.02, matrix_coefficients, distortion_coefficients)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

            # Extract x, y, z coordinates from tvec
            x, y, z = tvec[0][0]
            tag_id = ids[i][0]

            # Write x, y, z positions to the pose file
            pose_file.write(f"{frame_number}, {tag_id}, {x:.3f}, {y:.3f}, {z:.3f}\n")

            # Save the current frame in the specified folder
            frame_filename = os.path.join(folder_path, f"frame_{frame_number}.png")
            cv2.imwrite(frame_filename, frame)

    return frame

if __name__ == '__main__':

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
    ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
    ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
    args = vars(ap.parse_args())

    # Verify the ArUco dictionary type
    if ARUCO_DICT.get(args["type"], None) is None:
        print(f"ArUCo tag type '{args['type']}' is not supported")
        sys.exit(0)

    aruco_dict_type = ARUCO_DICT[args["type"]]
    calibration_matrix_path = args["K_Matrix"]
    distortion_coefficients_path = args["D_Coeff"]

    # Load camera calibration parameters
    k = np.load(calibration_matrix_path)
    d = np.load(distortion_coefficients_path)

    # Generate a unique folder for this run using the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = f"run_{timestamp}"
    os.makedirs(folder_path, exist_ok=True)

    # Open pose data file in the new folder and write header
    pose_file_path = os.path.join(folder_path, "pose_data.txt")
    with open(pose_file_path, "w") as pose_file:
        pose_file.write("frame_number, tag_id, x, y, z\n")

        # Start capturing video
        video = cv2.VideoCapture(0)
        time.sleep(2.0)
        frame_number = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            # Estimate pose and save results
            output = pose_estimation(frame, aruco_dict_type, k, d, frame_number, folder_path, pose_file)

            # Display the frame with pose estimation
            cv2.imshow('Estimated Pose', output)

            frame_number += 1

            # Exit on pressing 'q'
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    # Release resources
    video.release()
    cv2.destroyAllWindows()
    print(f"Results saved in folder: {folder_path}")


# '''
# Sample Usage:-
# python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
# '''


# import numpy as np
# import cv2
# import sys
# from utils import ARUCO_DICT
# import argparse
# import time
# import os


# def pose_esitmation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):

#     '''
#     frame - Frame from the video stream
#     matrix_coefficients - Intrinsic matrix of the calibrated camera
#     distortion_coefficients - Distortion coefficients associated with your camera

#     return:-
#     frame - The frame with the axis drawn on it
#     '''

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.aruco_dict = cv2.aruco.Dictionary_get(aruco_dict_type)
#     parameters = cv2.aruco.DetectorParameters_create()


#     corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, cv2.aruco_dict,parameters=parameters,
#         cameraMatrix=matrix_coefficients,
#         distCoeff=distortion_coefficients)

#     if len(corners) > 0:
#         for i in range(0, len(ids)):
#             # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
#             rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
#                                                                        distortion_coefficients)
#             # Draw a square around the markers
#             cv2.aruco.drawDetectedMarkers(frame, corners)

#             # Draw Axis
#             cv2.aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

#             # Extract x, y, z coordinates from tvec
#             x, y, z = tvec[0][0]
#             tag_id = ids[i][0]
#             with open("pose_data.txt", "a") as f:
#             # f.write("x, y, z\n")  # Write the header for clarity
#         # If markers are detected
#             # Write x, y, z positions to the file
#                 f.write(f"{frame_number}, {tag_id}, {x:.3f}, {y:.3f}, {z:.3f}\n")
#                 # f.write(f"{x}, {y}, {z}\n")

#             os.makedirs("frames", exist_ok=True)
#             frame_filename = f"frames/frame_{frame_number}.png"
#             cv2.imwrite(frame_filename, frame)

#     return frame

# if __name__ == '__main__':

#     ap = argparse.ArgumentParser()
#     ap.add_argument("-k", "--K_Matrix", required=True, help="Path to calibration matrix (numpy file)")
#     ap.add_argument("-d", "--D_Coeff", required=True, help="Path to distortion coefficients (numpy file)")
#     ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to detect")
#     args = vars(ap.parse_args())




#     if ARUCO_DICT.get(args["type"], None) is None:
#         print(f"ArUCo tag type '{args['type']}' is not supported")
#         sys.exit(0)

#     aruco_dict_type = ARUCO_DICT[args["type"]]
#     calibration_matrix_path = args["K_Matrix"]
#     distortion_coefficients_path = args["D_Coeff"]

#     k = np.load(calibration_matrix_path)
#     d = np.load(distortion_coefficients_path)

#     video = cv2.VideoCapture(0)
#     time.sleep(2.0)
#     frame_number = 0
#         #Open pose data file and write header
#     with open("pose_data.txt", "w") as f:
#         f.write("frame_number, tag_id, x, y, z\n")

#     while True:
#         ret, frame = video.read()

#         if not ret:
#             break

#         output = pose_esitmation(frame, aruco_dict_type, k, d)

#         cv2.imshow('Estimated Pose', output)

#         frame_number += 1

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             break

#     video.release()
#     cv2.destroyAllWindows()