# '''
# Sample Command:-
# python generate_aruco_tags.py --id 24 --type DICT_5X5_100 -o tags/
# '''


# import numpy as np
# import argparse
# from utils import ARUCO_DICT
# import cv2
# import sys


# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True, help="path to output folder to save ArUCo tag")
# ap.add_argument("-i", "--id", type=int, required=True, help="ID of ArUCo tag to generate")
# ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="type of ArUCo tag to generate")
# ap.add_argument("-s", "--size", type=int, default=200, help="Size of the ArUCo tag")
# args = vars(ap.parse_args())


# # Check to see if the dictionary is supported
# if ARUCO_DICT.get(args["type"], None) is None:
# 	print(f"ArUCo tag type '{args['type']}' is not supported")
# 	sys.exit(0)

# arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])

# print("Generating ArUCo tag of type '{}' with ID '{}'".format(args["type"], args["id"]))
# tag_size = args["size"]
# tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
# cv2.aruco.drawMarker(arucoDict, args["id"], tag_size, tag, 1)

# # Save the tag generated
# tag_name = f'{args["output"]}/{args["type"]}_id_{args["id"]}.png'
# cv2.imwrite(tag_name, tag)
# cv2.imshow("ArUCo Tag", tag)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#python generate_aruco_tags.py --number 10 --type DICT_5X5_100 -o tags/

# import numpy as np
# import argparse
# import cv2
# import sys
# from utils import ARUCO_DICT
# import os

# # Set up argument parser
# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True, help="Path to output folder to save ArUCo tags")
# ap.add_argument("-n", "--number", type=int, required=True, help="Number of ArUCo tags to generate")
# ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to generate")
# ap.add_argument("-s", "--size", type=int, default=100, help="Size of the ArUCo tag")
# args = vars(ap.parse_args())

# # Check if the specified dictionary type is supported
# if ARUCO_DICT.get(args["type"], None) is None:
#     print(f"ArUCo tag type '{args['type']}' is not supported")
#     sys.exit(0)

# # Set up the dictionary and output folder
# aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
# output_folder = args["output"]
# os.makedirs(output_folder, exist_ok=True)
# tag_size = args["size"]

# # Generate the specified number of tags with unique IDs
# for i in range(args["number"]):
#     tag_id = i  # Assign a unique ID starting from 0 up to the number specified
#     print(f"Generating ArUCo tag of type '{args['type']}' with ID '{tag_id}'")

#     # Create a blank tag and draw the marker
#     tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
#     cv2.aruco.drawMarker(aruco_dict, tag_id, tag_size, tag, 1)

#     # Save the generated tag with a unique filename
#     tag_name = f"{output_folder}/{args['type']}_id_{tag_id}.png"
#     cv2.imwrite(tag_name, tag)

#     # Display the generated tag (optional)
#     cv2.imshow("ArUCo Tag", tag)
#     cv2.waitKey(500)  # Display each tag for half a second (adjust as needed)

# cv2.destroyAllWindows()
# print("ArUCo tag generation complete.")

# import numpy as np
# import argparse
# import cv2
# import sys
# import os
# from utils import ARUCO_DICT

# # Constants for A4 size and DPI
# A4_WIDTH, A4_HEIGHT = 2480, 3508  # A4 at 300 DPI in pixels
# DPI = 300  # Adjust as necessary for desired resolution

# # Set up argument parser
# ap = argparse.ArgumentParser()
# ap.add_argument("-o", "--output", required=True, help="Path to output folder to save the A4 sheet with ArUCo tags")
# ap.add_argument("-n", "--number", type=int, required=True, help="Number of ArUCo tags to generate")
# ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to generate")
# ap.add_argument("-s", "--size", type=int, default=200, help="Size of each ArUCo tag in pixels")
# ap.add_argument("-r", "--rows", type=int, default=5, help="Number of rows on A4 sheet")
# ap.add_argument("-c", "--cols", type=int, default=4, help="Number of columns on A4 sheet")
# args = vars(ap.parse_args())

# # Check if the specified dictionary type is supported
# if ARUCO_DICT.get(args["type"], None) is None:
#     print(f"ArUCo tag type '{args['type']}' is not supported")
#     sys.exit(0)

# # Calculate grid cell size and check layout fits within A4 constraints
# aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
# output_folder = args["output"]
# os.makedirs(output_folder, exist_ok=True)
# tag_size = args["size"]
# rows, cols = args["rows"], args["cols"]

# # Create an A4 canvas
# a4_canvas = np.ones((A4_HEIGHT, A4_WIDTH, 3), dtype="uint8") * 255  # White background for A4

# # Set spacing between tags to center them on A4 sheet
# horizontal_spacing = (A4_WIDTH - cols * tag_size) // (cols + 1)
# vertical_spacing = (A4_HEIGHT - rows * tag_size) // (rows + 1)

# # Generate tags and place them on the A4 sheet
# tag_id = 0
# for r in range(rows):
#     for c in range(cols):
#         if tag_id >= args["number"]:
#             break

#         print(f"Generating ArUCo tag of type '{args['type']}' with ID '{tag_id}'")

#         # Create tag and draw marker
#         tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
#         cv2.aruco.drawMarker(aruco_dict, tag_id, tag_size, tag, 1)

#         # Calculate position on A4 sheet
#         x = horizontal_spacing + c * (tag_size + horizontal_spacing)
#         y = vertical_spacing + r * (tag_size + vertical_spacing)

#         # Place the tag on the A4 canvas
#         a4_canvas[y:y + tag_size, x:x + tag_size] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)

#         # Increment tag ID
#         tag_id += 1

# # Save the A4 sheet with all tags
# a4_filename = os.path.join(output_folder, f"A4_sheet_{args['type']}_tags.png")
# cv2.imwrite(a4_filename, a4_canvas)

# print(f"A4 sheet with ArUCo tags saved at: {a4_filename}")


import numpy as np
import argparse
import cv2
import sys
import os
from utils import ARUCO_DICT

# Constants for A4 size and DPI
A4_WIDTH, A4_HEIGHT = 2480, 3508  # A4 at 300 DPI in pixels
DPI = 300  # Adjust as necessary for desired resolution

# Set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="Path to output folder to save the A4 sheet with ArUCo tags")
ap.add_argument("-n", "--number", type=int, required=True, help="Number of ArUCo tags to generate")
ap.add_argument("-t", "--type", type=str, default="DICT_ARUCO_ORIGINAL", help="Type of ArUCo tag to generate")
ap.add_argument("-s", "--size", type=int, default=200, help="Size of each ArUCo tag in pixels")
ap.add_argument("-r", "--rows", type=int, default=10, help="Number of rows on A4 sheet")
ap.add_argument("-c", "--cols", type=int, default=4, help="Number of columns on A4 sheet")
args = vars(ap.parse_args())

# Check if the specified dictionary type is supported
if ARUCO_DICT.get(args["type"], None) is None:
    print(f"ArUCo tag type '{args['type']}' is not supported")
    sys.exit(0)

# Set up the ArUco dictionary and output folder
aruco_dict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
output_folder = args["output"]
os.makedirs(output_folder, exist_ok=True)
tag_size = args["size"]
rows, cols = args["rows"], args["cols"]

# Create an A4 canvas
a4_canvas = np.ones((A4_HEIGHT, A4_WIDTH, 3), dtype="uint8") * 255  # White background for A4

# Set spacing between tags to center them on A4 sheet
horizontal_spacing = (A4_WIDTH - cols * tag_size) // (cols + 1)
vertical_spacing = (A4_HEIGHT - rows * tag_size) // (rows + 1)

# Generate tags and place them on the A4 sheet
tag_id = 0
for r in range(rows):
    for c in range(cols):
        if tag_id >= args["number"]:
            break

        print(f"Generating ArUCo tag of type '{args['type']}' with ID '{tag_id}'")

        # Create tag and draw marker
        tag = np.zeros((tag_size, tag_size, 1), dtype="uint8")
        cv2.aruco.drawMarker(aruco_dict, tag_id, tag_size, tag, 1)

        # Calculate position on A4 sheet
        x = horizontal_spacing + c * (tag_size + horizontal_spacing)
        y = vertical_spacing + r * (tag_size + vertical_spacing)

        # Place the tag on the A4 canvas
        a4_canvas[y:y + tag_size, x:x + tag_size] = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)

        # Add the tag ID text next to each tag (below or beside the marker)
        text_x = x + int(tag_size * 0.1)  # Slightly adjust for padding
        text_y = y + tag_size + 30  # Position below the tag
        cv2.putText(a4_canvas, f"ID: {tag_id}", (text_x, text_y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                    color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        # Increment tag ID
        tag_id += 1

# Save the A4 sheet with all tags and IDs
a4_filename = os.path.join(output_folder, f"A4_sheet_{args['type']}_tags.png")
cv2.imwrite(a4_filename, a4_canvas)

print(f"A4 sheet with ArUCo tags saved at: {a4_filename}")
