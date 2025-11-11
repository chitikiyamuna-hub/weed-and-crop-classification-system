import os

image_files = []
# Change directory to where the images are extracted
# Assumes the Kaggle data extract results in data/agri_data/data/
os.chdir(os.path.join("data", "agri_data/data")) 
for filename in os.listdir(os.getcwd()):
    # Check for image files and their corresponding label files (which aren't handled here but should exist)
    if filename.endswith(".jpeg"):
        # The path should be relative to the Darknet root for the detector to find it
        image_files.append("data/agri_data/data/" + filename) 
os.chdir("..") # Move back to the 'data' directory
with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")
    # outfile.close() # Closing is redundant when using 'with open(...)', but harmless
os.chdir("..") # Move back to the Darknet root directory
