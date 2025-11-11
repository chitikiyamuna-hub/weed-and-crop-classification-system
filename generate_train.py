import os

image_files = []
# Change directory to where the images are extracted in the Darknet setup
os.chdir(os.path.join("data", "agri_data/data")) 
for filename in os.listdir(os.getcwd()):
    # Collect all image files
    if filename.endswith(".jpeg"):
        # Path relative to the Darknet root directory
        image_files.append("data/agri_data/data/" + filename) 
os.chdir("..") # Move back to the 'data' directory

with open("train.txt", "w") as outfile:
    for image in image_files:
        outfile.write(image)
        outfile.write("\n")

os.chdir("..") # Move back to the Darknet root directory
