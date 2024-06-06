import os
folderPath = f"extracted_text"
def remove_files_in_folder(folderPath):
        # loop through all the contents of folder
        for filename in os.listdir(folderPath):
            # remove the file
            os.remove(f"{folderPath}/{filename}")

def rename_files_in_folder(folderPath):
        # loop through all the contents of folder
        for filename in os.listdir(folderPath):
            # remove the file
            os.rename(f"{folderPath}/{filename}",f"{folderPath}/0_{filename}")
# remove_files_in_folder(folderPath)
rename_files_in_folder("extracted_diagram")