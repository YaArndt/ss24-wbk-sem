import os
import os
import random
import shutil

def count_image_files(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    count = 0

    for file_name in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, file_name)):
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                count += 1

    return count

def select_and_move_images(input_folder, output_folder, num_images):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif']
    image_files = []

    # Get a list of all image files in the input folder
    for file_name in os.listdir(input_folder):
        if os.path.isfile(os.path.join(input_folder, file_name)):
            if any(file_name.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file_name)

    # Select random images from the list
    selected_images = random.sample(image_files, num_images)

    # Move the selected images to the output folder
    for image in selected_images:
        source_path = os.path.join(input_folder, image)
        destination_path = os.path.join(output_folder, image)
        shutil.move(source_path, destination_path)

files_IO = count_image_files(r"Data\BilderNeu\IO")
files_NIO = count_image_files(r"Data\BilderNeu\NIO")

print("IO files: ", files_IO)
print("NIO files: ", files_NIO)
print("Total files: ", files_IO + files_NIO)

percentage = 0.375
files_to_move_IO = int(files_IO * percentage)
files_to_move_NIO = int(files_NIO * percentage)

print("Files to move from IO: ", files_to_move_IO)
select_and_move_images(r"Data\BilderNeu\IO", r"Data\TestSet\IO", files_to_move_IO)

print("Files to move from NIO: ", files_to_move_NIO)
select_and_move_images(r"Data\BilderNeu\NIO", r"Data\TestSet\NIO", files_to_move_NIO)

test_set_files_IO = count_image_files(r"Data\TestSet\IO")
test_set_files_NIO = count_image_files(r"Data\TestSet\NIO")

print("TestSet IO files: ", test_set_files_IO)
print("TestSet NIO files: ", test_set_files_NIO)
print("Total TestSet files: ", test_set_files_IO + test_set_files_NIO)

print("Files moved successfully!")