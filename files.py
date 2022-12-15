import os
import shutil

# GLOBAL VARIABLES
destination_train_blur = r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/VIP 47922 - Lunabotics/Project/Deblurring/GitHub/DeblurGANv2/gopro_data/train/blur"
destination_train_sharp = r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/VIP 47922 - Lunabotics/Project/Deblurring/GitHub/DeblurGANv2/gopro_data/train/sharp"

destination_test_blur = r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/VIP 47922 - Lunabotics/Project/Deblurring/GitHub/DeblurGANv2/gopro_data/test/blur"
destination_test_sharp = r"/Users/nikitaravi/Documents/Academics/Year 4/Semester 2/VIP 47922 - Lunabotics/Project/Deblurring/GitHub/DeblurGANv2/gopro_data/test/sharp"


def copy_files(source_blur, source_sharp, dest_blur, dest_sharp):
	for blur_path, sharp_path in zip(source_blur, source_sharp):
		blur_directories = [folder for folder in os.listdir(blur_path) if os.path.isdir(os.path.join(blur_path, folder))]
		sharp_directories = [folder for folder in os.listdir(sharp_path) if os.path.isdir(os.path.join(sharp_path, folder))]		

		for idx, blur_folder in enumerate(blur_directories):
			blur_dir_path = os.path.join(blur_path, blur_folder)
			sharp_dir_path = os.path.join(sharp_path, blur_folder)

			for idx, image in enumerate([image for image in os.listdir(blur_dir_path) if image != "desktop.ini"]):
				blur_image_path = os.path.join(blur_dir_path, image)
				sharp_image_path = os.path.join(sharp_dir_path, image)

				if(not os.path.exists(os.path.join(dest_blur, image)) and not os.path.exists(os.path.join(dest_sharp, image))):
					print("Copied: ", blur_image_path, " to destination")
					print("Copied: ", sharp_image_path, " to destination")

					shutil.move(blur_image_path, dest_blur)
					shutil.move(sharp_image_path, dest_sharp)




if __name__ == '__main__':
	source_train_blur = [r"/Users/nikitaravi/Downloads/GOPRO/GOPRO_3840FPS_AVG_3-21/train/blur"] # r"/Users/nikitaravi/Downloads/DVD/DVD_3840FPS_AVG_3-21/train/blur", r"/Users/nikitaravi/Downloads/NFS/NFS_3840FPS_AVG_3-21/train/blur",
	source_train_sharp = [r"/Users/nikitaravi/Downloads/GOPRO/GOPRO_3840FPS_AVG_3-21/train/sharp"] # r"/Users/nikitaravi/Downloads/DVD/DVD_3840FPS_AVG_3-21/train/sharp", r"/Users/nikitaravi/Downloads/NFS/NFS_3840FPS_AVG_3-21/train/sharp", 
	
	source_test_blur = [r"/Users/nikitaravi/Downloads/GOPRO/GOPRO_3840FPS_AVG_3-21/test/blur"] # r"/Users/nikitaravi/Downloads/DVD/DVD_3840FPS_AVG_3-21/test/blur", r"/Users/nikitaravi/Downloads/NFS/NFS_3840FPS_AVG_3-21/test/blur", 
	source_test_sharp = [r"/Users/nikitaravi/Downloads/GOPRO/GOPRO_3840FPS_AVG_3-21/test/sharp"] # r"/Users/nikitaravi/Downloads/DVD/DVD_3840FPS_AVG_3-21/test/sharp", r"/Users/nikitaravi/Downloads/NFS/NFS_3840FPS_AVG_3-21/test/sharp", 


	copy_files(source_train_blur, source_train_sharp, destination_train_blur, destination_train_sharp)
	copy_files(source_test_blur, source_test_sharp, destination_test_blur, destination_test_sharp)
