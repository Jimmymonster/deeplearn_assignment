from augmenter import Augmenter
from PIL import Image
import os
def load_images_from_directory(directory_path: str) -> list:
    images = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                image = Image.open(file_path).convert('RGB')
                images.append(image)
            except Exception as e:
                print(f"Failed to load image {file_path}: {e}")
    return images, os.listdir(directory_path)

path = "/home/jim/inceptionv3/data/banana/banana_splited_augmented/train"
folder_list = os.listdir(path)

augmenter = Augmenter()

perspec_directions = [(0,0),(0,1),(0,-1),(1,0),(-1,0)]
perspec_angles = [25]
rotation_angles = [-45,-15,0,15,45]
augmenter.add_augmentation('set_resolution', max_resolution=(400,400))
augmenter.add_augmentation('hue_color',brightness_range = (0.9,1.1), contrast_range = (0.9,1.1), hue_range = (-0.05,0.05), saturation_range=(0.95,1.05))
augmenter.add_augmentation('noise',min_noise_level=0,max_noise_level=10)
i=0

for rotation_angle in rotation_angles:
    for perspec_angle in perspec_angles:
            for (x,y) in perspec_directions:
                augmenter.add_augmentation('rotation',angle_range=(rotation_angle,rotation_angle),image_range=(i,i))
                augmenter.add_augmentation('set_perspective',angle=perspec_angle,direction=(x,y),image_range=(i,i))
                i+=1

for folder in folder_list:
    print("====== augmenting : " + folder + " =======")
    path_to_images = os.path.join(path,folder)
    images, names = load_images_from_directory(path_to_images)
    idx = 0
    for image,name in zip(images,names):
        print(f"====== augmenting {name} in " + folder + f" =======")
        augmenter.add_dict(folder,[image])
        augmented_images = augmenter.augment(folder, i, random=True)

        for img in augmented_images:
             name = folder + "augmented" + str(idx) + ".png"
             img.save(os.path.join(path_to_images,name))
             idx+=1

        augmenter.clear_dict()

print("augment finished")
        