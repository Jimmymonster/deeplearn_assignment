import cv2
import numpy as np
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter
import random

class Augmenter:
    def __init__(self, image_dict = None):
        self.image_dict = image_dict if image_dict is not None else {}   # images should be a list of PIL Image objects
        self.augmentations = []  # List to store the sequence of augmentations

    def add_dict(self, category, images):
        """Add or update a category with a list of PIL images."""
        if not isinstance(images, list) or not all(isinstance(img, Image.Image) for img in images):
            raise ValueError("Images should be provided as a list of PIL Image objects.")
        self.image_dict[category] = images

    def remove_dict(self, category):
        """Remove a category from the dictionary."""
        if category in self.image_dict:
            del self.image_dict[category]
        else:
            raise ValueError(f"Category '{category}' does not exist.")
        
    def clear_dict(self):
        self.image_dict = {}

    def update_dict(self, category, images):
        """Update the images in an existing category."""
        if category in self.image_dict:
            if not isinstance(images, list) or not all(isinstance(img, Image.Image) for img in images):
                raise ValueError("Images should be provided as a list of PIL Image objects.")
            self.image_dict[category] = images
        else:
            raise ValueError(f"Category '{category}' does not exist.")    
    
    def add_augmentation(self, aug_method_name, *args, image_range=None, **kwargs):
        """Add an augmentation method to the sequence with arguments and optional image range."""
        if hasattr(self, aug_method_name):
            self.augmentations.append((getattr(self, aug_method_name), args, kwargs, image_range))
        else:
            raise ValueError(f"Augmentation method '{aug_method_name}' does not exist.")
    
    def clear_augmentations(self):
        self.add_augmentation = []

    def set_augmentations(self, aug_method_data):
        """Set the sequence of augmentations with methods and their arguments."""
        self.augmentations = []
        for method_name, args, kwargs in aug_method_data:
            self.add_augmentation(method_name, *args, **kwargs)
        
    def _apply_augmentations(self, images):
        """Apply all augmentations in the specified order."""
        for aug, args, kwargs, image_range in self.augmentations:
            if image_range is None:
                # Apply to all images
                for i in range(len(images)):
                    images[i] = aug(images[i] ,*args, **kwargs)
            else:
                start, end = image_range
                if(len(images)<start):
                    continue
                if(len(images)-1<end):
                    end = len(images)-1
                for i in range(start,end+1):
                    images[i] = aug(images[i]  ,*args, **kwargs)
                
                # Apply to a specific range of images
                # augmented_images = []
                # for i, img in enumerate(images):
                #     if start <= i <= end:
                #         augmented_images.append(aug(img, *args, **kwargs))
                #     else:
                #         augmented_images.append(img)
                # images = augmented_images
        return images

    def augment(self, category, num_images, random=True):
        """Select and augment a number of images from the specified category."""
        if category not in self.image_dict:
            raise ValueError(f"Category '{category}' does not exist in the image dictionary.")
        images = self.image_dict[category]
        if not images:
            raise ValueError(f"No images available in category '{category}'.")
        # Ensure num_images is a positive integer
        if num_images <= 0:
            raise ValueError("The number of images to augment must be a positive integer.")
        # Select images evenly, looping over if necessary
        if(random == True):
            selected_images = [images[i % len(images)] for i in range(num_images)]
        else:
            repetitions = num_images // len(images)
            remainder = num_images % len(images)
            selected_images = []
            for idx, img in enumerate(images):
                count = repetitions + (1 if idx < remainder else 0)
                selected_images.extend([img] * count)
        # Apply augmentations to the selected images
        augmented_images =  self._apply_augmentations(selected_images)
        return augmented_images
    
    #=========================== utility function =====================================
    def pil_to_cv(self, pil_image):
        cv_image = np.array(pil_image)
        # Check if the image has an alpha channel (RGBA)
        if cv_image.shape[2] == 4:  # RGBA
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGRA)
        else:  # RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    
        return cv_image

    def cv_to_pil(self, cv_image):
        if cv_image.shape[2] == 4:  # BGRA
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGRA2RGBA))
        else:  # BGR
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        return pil_image
    
    def reset_obbox(self, image):
        oriented_bbox=[(0,0),(image.size[0]-1,0),(image.size[0]-1,image.size[1]-1),(0,image.size[1]-1)]
        return oriented_bbox
    
    def cut_image_width(self, image: Image.Image, percentage: float, side: str = 'right') -> Image.Image:
        if not 0 < percentage < 1:
            raise ValueError("Percentage must be between 0 and 1 (exclusive).")
        width, height = image.size
        cut_width = int(width * percentage)
        if side == 'left':
            # Cut from the left
            cropped_image = image.crop((cut_width, 0, width, height))
        elif side == 'right':
            # Cut from the right
            cropped_image = image.crop((0, 0, width - cut_width, height))
        else:
            raise ValueError("Side must be 'left' or 'right'.")
        oriented_bbox= self.reset_obbox(cropped_image)
        return cropped_image, oriented_bbox
    
    def trim_image(self, image: Image.Image, direction: str, pixels: int) -> Image.Image:
        width, height = image.size
        if direction == 'top':
            trimmed_image = image.crop((0, pixels, width, height))
        elif direction == 'bottom':
            trimmed_image = image.crop((0, 0, width, height - pixels))
        elif direction == 'left':
            trimmed_image = image.crop((pixels, 0, width, height))
        elif direction == 'right':
            trimmed_image = image.crop((0, 0, width - pixels, height))
        else:
            raise ValueError("Direction must be 'top', 'bottom', 'left', or 'right'.")
        oriented_bbox= self.reset_obbox(trimmed_image)
        return trimmed_image, oriented_bbox
    
    def vertical_lighting(self, img: Image.Image, lighting_angle=0, lighting_strokeWidth=0, lighting_luminance=0, lighting_opacity=0):
        if(lighting_strokeWidth==0 or lighting_opacity==0):
            return img
        width, height = img.size
        # Create a blank image for the lighting effect
        lighting_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(lighting_layer)
        # Calculate the x-position for the light based on the angle
        angle_rad = np.radians(lighting_angle)
        x_center = width / 2 + np.tan(angle_rad) * (height / 2)
        # Adjust stroke width based on luminance
        lighting_strokeWidth = max(1, int(lighting_strokeWidth * lighting_luminance))
        # Draw the lighting effect
        draw.line(
            [(x_center, 0), (x_center, height)],
            fill=(255, 255, 255, int(lighting_opacity * 255)),
            width=lighting_strokeWidth
        )
        # Apply a Gaussian blur to simulate light spread
        lighting_layer = lighting_layer.filter(ImageFilter.GaussianBlur(radius=lighting_luminance))
        # Combine the lighting effect with the original image
        combined = Image.alpha_composite(img, lighting_layer)

        return combined
    
    def curve_image_y_axis(self, image: Image.Image, direction: str = 'down', curvature: float = 0.5) -> Image.Image:
        # Convert image to numpy array
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        # Set direction multiplier
        direction_multiplier = 1 if direction == 'up' else -1
        # Calculate the middle of the image (x=0 in parabolic function)
        x_mid = width // 2
        # Determine the maximum shift to pad the image
        max_shift = int(curvature * (x_mid ** 2))
        # Create a new image with padded height to accommodate maximum shift
        padded_height = height + 2 * max_shift
        new_img_array = np.zeros((padded_height, width, 4), dtype=np.uint8)
        # Iterate over each column (x-coordinate) in the image
        for x in range(width):
            # Calculate the parabolic shift for this x-coordinate
            x_offset = x - x_mid
            y_shift = int(curvature * (x_offset ** 2) * direction_multiplier)
            # Apply the same y_shift to all pixels in this column
            for y in range(height):
                new_y = y + y_shift + max_shift
                # If new position is within bounds, place the pixel in the new image array
                if 0 <= new_y < padded_height:
                    new_img_array[new_y, x] = img_array[y, x]
        # Convert back to PIL image
        new_image = Image.fromarray(new_img_array, mode='RGBA')
        if(direction == 'up'):
            new_image, oriented_bbox = self.trim_image(new_image,'top',max_shift)
        elif(direction == 'down'):
            new_image, oriented_bbox = self.trim_image(new_image,'bottom',max_shift)
        oriented_bbox = self.reset_obbox(new_image)
        return new_image, oriented_bbox
    
    def recalculate_width_height(self,image):
        width, height = image.size
        image = np.array(image)
         # Convert to grayscale and find non-zero points
        gray_warped = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        non_zero_points = cv2.findNonZero(gray_warped)
        if non_zero_points is None:
            return img  # No valid transformation, return original image
        # Compute the bounding box
        x_min, y_min = np.min(non_zero_points, axis=0).flatten()
        x_max, y_max = np.max(non_zero_points, axis=0).flatten()
        # Ensure coordinates are within bounds
        x_min, y_min = int(max(0, x_min)), int(max(0, y_min))
        x_max, y_max = int(min(width - 1, x_max)), int(min(height - 1, y_max))
        # Crop the warped image to the bounding box
        cropped_warped_image = image[y_min:y_max+1, x_min:x_max+1]
        img = Image.fromarray(cropped_warped_image)

        oriented_bbox = self.reset_obbox(img)
        return img, oriented_bbox
    #======================================== Augment function ======================
        
    def flip_horizontal(self, img):
        # Flip the image horizontally
        flipped_img = ImageOps.mirror(img)
        # Calculate the new bbox by flipping x-coordinates
        width = img.size[0]
        return flipped_img

    def flip_vertical(self, img):
        # Flip the image vertically
        flipped_img = ImageOps.flip(img)
        # Calculate the new bbox by flipping y-coordinates
        height = img.size[1]
        return flipped_img

    def resize(self, image, scale_range=(0.9, 1.5)):
        scale = np.random.uniform(scale_range[0], scale_range[1])
        new_size = (int(image.width * scale), int(image.height * scale))
        return image.resize(new_size, Image.LANCZOS)

    def set_resolution(self, image, max_resolution=(80, 80)):
        original_width, original_height = image.size
        scale = min(max_resolution[0] / original_width, max_resolution[1] / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        return image.resize((new_width, new_height), Image.LANCZOS)
    
    def set_area(self, image, max_area=40000):
        original_width, original_height = image.size
        original_area = original_width * original_height
        # Determine the scaling factor
        if original_area > max_area:
            # If the image is too large, scale down
            l, r = 0, 1.0
            while (r - l) > 1e-6:  # Precision threshold
                mid = (l + r) / 2
                new_area = (mid * original_width) * (mid * original_height)
                if new_area <= max_area:
                    l = mid
                else:
                    r = mid
            scale_factor = l
        else:
            # If the image is too small, scale up
            l, r = 1.0, 10.0  # Upper bound can be adjusted if needed
            while (r - l) > 1e-6:  # Precision threshold
                mid = (l + r) / 2
                new_area = (mid * original_width) * (mid * original_height)
                if new_area <= max_area:
                    l = mid
                else:
                    r = mid
            scale_factor = l

        # Compute new dimensions
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
        return image.resize((new_width, new_height), Image.LANCZOS)

    def noise(self, image_pil: Image.Image, min_noise_level: float = 25.0, max_noise_level: float = 50.0) -> Image.Image:
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')
        image_np = np.array(image_pil)
        rgb = image_np[:, :, :3]
        alpha = image_np[:, :, 3]
        noise_level = random.uniform(min_noise_level, max_noise_level)
        noise = np.random.normal(0, noise_level, rgb.shape)
        noisy_rgb = np.clip(rgb + noise, 0, 255).astype(np.uint8)
        noisy_image_np = np.dstack((noisy_rgb, alpha))
        noisy_image_pil = Image.fromarray(noisy_image_np, 'RGBA')
        return noisy_image_pil

    def occlusions(self, image_pil: Image.Image, occlusion_images: list, num_occlusions: int = 3) -> Image.Image:
        if image_pil.mode != 'RGBA':
            image_pil = image_pil.convert('RGBA')
        result_image = image_pil.copy()
        max_width, max_height = result_image.size
        for _ in range(num_occlusions):
            occlusion_pil = random.choice(occlusion_images)
            scale_factor = random.uniform(0.5, 1.0)
            occlusion_width, occlusion_height = occlusion_pil.size
            new_width = int(max_width * scale_factor)
            new_height = int(occlusion_height * new_width / occlusion_width)
            occlusion_pil = occlusion_pil.resize((new_width, new_height), Image.LANCZOS)
            center_x = max_width / 2
            center_y = max_height / 2
            margin = 20
            min_x = -new_width // 2
            max_x = max_width - new_width // 2
            min_y = -new_height // 2
            max_y = max_height - new_height // 2
            while True:
                x = random.randint(min_x, max_x)
                y = random.randint(min_y, max_y)
                occlusion_center_x = x + new_width / 2
                occlusion_center_y = y + new_height / 2
                if not (center_x - margin < occlusion_center_x < center_x + margin and
                        center_y - margin < occlusion_center_y < center_y + margin):
                    break
            result_image.paste(occlusion_pil, (x, y), occlusion_pil)
        return result_image

    def blur(self, pil_image, scale_factor):
        original_width, original_height = pil_image.size
        new_width = int(original_width / scale_factor)
        new_height = int(original_height / scale_factor)
        downscaled_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
        upscaled_image = downscaled_image.resize((original_width, original_height), Image.LANCZOS)
        return upscaled_image

    def perspective(self, image, max_warp=0.2):
        width, height = image.size
        pts1 = np.float32([[0,0], [width-1,0], [0,height-1], [width-1,height-1]])
        pts2 = np.float32([
            [np.random.uniform(0, max_warp * width), np.random.uniform(0, max_warp * height)],
            [width-1 - np.random.uniform(0, max_warp * width), np.random.uniform(0, max_warp * height)],
            [np.random.uniform(0, max_warp * width), height-1 - np.random.uniform(0, max_warp * height)],
            [width-1 - np.random.uniform(0, max_warp * width), height-1 - np.random.uniform(0, max_warp * height)]
        ])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Apply perspective transform to the image
        image_np = np.array(image)
        warped_image = cv2.warpPerspective(image_np, matrix, (width, height))
        # Convert to grayscale and find non-zero points
        gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        non_zero_points = cv2.findNonZero(gray_warped)
        if non_zero_points is None:
            return image  # No valid transformation, return original image
        # Compute the bounding box
        x_min, y_min = np.min(non_zero_points, axis=0).flatten()
        x_max, y_max = np.max(non_zero_points, axis=0).flatten()
        # Ensure coordinates are within bounds
        x_min, y_min = int(max(0, x_min)), int(max(0, y_min))
        x_max, y_max = int(min(width - 1, x_max)), int(min(height - 1, y_max))
        # Crop the warped image to the bounding box
        cropped_warped_image = warped_image[y_min:y_max+1, x_min:x_max+1]
        result_image = Image.fromarray(cropped_warped_image)

        return result_image
    
    def set_perspective(self, image, angle=0, direction=(0,0)):
        width, height = image.size
        if(direction[0]<-1 or direction[0]>1 or direction[1]<-1 or direction[1]>1):
            raise ValueError("[set perspective] : wrong direction input only accept value -1 <= x,y <= 1")
        if(angle<0 or angle > 80):
            raise ValueError("[set perspective] : wrong angle input only accept value 0 <= x,y <= 80")
        pts1 = np.float32([[0,0], [width-1,0], [0,height-1], [width-1,height-1]])
        
        wrapx = (width*angle)/180
        wrapy = (height*angle)/180 

        # Define destination points
        if(direction==(-1,-1) or direction==(1,1)):
            pts2 = np.float32([
                [0, 0],
                [width-1-wrapx, wrapy],
                [wrapx, height-1-wrapy],
                [width-1, height-1]
            ])
        elif(direction==(-1,1) or direction==(1,-1)):
            pts2 = np.float32([
                [wrapx, wrapy],
                [width-1, 0],
                [0, height-1],
                [width-1-wrapx, height-1-wrapy]
            ])
        elif(direction==(-1,0)):
            pts2 = np.float32([
                [wrapx, wrapy],
                [width-1-wrapx, 0],
                [wrapx, height-1-wrapy],
                [width-1-wrapx, height-1]
            ])
        elif(direction==(1,0)):
            pts2 = np.float32([
                [wrapx, 0],
                [width-1-wrapx, wrapy],
                [wrapx, height-1],
                [width-1-wrapx, height-1-wrapy]
            ])
        elif(direction==(0,1)):
            pts2 = np.float32([
                [wrapx, wrapy],
                [width-1-wrapx, wrapy],
                [0, height-1-wrapy],
                [width-1, height-1-wrapy]
            ])
        elif(direction==(0,-1)):
            pts2 = np.float32([
                [0, wrapy],
                [width-1, wrapy],
                [wrapx, height-1-wrapy],
                [width-1-wrapx, height-1-wrapy]
            ])
        elif(direction==(0,0)):
            pts2 = np.float32([
                [0, 0],
                [width-1, 0],
                [0, height-1],
                [width-1, height-1]
            ])
        
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Apply perspective transform to the image
        image_np = np.array(image)
        warped_image = cv2.warpPerspective(image_np, matrix, (width, height))
        # Convert to grayscale and find non-zero points
        gray_warped = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
        non_zero_points = cv2.findNonZero(gray_warped)
        if non_zero_points is None:
            return image  # No valid transformation, return original image
        # Compute the bounding box
        x_min, y_min = np.min(non_zero_points, axis=0).flatten()
        x_max, y_max = np.max(non_zero_points, axis=0).flatten()
        # Ensure coordinates are within bounds
        x_min, y_min = int(max(0, x_min)), int(max(0, y_min))
        x_max, y_max = int(min(width - 1, x_max)), int(min(height - 1, y_max))
        # Crop the warped image to the bounding box
        cropped_warped_image = warped_image[y_min:y_max+1, x_min:x_max+1]
        result_image = Image.fromarray(cropped_warped_image)

        return result_image
    
    def cylindrical(self, img, focal_len_x=100, focal_len_y=100, rotation_angle=0, perspective_angle=0, lighting_angle=0, lighting_strokeWidth=0, lighting_luminance=0, lighting_opacity=0):
        width, height = img.size
        centerX, centerY = width/2,height/2
        
        # cut image if rotate the cylinder shape
        if(rotation_angle!=0):
            rotation_angle%=360
            side = "right"
            if(rotation_angle>180):
                rotation_angle = 360-rotation_angle
                side = "left"
                centerX += width * (rotation_angle / 180)
            else:
                centerX -= width * (rotation_angle / 180)
            percentage = rotation_angle/180
            img, oriented_bbox = self.cut_image_width(img, percentage, side)
        img = self.vertical_lighting(img=img,lighting_angle=lighting_angle,lighting_strokeWidth=lighting_strokeWidth,lighting_luminance=lighting_luminance,lighting_opacity=lighting_opacity)
        # Convert PIL to OpenCV image
        img = self.pil_to_cv(img)
        # height, width = img.shape[:2]
        # Calculate focal lengths as a percentage of image dimensions
        focal_len_x = (focal_len_x / 100.0) * width
        focal_len_y = (focal_len_y / 100.0) * height
        # Define the intrinsic camera matrix with dynamic focal lengths
        K = np.array([[focal_len_x, 0, centerX],
                [0, focal_len_y, centerY],
                [0, 0, 1]])
        # Pixel coordinates
        y_i, x_i = np.indices((height, width))
        X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(height * width, 3)  # to homog
        Kinv = np.linalg.inv(K)
        X = Kinv.dot(X.T).T  # normalized coords
        # Calculate cylindrical coords (sin(theta), h, cos(theta))
        A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(height * width, 3)
        B = K.dot(A.T).T  # project back to image-pixels plane
        # Back from homog coords
        B = B[:, :-1] / B[:, [-1]]
        # Ensure warp coords only within image bounds
        B[(B[:, 0] < 0) | (B[:, 0] >= width) | (B[:, 1] < 0) | (B[:, 1] >= height)] = -1
        B = B.reshape(height, width, -1)
        # Convert image to RGBA
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) if img.shape[2] == 3 else img
        # Warp the image according to cylindrical coords
        warped_img = cv2.remap(img_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        # Convert back to PIL image
        img = self.cv_to_pil(warped_img)

        if(perspective_angle!=0):
            if(perspective_angle<0):
                perspective_angle= - perspective_angle
                perspective_angle%=360
                img, oriented_bbox = self.curve_image_y_axis(img, 'up', curvature=perspective_angle/7200)
            else:
                perspective_angle%=360
                img, oriented_bbox = self.curve_image_y_axis(img, 'down', curvature=perspective_angle/7200)
            
            if(perspective_angle>180):
                angle = 360 - perspective_angle
            else:
                angle = perspective_angle

            width, height = img.size
            wrapx = (width*angle)/180
            wrapy = (height*angle)/180 
            pts1 = np.float32([[0,0], [width-1,0], [0,height-1], [width-1,height-1]])
            pts2 = pts2 = np.float32([
                [0, wrapy],
                [width-1, wrapy],
                [0, height-1-wrapy],
                [width-1, height-1-wrapy]
            ])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            # Apply perspective transform to the image
            image_np = np.array(img)
            warped_image = cv2.warpPerspective(image_np, matrix, (width, height))
            # Convert to grayscale and find non-zero points
            img = Image.fromarray(warped_image)
        return img 

    def rotation(self, image, angle_range=(-60, 60)):
        angle = np.random.uniform(angle_range[0], angle_range[1])
        width, height = image.size
        angle_rad = np.radians(angle)
        cos_a = np.abs(np.cos(angle_rad))
        sin_a = np.abs(np.sin(angle_rad))
        new_width = int(width * cos_a + height * sin_a)
        new_height = int(width * sin_a + height * cos_a)
        rotated_image = image.rotate(angle, resample=Image.Resampling.BICUBIC, expand=True)
        resized_image = rotated_image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)

        original_center = np.array([width / 2, height / 2])
        rotated_center = np.array([rotated_image.width / 2, rotated_image.height / 2])

        return resized_image
        
    def stretch(self, image_pil: Image.Image, scale_range= (0.5,1.5)) -> Image.Image:
        original_width, original_height = image_pil.size
        scale_x=1.0
        scale_y=1.0
        if(scale_range[0] > scale_range[1]):
            raise ValueError("[stretch] : wrong input scale range start should < scale range end")
        
        if(random.randint(0, 1) != 0):
            scale_x = random.uniform(scale_range[0],scale_range[1])

        new_width = int(original_width * scale_x)
        new_height = int(original_height * scale_y)
        stretched_image = image_pil.resize((new_width, new_height), Image.Resampling.BILINEAR)
        return stretched_image

    def hue_color(self, img: Image.Image, brightness_range = (1.0,1.0), contrast_range = (1.0,1.0), hue_range = (0.0,0.0), saturation_range=(1.0,1.0), gamma_range=(1.0,1.0)) -> Image.Image:
        if img.mode not in ('RGB', 'RGBA'):
            raise ValueError("Image mode must be RGB or RGBA")
        alpha = None
        if img.mode == 'RGBA':
            img, alpha = img.convert('RGB'), img.getchannel('A')
        brightness_factor = random.uniform(brightness_range[0],brightness_range[1])
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(brightness_factor)
        contrast_factor = random.uniform(contrast_range[0],contrast_range[1])
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)
        hue_factor = random.uniform(hue_range[0],hue_range[1])
        saturation_factor = random.uniform(saturation_range[0],saturation_range[1])

        def adjust_hue_saturation(image, hue_factor, saturation_factor):
            hsv_image = image.convert('HSV')
            h, s, v = hsv_image.split()
            if h.mode != 'L' or s.mode != 'L' or v.mode != 'L':
                raise ValueError("HSV channels are not in the correct mode")
            enhancer = ImageEnhance.Color(hsv_image)
            s = enhancer.enhance(saturation_factor).split()[1]
            h = h.point(lambda p: (p + hue_factor * 255) % 255)
            hsv_image = Image.merge('HSV', (h, s, v))
            return hsv_image.convert('RGB')
        img = adjust_hue_saturation(img, hue_factor, saturation_factor)
        # Randomly adjust gamma
        # gamma_factor = random.uniform(0.5, 2.0)
        gamma_factor = random.uniform(gamma_range[0],gamma_range[1])
        img = ImageEnhance.Brightness(img).enhance(gamma_factor)
        # Reattach alpha channel if it was separated
        if alpha:
            img = Image.merge('RGBA', (img.convert('RGB').split() + (alpha,)))
        return img
    
    def rgb_color(self, image, target_color_range=((0, 255), (0, 255), (0, 255)), random_color_range=((0, 255), (0, 255), (0, 255))):
        if image.mode == 'RGBA':
            image = image.convert('RGBA')
        elif image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Get the image data
        pixels = image.load()
        
        # Extract the color range
        (r_min, r_max), (g_min, g_max), (b_min, b_max) = target_color_range
        new_r = random.randint(random_color_range[0][0], random_color_range[0][1])
        new_g = random.randint(random_color_range[1][0], random_color_range[1][1])
        new_b = random.randint(random_color_range[2][0], random_color_range[2][1])
        
        # Iterate through each pixel
        for i in range(image.width):
            for j in range(image.height):
                r, g, b, *alpha = pixels[i, j]
                alpha = alpha[0] if alpha else 255  # Extract alpha if available
                
                # Check if the pixel is white or nearly white (considering transparency)
                if r_min<= r <= r_max and g_min<= g <= g_max and b_min<= b <=b_max:
                    # Generate a random color within the range
                    # Set the new color, preserving alpha
                    pixels[i, j] = (new_r, new_g, new_b, alpha) if image.mode == 'RGBA' else (new_r, new_g, new_b)
        
        return image
    
    def adjust_opacity(self, pil_image: Image.Image, opacity: float) -> Image.Image:
        if pil_image.mode != 'RGBA':
            raise ValueError("Image must have 'RGBA' mode to adjust opacity.")
        # Convert image to 'RGBA' if it's not already
        pil_image = pil_image.convert('RGBA')
        # Split the image into its component bands
        r, g, b, a = pil_image.split()
        # Create an 'A' band with the new opacity level
        alpha = a.point(lambda p: int(p * opacity))
        # Merge the bands back together
        pil_image = Image.merge('RGBA', (r, g, b, alpha))
        return pil_image
    
    def adjust_background_opacity(self, pil_image: Image.Image, rgb_color: tuple, background_opacity: float) -> Image.Image:
        if pil_image.mode != 'RGBA':
            pil_image = pil_image.convert('RGBA')
        # Split the image into its component bands
        r, g, b, a = pil_image.split()
        # Create a new image filled with the specified RGB color and the desired background opacity
        background = Image.new('RGBA', pil_image.size, (*rgb_color, int(255 * background_opacity)))
        # Composite the background and the original image
        result_image = Image.alpha_composite(background, pil_image)

        return result_image