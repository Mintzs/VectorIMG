# imports needed for the assignment
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt

class ImageProcessor():

    def __init__(self):
        self.image = None
        self.color_map = None
        self.current_mode = None

    def is_RGB_mode(self):

        """
        This Method returns the boolean value of the current mode (RGB or CM).
        """
        return self.current_mode

    def get_color_map(self):

        """
        This Method returns the color map.
        """
        return self.color_map

    def get_array(self):

        """
        This Method returns the image array.
        """
        return self.image

    def shape(self):

        """
        This Method returns the height and width of the image array.
        """
        if self.image is None:
            raise ValueError("No image loaded!")
        return (self.image.shape[0], self.image.shape[1])

    def load(self, filepath):

        """
        This Method loads an image from a file in the specified format (PNG or PKL).
        """
        extension = filepath.rsplit(".", 1)[-1]  # Splits the filepath at the last "." and takes the ext

        if extension.lower() == "png":
            img_pil = Image.open(filepath).convert("RGB")
            self.image = np.array(img_pil, dtype=np.uint8)
            self.current_mode = True
            self.color_map = None  # Reset color_map for RGB images in case it was set before
        elif extension.lower() == "pkl":
            with open(filepath, "rb") as file:  # "rb" since pickle files are binary
                data = pickle.load(file)
                self.image, self.color_map = data
            self.current_mode = False
        else:
            raise ValueError("No image loaded!")

    def extension_check(self, filepath):

        """
        This helper function checks if the specified "saving" filepath has an extension.
        If it doesn't, it adds the appropriate extension based on the current mode (RGB or CM).

        Method returns the filepath with the appropriate extension.
        """
        parts = filepath.rsplit(".", 1)
        if len(parts) == 1 or "/" in parts[-1] or "\\" in parts[-1]:
            if self.current_mode:
                filepath = filepath + ".png"
            else:
                filepath = filepath + ".pkl"
        return filepath

    def save(self, filepath):

        """
        This Method saves the image to a file in the specified format.
        """
        if self.image is None:
            raise ValueError("No image loaded!")

        # Check if filepath to save image to has an extension
        filepath = self.extension_check(filepath)
        extension = filepath.rsplit(".", 1)[-1]
        if extension.lower() == "png":
            pil_image = Image.fromarray(self.image)  # Reconstructs the image from the array
            pil_image.save(filepath)
        elif extension.lower() == "pkl":
            with open(filepath, "wb") as file:  # "wb" since pickle files are binary
                data = (self.image, self.color_map)  # Save as tuple
                pickle.dump(data, file)

    def RGB_to_CM_helper(self, bins: int = 2):

        """
        This Method converts an RGB image to a color map image by binning the RGB values into groups
        determined by the bins parameter. It then collects only unique RGB row combinations (non-repeating rows)
        and calculates the average color for each color bin by averaging the values of all rows that fall into that bin,
        and uses the average color as the color code for that bin in the new color map.

        The method then updates the image array with the new array of unique IDs that are assigned to the new color map.
        It returns nothing.
        """
        max_groups = bins ** 3
        # Choose smallest integer type for memory efficiency
        dtype = np.uint8 if max_groups <= 256 else (np.uint16 if max_groups <= 65536 else np.uint32)

        # Turns each channel value into a bin number by doing (color value / bin size)
        binned_image = (self.image // (256 / bins)).astype(dtype)
        height, width = self.image.shape[0], self.image.shape[1]
        # Reshapes the image into a 2D array where each row is a pixel/RGB (-1 is used for dynamic sizing in .reshape)
        pixels = self.image.reshape(-1, 3)
        binned_pixels = binned_image.reshape(-1, 3)

        # Collects only unique RGB rows (determined by bin number) and creates an array of indices that
        # assigns each original RGB row to its corresponding unique RGB combination in unique_binned
        unique_binned, unique_id = np.unique(binned_pixels, axis=0, return_inverse=True)
        cm_image = unique_id.reshape(height, width).astype(dtype)
        color_map = {}

        # For every unique RGB row combination, calculate the average color value of all pixels in that bin
        for group_id in range(len(unique_binned)):
            avg_color = pixels[unique_id == group_id].mean(axis=0)
            # Normalizes average RGB value to be between 0-1 according to standards of PKL for CodeGrade
            color_map[group_id] = np.array([avg_color[0] / 255.0, avg_color[1] / 255.0, avg_color[2] / 255.0])

        self.image = cm_image
        self.color_map = color_map
        self.current_mode = False

    def CM_to_RGB_helper(self):

        """
        This Method converts a color map image to an RGB image by using
        the color map to map each ID to its corresponding color code.

        Method returns nothing, but edits the image array and sets color map to None.
        """

        if self.color_map is None:
            raise ValueError("No image loaded!")

        # Turns the color map of IDs into an array of color codes,
        # where each row is the color code of an ID (i.e. row 0 = ID 0)
        CM_color_values = np.array([self.color_map[k] for k in sorted(self.color_map.keys())])

        # Check if values are normalized (0-1)
        if CM_color_values.max() <= 1.0:
            CM_color_values = (CM_color_values * 255).astype(np.uint8)  # Converted to uint8 for PNG
        else:
            CM_color_values = CM_color_values.astype(np.uint8)

        # Replaces each of the image array's ID with the corresponding color code row
        RGB_array = CM_color_values[self.image]

        self.image = RGB_array
        self.color_map = None
        self.current_mode = True

    def change_image_format(self, convert_to: bool, bins: int = 2):

        """
        This Method converts an image from RGB to CM or CM to RGB based on the convert_to parameter.
        It calls the corresponding helper function based on the current mode and the convert_to parameter.

        Method returns nothing.
        """
        if convert_to is False and self.current_mode:
            self.RGB_to_CM_helper(bins)
        elif convert_to is True and not self.current_mode:
            self.CM_to_RGB_helper()
        else:
            raise ValueError("No image loaded!")

    def rotate_colors(self):

        """
        This Method rotates the colors in the image array by 1 position.
        It calls the corresponding helper function based on the current mode.

        Method returns nothing, but edits the image array or color map depending on the current mode.
        """
        if self.current_mode:
            self.image = np.roll(self.image, shift=-1, axis=2)
        elif self.current_mode is None:
            raise ValueError("No image loaded!")
        else:
            color_id = sorted(self.color_map.keys())  # Sorted keys important if the IDs are not in order originally
            colors = [self.color_map[k] for k in color_id]
            rotated_colors = colors[-1:] + colors[:-1]  # Shifts colors by 1 by moving last color to the first index
            # Zips up the color ID with the new rotated colors as a tuple, then converts into dict
            self.color_map = dict(zip(color_id, rotated_colors))

    def blur_helper(self, radius: int, blurred_image):

        """
        This helper function is used by the blur_RGB_images method to calculate
        the average color of a pixel and its neighbors in blocks. It takes the radius
        of the block and the image array copy of zeroes as parameters.

        Method returns the blurred image array.
        """
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                top_edge = max(0, i - radius)  # Ensures row 0 is highest possible top edge
                left_edge = max(0, j - radius)
                # Adds 1 on top of radius to ensure the desired edge is included during SLICING
                bottom_edge = min(self.image.shape[0], i + radius + 1)
                right_edge = min(self.image.shape[1], j + radius + 1)

                area_submatrix = self.image[top_edge:bottom_edge, left_edge:right_edge]
                average_color = np.mean(area_submatrix, axis=(0, 1))  # Average only in x,y directions
                blurred_image[i, j] = average_color
        return blurred_image

    def blur_RGB_images(self, area=3):

        """
        This method blurs the image by calculating the average color of neighbors
        in blocks determined by the area parameter using the blur helper method.

        Method returns nothing, but edits the image array.
        """
        if self.current_mode is None:
            raise ValueError("No image loaded!")

        area = area if area % 2 == 1 else area + 1  # Ensures area edge is odd by rounding up
        radius = area // 2  # Radius from pixel to edge of area
        # Creates a copy of zeroes so calculating average color will include blurred values
        blurred_image = np.zeros_like(self.image)

        self.blur_helper(radius, blurred_image)
        self.image = blurred_image

    def common_value(self, pixelated_block):

        """
        This helper method calculates the common value of a block of pixels.
        It returns the mean of the block if the image is in RGB mode,
        otherwise it returns the mode color ID for color map mode. It takes
        a block array of the image array as a parameter.

        Method returns the common value.
        """
        if self.current_mode:
            common_value = pixelated_block.mean(axis=(0, 1)).astype(np.uint8)
        else:
            flat_block = pixelated_block.flatten()
            # Returns the mode color ID by counting (bincount) the IDs appearance as a list of frequencies
            # then returns the highest frequency ID (argmax) as the mode color
            common_value = np.bincount(flat_block).argmax()

        return common_value

    def pixelate_images(self, area: tuple[tuple[int, int], tuple[int, int]], block_size=10):

        """
        This method pixelates the image by assigning a specific section of the image array
        to the array's common value calculated by the helper function.

        Method returns nothing, but edits the image array.
        """
        if self.image is None:
            raise ValueError("No image loaded!")

        (xmin, xmax), (ymin, ymax) = area
        for y in range(ymin, ymax, block_size):  # Block size as step
            for x in range(xmin, xmax, block_size):
                # Provide dynamic shaping if box is against a corner and not full block size
                y_end = min(y + block_size, ymax)
                x_end = min(x + block_size, xmax)
                block = self.image[y:y_end, x:x_end]

                # Apply pixelation to this block
                self.image[y:y_end, x:x_end] = self.common_value(block)

    def show(self, filename=None):

        """
        This shows the images or saves the image if an filename is given.
        This works for both image formats.
        """
        if self.is_RGB_mode():
            img = self.get_array()
        else:
            img = np.vectorize(self.get_color_map().get, signature='()->(n)')(self.get_array())

        plt.imshow(img, interpolation='none')
        plt.axis('off')
        if filename is not None:
            plt.savefig(filename + ".png", bbox_inches='tight', pad_inches=0)
        else:
            plt.show()

if __name__ == "__main__":
    pass
