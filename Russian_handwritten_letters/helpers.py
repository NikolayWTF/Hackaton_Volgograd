from functools import reduce
from PIL import ImageChops
from PIL import Image
import math, operator


def sliding_window(image, stepSize, windowSize):

	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):

			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# def rmsdiff(im1, im2):
#     "Calculate the root-mean-square difference between two images"
#
#     h = ImageChops.difference(im1, im2).histogram()
#
#     # calculate rms
#     print(math.sqrt(reduce(operator.add,
#         map(lambda h, i: h*(i**2), h, range(256))
#     ) / (float(im1.size[0]) * im1.size[1])))
#
# im1 = Image.open("picture.png")
# mask1 = Image.open("maskA.png")
# mask2 = Image.open("maskSHA.png")
# im1 = im1.convert('1')
# mask1 = mask1.convert('1')
# mask2 = mask2.convert('1')
#
# rmsdiff(im1, mask1)
# rmsdiff(im1, mask2)

