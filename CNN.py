import numpy as np
import pandas as pd 
from IPython.core.display import display,Image






class initialize:

    def __init__(self, filter_n):
        self.num_filters = num_filters


        self.filters = np.random.randn(num_filters, 3, 3) / 9

    def iteration(self, sample):
        # generates all possible 3*3 image regions using valid padding

        dim1, dim2 = image.shape

        for i in range(dim1 - 2):
            for j in range(dim2 - 2):
                image_reg = image[i:(i + 3), j:(j + 3)]
                yield image_reg, i, j

    def ForwardPropagation(self, t_data):
        self.last_input = input

        dim1, dim2 = input.shape

        res = np.zeros((dim1 - 2, dim2 - 2, self.num_filters))

        for image_reg, i, j in self.iterate_regions(input):
            res[i, j] = np.sum(image_reg * self.filters, axis=(1, 2))
        return res

    def BackwardPropagtion(self, d_l_d_out, learn_rate):

        d_l_d_filters = np.zeros(self.filters.shape)

        for im_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.num_filters):
                d_l_d_filters[f] += d_l_d_out[i, j, f] * im_region

        # update filters
        self.filters -= learn_rate * d_l_d_filters

        return None




        def maxPool(self, sample):
            dim1, dim2, _ = sample.shape

            resize1 = dim1 // 2
            resize2 = dim2 // 2

            for i in range(resize1):
                for j in range(resize2):
                    image_reg = sample[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                    yield image_reg, i, j


